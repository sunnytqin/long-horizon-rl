# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-turn code-refinement agent loop with a MODEL-written "user" turn.

A variant of ``code_refine_agent.CodeRefineAgentLoop``. The structure is identical
(solver writes code, an oracle env grades it against ground-truth tests, binary
outcome reward), with ONE difference in the between-turns feedback:

  oracle loop : the failing GT cases are injected verbatim and the SOLVER is asked
                to reflect on them (3 bullets) before rewriting.
  this loop   : a second inference call -- the SAME policy run as a "user model" --
                reads (problem, failed code, failing cases) and writes the 3-bullet
                diagnosis. ONLY that diagnosis is injected as the next user turn; the
                raw failing cases are not shown to the solver.

Training is still SOLVER-ONLY. The user-model diagnosis is injected with mask=0 (no
gradient), exactly like the deterministic oracle feedback. The feedback inference is
a separate, throwaway sequence (its own request_id): its prompt/response tokens never
enter the solver trajectory and its weights-version is not tracked (masked tokens
carry no gradient, so staleness is irrelevant to them).

No new config knobs: the feedback call reuses the solver ``sampling_params`` (same
temperature/top_p) and the existing per-turn length cap ``max_new_tokens_per_turn``.
"""

import asyncio
import logging
import os
import re
from typing import Any
from uuid import uuid4

from codecontest import templates
from codecontest.env import GTOracleEnv
from codecontest.masking import TRAIN_TURNS_MODES, apply_train_turns_mask
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Same debug gate as env.py: when set, dump a head preview of each model-written
# diagnosis so we can eyeball what the "user" is actually saying.
_DEBUG_FEEDBACK = bool(int(os.getenv("CODECONTEST_DEBUG_FEEDBACK", "0") or "0"))
_DEBUG_FEEDBACK_PREVIEW = int(os.getenv("CODECONTEST_DEBUG_FEEDBACK_PREVIEW", "200") or "200")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Conservative chars/token ratio for the digit/whitespace-heavy CodeContests I/O; used to
# size char budgets from the token-based prompt_length cap.
_CHARS_PER_TOKEN = 3.0


@register("model_feedback_agent")
class ModelFeedbackAgentLoop(AgentLoopBase):
    """Multi-turn code-refinement loop where the feedback turn is written by the policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Number of assistant (solver) turns: turn 0 + refinements. Default 3.
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns or 3

        # CodeContests-specific knobs. Read from an optional `codecontest` config
        # block (add via `+codecontest.key=value` on the CLI); fall back to defaults.
        cc = {}
        try:
            cc = self.config.get("codecontest", {}) or {}
        except Exception:  # noqa: BLE001 - config may not define the block
            cc = {}
        # Per-turn generation cap (tokens). None -> use remaining response budget. The
        # user-model feedback call reuses this same cap (no separate knob).
        self.max_new_tokens_per_turn = cc.get("max_new_tokens_per_turn", None)
        # Overflow handling: "end_zero_reward" (default) or "discard_sample".
        self.on_overflow = cc.get("on_overflow", "end_zero_reward")
        # Env defaults (per-sample values in extra_info take precedence).
        self.default_exec_timeout = float(cc.get("exec_timeout", 6.0))
        self.default_max_failures_shown = int(cc.get("max_failures_shown", 3))
        self.default_max_gt_test = int(cc.get("max_gt_test", 20))
        # Combined char budget for the failing-case fields inside the FEEDBACK MODEL
        # PROMPT (problem + code + failures). Because those failures now live only in the
        # user model's single-turn prompt -- NOT the solver's cumulative conversation --
        # this can be set generously; the only ceiling is that problem+code+failures must
        # still fit prompt_length (else the agent loop left-truncates the problem). Derived
        # from prompt_length by default; pin an absolute value via +codecontest.max_feedback_chars.
        _FEEDBACK_BUDGET_FRACTION = 0.5
        _derived_max_feedback_chars = int(self.prompt_length * _CHARS_PER_TOKEN * _FEEDBACK_BUDGET_FRACTION)
        _cfg_max_feedback_chars = int(cc.get("max_feedback_chars", 0) or 0)
        self.default_max_feedback_chars = (
            _cfg_max_feedback_chars if _cfg_max_feedback_chars > 0 else _derived_max_feedback_chars
        )
        # Max NEW tokens the user model may generate for its diagnosis. Exact, hard bound on
        # the diagnosis length: the injected solver turn is the fixed skeleton (~50 tokens)
        # plus <= this, so it stays far under prompt_length and the agent loop never left-
        # truncates it. The cap only bites when the model WOULD write more (diagnoses are
        # prompted "Be concise", so it usually won't). IMPORTANT: the diagnosis lands in the
        # solver's response tail, so it shares response_length with ALL solver code turns +
        # all feedback turns. With response_length=8192 and up to 3 feedback turns, 2048 each
        # already reserves 6144 for feedback -- push higher only if you also raise
        # MAX_RESPONSE_LENGTH, else the solver starves and overflows (unsolved => reward 0).
        # Watch feedback_resp_len_mean (logged) for actual usage. Pin via
        # +codecontest.max_feedback_tokens.
        self.max_feedback_tokens = int(cc.get("max_feedback_tokens", 2048))
        # Hard wall on a single env.step (code grading). See code_refine_agent.
        self.env_step_timeout = float(cc.get("env_step_timeout", 180.0))
        # SET 2 gradient-masking study: which solver turns get trained. "all" (default)
        # reproduces prior behavior; "final_only" trains only the last solver turn (clean
        # credit -- see codecontest/masking.py for why "refinement_only" was dropped). The
        # masked user-model feedback turns are unaffected -- they are already mask=0.
        self.train_turns = cc.get("train_turns", "all")
        if self.train_turns not in TRAIN_TURNS_MODES:
            raise ValueError(f"codecontest.train_turns must be one of {TRAIN_TURNS_MODES}, got {self.train_turns!r}")

    async def _tokenize_uncapped(self, messages: list[dict]) -> list[int]:
        """Tokenize a chat-message list WITHOUT the solver's prompt_length cap.

        ``AgentLoopBase.apply_chat_template`` left-truncates any prompt over
        ``rollout.prompt_length`` -- correct for the solver's trajectory, wrong for the
        user model's throwaway feedback prompt (it would drop the problem statement). This
        variant returns the full ids; the caller bounds generation so prompt+gen fit the
        engine context. Text-only path (``self.processor`` is None for this model).
        """
        ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True),
        )
        return list(ids)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}

        # Problem text for the user-model prompt: the initial solver user turn (it already
        # wraps the problem statement via CODE_PROMPT_TEMPLATE). Captured once up front.
        problem_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )

        # Ground-truth tests: prefer extra_info, fall back to reward_model.ground_truth.
        gt = extra_info.get("ground_truth")
        if gt is None:
            gt = (kwargs.get("reward_model", {}) or {}).get("ground_truth", {})
        env = GTOracleEnv(
            test_input=list(gt["test_input"]),
            test_output=list(gt["test_output"]),
            test_time_limit=float(gt.get("test_time_limit", self.default_exec_timeout)),
            max_failures_shown=int(extra_info.get("max_failures_shown", self.default_max_failures_shown)),
            max_gt_test=int(extra_info.get("max_gt_test", self.default_max_gt_test)),
            max_feedback_chars=int(extra_info.get("max_feedback_chars", self.default_max_feedback_chars)),
            seed=int(kwargs.get("index", 0)),
        )

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self.apply_chat_template(messages)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        track_logprobs = True

        assistant_turns = 0
        user_turns = 0
        solved = False
        overflow = False
        solved_at_turn = -1
        # True solver-generated length per assistant turn, captured at generation time.
        solver_turn_lengths: list[int] = []
        # (start, end) span of each solver turn within response_mask, for the SET 2
        # gradient-masking policy applied after the loop (see codecontest/masking.py).
        solver_turn_spans: list[tuple[int, int]] = []
        # User-model diagnosis length per feedback turn, and degeneracy counters.
        feedback_turn_lengths: list[int] = []
        feedback_empty = 0
        feedback_overflow = 0

        # Off-policy staleness bookkeeping the trainer requires. ONLY the solver-turn
        # outputs update this -- the feedback turns are masked, so their weights-version
        # is irrelevant.
        min_global_steps = None
        max_global_steps = None

        for turn in range(self.max_assistant_turns):
            remaining = self.response_length - len(response_mask)
            if remaining <= 0:
                overflow = True
                break

            # Bound this turn's generation so one turn can't consume the whole budget.
            cap = remaining if self.max_new_tokens_per_turn is None else min(self.max_new_tokens_per_turn, remaining)
            turn_sampling_params = {**sampling_params, "max_new_tokens": cap}

            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_sampling_params,
                    image_data=None,
                    video_data=None,
                    audio_data=None,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

            # Track weights-version span across turns (oldest stays, freshest advances).
            turn_min = output.extra_fields.get("min_global_steps")
            turn_max = output.extra_fields.get("max_global_steps")
            if turn_min is not None and min_global_steps is None:
                min_global_steps = turn_min
            if turn_max is not None:
                max_global_steps = turn_max

            resp_ids = output.token_ids
            prompt_ids += resp_ids
            seg_start = len(response_mask)
            response_mask += [1] * len(resp_ids)
            solver_turn_spans.append((seg_start, len(response_mask)))
            solver_turn_lengths.append(len(resp_ids))
            if track_logprobs and output.log_probs:
                response_logprobs += output.log_probs
            else:
                track_logprobs = False

            assistant_turns += 1

            # Grade only the latest submission (env extracts the last ```python block).
            assistant_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            with simple_timer("env_step", metrics):
                try:
                    step = await asyncio.wait_for(
                        self.loop.run_in_executor(None, env.step, assistant_text),
                        timeout=self.env_step_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("env.step exceeded %.0fs; ending trajectory unsolved", self.env_step_timeout)
                    metrics["env_step_timeout"] = 1
                    break

            if step.solved:
                solved = True
                solved_at_turn = turn
                break

            # Last allowed turn: stop without injecting feedback we cannot act on.
            if turn == self.max_assistant_turns - 1:
                break

            # ── Build the user turn: a MODEL-written diagnosis (this is the only
            # ── difference from the oracle loop). When there is no parseable code to
            # ── diagnose (no failing cases surfaced), fall back to the env's fixed
            # ── "no code block" message rather than prompting the user model on nothing.
            if step.failures:
                fb_messages = templates.build_feedback_model_messages(
                    step.failures,
                    problem=problem_text,
                    code=step.code,
                    max_total_chars=(self.default_max_feedback_chars or None),
                )
                # The user-model prompt is a throwaway inference on a SEPARATE sequence, so
                # it does NOT go through the solver's prompt_length cap (which would left-
                # truncate the problem). Tokenize it uncapped; its only real ceiling is the
                # SGLang engine context window (prompt_length + response_length).
                fb_prompt_ids = await self._tokenize_uncapped(fb_messages)
                # Cap the diagnosis at max_feedback_tokens (exact token bound => the injected
                # solver turn is skeleton + <= this, well under prompt_length). Never exceed
                # the remaining solver response budget (the diagnosis lands in the response
                # tail) nor overflow the engine context on this generate call.
                engine_ctx = self.prompt_length + self.response_length
                fb_remaining = self.response_length - len(response_mask)
                fb_cap = min(self.max_feedback_tokens, fb_remaining, max(1, engine_ctx - len(fb_prompt_ids)))
                fb_sampling_params = {**sampling_params, "max_new_tokens": max(1, fb_cap)}
                with simple_timer("generate_feedback", metrics):
                    fb_output: TokenOutput = await self.server_manager.generate(
                        request_id=uuid4().hex,  # throwaway: not part of the solver sequence
                        prompt_ids=fb_prompt_ids,
                        sampling_params=fb_sampling_params,
                        image_data=None,
                        video_data=None,
                        audio_data=None,
                    )
                feedback_turn_lengths.append(len(fb_output.token_ids))
                analysis = self.tokenizer.decode(fb_output.token_ids, skip_special_tokens=True)
                analysis = _THINK_BLOCK.sub("", analysis).strip()
                if not analysis:
                    feedback_empty += 1
                    analysis = (
                        "Your previous solution failed some test cases. Reconsider your "
                        "approach and edge cases, then write a corrected solution."
                    )
                if _DEBUG_FEEDBACK:
                    # Distinct tag from env.py's [FEEDBACK_DBG] failing-case dump so the
                    # user-model diagnosis is unambiguous in the logs. This is the text the
                    # solver actually sees next turn.
                    n = _DEBUG_FEEDBACK_PREVIEW
                    logger.warning(
                        "[MODELFB_DBG] turn=%d n_failures=%d analysis_chars=%d analysis[:%d]=%r",
                        turn, len(step.failures), len(analysis), n, analysis[:n],
                    )
                user_content = templates.build_model_feedback_user_message(analysis)
            else:
                # No code / nothing to diagnose: use the env's fixed instruction verbatim.
                user_content = step.feedback

            feedback_ids = await self.apply_chat_template(
                [{"role": "user", "content": user_content}],
                remove_system_prompt=True,
            )
            # Need room for the feedback AND at least one response token next turn.
            if len(response_mask) + len(feedback_ids) >= self.response_length:
                overflow = True
                feedback_overflow += 1
                break
            prompt_ids += feedback_ids
            response_mask += [0] * len(feedback_ids)
            if track_logprobs:
                response_logprobs += [0.0] * len(feedback_ids)
            user_turns += 1

        # Binary outcome reward. Overflow while unsolved -> 0.
        reward = 1.0 if solved else 0.0

        # SET 2: restrict the training loss to the selected solver turns (no-op for
        # the "all" default). Applied before truncation so the spans stay valid.
        apply_train_turns_mask(response_mask, solver_turn_spans, self.train_turns)

        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids_out = prompt_ids[: len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids_out,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if track_logprobs and response_logprobs else None,
            reward_score=reward,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "min_global_steps": min_global_steps,
                "max_global_steps": max_global_steps,
                "solved": solved,
                "solved_at_turn": solved_at_turn,
                "num_assistant_turns": assistant_turns,
                "overflow": overflow,
                "solver_resp_len_mean": (
                    sum(solver_turn_lengths) / len(solver_turn_lengths) if solver_turn_lengths else 0.0
                ),
                # Per-conversation mean user-model diagnosis length (tokens). Averaged by
                # the trainer across conversations. 0 when no feedback turn ran.
                "feedback_resp_len_mean": (
                    sum(feedback_turn_lengths) / len(feedback_turn_lengths) if feedback_turn_lengths else 0.0
                ),
                # Degeneracy guards: empty diagnoses (user said nothing) and feedback
                # turns that overflowed the response budget.
                "feedback_empty": feedback_empty,
                "feedback_overflow": feedback_overflow,
            },
        )
        return output

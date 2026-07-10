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
"""ColBench multi-turn agent loop: solver vs. a FROZEN user simulator.

Structurally a sibling of ``codecontest.model_feedback_agent.ModelFeedbackAgentLoop`` (same
budget/overflow handling, weights-version bookkeeping, response-mask construction,
``AgentLoopOutput`` assembly, and ``masking.apply_train_turns_mask`` for the SET-2 study).
The differences:

  * The between-turns user turn is produced EVERY non-terminal turn (not only on failure) by
    ``env.generate_user_turn`` -- an HTTP call to the frozen sim server, NOT
    ``server_manager.generate`` (that is the Phase-2 co-training swap point). It is injected
    with mask=0 (no gradient); the hidden GT never enters the solver trajectory.
  * There is no mid-turn grading: the solver interacts to extract requirements and submits
    once (``I WANT TO ANSWER:``); the code is graded ONCE at the end and the trajectory
    reward is the fractional GT pass-rate.

Training is SOLVER-ONLY (Phase 1). The simulator is frozen and its turns are masked.
"""

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from colbench import templates
from colbench.env import ColBenchUserSimEnv
from codecontest.masking import TRAIN_TURNS_MODES, apply_train_turns_mask
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Conversation debug: dump the first N trajectories' full dialogues (see env.py for the sim
# side). Logged at WARNING so it survives the Ray rollout workers (print() is swallowed).
_DEBUG_CONVO = bool(int(os.getenv("COLBENCH_DEBUG_CONVO", "0") or "0"))
_DEBUG_CONVO_N = int(os.getenv("COLBENCH_DEBUG_CONVO_N", "3") or "3")
_DEBUG_PREVIEW = int(os.getenv("COLBENCH_DEBUG_CONVO_PREVIEW", "400") or "400")


@register("colbench_agent")
class ColBenchAgentLoop(AgentLoopBase):
    """Solver rollout against a frozen ColBench user simulator (fractional GT reward)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Total solver turns (clarify + submit). ColBench default 10 (sweet_rl max_steps).
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns or 10

        # Optional `colbench` config block (add via `+colbench.key=value`); minimal knobs,
        # reusing the solver sampling_params + per-turn cap rather than adding many.
        cc = {}
        try:
            cc = self.config.get("colbench", {}) or {}
        except Exception:  # noqa: BLE001 - config may not define the block
            cc = {}
        # Per-turn solver generation cap (tokens). None -> use remaining response budget.
        self.max_new_tokens_per_turn = cc.get("max_new_tokens_per_turn", None)
        # Hard wall (seconds) on a single blocking env call (sim HTTP turn or final grading).
        self.env_step_timeout = float(cc.get("env_step_timeout", 180.0))
        # Per-case exec timeout for the final GT grading.
        self.reward_time_limit = float(cc.get("reward_time_limit", 6.0))
        # SET 2 gradient-masking arm (shared with codecontest): "all" or "final_only". The
        # masked simulator turns are unaffected (already mask=0).
        self.train_turns = cc.get("train_turns", "all")
        if self.train_turns not in TRAIN_TURNS_MODES:
            raise ValueError(f"colbench.train_turns must be one of {TRAIN_TURNS_MODES}, got {self.train_turns!r}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        index = int(kwargs.get("index", 0))

        # The initial (public) problem turn = the last user message of the prompt. It seeds
        # the simulator's dialogue history (sweet_rl reset()) -- it carries NO ground truth.
        problem_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )

        # Task payload: prefer extra_info.ground_truth, fall back to reward_model.ground_truth.
        gt = extra_info.get("ground_truth")
        if gt is None:
            gt = (kwargs.get("reward_model", {}) or {}).get("ground_truth", {})
        env = ColBenchUserSimEnv(
            problem_description=gt.get("problem_description", problem_text),
            ground_truth=gt["ground_truth"],
            test_cases=list(gt.get("test_cases", []) or []),
            max_steps=self.max_assistant_turns,
            reward_time_limit=self.reward_time_limit,
        )

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self.apply_chat_template(messages)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        track_logprobs = True

        # Running dialogue for the SIMULATOR prompt (problem + solver turns + user replies).
        # Contains no GT; the GT is injected only inside env.generate_user_turn.
        sim_dialogue: list[dict] = [{"role": "user", "content": problem_text}]

        assistant_turns = 0
        user_turns = 0
        answered = False
        overflow = False
        answered_at_turn = -1
        reward = 0.0
        result: dict[str, Any] = {}
        answer_text = ""
        solver_turn_lengths: list[int] = []
        solver_turn_spans: list[tuple[int, int]] = []

        # Off-policy staleness bookkeeping the trainer requires. Only the solver turns update
        # this -- the simulator turns are masked, so their weights-version is irrelevant.
        min_global_steps = None
        max_global_steps = None

        for turn in range(self.max_assistant_turns):
            remaining = self.response_length - len(response_mask)
            if remaining <= 0:
                overflow = True
                break

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

            assistant_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            sim_dialogue.append({"role": "assistant", "content": assistant_text})

            is_last = turn == self.max_assistant_turns - 1
            has_answer, ans = env.is_answer(assistant_text, episode_done=is_last)

            if has_answer:
                answered = True
                answered_at_turn = turn
                answer_text = ans
                with simple_timer("env_score", metrics):
                    try:
                        result = await asyncio.wait_for(
                            self.loop.run_in_executor(None, env.score, ans),
                            timeout=self.env_step_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("env.score exceeded %.0fs; grading as 0", self.env_step_timeout)
                        metrics["env_score_timeout"] = 1
                        result = {"pass_rate": 0.0, "all_pass": False, "per_case": [], "n": 0}
                reward = float(result.get("pass_rate", 0.0))
                break

            # No answer yet. On the last allowed turn, stop without injecting a reply we can't
            # act on (reward stays 0 -- the solver never submitted anything usable).
            if is_last:
                break

            # ── The user (simulator) turn. THE Phase-1/Phase-2 seam. Blocking HTTP -> run in
            # ── the executor with a hard timeout so a hung sim can't stall the rollout.
            with simple_timer("generate_user_turn", metrics):
                try:
                    user_content = await asyncio.wait_for(
                        self.loop.run_in_executor(None, env.generate_user_turn, list(sim_dialogue)),
                        timeout=self.env_step_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("generate_user_turn exceeded %.0fs; ending trajectory", self.env_step_timeout)
                    metrics["sim_turn_timeout"] = 1
                    break
            sim_dialogue.append({"role": "user", "content": user_content})

            feedback_ids = await self.apply_chat_template(
                [{"role": "user", "content": user_content}],
                remove_system_prompt=True,
            )
            # Need room for the user turn AND at least one response token next turn.
            if len(response_mask) + len(feedback_ids) >= self.response_length:
                overflow = True
                break
            prompt_ids += feedback_ids
            response_mask += [0] * len(feedback_ids)
            if track_logprobs:
                response_logprobs += [0.0] * len(feedback_ids)
            user_turns += 1

        # SET 2: restrict the loss to the selected solver turns ("all" default = no-op).
        apply_train_turns_mask(response_mask, solver_turn_spans, self.train_turns)

        if _DEBUG_CONVO and index < _DEBUG_CONVO_N:
            self._dump_conversation(index, problem_text, sim_dialogue, answered, answer_text, reward, result)

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
                "answered": answered,
                "answered_at_turn": answered_at_turn,
                "num_assistant_turns": assistant_turns,
                "overflow": overflow,
                "pass_rate": reward,
                "all_pass": bool(result.get("all_pass", False)),
                "num_test_cases": int(result.get("n", 0)),
                "solver_resp_len_mean": (
                    sum(solver_turn_lengths) / len(solver_turn_lengths) if solver_turn_lengths else 0.0
                ),
            },
        )
        return output

    def _dump_conversation(self, index, problem_text, sim_dialogue, answered, answer_text, reward, result):
        """Dump one full trajectory for manual inspection (COLBENCH_DEBUG_CONVO)."""
        n = _DEBUG_PREVIEW
        logger.warning("[COLBENCH_CONVO] ===== trajectory index=%d =====", index)
        logger.warning("[COLBENCH_CONVO] problem[:%d]=%r", n, str(problem_text)[:n])
        for i, m in enumerate(sim_dialogue):
            logger.warning("[COLBENCH_CONVO] turn=%d role=%s content[:%d]=%r", i, m.get("role"), n, str(m.get("content"))[:n])
        logger.warning("[COLBENCH_CONVO] answered=%s answer[:%d]=%r", answered, n, str(answer_text)[:n])
        logger.warning(
            "[COLBENCH_CONVO] pass_rate=%.3f all_pass=%s per_case=%s n=%d",
            reward, result.get("all_pass"), result.get("per_case"), result.get("n", 0),
        )

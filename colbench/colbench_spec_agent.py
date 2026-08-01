"""ColBench SPEC-path multi-turn agent loop: solver vs. a SPEC-conditioned frozen user sim.

Sibling of ``colbench.colbench_agent.ColBenchAgentLoop`` (same budget/overflow bookkeeping,
weights-version handling, response-mask construction, ``AgentLoopOutput`` assembly, and
``masking.apply_train_turns_mask`` for the SET-2 study). The one thing that differs is the
TERMINATION STATE MACHINE and the env:

  * The frozen user-simulator is conditioned on a natural-language **spec**
    (``persona/scenario/requirements/plot``) instead of the hidden GT source, so a code leak is
    structurally impossible -- there is NO rejection-sampling-for-leaks / ``is_answer`` machinery
    here. The env is ``ColBenchSpecUserSimEnv``; the sim never sees GT.
  * The solver does NOT "submit" with a marker. It proposes a function inside a ```python block
    and the USER ends the conversation with ``[TERMINATE]``. The loop grades the LAST function the
    solver showed (``templates.extract_last_code``). Reward is 0 iff the solver never showed code.
  * The env still owns a small rejection sampler for the OPPOSITE leak (an ordinary user must never
    paste code): ``env.generate_user_turn`` re-queries the sim up to ``sim_max_tries`` if the reply
    contains a code fence, and flags ``last_sim_code_reject_exhausted`` if every try wrote code ->
    the loop aborts that conversation (``terminated_by="sim_code_reject"``).

Reference contract (MUST stay byte-identical): ``validate_colbench_spec.run_eval`` and the pinned
``tests/test_env_spec.py::drive``. Per turn, per active trajectory:
  1. Solver generates. If it contains code: record last_code, code_proposals += 1, showed_code.
  2. Turn cap (last turn) -> stop; grade last_code (terminated_by "turn_cap", or "no_code").
  3. Code cap (code_proposals >= max_code_proposals, default 2) -> stop; grade (terminated_by "code_cap").
  4. Else the sim replies:
       - exhausted (all tries wrote code) -> stop; terminated_by "sim_code_reject"; grade last_code.
       - elif sim_terminated -> stop; terminated_by "user" (or "no_code"); grade last_code.
       - else append the sim reply (mask=0) and continue.

Training is SOLVER-ONLY. The simulator is frozen and its turns are masked.
"""

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from codecontest.masking import TRAIN_TURNS_MODES, apply_train_turns_mask
from colbench import templates
from colbench.env_spec import ColBenchSpecUserSimEnv
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Conversation debug: dump the first N trajectories' full dialogues. Logged at WARNING so it
# survives the Ray rollout workers (print() is swallowed). Mirrors colbench_agent.
_DEBUG_CONVO = bool(int(os.getenv("COLBENCH_DEBUG_CONVO", "0") or "0"))
_DEBUG_CONVO_N = int(os.getenv("COLBENCH_DEBUG_CONVO_N", "3") or "3")
_DEBUG_PREVIEW = int(os.getenv("COLBENCH_DEBUG_CONVO_PREVIEW", "400") or "400")


@register("colbench_spec_agent")
class ColBenchSpecAgentLoop(AgentLoopBase):
    """Solver rollout against a SPEC-conditioned frozen ColBench user sim (fractional GT reward)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Total solver turns (clarify + propose). ColBench default 10 (sweet_rl max_steps).
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
        # SET 2 gradient-masking arm (shared with codecontest). Default "all" (train every solver
        # turn) is the first-run setting and a no-op in apply_train_turns_mask. "upto_last_code"
        # (Intervention 2) keeps turns [0 .. last-code-turn] and zeros the trailing post-code turns;
        # it is the spec-aware mode that consumes the last-code-turn index the loop records.
        # "final_only" (keep ONLY the last solver turn) is still NOT supported on the spec path:
        # unlike the GT path (where the loop breaks on submit so the last solver turn IS the graded
        # one), the spec sim can [TERMINATE] after a NON-code clarification turn, so the last solver
        # turn may not be the graded code turn -- masking to it would train the wrong tokens. Hard-
        # error rather than silently mis-mask; use "upto_last_code" instead.
        self.train_turns = cc.get("train_turns", "all")
        if self.train_turns not in TRAIN_TURNS_MODES:
            raise ValueError(f"colbench.train_turns must be one of {TRAIN_TURNS_MODES}, got {self.train_turns!r}")
        if self.train_turns == "final_only":
            raise NotImplementedError(
                "colbench.train_turns='final_only' is not supported on the spec path yet: the spec "
                "sim can [TERMINATE] after a non-code turn, so the last solver turn may not be the "
                "graded code turn. Use 'all', or add last-code-turn masking first."
            )
        # Length penalty (Intervention 1.5; OFF by default -> Int-1 leaves reward untouched). When
        # length_penalty_coef>0, subtract coef * clip((solver_tokens - soft_cap)/soft_cap, 0, 1)
        # from the trajectory reward. solver_tokens = total tokens the SOLVER generated across turns
        # (sum of the response-mask=1 spans). Targets the ~step-300 runaway-length degeneration.
        self.length_penalty_coef = float(cc.get("length_penalty_coef", 0.0) or 0.0)
        self.length_soft_cap = float(cc.get("length_soft_cap", 2048.0) or 2048.0)
        # Guardrail: max ```python proposals before the loop force-grades the last one (default 2,
        # reduced from 3 after eval). New spec-path knob.
        self.max_code_proposals = int(cc.get("max_code_proposals", 2) or 2)
        # Sim reject-sampling budget: re-query the sim up to N times if it writes code (an ordinary
        # user never pastes a function). On exhaustion the conversation is aborted. Default 8.
        self.sim_max_tries = int(cc.get("sim_max_tries", 8) or 8)
        # Terminate-on-all-pass (TRAINING-only rollout cleanup; needs the GT oracle). Once the
        # solver's latest code passes ALL tests, end the episode immediately (never consult the sim
        # again) so the frozen sim cannot press on correct code and elicit a post-code capitulation
        # ramble that the flat reward would reinforce (the free-ride). Purely additive early-exit;
        # the shared sim env is untouched, and eval (validate_colbench_spec) runs its own loop and
        # never sees this flag. OFF by default -> baseline byte-identical.
        _tap = cc.get("terminate_on_allpass", False)
        self.terminate_on_allpass = (
            _tap if isinstance(_tap, bool) else str(_tap).strip().lower() in ("1", "true", "yes", "on")
        )
        # Binary reward: reward = 1.0 iff the graded code passes ALL tests, else 0.0 (vs the default
        # fractional pass_rate). Cleaner objective proxy + suppresses partial-credit reinforcement of
        # mediocre-code-plus-ramble. The raw fractional pass_rate is ALWAYS kept as a metric. OFF by
        # default. Independent of terminate_on_allpass (they clean complementary contamination paths).
        _br = cc.get("binary_reward", False)
        self.binary_reward = _br if isinstance(_br, bool) else str(_br).strip().lower() in ("1", "true", "yes", "on")
        # GROUNDED sim (TRAINING-only): condition the frozen sim on the hidden GT function source +
        # spec["plot"] instead of persona/scenario/requirements. Tests whether the plot mechanism
        # survives when the 4B sim has a reliable artifact to read off (the spec-conditioned sim
        # collapses ~step 300; the GT-conditioned one does not). All other spec machinery --
        # user-driven [TERMINATE], max_code_proposals, grade-last-shown-code, sim_wrote_code
        # rejection, the solver prompt, the parquet -- is unchanged. OFF by default, and eval
        # (validate_colbench_spec) never reads it, so the golden eval stays spec-conditioned.
        _gs = cc.get("grounded_sim", False)
        self.grounded_sim = _gs if isinstance(_gs, bool) else str(_gs).strip().lower() in ("1", "true", "yes", "on")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        index = int(kwargs.get("index", 0))

        # The initial (public) problem turn = the last user message of the prompt. It seeds the
        # simulator's dialogue history -- it carries NO ground truth.
        problem_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

        # Task payload: prefer extra_info.ground_truth, fall back to reward_model.ground_truth.
        gt = extra_info.get("ground_truth")
        if gt is None:
            gt = (kwargs.get("reward_model", {}) or {}).get("ground_truth", {})
        # The authored spec the sim conditions on (persona/scenario/requirements/plot). NEVER GT.
        spec = extra_info.get("spec", {}) or {}
        # verl (HF datasets) hands test_cases as a plain list; a pandas reader would give an
        # np.ndarray. Convert via an explicit None check (an ndarray would raise under `or []`).
        _tc = gt.get("test_cases")
        env = ColBenchSpecUserSimEnv(
            problem_description=gt.get("problem_description", problem_text),
            spec=spec,
            ground_truth=gt["ground_truth"],
            test_cases=list(_tc) if _tc is not None else [],
            max_steps=self.max_assistant_turns,
            reward_time_limit=self.reward_time_limit,
            sim_max_tries=self.sim_max_tries,
            grounded=self.grounded_sim,
        )

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        prompt_ids = await self.apply_chat_template(messages)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        track_logprobs = True

        # Running dialogue for the SIMULATOR prompt (problem + solver turns + user replies).
        # Contains no GT; the GT is used only inside env.score.
        sim_dialogue: list[dict] = [{"role": "user", "content": problem_text}]

        assistant_turns = 0
        user_turns = 0
        overflow = False
        # Spec-path termination bookkeeping (mirrors validate_colbench_spec / drive).
        code_proposals = 0
        showed_code = False
        last_code = ""
        # Solver-turn ordinal of the last code proposal (index into solver_turn_spans). Stays
        # None until the solver shows code; consumed by the "upto_last_code" gradient mask.
        last_code_turn_idx = None
        first_code = ""
        sim_code_rejected = 0
        # Total characters of the sim replies actually injected into the solver's context (paired
        # with user_turns to give a mean). Watches whether the SIM_MAX_TOKENS bound + the prompt's
        # brevity instruction are holding -- the spec path has no post-hoc char cap.
        sim_reply_chars_total = 0
        terminated_by = None
        reward = 0.0
        result: dict[str, Any] = {}
        # Grade cached by the terminate-on-all-pass mid-loop check (for the current last_code), so
        # the end-of-episode grading reuses it instead of re-scoring the same code. None when the
        # flag is off or the mid-loop grade timed out.
        graded_result: dict[str, Any] | None = None
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
                terminated_by = "turn_cap" if showed_code else "no_code"
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

            # 1. Did this turn propose a function? Track the grading target.
            if templates.contains_code(assistant_text):
                showed_code = True
                last_code = templates.extract_last_code(sim_dialogue)
                # This turn's span was just appended above, so its ordinal is len-1. The LAST
                # time this fires is the turn extract_last_code grades -> kept span == graded code.
                last_code_turn_idx = len(solver_turn_spans) - 1
                code_proposals += 1
                if not first_code:
                    first_code = last_code

            # 1b. Terminate-on-all-pass (TRAINING-only; needs GT oracle). Grade the code just shown;
            #     if it passes ALL tests, end the episode NOW -- before the sim replies -- so no
            #     post-code free-ride ramble is generated. Additive early-exit; OFF by default. The
            #     grade is cached in `graded_result` (for this last_code) and reused at final grading
            #     to avoid a second exec call.
            if self.terminate_on_allpass and showed_code and last_code:
                with simple_timer("env_score", metrics):
                    try:
                        graded_result = await asyncio.wait_for(
                            self.loop.run_in_executor(None, env.score, last_code),
                            timeout=self.env_step_timeout,
                        )
                    except asyncio.TimeoutError:
                        graded_result = None
                if graded_result is not None and bool(graded_result.get("all_pass", False)):
                    terminated_by = "oracle_solved"
                    break

            # 2. Turn cap: last allowed solver turn -> stop, grade the last code shown.
            if turn == self.max_assistant_turns - 1:
                terminated_by = "turn_cap" if showed_code else "no_code"
                break

            # 3. Code cap: too many proposals -> force-grade the last one.
            if code_proposals >= self.max_code_proposals:
                terminated_by = "code_cap"
                break

            # 4. Else the sim replies. env.generate_user_turn does the (no-code) rejection
            #    sampling internally; blocking HTTP -> executor + hard timeout so a hung sim can't
            #    stall the rollout.
            with simple_timer("generate_user_turn", metrics):
                try:
                    reply = await asyncio.wait_for(
                        self.loop.run_in_executor(None, env.generate_user_turn, list(sim_dialogue)),
                        timeout=self.env_step_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("generate_user_turn exceeded %.0fs; ending trajectory", self.env_step_timeout)
                    metrics["sim_turn_timeout"] = 1
                    terminated_by = "turn_cap" if showed_code else "no_code"
                    break

            sim_code_rejected += env.last_sim_code_rejected
            raw = env.last_sim_raw

            if env.last_sim_code_reject_exhausted:
                # Every retry still wrote code -> the sim can only answer this turn WITH code.
                # Abort here; grade the last function the solver showed (0 if none).
                terminated_by = "sim_code_reject"
                break

            if templates.sim_terminated(raw):
                terminated_by = "user" if showed_code else "no_code"
                break

            # Continue: inject the sim reply (mask=0, no gradient).
            sim_dialogue.append({"role": "user", "content": reply})
            feedback_ids = await self.apply_chat_template(
                [{"role": "user", "content": reply}],
                remove_system_prompt=True,
            )
            # Need room for the user turn AND at least one response token next turn.
            if len(response_mask) + len(feedback_ids) >= self.response_length:
                overflow = True
                terminated_by = "turn_cap" if showed_code else "no_code"
                break
            prompt_ids += feedback_ids
            response_mask += [0] * len(feedback_ids)
            if track_logprobs:
                response_logprobs += [0.0] * len(feedback_ids)
            user_turns += 1
            sim_reply_chars_total += len(reply)

        if terminated_by is None:  # defensive: the turn-cap branch always sets it
            terminated_by = "turn_cap" if showed_code else "no_code"

        # Grade the last function shown, once, at whatever stop. Reward 0 iff no code was shown.
        first_code_pass_rate = 0.0
        raw_pass_rate = 0.0  # raw fractional pass-rate; always kept as a metric (even under binary reward)
        if showed_code and last_code:
            if graded_result is not None:
                # Reuse the terminate-on-all-pass mid-loop grade of this same last_code.
                result = graded_result
            else:
                with simple_timer("env_score", metrics):
                    try:
                        result = await asyncio.wait_for(
                            self.loop.run_in_executor(None, env.score, last_code),
                            timeout=self.env_step_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("env.score exceeded %.0fs; grading as 0", self.env_step_timeout)
                        metrics["env_score_timeout"] = 1
                        result = {"pass_rate": 0.0, "all_pass": False, "per_case": [], "n": 0}
            raw_pass_rate = float(result.get("pass_rate", 0.0))
            # Binary reward -> 1.0 iff all tests pass, else the fractional pass_rate. pass_rate metric
            # stays raw either way (see below).
            reward = float(bool(result.get("all_pass", False))) if self.binary_reward else raw_pass_rate
            # Feedback-lift diagnostic: pass-rate of the FIRST proposal. Reuse the raw pass_rate when
            # the solver only ever showed one function (the common case) to avoid a second exec call.
            if first_code and first_code != last_code:
                try:
                    fres = await asyncio.wait_for(
                        self.loop.run_in_executor(None, env.score, first_code),
                        timeout=self.env_step_timeout,
                    )
                    first_code_pass_rate = float(fres.get("pass_rate", 0.0))
                except asyncio.TimeoutError:
                    first_code_pass_rate = 0.0
            else:
                first_code_pass_rate = raw_pass_rate

        # Intervention 1.5: length penalty on the TRAJECTORY reward (no-op when coef==0, i.e. Int-1).
        # pass_rate keeps the raw graded quality (for the pass_rate metric); reward_score is the
        # penalized training signal. solver_tokens = total SOLVER-generated tokens (response_mask=1
        # spans, before the train_turns mask). Penalty = coef*clip((tok-cap)/cap, 0, 1).
        pass_rate = raw_pass_rate
        solver_tokens = sum(solver_turn_lengths)
        length_penalty = 0.0
        if self.length_penalty_coef > 0.0 and self.length_soft_cap > 0.0:
            over = (solver_tokens - self.length_soft_cap) / self.length_soft_cap
            length_penalty = self.length_penalty_coef * max(0.0, min(1.0, over))
            reward = reward - length_penalty

        # SET 2 / Intervention 2: restrict the loss to the selected solver turns. "all" is a no-op;
        # "upto_last_code" keeps [0 .. last_code_turn_idx] and zeros the trailing post-code turns
        # (removes the reward-irrelevant post-code ramble free-ride). "final_only" is still rejected
        # by the __init__ guard on the spec path.
        apply_train_turns_mask(response_mask, solver_turn_spans, self.train_turns, last_code_idx=last_code_turn_idx)

        if _DEBUG_CONVO and index < _DEBUG_CONVO_N:
            self._dump_conversation(index, problem_text, spec, sim_dialogue, last_code, reward, terminated_by, result)

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids_out = prompt_ids[: len(prompt_ids) - len(response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids_out,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length]
            if track_logprobs and response_logprobs
            else None,
            reward_score=reward,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "min_global_steps": min_global_steps,
                "max_global_steps": max_global_steps,
                # `overflow` is consumed at top-level by verl core (-> val-aux/solve/overflow_rate).
                "overflow": overflow,
                # Outcome diagnostics. verl does NOT log arbitrary top-level extra_fields keys --
                # compute_data_metrics only aggregates a hardcoded whitelist. The one supported
                # channel is the nested `reward_extra_info` dict: verl means every key here into
                # val-core/val-aux metrics (per test_freq) and per-step rollout_data_dir dumps.
                # So every scalar we want on the dashboards MUST live here. terminated_by is a
                # string, so it is one-hot expanded into 0/1 scalars per category. Keys must be
                # identical across all rollouts (verl reads the key set from the first sample).
                "reward_extra_info": {
                    "showed_code": float(showed_code),
                    "code_proposals": float(code_proposals),
                    "sim_code_rejected": float(sim_code_rejected),
                    "first_code_pass_rate": float(first_code_pass_rate),
                    "pass_rate": float(pass_rate),
                    "length_penalty": float(length_penalty),
                    "solver_tokens": float(solver_tokens),
                    "all_pass": float(bool(result.get("all_pass", False))),
                    "num_test_cases": float(result.get("n", 0)),
                    "num_assistant_turns": float(assistant_turns),
                    "solver_resp_len_mean": (
                        sum(solver_turn_lengths) / len(solver_turn_lengths) if solver_turn_lengths else 0.0
                    ),
                    "term_user": float(terminated_by == "user"),
                    "term_no_code": float(terminated_by == "no_code"),
                    "term_turn_cap": float(terminated_by == "turn_cap"),
                    "term_code_cap": float(terminated_by == "code_cap"),
                    "term_sim_code_reject": float(terminated_by == "sim_code_reject"),
                    "term_oracle_solved": float(terminated_by == "oracle_solved"),
                    # Did the SIM end the episode on code that actually passes? Divided by term_user
                    # this is P(all_pass | user-terminated) -- the load-bearing assumption of
                    # user-driven termination, and the thing grounding the sim is meant to buy.
                    "user_term_and_allpass": float(terminated_by == "user" and bool(result.get("all_pass", False))),
                    # Mean chars per injected sim reply (0 if the sim never got to speak).
                    "sim_reply_chars": float(sim_reply_chars_total / user_turns) if user_turns else 0.0,
                },
            },
        )
        return output

    def _dump_conversation(self, index, problem_text, spec, sim_dialogue, last_code, reward, terminated_by, result):
        """Dump one full trajectory for manual inspection (COLBENCH_DEBUG_CONVO)."""
        n = _DEBUG_PREVIEW
        logger.warning("[COLBENCH_SPEC_CONVO] ===== trajectory index=%d =====", index)
        logger.warning("[COLBENCH_SPEC_CONVO] problem[:%d]=%r", n, str(problem_text)[:n])
        logger.warning("[COLBENCH_SPEC_CONVO] spec.requirements[:%d]=%r", n, str(spec.get("requirements"))[:n])
        for i, m in enumerate(sim_dialogue):
            logger.warning(
                "[COLBENCH_SPEC_CONVO] turn=%d role=%s content[:%d]=%r", i, m.get("role"), n, str(m.get("content"))[:n]
            )
        logger.warning("[COLBENCH_SPEC_CONVO] terminated_by=%s last_code[:%d]=%r", terminated_by, n, str(last_code)[:n])
        logger.warning(
            "[COLBENCH_SPEC_CONVO] pass_rate=%.3f all_pass=%s n=%d",
            reward,
            result.get("all_pass"),
            result.get("n", 0),
        )

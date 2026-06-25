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
"""Multi-turn oracle code-refinement agent loop for CodeContests.

A custom ``AgentLoopBase`` that reproduces ``run_oracle_iterative_eval`` as an RL
rollout:

  turn 0      : the solver writes code from the problem statement (assistant, mask=1)
  turn 1..N-1 : env runs the code vs ground-truth tests; on failure it injects the
                failing cases as a user turn (mask=0) and the solver refines.
  termination : all GT tests pass (early stop), max assistant turns reached, or the
                response budget would overflow.

Final reward (written to the last token via ``AgentLoopOutput.reward_score``):
  1.0 iff the model's final code passed all GT tests, else 0.0.

Token masking is the whole training signal: assistant-generated tokens are
trainable (mask=1); injected oracle/user feedback is not (mask=0). A future
tester turn produced by the *same* policy would simply be appended with mask=1.

Context-overflow policy: we never trim old turns (that corrupts the train-time
token/mask alignment). If a turn would exceed ``rollout.response_length`` while
still unsolved, we stop and assign reward 0 (``on_overflow=end_zero_reward``,
default) -- we never reward a run that did not cleanly solve within budget.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from codecontest.env import GTOracleEnv
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("code_refine_agent")
class CodeRefineAgentLoop(AgentLoopBase):
    """Oracle multi-turn code-refinement loop (GT-test feedback, binary outcome reward)."""

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
        # Per-turn generation cap (tokens). None -> use remaining response budget.
        self.max_new_tokens_per_turn = cc.get("max_new_tokens_per_turn", None)
        # Overflow handling: "end_zero_reward" (default) or "discard_sample".
        self.on_overflow = cc.get("on_overflow", "end_zero_reward")
        # Env defaults (per-sample values in extra_info take precedence).
        self.default_exec_timeout = float(cc.get("exec_timeout", 6.0))
        self.default_max_failures_shown = int(cc.get("max_failures_shown", 3))
        self.default_max_gt_test = int(cc.get("max_gt_test", 20))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}

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

        # Off-policy staleness bookkeeping the trainer requires (see
        # trainer_base._compute_metrics). Each server.generate() tags the output with the
        # weights version it was produced on; we keep the oldest (min) and freshest (max)
        # across turns. Must be plain ints, not None, or np.array(dtype=int) blows up.
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
            response_mask += [1] * len(resp_ids)
            if track_logprobs and output.log_probs:
                response_logprobs += output.log_probs
            else:
                track_logprobs = False

            assistant_turns += 1

            # Grade only the latest submission (env extracts the last ```python block).
            assistant_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            with simple_timer("env_step", metrics):
                step = await self.loop.run_in_executor(None, env.step, assistant_text)

            if step.solved:
                solved = True
                solved_at_turn = turn
                break

            # Last allowed turn: stop without injecting feedback we cannot act on.
            if turn == self.max_assistant_turns - 1:
                break

            # Inject the oracle feedback as a user turn (mask=0, not trained).
            feedback_ids = await self.apply_chat_template(
                [{"role": "user", "content": step.feedback}],
                remove_system_prompt=True,
            )
            # Need room for the feedback AND at least one response token next turn.
            if len(response_mask) + len(feedback_ids) >= self.response_length:
                overflow = True
                break
            prompt_ids += feedback_ids
            response_mask += [0] * len(feedback_ids)
            if track_logprobs:
                response_logprobs += [0.0] * len(feedback_ids)
            user_turns += 1

        # Binary outcome reward. Overflow while unsolved -> 0 (never reward a run that
        # did not cleanly solve within budget). "discard_sample" is treated as 0 here
        # because the agent loop cannot drop a sample without breaking GRPO grouping;
        # true discarding, if ever needed, must happen upstream in the trainer.
        reward = 1.0 if solved else 0.0

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
            },
        )
        return output

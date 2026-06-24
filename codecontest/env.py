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
"""Environment abstraction for the CodeContests multi-turn loop.

The env is the pluggable "what happens between assistant turns" component. It is a
direct analog of the (now-deprecated) ``BaseInteraction.generate_response``: given
the conversation so far, it scores the latest submission and decides whether to
stop and what feedback (the next user turn) to inject.

- ``GTOracleEnv`` (used now): runs the model's code against ground-truth tests.
  Feedback = a few failing cases; terminates early when all tests pass. This is
  the RL analog of ``iterative_eval.run_oracle_iterative_eval``.
- ``TesterPolicyEnv`` (future hook, not implemented): the next user turn would be
  produced by a *tester* — either the same policy with a tester prompt
  (shared-policy / CURE-style) or a separate model. Kept as a stub so the agent
  loop's interface and token masking already accommodate it.

Reward convention: a turn returns ``solved`` (did the latest code pass all GT
tests). The agent loop turns the final ``solved`` into the binary 0/1 trajectory
reward.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

from codecontest import local_exec, templates


@dataclass
class StepResult:
    """Outcome of evaluating one assistant submission.

    Attributes:
        solved: True iff the extracted code passed all executed GT tests.
        should_terminate: True iff the conversation should stop now (solved, or no
            code to refine).
        feedback: the next user-turn text to inject when not terminating (else "").
        num_failures_shown: how many failing cases were included in ``feedback``.
        had_code: whether a ```python block was found in the submission.
    """

    solved: bool
    should_terminate: bool
    feedback: str = ""
    num_failures_shown: int = 0
    had_code: bool = False


class BaseEnv:
    """Interface for a between-turns environment (cf. BaseInteraction.generate_response)."""

    def step(self, assistant_text: str) -> StepResult:
        raise NotImplementedError


@dataclass
class GTOracleEnv(BaseEnv):
    """Oracle env: grade the latest code against ground-truth stdin/stdout tests.

    Args:
        test_input / test_output: ground-truth cases (same set used for feedback and
            final reward, per the eval protocol).
        test_time_limit: per-case execution timeout (seconds).
        max_failures_shown: max failing cases revealed per turn (randomly sampled).
        max_gt_test: cap on number of GT cases executed per turn.
        seed: RNG seed for sampling which failures to show (deterministic per env).
    """

    test_input: list
    test_output: list
    test_time_limit: float = 6.0
    max_failures_shown: int = 3
    max_gt_test: int = 20
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def step(self, assistant_text: str) -> StepResult:
        code = local_exec.extract_code(assistant_text)
        if code is None:
            # No parseable submission: nothing to grade or refine against.
            return StepResult(
                solved=False,
                should_terminate=False,
                feedback=(
                    "Your previous response did not contain a ```python code block. "
                    "Please provide a complete Python 3 solution in a single "
                    "```python ... ``` code block."
                ),
                num_failures_shown=0,
                had_code=False,
            )

        all_pass, _per_case, failures = local_exec.eval_code_on_tests(
            code,
            self.test_input,
            self.test_output,
            time_limit=self.test_time_limit,
            max_gt_test=self.max_gt_test,
        )

        if all_pass:
            return StepResult(
                solved=True,
                should_terminate=True,
                feedback=templates.SOLVER_CORRECT_MESSAGE,
                num_failures_shown=0,
                had_code=True,
            )

        shown = failures
        if len(failures) > self.max_failures_shown:
            shown = self._rng.sample(failures, self.max_failures_shown)
        feedback = templates.build_feedback_message(shown)
        return StepResult(
            solved=False,
            should_terminate=False,
            feedback=feedback,
            num_failures_shown=len(shown),
            had_code=True,
        )


@dataclass
class TesterPolicyEnv(BaseEnv):
    """Future hook: feedback produced by a tester policy instead of GT tests.

    Not implemented yet. When co-training a tester+solver, ``step`` would (a) ask a
    tester to generate targeted tests for the latest code, (b) run them, and (c)
    return the results as feedback. For the shared-policy (CURE-style) setting the
    tester turns are generated by the same actor and masked as trainable in the
    agent loop; for separate policies a second model would be queried here. The
    surrounding loop/masking interface is already compatible with this.
    """

    test_input: Optional[list] = None
    test_output: Optional[list] = None

    def step(self, assistant_text: str) -> StepResult:  # pragma: no cover - stub
        raise NotImplementedError("TesterPolicyEnv is a placeholder for future tester+solver co-training.")

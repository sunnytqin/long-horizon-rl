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
"""CPU unit tests for codecontest.env.GTOracleEnv (no GPU, no network)."""

from codecontest import templates
from codecontest.env import GTOracleEnv

GT_IN = ["2 3\n", "10 20\n", "1 1\n", "4 5\n"]
GT_OUT = ["5\n", "30\n", "2\n", "9\n"]

GOOD = "```python\nimport sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)\n```"
BAD = "Here is my attempt:\n```python\nprint(0)\n```"
NO_CODE = "I think the answer is to add the numbers."


def _env(**kw):
    return GTOracleEnv(test_input=GT_IN, test_output=GT_OUT, test_time_limit=4.0, **kw)


def test_passing_submission_solves_and_terminates():
    res = _env().step(GOOD)
    assert res.solved is True
    assert res.should_terminate is True
    assert res.had_code is True
    assert res.feedback == templates.SOLVER_CORRECT_MESSAGE


def test_failing_submission_gives_feedback():
    res = _env(max_failures_shown=2).step(BAD)
    assert res.solved is False
    assert res.should_terminate is False
    assert res.had_code is True
    assert 1 <= res.num_failures_shown <= 2
    assert "Test 1:" in res.feedback
    assert "Expected output:" in res.feedback


def test_no_code_block_requests_code():
    res = _env().step(NO_CODE)
    assert res.had_code is False
    assert res.solved is False
    assert res.should_terminate is False
    assert "python" in res.feedback.lower()


def test_max_failures_shown_caps_feedback():
    # BAD fails 3 of 4 cases; cap to 1.
    res = _env(max_failures_shown=1).step(BAD)
    assert res.num_failures_shown == 1


def test_failure_sampling_is_deterministic_per_seed():
    a = _env(max_failures_shown=1, seed=123).step(BAD).feedback
    b = _env(max_failures_shown=1, seed=123).step(BAD).feedback
    assert a == b


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all env tests passed")

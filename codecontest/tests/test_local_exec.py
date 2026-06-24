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
"""CPU unit tests for codecontest.local_exec (no GPU, no network)."""

from codecontest import local_exec

# Read two ints from stdin, print their sum.
GOOD_CODE = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)"
BAD_CODE = "print(0)"  # ignores input, always prints 0
TIMEOUT_CODE = "while True:\n    pass"

GT_IN = ["2 3\n", "10 20\n", "0 0\n"]
GT_OUT = ["5\n", "30\n", "0\n"]


def test_extract_code_takes_last_block():
    text = "first\n```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```"
    assert local_exec.extract_code(text) == "print(2)"
    assert local_exec.extract_code("no code here") is None


def test_outputs_match_whitespace_insensitive():
    assert local_exec.outputs_match("5\n", "5")
    assert local_exec.outputs_match(" 5  \n", "5\n")
    assert not local_exec.outputs_match("5", "6")


def test_good_code_passes_all():
    all_pass, per_case, failures = local_exec.eval_code_on_tests(GOOD_CODE, GT_IN, GT_OUT)
    assert all_pass is True
    assert per_case == [True, True, True]
    assert failures == []


def test_bad_code_fails_with_failures():
    all_pass, per_case, failures = local_exec.eval_code_on_tests(BAD_CODE, GT_IN, GT_OUT)
    assert all_pass is False
    # "0" matches the third case (0+0=0) but not the first two.
    assert per_case == [False, False, True]
    assert len(failures) == 2
    inp, actual, expected = failures[0]
    assert actual.strip() == "0" and expected.strip() == "5"


def test_none_code_returns_unsolved():
    all_pass, per_case, failures = local_exec.eval_code_on_tests(None, GT_IN, GT_OUT)
    assert all_pass is False and per_case == [] and failures == []


def test_timeout_is_handled():
    all_pass, per_case, failures = local_exec.eval_code_on_tests(
        TIMEOUT_CODE, ["1 1\n"], ["2\n"], time_limit=1.0
    )
    assert all_pass is False
    assert per_case == [False]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all local_exec tests passed")

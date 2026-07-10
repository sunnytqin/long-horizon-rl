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
"""CPU unit tests for colbench.reward.grade (functional-equivalence pass-rate).

Needs the exec sidecar up, OR CODECONTEST_ALLOW_INPROCESS=1 (set here) to grade via the
in-process fallback -- no GPU, no server. Run:
    CODECONTEST_ALLOW_INPROCESS=1 pytest colbench/tests/test_reward.py
"""

import os

# Grade via the in-process exec fallback (no sidecar). Set BEFORE reward/exec_client run.
os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

from colbench import reward  # noqa: E402

# A GT with a hidden branch (x >= 10) so a "sum only" candidate matches PART of the cases.
GT = "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n        return x - y\n"
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]  # 2 take the else branch, 2 the if branch


def test_exact_copy_scores_full():
    res = reward.grade(GT, GT, CALLS)
    assert res["pass_rate"] == 1.0
    assert res["all_pass"] is True
    assert res["n"] == 4


def test_partial_impl_scores_fraction():
    # Misses the hidden else branch: matches only the 2 cases where x >= 10.
    candidate = "def f(x, y):\n    return x + y\n"
    res = reward.grade(candidate, GT, CALLS)
    assert res["pass_rate"] == 0.5
    assert res["all_pass"] is False
    assert sum(res["per_case"]) == 2


def test_wrong_impl_scores_zero():
    candidate = "def f(x, y):\n    return 0\n"
    res = reward.grade(candidate, GT, CALLS)
    assert res["pass_rate"] == 0.0
    assert res["all_pass"] is False


def test_candidate_stdout_is_suppressed():
    # A correct impl that ALSO prints must still score 1.0 -- the harness suppresses candidate
    # stdout so the sole harness output stays the boolean the sidecar compares to "True".
    candidate = (
        "def f(x, y):\n"
        "    print('noisy candidate output')\n"
        "    return x + y if x >= 10 else x - y\n"
    )
    res = reward.grade(candidate, GT, CALLS)
    assert res["pass_rate"] == 1.0


def test_call_string_with_escaped_newline_not_corrupted():
    # base64-on-stdin must survive local_exec.normalise's `\n`->newline rewrite. A call whose
    # arg is a string literal containing \n would break if passed raw; here GT==candidate so a
    # correct decode yields a match, a corrupted decode raises -> 0.
    gt = "def g(s):\n    return s.count(chr(10))\n"
    calls = ['g("a\\nb\\nc")']  # the literal 4-char sequences a \n b ...
    res = reward.grade(gt, gt, calls)
    assert res["pass_rate"] == 1.0


def test_missing_candidate_or_no_cases_scores_zero():
    assert reward.grade("", GT, CALLS)["pass_rate"] == 0.0
    assert reward.grade(GT, GT, [])["pass_rate"] == 0.0
    assert reward.grade(GT, GT, [None])["pass_rate"] == 0.0  # None-only cases filtered out


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all reward tests passed")

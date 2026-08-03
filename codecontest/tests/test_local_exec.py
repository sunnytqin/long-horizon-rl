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

import importlib
import os
import time

from codecontest import local_exec

# Read two ints from stdin, print their sum.
GOOD_CODE = (
    "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)"
)
BAD_CODE = "print(0)"  # ignores input, always prints 0
TIMEOUT_CODE = "while True:\n    pass"

GT_IN = ["2 3\n", "10 20\n", "0 0\n"]
GT_OUT = ["5\n", "30\n", "0\n"]


def test_extract_code_takes_last_block():
  """``extract_code`` grades the most recent submission, not the first."""
  text = "first\n```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```"
  assert local_exec.extract_code(text) == "print(2)"
  assert local_exec.extract_code("no code here") is None


def test_outputs_match_whitespace_insensitive():
  """``outputs_match`` ignores differences in whitespace runs."""
  assert local_exec.outputs_match("5\n", "5")
  assert local_exec.outputs_match(" 5  \n", "5\n")
  assert not local_exec.outputs_match("5", "6")


def test_good_code_passes_all():
  """Correct code passes every ground-truth case with no failures."""
  all_pass, per_case, failures = local_exec.eval_code_on_tests(
      GOOD_CODE, GT_IN, GT_OUT
  )
  assert all_pass
  assert per_case == [True, True, True]
  assert not failures


def test_bad_code_fails_with_failures():
  """Wrong code reports per-case results and the cases that failed."""
  all_pass, per_case, failures = local_exec.eval_code_on_tests(
      BAD_CODE, GT_IN, GT_OUT
  )
  assert not all_pass
  # "0" matches the third case (0+0=0) but not the first two.
  assert per_case == [False, False, True]
  assert len(failures) == 2
  _, actual, expected = failures[0]
  assert actual.strip() == "0" and expected.strip() == "5"


def test_none_code_returns_unsolved():
  """Unextractable code grades as unsolved rather than raising."""
  all_pass, per_case, failures = local_exec.eval_code_on_tests(
      None, GT_IN, GT_OUT
  )
  assert not all_pass and not per_case and not failures


def test_timeout_is_handled():
  """A non-terminating program is killed at the time limit and fails."""
  all_pass, per_case, _ = local_exec.eval_code_on_tests(
      TIMEOUT_CODE, ["1 1\n"], ["2\n"], time_limit=1.0
  )
  assert not all_pass
  assert per_case == [False]


def test_memory_bomb_is_contained():
  """A memory bomb must hit the RLIMIT_AS cap, not consume host RAM.

  Uses a small cap (0.5GB) and a modest bomb (1.5GB) so the test is safe even
  if the cap somehow doesn't apply on this platform.
  """
  os.environ["CODECONTEST_EXEC_MEM_GB"] = "0.5"
  importlib.reload(local_exec)
  try:
    # Single allocation ~1.5GB > 0.5GB headroom -> MemoryError in the child.
    bomb = "x = bytearray(1536 * 1024 * 1024)\nprint(len(x))"
    all_pass, per_case, _ = local_exec.eval_code_on_tests(
        bomb, ["1\n"], ["1610612736\n"], time_limit=15.0
    )
    assert not all_pass
    assert per_case == [False]
    # Normal small code must still run fine under the tight cap.
    ok_pass, ok_per_case, _ = local_exec.eval_code_on_tests(
        GOOD_CODE, ["2 3\n"], ["5\n"]
    )
    assert ok_pass and ok_per_case == [True]
  finally:
    os.environ.pop("CODECONTEST_EXEC_MEM_GB", None)
    importlib.reload(local_exec)


def test_concurrency_cap_bounds_live_processes():
  """Sleepy cases must serialize through the concurrency cap.

  With EXEC_CONCURRENCY=2, 6 cases / 2 slots * ~1s each => well over 1s total
  (vs ~1s if the cap were not applied).
  """
  os.environ["CODECONTEST_EXEC_CONCURRENCY"] = "2"
  importlib.reload(local_exec)
  try:
    sleepy = "import time\ntime.sleep(1.0)\nprint('done')"
    n = 6
    t0 = time.time()
    results = local_exec.run_code_batch([sleepy] * n, [""] * n, [10.0] * n)
    elapsed = time.time() - t0
    assert all(r.strip() == "done" for r in results)
    # 6 cases through 2 slots = 3 waves * ~1s => >=2.5s. Unbounded would be ~1s.
    assert elapsed >= 2.5, f"expected serialization >=2.5s, got {elapsed:.2f}s"
  finally:
    os.environ.pop("CODECONTEST_EXEC_CONCURRENCY", None)
    importlib.reload(local_exec)


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"PASS {name}")
  print("all local_exec tests passed")

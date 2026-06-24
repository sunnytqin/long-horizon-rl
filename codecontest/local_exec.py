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
"""Local subprocess sandbox for executing model-generated Python on stdin/stdout.

This replaces the internal "xbox" execution server used by the tunix eval harness
with a self-contained local executor, reusing the multiprocessing-based exec
pattern from ``codecontest/eval_example.py``. It is used both for mid-conversation
oracle feedback and for the final binary reward.

Security note: this runs untrusted model output in a child process guarded only by
a wall-clock timeout (same risk profile as the existing ``eval_example.py``).
Run it on an isolated research node; rlimit/seccomp/container hardening (or a swap
to SandboxFusion) is a follow-up if stronger isolation is needed.
"""

import io
import multiprocessing as mp
import re
import sys
import time
import typing

# A spawn/fork-safe context. "fork" is fastest on Linux and matches eval_example.py.
_MP_CTX = mp.get_context("fork")

# ── output normalization & comparison (matches eval_example.modify / test_if_eq) ──


def normalise(s: str) -> str:
    """Normalize an input/output blob the way the eval harness does."""
    s = s.replace("plaintext\n", "").replace("\\n", "\n")
    if not s.endswith("\n"):
        s += "\n"
    return s


def outputs_match(actual: str, expected: str) -> bool:
    """Whitespace-insensitive equality, identical to eval_example.test_if_eq."""
    return " ".join(str(actual).split()) == " ".join(str(expected).split())


def extract_code(text: str) -> typing.Optional[str]:
    """Return the content of the LAST ```python ... ``` block, or None.

    Mirrors code_util.extract_code from the tunix harness. Taking the *last* block
    means oracle-feedback text (which never contains python fences) is ignored and
    we always grade the model's most recent submission.
    """
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


# ── subprocess execution (ported from eval_example.worker / run_scripts_with_timeout) ──


def _worker(script: str, stdin_str: str, output_queue):
    """Execute ``script`` with ``stdin_str`` on stdin; put stdout (or error) on queue."""
    input_lines = iter(stdin_str.splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    stdout_capture = io.StringIO()
    original_stdout, original_stdin = sys.stdout, sys.stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(stdin_str)

    context = {
        "__name__": "__main__",
        "input": fake_input,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Optional": typing.Optional,
    }
    try:
        exec(script, context)
        output_queue.put(stdout_capture.getvalue())
    except SystemExit:
        output_queue.put(stdout_capture.getvalue())
    except BaseException as e:  # noqa: BLE001 - sandbox: report any failure as text
        output_queue.put(f"error: {e}")
    finally:
        sys.stdout, sys.stdin = original_stdout, original_stdin


def run_code_batch(codes, stdins, time_limits):
    """Run a batch of (code, stdin) pairs in parallel child processes.

    Args:
        codes: list[str] python sources.
        stdins: list[str] stdin for each code.
        time_limits: list[float] per-case wall-clock timeout (seconds).

    Returns:
        list[str]: stdout for each case, or "Timeout Error" / "error: ..." on failure.
    """
    n = len(codes)
    results: list = [None] * n
    processes, queues, deadlines = [], [], []
    for i in range(n):
        q = _MP_CTX.Queue()
        p = _MP_CTX.Process(target=_worker, args=(codes[i], stdins[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:  # noqa: BLE001
                results[i] = f"Execution Error: {e}"
        p.join(timeout=0.1)
    return results


def eval_code_on_tests(code, test_input, test_output, time_limit=6.0, max_gt_test=20):
    """Run ``code`` against ground-truth (stdin -> expected stdout) cases.

    Args:
        code: python source (may be None if extraction failed).
        test_input: list[str] stdin strings.
        test_output: list[str] expected stdout strings.
        time_limit: per-case timeout in seconds (or a per-case list).
        max_gt_test: cap on number of cases actually executed.

    Returns:
        (all_pass, per_case, failures) where
          all_pass : bool  - True iff every executed case matched (and >=1 case ran),
          per_case : list[bool] - pass/fail per executed case,
          failures : list[(inp, actual, expected)] - the cases that did not match.
    """
    if code is None:
        return False, [], []

    n = min(len(test_input), len(test_output), max_gt_test)
    if n == 0:
        return False, [], []

    inps = [normalise(x) for x in test_input[:n]]
    exps = test_output[:n]
    if isinstance(time_limit, (list, tuple)):
        tls = [float(t) for t in time_limit[:n]]
    else:
        tls = [float(time_limit)] * n

    actuals = run_code_batch([code] * n, inps, tls)

    per_case, failures = [], []
    for inp, actual, expected in zip(inps, actuals, exps):
        ok = outputs_match(actual, expected)
        per_case.append(ok)
        if not ok:
            failures.append((inp, actual, expected))

    all_pass = bool(per_case) and all(per_case)
    return all_pass, per_case, failures

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

Security note: this runs untrusted model output in a child process guarded by a
wall-clock timeout AND a per-process address-space cap (``RLIMIT_AS``), plus a
global cap on how many executions run concurrently. The memory/concurrency caps
keep a memory-bomb generation (e.g. ``x = [0]*10**10``) from exhausting host RAM
and tripping the Ray OOM killer during large multi-turn rollouts. Run it on an
isolated research node; seccomp/container hardening (or a swap to SandboxFusion)
is a follow-up if stronger isolation is needed.

Tunables (env vars, read at import):
  CODECONTEST_EXEC_MEM_GB        per-process address-space headroom cap (default 2)
  CODECONTEST_EXEC_CONCURRENCY   max concurrent child executions (default 64)
"""

import io
import multiprocessing as mp
import os
import re
import resource
import sys
import threading
import time
import typing

# A spawn/fork-safe context. "fork" is fastest on Linux and matches eval_example.py.
# We keep fork (no torch/CUDA re-import tax that `spawn` would incur from the trainer
# process) and rely on a *relative* RLIMIT_AS in the child for memory safety.
_MP_CTX = mp.get_context("fork")

# Per-process memory headroom: a forked child inherits the parent's (possibly huge,
# CUDA-reserved) virtual address space, so an absolute cap is meaningless. Instead we
# cap *growth* beyond the child's startup VSZ, which a memory bomb trips as MemoryError.
_EXEC_MEM_LIMIT_BYTES = int(float(os.environ.get("CODECONTEST_EXEC_MEM_GB", "2")) * (1024**3))
# Global ceiling on concurrently-alive child processes across all agent-loop threads,
# so a 200+-way rollout fan-out can't launch thousands of exec processes at once.
_EXEC_CONCURRENCY = int(os.environ.get("CODECONTEST_EXEC_CONCURRENCY", "64"))
_EXEC_SLOTS = threading.BoundedSemaphore(_EXEC_CONCURRENCY)


def _apply_mem_limit() -> None:
    """Cap this (child) process's address-space growth to +_EXEC_MEM_LIMIT_BYTES.

    Best-effort: any failure leaves the process uncapped rather than blocking exec.
    Relative to startup VSZ so it works regardless of inherited CUDA VA reservations.
    """
    if _EXEC_MEM_LIMIT_BYTES <= 0:
        return
    try:
        with open("/proc/self/statm") as f:
            vsz_pages = int(f.read().split()[0])
        current = vsz_pages * resource.getpagesize()
        soft = current + _EXEC_MEM_LIMIT_BYTES
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            soft = min(soft, hard)
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    except Exception:  # noqa: BLE001 - never let rlimit setup break execution
        pass

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
    _apply_mem_limit()  # bound this child's RAM growth before running untrusted code
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


def _run_single(code: str, stdin_str: str, time_limit: float) -> str:
    """Run one (code, stdin) in a child process, holding one global concurrency slot.

    Returns stdout, or "Timeout Error" / "error: ..." on failure. The slot bounds how
    many child processes are alive at once across all calling threads; each call is
    self-contained (acquire -> start -> reap -> release) so it can't deadlock against
    other in-flight cases the way a batch holding several slots could.
    """
    with _EXEC_SLOTS:
        q = _MP_CTX.Queue()
        p = _MP_CTX.Process(target=_worker, args=(code, stdin_str, q))
        p.start()
        deadline = time.time() + time_limit
        result = None
        try:
            while p.is_alive() and time.time() < deadline:
                time.sleep(0.001)
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)
                if p.is_alive():
                    p.kill()
                result = "Timeout Error"
            else:
                try:
                    result = q.get_nowait()
                except Exception as e:  # noqa: BLE001
                    result = f"Execution Error: {e}"
            p.join(timeout=0.1)
            return result
        finally:
            try:
                q.close()
            except Exception:  # noqa: BLE001
                pass


def run_code_batch(codes, stdins, time_limits):
    """Run a batch of (code, stdin) pairs in parallel, bounded by the global slot cap.

    Args:
        codes: list[str] python sources.
        stdins: list[str] stdin for each code.
        time_limits: list[float] per-case wall-clock timeout (seconds).

    Returns:
        list[str]: stdout for each case, or "Timeout Error" / "error: ..." on failure.
    """
    n = len(codes)
    results: list = [None] * n

    def runner(i):
        results[i] = _run_single(codes[i], stdins[i], time_limits[i])

    # One thread per case; each blocks on the global semaphore before spawning its
    # child, so within-trajectory parallelism is preserved up to the global ceiling.
    threads = [threading.Thread(target=runner, args=(i,), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
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

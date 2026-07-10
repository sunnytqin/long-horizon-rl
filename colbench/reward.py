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
"""Functional-equivalence reward for ColBench, via the EXISTING code-exec sidecar.

ColBench grades a submitted function against a hidden ground-truth function by CALLING both
on the same argument tuples and comparing return values with Python ``==`` (sweet_rl's
``code_utils.check_correctness``). We reuse the ``codecontest`` sandbox sidecar UNCHANGED
rather than exec'ing untrusted code in the trainer:

  * We build a self-contained COMPARISON HARNESS script that embeds the GT and candidate
    sources (base64, so arbitrary source -- quotes, triple-quotes -- round-trips cleanly),
    reads ONE call-string from stdin, evaluates it in two separate namespaces, and prints
    ``repr(gt_out == cand_out and gt_out is not None)`` -- with the candidate's own stdout
    suppressed so the sole stdout line is that boolean.
  * We hand the harness to ``exec_client.eval_code_on_tests(code=harness,
    test_input=<call-strings>, test_output=["True", ...])``. The sidecar runs the harness
    once per call-string (each in its own RLIMIT/cgroup-bounded child), and the returned
    ``per_case`` bools ARE the per-test-case pass/fail. ``pass_rate = mean(per_case)`` is the
    reward (fractional, matching sweet_rl's stored reward); ``all_pass`` is a diagnostic.

The compare happens INSIDE the sandboxed child (no live objects cross the process boundary),
so we keep sweet_rl's faithful ``==``-on-return-values semantics while reusing all existing
isolation + concurrency + retry. No sidecar/server change.

Call-strings are base64-encoded onto stdin: ``local_exec.normalise`` rewrites ``\\n`` -> real
newline on every ``test_input``, which would corrupt a call whose args contain string
literals with ``\\n``. base64 (no backslashes) passes through ``normalise`` untouched.
"""

import base64
import logging
import os

from codecontest import exec_client

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _b64(s: str) -> str:
    return base64.b64encode((s or "").encode("utf-8")).decode("ascii")


def build_harness(ground_truth_src: str, candidate_src: str) -> str:
    """Build the stdin-driven comparison harness (see module docstring).

    The harness reads a base64-encoded call-string from stdin, evaluates it in a GT namespace
    and a candidate namespace (each ``exec``'d fresh), and prints the boolean equivalence for
    that one call. Both sources are embedded base64 so any source text round-trips.
    """
    gt_b64 = _b64(ground_truth_src)
    cand_b64 = _b64(candidate_src)
    # NOTE: ``{{}}`` -> literal ``{}`` under .format-free f-string; the only interpolations are
    # the two base64 blobs. Everything the candidate prints is captured into _buf and dropped;
    # only the final repr(bool) reaches real stdout, which the sidecar compares to "True".
    return f'''import sys, base64, contextlib, io

_GT_SRC = base64.b64decode("{gt_b64}").decode("utf-8")
_CAND_SRC = base64.b64decode("{cand_b64}").decode("utf-8")


def _get_output(src, call):
    try:
        ns = {{}}
        exec(src, ns)
        return eval(call, ns)
    except Exception:
        return None


_call = base64.b64decode(sys.stdin.read().strip()).decode("utf-8")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _gt = _get_output(_GT_SRC, _call)
    _cand = _get_output(_CAND_SRC, _call)
    try:
        _ok = bool(_gt is not None and (_gt == _cand))
    except Exception:
        _ok = False
print(repr(_ok))
'''


def grade(candidate_code, ground_truth_src, test_calls, time_limit=6.0):
    """Grade ``candidate_code`` against ``ground_truth_src`` on ``test_calls``.

    Args:
        candidate_code: the solver's submitted function source (already fence-stripped).
        ground_truth_src: the hidden GT function source.
        test_calls: list of call-strings, e.g. ``"f(1969, 140, 500)"``.
        time_limit: per-case wall-clock timeout (seconds), passed to the sidecar.

    Returns:
        dict with ``pass_rate`` (float in [0,1] -- the reward), ``all_pass`` (bool),
        ``per_case`` (list[bool]), and ``n`` (cases executed). A missing candidate / no test
        cases / an unreachable sidecar all yield ``pass_rate=0.0`` (never raises).
    """
    calls = [str(c) for c in (test_calls or []) if c]
    if not candidate_code or not calls:
        return {"pass_rate": 0.0, "all_pass": False, "per_case": [], "n": 0}

    harness = build_harness(ground_truth_src, candidate_code)
    test_input = [_b64(c) for c in calls]
    test_output = ["True"] * len(calls)
    # max_gt_test = len(calls): grade EVERY case so the fraction denominator matches sweet_rl
    # (which divides by len(test_cases)); the default cap of 20 would silently drop cases.
    all_pass, per_case, _failures = exec_client.eval_code_on_tests(
        harness,
        test_input,
        test_output,
        time_limit=time_limit,
        max_gt_test=len(calls),
    )
    n = len(per_case)
    pass_rate = (sum(1 for p in per_case if p) / n) if n else 0.0
    return {"pass_rate": pass_rate, "all_pass": bool(all_pass), "per_case": per_case, "n": n}

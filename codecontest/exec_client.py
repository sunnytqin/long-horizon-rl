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
"""Client shim: route untrusted code grading to the sandbox exec server.

``GTOracleEnv`` calls ``eval_code_on_tests`` here instead of ``local_exec`` so the
actual ``exec()`` of model output happens inside an isolated sibling container, not
in the trainer process. The wire format mirrors ``local_exec.eval_code_on_tests``
exactly (same args in, same ``(all_pass, per_case, failures)`` out, with the same
``normalise``/``outputs_match`` comparison run server-side), so reward semantics are
identical to the in-process path.

In-process mode is STRICTLY WORSE (it is the mode where bad model code can kill a
rollout worker), so it is never used silently. It runs only when explicitly opted in
via ``CODECONTEST_ALLOW_INPROCESS=1`` -- which the no-sidecar smoke/dev run sets. A
real training job instead guarantees the sidecar is up (entrypoint.sh hard-fails if
it is not), so ``CODECONTEST_EXEC_URL`` is always set there.

Failure policy: a grade request must NEVER raise. The agent loop
(``code_refine_agent.run``) only catches ``asyncio.TimeoutError`` around ``env.step``;
any other exception would crash the whole rollout. So on a server blip we retry, and
if still unreachable we grade the turn as unsolved-with-no-feedback. A persistently
down server therefore shows up as all-zero reward (loudly visible) rather than a
crash.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

from codecontest import local_exec

logger = logging.getLogger(__name__)

_GRADE_PATH = "/grade"
# Client-side wall on one grade request. Keep under the agent loop's env_step_timeout
# (default 180s) so a slow server surfaces as a normal unsolved turn, not a hang.
_HTTP_TIMEOUT = float(os.environ.get("CODECONTEST_EXEC_HTTP_TIMEOUT", "150"))
_RETRIES = int(os.environ.get("CODECONTEST_EXEC_HTTP_RETRIES", "3"))

# One-shot guard so a misconfigured run logs the refusal once, not once per grade.
_warned_no_url = [False]


def _server_url():
    url = os.environ.get("CODECONTEST_EXEC_URL")
    return url.rstrip("/") if url else None


def eval_code_on_tests(code, test_input, test_output, time_limit=6.0, max_gt_test=20):
    """Drop-in remote replacement for ``local_exec.eval_code_on_tests``.

    Returns ``(all_pass, per_case, failures)`` where ``failures`` is a list of
    ``(inp, actual, expected)`` tuples -- identical shape to the local path.
    """
    url = _server_url()
    if not url:
        # No sidecar configured. In-process exec (Phase 0) is strictly worse -- it's the
        # mode where bad code can kill a rollout worker -- so we never use it silently.
        # Allowed only as an explicit opt-in for the no-sidecar smoke/dev run.
        if os.environ.get("CODECONTEST_ALLOW_INPROCESS") == "1":
            return local_exec.eval_code_on_tests(code, test_input, test_output, time_limit, max_gt_test)
        if not _warned_no_url[0]:
            logger.error(
                "CODECONTEST_EXEC_URL is unset and CODECONTEST_ALLOW_INPROCESS != 1: refusing "
                "the strictly-worse in-process exec and grading every turn as UNSOLVED. Start "
                "the sidecar (entrypoint.sh) or set CODECONTEST_ALLOW_INPROCESS=1 for dev/smoke."
            )
            _warned_no_url[0] = True
        return False, [], []

    payload = json.dumps(
        {
            "code": code,
            "test_input": list(test_input),
            "test_output": list(test_output),
            "time_limit": time_limit,
            "max_gt_test": max_gt_test,
        }
    ).encode("utf-8")

    last_err = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(
                url + _GRADE_PATH,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            # Rebuild tuples so callers (templates.format_oracle_feedback) see the
            # exact same shape the local path returns.
            return obj["all_pass"], obj["per_case"], [tuple(f) for f in obj["failures"]]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:  # noqa: PERF203
            last_err = e
            if attempt + 1 < _RETRIES:
                time.sleep(0.5 * (attempt + 1))

    logger.warning("exec server %s unreachable (%r); grading turn as unsolved", url, last_err)
    return False, [], []

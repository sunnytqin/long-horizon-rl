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
"""HTTP wrapper around ``local_exec`` for the sandboxed code-exec sidecar.

Runs inside a slim, isolated container (no torch/CUDA -- stdlib only). The trainer's
env POSTs a ``(code, tests)`` payload; we run the untrusted code with ``local_exec``'s
existing multiprocessing + per-process RLIMIT guards and return the SAME
``(all_pass, per_case, failures)`` that ``local_exec.eval_code_on_tests`` produces --
so reward semantics are byte-identical to the in-process path, but the ``exec()`` now
happens behind the container boundary instead of inside the trainer process.

Strong isolation is the *container's* job (cgroup ``--memory`` / ``--pids-limit`` /
``--cpus``, ``--network`` internal-or-none, ``--read-only`` + ``--tmpfs``). This process
only adds per-case timeouts and the global concurrency cap via ``local_exec``.

Env vars:
  CODECONTEST_EXEC_HOST          bind address (default 0.0.0.0)
  CODECONTEST_EXEC_PORT          bind port (default 8088)
  CODECONTEST_EXEC_CONCURRENCY   max concurrent child execs (read by local_exec)
  CODECONTEST_EXEC_MEM_GB        per-exec address-space cap (read by local_exec)

Endpoints:
  GET  /health  -> {"status": "ok", "concurrency": int}
  POST /grade   -> body {code, test_input, test_output, time_limit?, max_gt_test?}
                   reply {all_pass: bool, per_case: [bool], failures: [[inp,act,exp]]}
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from codecontest import local_exec


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/health":
            self._send(200, {"status": "ok", "concurrency": local_exec._EXEC_CONCURRENCY})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802 - http.server API
        if self.path != "/grade":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - malformed request, not our problem
            self._send(400, {"error": f"bad request: {e!r}"})
            return
        try:
            all_pass, per_case, failures = local_exec.eval_code_on_tests(
                payload.get("code"),
                payload.get("test_input", []),
                payload.get("test_output", []),
                time_limit=payload.get("time_limit", 6.0),
                max_gt_test=payload.get("max_gt_test", 20),
            )
            self._send(
                200,
                {
                    "all_pass": all_pass,
                    "per_case": per_case,
                    "failures": [list(f) for f in failures],  # tuples -> JSON arrays
                },
            )
        except Exception as e:  # noqa: BLE001 - one bad grade must not kill the server
            self._send(500, {"error": repr(e)})

    def log_message(self, *args):  # silence per-request stderr spam
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True  # don't let in-flight grades block shutdown
    allow_reuse_address = True


def main() -> None:
    host = os.environ.get("CODECONTEST_EXEC_HOST", "0.0.0.0")
    port = int(os.environ.get("CODECONTEST_EXEC_PORT", "8088"))
    server = _Server((host, port), _Handler)
    print(
        f"[exec_server] listening on {host}:{port} "
        f"(concurrency={local_exec._EXEC_CONCURRENCY}, "
        f"mem_gb={local_exec._EXEC_MEM_LIMIT_BYTES / (1024 ** 3):.1f})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

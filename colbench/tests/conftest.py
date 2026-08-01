# Test-only compatibility shims so the spec agent-loop tests run on Python 3.10
# dev envs (openrlhf / verl conda) in addition to the py3.11 container.
#
#  1. verl.experimental.agent_loop uses ``enum.StrEnum`` (a 3.11 builtin).
#     Provide a minimal backport when absent; a strict no-op on 3.11+.
#  2. Grading (``env.score`` -> codecontest.exec_client) uses the exec-sidecar
#     HTTP service when ``CODECONTEST_EXEC_URL`` is set (the container), else
#     falls back to in-process CPU exec only when
#     ``CODECONTEST_ALLOW_INPROCESS=1``. Enable that fallback for local runs (no
#     GPU / no sidecar). ``setdefault`` respects an explicit override and is
#     ignored when the URL is set.
import enum
import os

if not hasattr(enum, "StrEnum"):

  class StrEnum(str, enum.Enum):

    def __str__(self):
      return str(self.value)

  enum.StrEnum = StrEnum

os.environ.setdefault("CODECONTEST_ALLOW_INPROCESS", "1")

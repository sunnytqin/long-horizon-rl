#!/usr/bin/env bash
# NOTE: NOT used by the XManager/XCloud deployment (that runs ONE managed container,
# so the sidecar is started as a background PROCESS in entrypoint.sh instead). Keep
# this only for a plain-VM / future separate-POD deployment where you control the
# Docker host and want the sandbox in its own resource cgroup.
#
# Build + launch the slim code-exec SANDBOX SIDECAR as an isolated sibling container.
#
# Untrusted model-generated Python runs HERE, not in the trainer. A memory bomb / fork
# bomb / busy loop is then confined to this container's cgroup (--memory / --pids-limit
# / --cpus) and dies without touching the trainer or tripping the Ray OOM killer.
# Reward semantics are identical to the in-process path (same local_exec grading code).
#
# Networking (trainer runs inside its own verl/SGLang container -- "Option A"):
#   We use a PRIVATE, --internal bridge network with NO internet gateway. The sandbox
#   joins ONLY this network, so it physically cannot reach the internet. The trainer
#   keeps its own network (HF downloads etc.) AND joins this one to reach the sandbox
#   by name at http://cc-sandbox:8088.
#
# Usage:
#   bash codecontest/run_sandbox.sh                 # build (if needed) + (re)start sidecar
#   TRAINER_CONTAINER=verl-trainer bash codecontest/run_sandbox.sh   # also wire trainer
#
# Then inside the trainer set:  export CODECONTEST_EXEC_URL=http://cc-sandbox:8088
set -xeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE=${SANDBOX_IMAGE:-cc-sandbox}
NAME=${SANDBOX_NAME:-cc-sandbox}
NET=${SANDBOX_NET:-cc-net}
PORT=${SANDBOX_PORT:-8088}

# Worker pool / resource caps. With ~1 TB host RAM, RAM is NOT the constraint -- PID
# and CPU are. So --pids-limit and --cpus are the real guards; --memory is a fast
# fail-stop that kills a memory bomb in milliseconds rather than protecting the host.
CONCURRENCY=${SANDBOX_CONCURRENCY:-64}     # max concurrent child execs inside the sidecar
EXEC_MEM_GB=${SANDBOX_EXEC_MEM_GB:-1}      # per-exec address-space cap (leetcode-sized; was 2)
CTR_MEM=${SANDBOX_CTR_MEM:-96g}            # container-wide memory tripwire (>= CONCURRENCY*EXEC_MEM_GB + margin)
CTR_PIDS=${SANDBOX_CTR_PIDS:-1024}         # hard ceiling on processes/threads -> kills fork bombs
CTR_CPUS=${SANDBOX_CTR_CPUS:-16}           # CPU cores the pool may use (keep headroom for the trainer)

# 1) Build the slim image (no torch/CUDA). Context = this dir so the COPYs resolve.
docker build -f "${HERE}/Dockerfile.sandbox" -t "${IMAGE}" "${HERE}"

# 2) Private, internet-less network (idempotent).
docker network inspect "${NET}" >/dev/null 2>&1 || docker network create --internal "${NET}"

# 3) (Re)start the sidecar with full hardening.
docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d --name "${NAME}" \
  --network "${NET}" \
  --read-only --tmpfs /tmp:rw,size=256m \
  --memory "${CTR_MEM}" --memory-swap "${CTR_MEM}" \
  --pids-limit "${CTR_PIDS}" \
  --cpus "${CTR_CPUS}" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --restart unless-stopped \
  -e CODECONTEST_EXEC_PORT="${PORT}" \
  -e CODECONTEST_EXEC_CONCURRENCY="${CONCURRENCY}" \
  -e CODECONTEST_EXEC_MEM_GB="${EXEC_MEM_GB}" \
  "${IMAGE}"

# 4) Attach the trainer container to this network too, if its name was given.
#    (The trainer keeps its primary network/internet; this just adds reachability.)
if [ -n "${TRAINER_CONTAINER:-}" ]; then
  docker network connect "${NET}" "${TRAINER_CONTAINER}" 2>/dev/null \
    || echo "trainer '${TRAINER_CONTAINER}' already on ${NET} (or not running yet)"
fi

# 5) Smoke-check the sidecar is up (from a throwaway container on the same network).
sleep 1
docker run --rm --network "${NET}" "${IMAGE}" \
  python -c "import urllib.request,sys; print(urllib.request.urlopen('http://${NAME}:${PORT}/health',timeout=5).read().decode())"

echo "sandbox '${NAME}' up on network '${NET}'. In the trainer set:"
echo "    export CODECONTEST_EXEC_URL=http://${NAME}:${PORT}"

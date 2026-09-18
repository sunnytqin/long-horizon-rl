#!/bin/bash
# ============================================================================
# STEP 0 of the Slurm migration: prove the verl SGLang container runs on a
# FASRC GPU node under Singularity before any RL code is ported.
#
# It (1) rebuilds the Singularity sandbox from the *cached* image (no download)
# into a durable, writable location, then (2) runs checks covering every
# container assumption the ported entrypoint will rely on: --nv GPU
# passthrough, real CUDA compute, /dev/shm, the read-only-rootfs / writable-bind
# split, the repo bind + PYTHONPATH (verl import + agent registration), Ray with
# a node-local temp dir, and a live SGLang OpenAI server + chat completion.
#
# Submit:  sbatch verl/slurm_setup/gpu_smoke.sbatch
# Logs:    /n/netscratch/barak_lab/Lab/sqin/slurm_logs/verl_gpu_smoke_<jobid>.{out,err}
#
# SANDBOX LOCATION HISTORY (why it is where it is):
#   - /n/netscratch/.../docker_images/  -> eaten by retention TWICE (2026-07-17,
#     and again by 2026-09-04: /usr/bin down to 208 entries, no python3.12).
#     Partial decay is the nasty mode: the sandbox still "exists" but is broken.
#   - /n/holylabs/.../Lab/sqin/         -> NOT writable by sqin (owner `dam`, ACL
#     grants rwx only to group dam_lab_admin). Job 44399496 died on this.
#   - /n/lab_storage/dam_lab/Lab/sqin/  -> writable + mounted on compute nodes
#     (proved by job 44400009's preflight), but the sandbox build there is
#     I/O-STARVED: 2m40s of CPU in 95min of wall clock, extracting ~500k small
#     files over Isilon NFS. Cancelled. Kept only as the fallback location.
#   - BACK ON /n/netscratch/.../docker_images/ : the MEASURED path (~40 min for
#     this exact build). Retention is the known cost, and the observed decay is
#     per-file atime reclamation (usr/bin thinned to 208 entries while
#     site-packages survived) -> a periodic touch, or just using it, keeps it warm.
# Durability (single-file .sif vs sandbox) is deliberately DEFERRED until the
# container is proven to work on a GPU node -- one question at a time.
# ============================================================================

#SBATCH -J verl_gpu_smoke
#SBATCH -p kempner_h100
#SBATCH --account=kempner_dam_lab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /n/netscratch/barak_lab/Lab/sqin/slurm_logs/verl_gpu_smoke_%j.out
#SBATCH -e /n/netscratch/barak_lab/Lab/sqin/slurm_logs/verl_gpu_smoke_%j.err

# NOT set -e: every check must run so the log shows the full picture.
set -uo pipefail

REPO=/n/home05/sqin/long-horizon-RL/verl
LAB_STORE=/n/lab_storage/dam_lab/Lab/sqin
SANDBOX=${SANDBOX:-/n/netscratch/barak_lab/Lab/sqin/docker_images/verl-sgl0512-dev2}
FALLBACK_SANDBOX=$LAB_STORE/containers/verl-sgl0512-dev2
IMAGE=docker://verlai/verl:sgl0512.dev2
CACHE=/n/netscratch/barak_lab/Lab/sqin/docker_images/sing_cache
MODEL=/n/netscratch/barak_lab/Lab/sqin/models/qwen/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
WORK=/n/netscratch/barak_lab/Lab/sqin/slurm_smoke
NODE_TMP=/scratch/${USER}_${SLURM_JOB_ID:-manual}
PORT=$(( 30000 + ${SLURM_JOB_ID:-0} % 2000 ))
SIM_NAME=smoke-sim

PASSES=0; FAILS=0
banner() { echo; echo "================ $* ================"; }
ok()   { echo "✅ PASS: $*"; PASSES=$((PASSES+1)); }
bad()  { echo "❌ FAIL: $*"; FAILS=$((FAILS+1)); }
summary_and_exit() {
  banner "SUMMARY"
  echo "passed=$PASSES  failed=$FAILS"
  echo "sandbox: $SANDBOX"
  rm -rf "$NODE_TMP"
  if [ "$FAILS" = "0" ]; then echo "🎉 container is GPU-ready under Slurm"; exit 0
  else echo "⚠️  fix the FAILs above before porting the entrypoint"; exit 1; fi
}

banner "[0] Job context"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-none}  gpus_alloc=${SLURM_GPUS_ON_NODE:-?}  port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || bad "host nvidia-smi"
singularity --version

# ---------------------------------------------------------------------------
# PREFLIGHT. Job 44399496 wasted a queue slot and reported 10 bogus failures
# because ONE unwritable parent dir meant there was no container to exec. Prove
# every write target is usable ON THIS NODE first, and never let a missing
# container cascade into fake per-check results.
# ---------------------------------------------------------------------------
banner "[0b] Preflight: are the write targets usable on this node?"
writable() { mkdir -p "$1" 2>/dev/null && touch "$1/.wt" 2>/dev/null && rm -f "$1/.wt"; }
if writable "$(dirname "$SANDBOX")"; then
  ok "sandbox parent writable: $(dirname "$SANDBOX")"
else
  echo "⚠️  $(dirname "$SANDBOX") not writable/mounted on this node -> FALLING BACK to netscratch"
  echo "    (lab_storage may not be mounted on compute nodes; this run still answers the GPU question)"
  SANDBOX="$FALLBACK_SANDBOX"
  if writable "$(dirname "$SANDBOX")"; then ok "fallback sandbox parent writable"
  else bad "no writable sandbox location at all"; summary_and_exit; fi
fi
for d in "$WORK/fakehome" "$NODE_TMP/ray" "$NODE_TMP/triton" "$NODE_TMP/inductor" "$NODE_TMP/xdg" "$NODE_TMP/outlines"; do
  writable "$d" || { bad "not writable: $d"; summary_and_exit; }
done
ok "scratch + node-local dirs writable"
df -h "$(dirname "$SANDBOX")" | tail -1

# --- The exec recipe the ported entrypoint will use ------------------------
# --nv          : GPU passthrough (the Docker `--gpus all` equivalent)
# --pwd $REPO   : mimics the Docker WORKDIR=/workspace/verl the entrypoints assume
# NO --cleanenv : it drops the image env -> falls back to /usr/bin/python (no sglang)
# NO `bash -lc` : it sources the host ~/.bashrc and leaks the host conda env
run() {
  singularity exec --nv \
    --pwd "$REPO" \
    --bind "$REPO" --bind /n/netscratch --bind /n/lab_storage --bind "$NODE_TMP" \
    --env PYTHONPATH="$REPO" \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    --home "$WORK/fakehome" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env RAY_TMPDIR="$NODE_TMP/ray" \
    --env TRITON_CACHE_DIR="$NODE_TMP/triton" \
    --env TORCHINDUCTOR_CACHE_DIR="$NODE_TMP/inductor" \
    --env XDG_CACHE_HOME="$NODE_TMP/xdg" \
    --env OUTLINES_CACHE_DIR="$NODE_TMP/outlines" \
    "$SANDBOX" "$@"
}

# ---------------------------------------------------------------------------
banner "[1] Sandbox build (download-free from the 16G cache)"
if [ -x "$SANDBOX/usr/bin/python" ] && [ -x "$SANDBOX/bin/bash" ]; then
  echo "[build] complete sandbox already present -- skipping"
  ok "sandbox present"
else
  echo "[build] building/rebuilding at $SANDBOX ..."
  export SINGULARITY_DISABLE_CACHE=false
  export SINGULARITY_CACHEDIR="$CACHE"    # holds the image layers -> no download
  export SINGULARITY_TMPDIR=/scratch      # node-local: extracting 37G onto NFS is slow
  time singularity build --force --sandbox "$SANDBOX" "$IMAGE"
  if [ -x "$SANDBOX/usr/bin/python" ]; then ok "sandbox built"; else bad "sandbox build (no /usr/bin/python)"; fi
fi

# HARD GATE: without a usable container every check below is meaningless noise.
if ! run true 2>/dev/null; then
  bad "container is not executable -- skipping checks 2-8 (they would all be false failures)"
  summary_and_exit
fi

# ---------------------------------------------------------------------------
banner "[2] Container versions"
if run python -c "
import sys, sglang, torch, ray
print('python ', sys.version.split()[0])
print('sglang ', sglang.__version__)
print('torch  ', torch.__version__)
print('ray    ', ray.__version__)
assert sys.version_info >= (3, 11), 'verl v0.8 needs py>=3.11 (enum.StrEnum)'
"; then ok "versions + py>=3.11"; else bad "versions"; fi

# ---------------------------------------------------------------------------
banner "[3] --nv GPU passthrough + real CUDA compute"
run nvidia-smi -L || bad "nvidia-smi inside container (--nv broken)"
if run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available inside container'
n = torch.cuda.device_count()
print('device_count', n, '| name', torch.cuda.get_device_name(0))
a = torch.randn(4096, 4096, device='cuda'); b = torch.randn(4096, 4096, device='cuda')
c = (a @ b).sum().item(); torch.cuda.synchronize()
print('matmul ok, sum =', c)
print('free/total GiB', [round(x/2**30, 1) for x in torch.cuda.mem_get_info()])
"; then ok "GPU visible + matmul"; else bad "GPU compute"; fi

# ---------------------------------------------------------------------------
banner "[4] /dev/shm (Docker's 64MB default is the classic NCCL/sglang killer)"
run df -h /dev/shm
if run python -c "
import shutil
total = shutil.disk_usage('/dev/shm').total / 2**30
print(f'/dev/shm total = {total:.1f} GiB')
assert total >= 8, 'too small for sglang/NCCL shared memory'
"; then ok "/dev/shm >= 8GiB"; else bad "/dev/shm too small"; fi

# ---------------------------------------------------------------------------
banner "[5] Read-only rootfs vs writable binds (the sandbox is NOT writable)"
if run touch /usr/local/lib/_probe 2>/dev/null; then
  bad "container rootfs is WRITABLE (unexpected -- the port's write-redirect assumption is off)"
  run rm -f /usr/local/lib/_probe
else
  ok "container rootfs read-only (as expected -> all writes must go to binds)"
fi
if run bash -c "echo hi > $WORK/_probe && cat $WORK/_probe && rm -f $WORK/_probe"; then
  ok "bound netscratch dir writable from inside"
else bad "bound netscratch dir NOT writable from inside"; fi
if run bash -c "echo hi > $NODE_TMP/_probe && rm -f $NODE_TMP/_probe"; then
  ok "node-local /scratch bind writable from inside"
else bad "node-local /scratch NOT writable from inside"; fi

# ---------------------------------------------------------------------------
banner "[6] Repo bind + PYTHONPATH: verl import + agent-loop registration"
if run python -c "
import verl
from colbench.colbench_agent import ColBenchAgentLoop
from colbench.colbench_spec_agent import ColBenchSpecAgentLoop
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
print('registry:', sorted(_agent_loop_registry))
assert 'colbench_agent' in _agent_loop_registry
assert 'colbench_spec_agent' in _agent_loop_registry
"; then ok "verl + colbench agent loops registered"; else bad "verl/colbench import or registration"; fi

# ---------------------------------------------------------------------------
banner "[7] Ray init with a node-local temp dir (never Ray on NFS)"
if run python -c "
import os, ray
ray.init(num_gpus=1, include_dashboard=False, _temp_dir=os.environ['RAY_TMPDIR'], logging_level='WARNING')
print('ray resources:', {k: v for k, v in ray.cluster_resources().items() if k in ('CPU', 'GPU')})
@ray.remote(num_gpus=1)
def probe():
    import torch
    return torch.cuda.get_device_name(0)
print('ray GPU actor sees:', ray.get(probe.remote()))
ray.shutdown()
"; then ok "ray + GPU actor"; else bad "ray"; fi

# ---------------------------------------------------------------------------
banner "[8] Live SGLang OpenAI server (the frozen-sim path) on 1 GPU"
SERVER_LOG="$WORK/sglang_${SLURM_JOB_ID:-manual}.log"
run python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SIM_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tp 1 --mem-fraction-static 0.85 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

# Singularity shares the host network namespace, so 127.0.0.1 is reachable from here.
READY=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
  kill -0 $SERVER_PID 2>/dev/null || { echo "[sglang] server process died after ~$((i*5))s"; break; }
  sleep 5
done
if [ "$READY" = "1" ]; then
  ok "sglang /health up (127.0.0.1:$PORT reachable from the host)"
  echo "--- chat completion ---"
  if curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SIM_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: PONG\"}],\"max_tokens\":16,\"temperature\":0}" \
      | tee "$WORK/completion_${SLURM_JOB_ID:-manual}.json" | grep -qi pong; then
    ok "chat completion returned PONG"
  else bad "chat completion (see $WORK/completion_*.json)"; fi
else
  bad "sglang server never became healthy -- tail of $SERVER_LOG:"
  tail -40 "$SERVER_LOG"
fi
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
trap - EXIT

echo "sglang server log: $SERVER_LOG"
summary_and_exit

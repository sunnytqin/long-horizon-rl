#!/bin/bash
# Minimal single-GPU Slurm harness for Phase-0 spec generation + diagnostic on FASRC (Cannon).
#
# Uses the persistent `openrlhf` conda env (holylabs, NOT scratch) to serve ONE vllm
# OpenAI endpoint, then runs generate_specs.py + diagnose_specs.py against it. Grading uses
# the in-process fallback (CODECONTEST_ALLOW_INPROCESS=1) -- no exec sidecar / container.
#
# vllm vs sglang is irrelevant here: the scripts are plain OpenAI-API clients (top_k/min_p go
# through extra_body, which vllm accepts; Qwen3-4B-Instruct-2507 is non-thinking so no thinking
# kwarg). We use the conda env instead of the Singularity sandbox because the sandbox lives on
# netscratch and got purged (dangling symlinks, missing /usr/bin/python) -- the env is durable.
#
# The SAME served model plays both roles (author + diagnostic solver): a true self-play run.
#
# Submit:   sbatch verl/colbench/selfplay/run_specs_slurm.sh
# Override: MAX_ROWS=200 BACKEND=strong sbatch ... run_specs_slurm.sh
#
#SBATCH -c 16
#SBATCH -t 0-02:00
#SBATCH -p kempner_h100
#SBATCH --mem=80G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=colbench_specs
#SBATCH -o /n/home05/sqin/long-horizon-RL/verl/colbench/selfplay/slurm_out/specs-%j.out
#SBATCH -e /n/home05/sqin/long-horizon-RL/verl/colbench/selfplay/slurm_out/specs-%j.out

set -euo pipefail

# ── Config (all overridable via env) ─────────────────────────────────────────
REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
CONDA_SH=${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf}
DATA_FILE=${DATA_FILE:-/n/home05/sqin/long-horizon-RL/InfoPO/data/colbench_code/train.parquet}
OUT_DIR=${OUT_DIR:-/n/netscratch/dam_lab/Lab/sqin/colbench_specs}

# HF cache dir for the model to serve; resolve the snapshot (weights) subdir.
# Default = Qwen3-4B-Instruct-2507 (the ColBench training model) -> BACKEND=selfplay is genuine
# self-play authoring. It is the non-thinking Instruct variant (enable_thinking left unset).
MODEL_HFDIR=${MODEL_HFDIR:-/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen3-4B-Instruct-2507}
MODEL=${MODEL:-$(ls -d "$MODEL_HFDIR"/snapshots/*/ 2>/dev/null | head -1)}
SERVED_NAME=${SERVED_NAME:-specgen}

BACKEND=${BACKEND:-selfplay}        # label written into the spec records: strong | selfplay
MODE=${MODE:-static}                # static (complete-requirements) | plot (requirements + tailored plot)
MAX_ROWS=${MAX_ROWS:-100}           # small slice for the Phase-0 gate
N_SAMPLES=${N_SAMPLES:-1}           # solver samples per spec in the diagnostic (pass@n)
PORT=${PORT:-30000}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}

# Author endpoint: default = the local vLLM server (self-play). Set GEN_VENDOR=openai to author
# with an external teacher (e.g. gpt-5.4-mini) while the DIAGNOSTIC still uses the local solver,
# so strong-vs-self specs are graded by the SAME solver. Key is read from a file (never argv/log).
GEN_VENDOR=${GEN_VENDOR:-vllm}
GEN_OPENAI_MODEL=${GEN_OPENAI_MODEL:-gpt-5.4-mini}
GEN_OPENAI_BASE_URL=${GEN_OPENAI_BASE_URL:-https://api.openai.com/v1}
GEN_API_KEY_FILE=${GEN_API_KEY_FILE:-$HOME/.openai_key}
GEN_CONCURRENCY=${GEN_CONCURRENCY:-8}   # gentler default for a rate-limited external API

STEM=$(basename "$DATA_FILE" .parquet)
# static -> stem.backend.jsonl ; plot -> stem.backend.plot.jsonl (both diagnose on requirements)
if [ "$MODE" = "static" ]; then TAG="$BACKEND"; else TAG="$BACKEND.$MODE"; fi
# RUN_TAG (optional): suffix for one-off experiments so the primary deliverables aren't clobbered.
[ -n "${RUN_TAG:-}" ] && TAG="$TAG.$RUN_TAG"
SPECS_OUT="$OUT_DIR/specs/${STEM}.${TAG}.jsonl"
DIAG_OUT="$OUT_DIR/specs/${STEM}.${TAG}.diagnostic.json"
mkdir -p "$OUT_DIR/specs"

echo "[harness] MODEL=$MODEL"
echo "[harness] DATA_FILE=$DATA_FILE  MAX_ROWS=$MAX_ROWS  BACKEND=$BACKEND  MODE=$MODE"
echo "[harness] SPECS_OUT=$SPECS_OUT"
[ -z "$MODEL" ] && { echo "[harness] ERROR: no model snapshot found under $MODEL_HFDIR"; exit 1; }

# ── Environment ──────────────────────────────────────────────────────────────
# conda activation scripts (MKL etc.) are not `set -u` clean -> relax nounset around them.
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO"
export CODECONTEST_ALLOW_INPROCESS=1
echo "[harness] python=$(which python)  vllm=$(python -c 'import vllm;print(vllm.__version__)')"

# ── 1. Launch the vllm OpenAI server (background) ────────────────────────────
echo "[harness] starting vllm server on 127.0.0.1:$PORT ..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" --gpu-memory-utilization 0.85 \
  --max-model-len "$MAX_MODEL_LEN" --enforce-eager &
SERVER_PID=$!
trap 'echo "[harness] tearing down server $SERVER_PID"; kill $SERVER_PID 2>/dev/null || true' EXIT

# ── 2. Wait for health ───────────────────────────────────────────────────────
echo "[harness] waiting for /health ..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "[harness] server up (after ${i}0s)"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[harness] ERROR: server died during startup"; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "[harness] ERROR: server never became healthy"; exit 1; }

BASE_URL="http://127.0.0.1:$PORT/v1"

# ── 3. Generate specs (resumable) ────────────────────────────────────────────
# Author endpoint: local vLLM (self-play) OR external OpenAI teacher (strong). Solver stays local.
if [ "$GEN_VENDOR" = "openai" ]; then
  GEN_ARGS=(--gen_vendor openai --gen_base_url "$GEN_OPENAI_BASE_URL" --gen_model "$GEN_OPENAI_MODEL"
            --gen_api_key_file "$GEN_API_KEY_FILE" --concurrency "$GEN_CONCURRENCY")
  echo "[harness] author endpoint = OpenAI ($GEN_OPENAI_MODEL); key from $GEN_API_KEY_FILE"
  [ -s "$GEN_API_KEY_FILE" ] || { echo "[harness] ERROR: key file $GEN_API_KEY_FILE is empty/missing"; exit 1; }
else
  GEN_ARGS=(--gen_vendor vllm --gen_base_url "$BASE_URL" --gen_model "$SERVED_NAME")
  echo "[harness] author endpoint = local vLLM ($SERVED_NAME)"
fi
echo "[harness] === generate_specs (backend=$BACKEND mode=$MODE) ==="
python -m colbench.selfplay.generate_specs \
  --data_file "$DATA_FILE" --max_rows "$MAX_ROWS" \
  --backend "$BACKEND" --mode "$MODE" \
  "${GEN_ARGS[@]}" \
  --out "$SPECS_OUT"

# ── 4. Full-spec solve-rate diagnostic (in-process grading) ──────────────────
# Grades the authored 'requirements' (does a solver reconstruct GT behavior from them alone).
# The 'plot' shapes only the Phase-1 dialogue and is eyeballed, not graded here.
# SKIP_DIAG=1 skips this (e.g. full-dataset generation, where faithfulness is already validated).
if [ -n "${SKIP_DIAG:-}" ]; then
  echo "[harness] SKIP_DIAG set -> skipping diagnostic (generation only)."
else
  echo "[harness] === diagnose_specs ==="
  python -m colbench.selfplay.diagnose_specs \
    --data_file "$DATA_FILE" --max_rows "$MAX_ROWS" \
    --specs "$SPECS_OUT" \
    --solver_base_url "$BASE_URL" --solver_model "$SERVED_NAME" \
    --n_samples "$N_SAMPLES" \
    --out "$DIAG_OUT"
fi

echo "[harness] DONE. specs -> $SPECS_OUT ; diagnostic -> $DIAG_OUT"

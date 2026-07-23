#!/bin/bash
# Single-GPU Slurm harness to EYEBALL the spec-path simulation on FASRC (Cannon).
#
# Serves ONE vLLM OpenAI endpoint (base Qwen3-4B) and runs colbench/validate_colbench_spec.py
# with --solver_backend openai, so BOTH roles -- the assistant (solver) and the user-simulator --
# hit the SAME served model, differing only by system prompt (COLBENCH_SPEC_AGENT_SYSTEM_PROMPT
# vs SPEC_SIM_SYSTEM_PROMPT) and sampling. This produces a batch of real solver<->sim
# conversations + spec metrics (terminated_by, showed_code_rate, false_terminate_rate) to read.
#
# Uses the durable `openrlhf` conda env (NOT the container): the harness is pure OpenAI-API +
# in-process grading (CODECONTEST_ALLOW_INPROCESS=1), no exec sidecar. Mirrors run_specs_slurm.sh.
#
# Submit:   sbatch verl/colbench/run_validate_spec_slurm.sh
# Override: VAL_FILE=~/data/colbench_spec/selfplay_1k.parquet MAX_PROBLEMS=100 N_SAMPLES=4 \
#           TEMPERATURE=0.6 sbatch verl/colbench/run_validate_spec_slurm.sh
#
#SBATCH -c 16
#SBATCH -t 0-02:00
#SBATCH -p kempner
#SBATCH --mem=80G
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=colbench_spec_eval
#SBATCH -o /n/home05/sqin/long-horizon-RL/verl/colbench/slurm_out/spec-eval-%j.out
#SBATCH -e /n/home05/sqin/long-horizon-RL/verl/colbench/slurm_out/spec-eval-%j.out

set -euo pipefail

# ── Config (all overridable via env) ─────────────────────────────────────────
REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
CONDA_SH=${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf}

MODEL_HFDIR=${MODEL_HFDIR:-/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen3-4B-Instruct-2507}
MODEL=${MODEL:-$(ls -d "$MODEL_HFDIR"/snapshots/*/ 2>/dev/null | head -1)}
SERVED_NAME=${SERVED_NAME:-colbench-sim}

VAL_FILE=${VAL_FILE:-$HOME/data/colbench_spec/selfplay_cond30.parquet}
OUT_DIR=${OUT_DIR:-$REPO/colbench/runs/spec}
RUN_TAG=${RUN_TAG:-$(basename "$VAL_FILE" .parquet)}
OUT="$OUT_DIR/spec_eval_${RUN_TAG}.json"

# Simulation knobs (defaults mirror the spec design / run_colbench_grpo.sh where applicable).
N_SAMPLES=${N_SAMPLES:-2}
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_PROBLEMS=${MAX_PROBLEMS:-}                 # empty = all rows in VAL_FILE
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-10}
MAX_CODE_PROPOSALS=${MAX_CODE_PROPOSALS:-2}
MAX_NEW_TOKENS_PER_TURN=${MAX_NEW_TOKENS_PER_TURN:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-14336}
PORT=${PORT:-30000}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
# Sim sampling (Qwen3-Instruct recommendation; read by env_spec.openai_sim_backend).
export SIM_TEMPERATURE=${SIM_TEMPERATURE:-0.7}
export SIM_TOP_P=${SIM_TOP_P:-0.8}
# Generation-time token bound on each user (sim) turn (replaces the old 400-char post-hoc slice
# that chopped verbose replies mid-sentence). Brevity is driven by the prompt; this just caps runaway.
export SIM_MAX_TOKENS=${SIM_MAX_TOKENS:-256}

# ── Frozen user-simulator backend (comparison study) ─────────────────────────
# SIM_BACKEND=local (default): sim is the SAME local Qwen server as the solver (self-play). Here on
#   FASRC that local server is vLLM; the container serves SGLang instead -- same transport, so the
#   label names the transport, not the engine. ('vllm' still works as a legacy alias.)
# SIM_BACKEND=openai        : sim is a hosted GPT (SIM_OPENAI_MODEL); SOLVER stays on local Qwen.
#   Needs OPENAI_API_KEY -- read from GEN_API_KEY_FILE (~/.openai_key, same as spec gen).
#   Example: SIM_BACKEND=openai SIM_OPENAI_MODEL=gpt-5.4-mini sbatch run_validate_spec_slurm.sh
SIM_BACKEND=${SIM_BACKEND:-local}
SIM_OPENAI_MODEL=${SIM_OPENAI_MODEL:-gpt-5.4-mini}
SIM_OPENAI_BASE_URL=${SIM_OPENAI_BASE_URL:-https://api.openai.com/v1}
GEN_API_KEY_FILE=${GEN_API_KEY_FILE:-$HOME/.openai_key}
# Tag the output so vllm-sim vs openai-sim runs land in distinct files.
[ "$SIM_BACKEND" = "openai" ] && OUT="$OUT_DIR/spec_eval_${RUN_TAG}_sim-${SIM_OPENAI_MODEL}.json"

mkdir -p "$OUT_DIR" "$REPO/colbench/slurm_out"
echo "[harness] MODEL=$MODEL"
echo "[harness] VAL_FILE=$VAL_FILE  N_SAMPLES=$N_SAMPLES  TEMPERATURE=$TEMPERATURE  OUT=$OUT"
[ -z "$MODEL" ] && { echo "[harness] ERROR: no model snapshot found under $MODEL_HFDIR"; exit 1; }

# ── Environment ──────────────────────────────────────────────────────────────
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
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "[harness] ERROR: server never healthy"; exit 1; }

BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_BASE_URL="$BASE_URL"          # the sim (env_spec) reads these two
export MULTITURN_MODEL_NAME="$SERVED_NAME"

# ── 3. Run the full solver<->sim simulation + eval ───────────────────────────
EXTRA=()
[ -n "$MAX_PROBLEMS" ] && EXTRA+=(--max_problems "$MAX_PROBLEMS")
if [ "$SIM_BACKEND" = "openai" ]; then
  [ -f "$GEN_API_KEY_FILE" ] || { echo "[harness] ERROR: no OpenAI key at $GEN_API_KEY_FILE"; exit 1; }
  export OPENAI_API_KEY="$(tr -d ' \t\r\n' < "$GEN_API_KEY_FILE")"
  EXTRA+=(--sim_backend openai --sim_model "$SIM_OPENAI_MODEL" --sim_base_url "$SIM_OPENAI_BASE_URL"
          --sim_temperature "$SIM_TEMPERATURE" --sim_top_p "$SIM_TOP_P")
  echo "[harness] === validate_colbench_spec (Qwen solver @ $SERVED_NAME  vs  OpenAI sim '$SIM_OPENAI_MODEL') ==="
else
  echo "[harness] === validate_colbench_spec (openai solver; both roles on $SERVED_NAME) ==="
fi
python "$REPO/colbench/validate_colbench_spec.py" \
  --solver_backend openai --base_url "$BASE_URL" --served_model "$SERVED_NAME" \
  --model "$MODEL" \
  --val_file "$VAL_FILE" --out "$OUT" \
  --n_samples "$N_SAMPLES" --temperature "$TEMPERATURE" \
  --max_assistant_turns "$MAX_ASSISTANT_TURNS" --max_code_proposals "$MAX_CODE_PROPOSALS" \
  --max_new_tokens_per_turn "$MAX_NEW_TOKENS_PER_TURN" --max_response_length "$MAX_RESPONSE_LENGTH" \
  "${EXTRA[@]}"

echo "[harness] DONE. eval -> $OUT"

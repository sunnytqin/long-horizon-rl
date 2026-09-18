#!/bin/bash
# Phase B of the sim-training PoC: collect prefixes, then draw candidates. ONE GPU, ~1 hour.
#
# Serves TWO vLLM OpenAI endpoints on a single card:
#   :30000  the PARTNER assistant -- the MERGED global_step_200 checkpoint, i.e. the
#           demonstrated hacker (val reward 0.932, sim_leak_frac 0.635 against 0.251 at init).
#           Step 300 is post-collapse and is NOT the partner.
#   :30001  the frozen BASE simulator -- Qwen3-4B-Instruct-2507, the same weights the training
#           runs serve, so the collected dialogues and the candidate draws come from the sim
#           this project is actually trying to improve.
# ONE CARD EACH, not two servers on one card. vLLM's --gpu-memory-utilization is a fraction of
# TOTAL device memory but is CHECKED against FREE memory at init, so a second engine started on a
# card the first one already occupies computes a negative cache budget and dies with
# "No available memory for the cache blocks" (job 45701054) -- and the arithmetic that avoids
# that (second server at 0.85 so its share nets out to 0.45) is fragile and silent when wrong.
# Two 4B models for an hour is not worth that; pin one per device instead.
#
# WHY ONE JOB FOR BOTH STAGES: candidate drawing needs only the sim server, which this job
# already has up. Splitting it into a second submission buys nothing and pays a second GPU
# queue wait. STAGE=collect or STAGE=candidates runs either half alone (e.g. to re-draw at a
# different --k without re-running the dialogues).
#
# Uses the durable `openrlhf` conda env, not the container: both stages are pure OpenAI-API
# clients with no grading, no exec sidecar and no in-process engine. Mirrors
# run_validate_spec_slurm.sh.
#
# Submit:  sbatch verl/colbench/simtrain/run_collect_slurm.sh
# Resume:  re-submit the SAME command. Both stages skip what is already on disk (episodes for
#          the collector, prefix_ids for the candidates), so a preempted job costs one episode
#          per worker and nothing else.
# Pilot:   END=40 sbatch ...            (a few dozen episodes, to read turns/episode off)
#
#SBATCH -c 16
#SBATCH -t 0-04:00
#SBATCH -p kempner_h100
#SBATCH --mem=100G
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=simtrain_collect
#SBATCH -o /n/home05/sqin/long-horizon-RL/verl/colbench/simtrain/slurm_out/collect-%j.out
#SBATCH -e /n/home05/sqin/long-horizon-RL/verl/colbench/simtrain/slurm_out/collect-%j.out

set -euo pipefail

REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
CONDA_SH=${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf}

# shellcheck source=../../slurm_setup/paths.sh
source "$REPO/slurm_setup/paths.sh"

STAGE=${STAGE:-both}                       # collect | candidates | both

# ── ARM ───────────────────────────────────────────────────────────────────────
# gt       = the hidden-code sim (env.ColBenchUserSimEnv), the original PoC.
# grounded = the spec path's grounded sim: conditioned on the GT SOURCE + the
#            plot (+colbench.grounded_sim). This is the arm `qwen3_4b_spec_rej8`
#            trains against, so its defaults below are lifted from that run's
#            own launch record rather than guessed:
#              --model=Qwen/Qwen3-4B --spec_author=gpt-5.4 --grounded_sim
#              --sim_reject_max_tries=8 --sim_code_leak_detector=a0_strict
#              --noearly_term_guard --max_code_proposals=2
# Every one of those is a distribution knob: collecting under a different value
# yields episodes the training run never saw.
ARM=${ARM:-gt}
case "$ARM" in gt|spec|grounded) ;; *) echo "[harness] ERROR: ARM must be gt|spec|grounded"; exit 1 ;; esac

# ── Models ────────────────────────────────────────────────────────────────────
# PARTNER_MODEL must be a MERGED HF directory. The raw checkpoint under
# .../global_step_200/actor is FSDP shards plus a huggingface/ dir that holds config and
# tokenizer and NO WEIGHTS -- vLLM will happily start on it and serve nonsense, so this script
# checks for a weights file rather than for the directory.
# The role_restraint run's step 200: reward 0.820, sim_leak_frac 0.440 (vs 0.022 at init),
# entropy 0.41 -- a demonstrated hacker, still competent, and BEFORE the entropy explosion
# that starts around step 240. NOT the old baseline's step 200: a partner trained against the
# default sim learned to hack a simulator this arm no longer uses.
if [ "$ARM" != gt ]; then
  # THE PARTNER IS THE BASE MODEL, not a trained checkpoint. Training the sim
  # against a partner already trained to hack it is ill-posed as the first step
  # of alternating optimisation -- and on the GT path the swap alone moved the
  # sim's code-leak rate 0.404 -> 0.034, while the hacker prefixes yielded zero
  # usable targets. Same argument here.
  # Resolve through the REGISTRY, not by constructing a cache path: paths.sh
  # maps the training run's `--model=Qwen/Qwen3-4B` onto the Qwen3-4B-Instruct-2507
  # snapshot, so a hand-built models--Qwen--Qwen3-4B path points at nothing.
  # The registry is what the training run itself resolves through, so this is
  # the same weights by construction rather than by coincidence.
  BASE_MODEL_ID=${BASE_MODEL_ID:-Qwen/Qwen3-4B}
  _BASE_META=$(resolve_model_meta "$BASE_MODEL_ID")
  [ -n "$_BASE_META" ] || { echo "[harness] ERROR: $BASE_MODEL_ID is not in paths.sh::resolve_model_meta"; exit 1; }
  read -r _BASE_SNAP _ <<< "$_BASE_META"
  PARTNER_MODEL=${PARTNER_MODEL:-$_BASE_SNAP}
  PARTNER_NAME=${PARTNER_NAME:-partner-base}
  # The grounded arm serves NO separate sim model: launch_slurm.sh resolves
  # `_sim_identity="${SIM_MODEL:-${MODEL}}"`, and qwen3_4b_spec_rej8 set no
  # --sim_model, so the frozen sim IS the same base weights as the partner.
  SIM_MODEL=${SIM_MODEL:-$_BASE_SNAP}
  DATA_FILE=${DATA_FILE:-$DATA_ROOT/colbench_spec/train.gpt-5.4.parquet}
  RUN_TAG=${RUN_TAG:-grounded_base}
  # The grounded sim's system prompt is built by templates.build_grounded_sim_messages
  # from the GT source + plot. resolve_sim_system's role/role_restraint teachers
  # are GT-PATH prompts carrying neither, so they must never be applied here;
  # `default` replays each prefix's own recorded system string.
  SIM_SYSTEM=${SIM_SYSTEM:-default}
  if [ "$SIM_SYSTEM" != default ]; then
    echo "[harness] ERROR: ARM=$ARM needs SIM_SYSTEM=default."
    echo "[harness]   role/role_restraint are GT-path prompts with no ground truth"
    echo "[harness]   and no plot in them; applying one here would draw candidates"
    echo "[harness]   from a simulator that cannot see what it is answering from."
    exit 1
  fi
  SIM_MAX_TRIES=${SIM_MAX_TRIES:-8}
  MAX_CODE_PROPOSALS=${MAX_CODE_PROPOSALS:-2}
  SIM_CODE_LEAK_DETECTOR=${SIM_CODE_LEAK_DETECTOR:-a0_strict}
  # qwen3_4b_spec_rej8 launched with --noearly_term_guard. Collecting WITH the
  # guard would drop exactly the premature terminations that run saw.
  EARLY_TERM_GUARD=${EARLY_TERM_GUARD:-False}
fi
PARTNER_MODEL=${PARTNER_MODEL:-$SCRATCH_ROOT/models/local/qwen3_4b_restraint_gs200}
PARTNER_NAME=${PARTNER_NAME:-partner-restraint-gs200}
SIM_MODEL_DIR=${SIM_MODEL_DIR:-$MODEL_ROOT/models--Qwen--Qwen3-4B-Instruct-2507}
SIM_MODEL=${SIM_MODEL:-$(resolve_hf_snapshot "$SIM_MODEL_DIR")}
SIM_NAME=${SIM_NAME:-colbench-sim}

DATA_FILE=${DATA_FILE:-$DATA_ROOT/colbench/train.fence.parquet}
RUN_TAG=${RUN_TAG:-gs200}
OUT_DIR=${OUT_DIR:-$SIMTRAIN_ROOT/$RUN_TAG}
START=${START:-0}
END=${END:-600}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-1}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-10}
CONCURRENCY=${CONCURRENCY:-16}
K=${K:-8}
MAX_PREFIXES=${MAX_PREFIXES:-}             # empty = all; set for a pilot slice
# WHICH system prompt the simulator gets, for BOTH stages. This must match the arm the
# production training run serves: episodes played against one simulator are off-distribution
# context for another, and candidates drawn under a different system prompt would answer the
# same materialized user message in a different voice. role_restraint is the arm as of
# 2026-09-10 (see slurm_setup/README.md 5.2b and the job-45772513 result).
SIM_SYSTEM=${SIM_SYSTEM:-role_restraint}

PARTNER_PORT=${PARTNER_PORT:-30000}
SIM_PORT=${SIM_PORT:-30001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_FRAC=${GPU_FRAC:-0.85}

# Sampling. Defaults are run_colbench_grpo.sh's rollout config for the solver and
# env._sim_sampling + SIM_MAX_TOKENS=256 for the sim, so the collected data comes from the
# same distributions training does.
SOLVER_TEMPERATURE=${SOLVER_TEMPERATURE:-0.7}
SOLVER_TOP_P=${SOLVER_TOP_P:-0.8}
SOLVER_TOP_K=${SOLVER_TOP_K:-20}
SOLVER_MAX_TOKENS=${SOLVER_MAX_TOKENS:-1024}
SIM_TEMPERATURE_ARG=${SIM_TEMPERATURE_ARG:-0.7}
SIM_TOP_P_ARG=${SIM_TOP_P_ARG:-0.8}
SIM_TOP_K_ARG=${SIM_TOP_K_ARG:-20}
SIM_MAX_TOKENS_ARG=${SIM_MAX_TOKENS_ARG:-256}

# THE SILENT ONE. env._finalize_reply reads SIM_CHAR_LIMIT per call and DEFAULTS IT TO 400.
# run_colbench_grpo.sh exports 0, so leaving this unset would build the entire dataset from
# 400-char-truncated replies that training never sees. Both scripts hard-fail if it is unset;
# exporting it here is what makes that check pass.
export SIM_CHAR_LIMIT=${SIM_CHAR_LIMIT:-0}
export PYTHONUNBUFFERED=1

STEM=$(basename "$DATA_FILE" .parquet)
TAG=${TAG:-${STEM}.c1}
PREFIX_FILE="$OUT_DIR/prefixes.${TAG}.jsonl"
CAND_FILE="$OUT_DIR/candidates.${TAG}.jsonl"

mkdir -p "$OUT_DIR" "$REPO/colbench/simtrain/slurm_out"
echo "[harness] node=$(hostname -s)  stage=$STAGE"
echo "[harness] partner=$PARTNER_MODEL"
echo "[harness] sim=$SIM_MODEL"
echo "[harness] data=$DATA_FILE rows [$START,$END)  out=$OUT_DIR"
echo "[harness] SIM_CHAR_LIMIT=$SIM_CHAR_LIMIT  SIM_SYSTEM=$SIM_SYSTEM  ARM=$ARM"
[ "$ARM" = gt ] || echo "[harness] grounded: sim_max_tries=${SIM_MAX_TRIES:-} max_code_proposals=${MAX_CODE_PROPOSALS:-} detector=${SIM_CODE_LEAK_DETECTOR:-} early_term_guard=${EARLY_TERM_GUARD:-}"

# A merged HF dir has weights; the raw FSDP checkpoint's huggingface/ subdir does not.
# (On the grounded arm the partner is a plain HF snapshot, so this only ever
# catches a bad path -- but an empty directory must still fail loudly rather
# than have vLLM start and serve nonsense.)
_n_weights=$(find "$PARTNER_MODEL" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l)
if [ ! -f "$PARTNER_MODEL/config.json" ] || [ "$_n_weights" -eq 0 ]; then
  if [ "$ARM" != gt ]; then
    echo "[harness] ERROR: no weights at PARTNER_MODEL=$PARTNER_MODEL"
    echo "[harness]   Stage the base solver: bash slurm_setup/stage_model.sh Qwen/Qwen3-4B"
    exit 1
  fi
  echo "[harness] ERROR: $PARTNER_MODEL has no weights. Merge the checkpoint first:"
  echo "[harness]   python -m verl.model_merger merge --backend fsdp \\"
  echo "[harness]     --local_dir $RUN_ROOT_BASE/colbench_mt/qwen3_4b_nonspec_role_restraint/checkpoints/global_step_200/actor \\"
  echo "[harness]     --target_dir $PARTNER_MODEL"
  exit 1
fi
[ -n "$SIM_MODEL" ] || { echo "[harness] ERROR: no snapshot under $SIM_MODEL_DIR"; exit 1; }

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO"
echo "[harness] python=$(which python)  vllm=$(python -c 'import vllm;print(vllm.__version__)')"

PIDS=()
trap 'echo "[harness] tearing down servers ${PIDS[*]:-}"; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done' EXIT

serve() {  # serve <model_path> <served_name> <port> <cuda_device>
  echo "[harness] starting vllm '$2' on 127.0.0.1:$3 (cuda:$4) ..."
  CUDA_VISIBLE_DEVICES="$4" python -m vllm.entrypoints.openai.api_server \
    --model "$1" --served-model-name "$2" \
    --host 127.0.0.1 --port "$3" \
    --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_FRAC" \
    --max-model-len "$MAX_MODEL_LEN" --enforce-eager &
  PIDS+=($!)
}

wait_health() {  # wait_health <port> <pid>
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { echo "[harness] :$1 up after ${i}0s"; return 0; }
    kill -0 "$2" 2>/dev/null || { echo "[harness] ERROR: server on :$1 died during startup"; exit 1; }
    sleep 10
  done
  echo "[harness] ERROR: server on :$1 never healthy"; exit 1
}

# The sim server is needed by BOTH stages; the partner only by the collector, so a
# STAGE=candidates re-run leaves the whole card to the sim.
serve "$SIM_MODEL" "$SIM_NAME" "$SIM_PORT" 0
SIM_PID=${PIDS[-1]}
if [ "$STAGE" != "candidates" ]; then
  serve "$PARTNER_MODEL" "$PARTNER_NAME" "$PARTNER_PORT" 1
  PARTNER_PID=${PIDS[-1]}
  wait_health "$PARTNER_PORT" "$PARTNER_PID"
fi
wait_health "$SIM_PORT" "$SIM_PID"

if [ "$STAGE" != "candidates" ]; then
  echo "[harness] === Stage 1: collect_prefixes ==="
  python -m colbench.simtrain.collect_prefixes \
    --data_file "$DATA_FILE" --start "$START" --end "$END" \
    --episodes_per_task "$EPISODES_PER_TASK" \
    --out_dir "$OUT_DIR" --tag "$TAG" \
    --max_assistant_turns "$MAX_ASSISTANT_TURNS" --concurrency "$CONCURRENCY" \
    --solver_base_url "http://127.0.0.1:$PARTNER_PORT/v1" --solver_model "$PARTNER_NAME" \
    --solver_temperature "$SOLVER_TEMPERATURE" --solver_top_p "$SOLVER_TOP_P" \
    --solver_top_k "$SOLVER_TOP_K" --solver_max_tokens "$SOLVER_MAX_TOKENS" \
    --sim_base_url "http://127.0.0.1:$SIM_PORT/v1" --sim_model "$SIM_NAME" \
    --sim_temperature "$SIM_TEMPERATURE_ARG" --sim_top_p "$SIM_TOP_P_ARG" \
    --sim_top_k "$SIM_TOP_K_ARG" --sim_max_tokens "$SIM_MAX_TOKENS_ARG" \
    --sim_system "$SIM_SYSTEM" \
    --arm "$ARM" \
    ${SIM_MAX_TRIES:+--sim_max_tries "$SIM_MAX_TRIES"} \
    ${MAX_CODE_PROPOSALS:+--max_code_proposals "$MAX_CODE_PROPOSALS"} \
    ${SIM_CODE_LEAK_DETECTOR:+--sim_code_leak_detector "$SIM_CODE_LEAK_DETECTOR"} \
    ${EARLY_TERM_GUARD:+$([ "$EARLY_TERM_GUARD" = False ] && echo --noearly_term_guard || true)}
fi

if [ "$STAGE" != "collect" ]; then
  echo "[harness] === Stage 3a: collect_candidates (k=$K) ==="
  python -m colbench.simtrain.collect_candidates \
    --prefixes "$PREFIX_FILE" --out "$CAND_FILE" --k "$K" \
    ${MAX_PREFIXES:+--max_prefixes "$MAX_PREFIXES"} \
    --concurrency "$CONCURRENCY" \
    --sim_base_url "http://127.0.0.1:$SIM_PORT/v1" --sim_model "$SIM_NAME" \
    --sim_temperature "$SIM_TEMPERATURE_ARG" --sim_top_p "$SIM_TOP_P_ARG" \
    --sim_top_k "$SIM_TOP_K_ARG" --sim_max_tokens "$SIM_MAX_TOKENS_ARG" \
    --sim_system "$SIM_SYSTEM"
fi

echo "[harness] DONE."
echo "[harness]   prefixes   -> $PREFIX_FILE ($( [ -f "$PREFIX_FILE" ] && wc -l < "$PREFIX_FILE" || echo 0 ) rows)"
echo "[harness]   candidates -> $CAND_FILE  ($( [ -f "$CAND_FILE" ] && wc -l < "$CAND_FILE" || echo 0 ) rows)"
if [ "$ARM" = gt ]; then
  echo "[harness] next: the PILOT judge pass (CPU, ~\$0.60) ->"
  echo "[harness]   MAX_PREFIXES=50 PREFIX_FILE=$PREFIX_FILE CAND_FILE=$CAND_FILE \\"
  echo "[harness]     sbatch $REPO/colbench/simtrain/run_judge_slurm.sh"
else
  echo "[harness] next: the PILOT judge pass under rubric g1, on the LOCAL 235B ->"
  echo "[harness]   bash slurm_setup/launch_eval_slurm.sh --mode judge --exp_name g1_pilot \\"
  echo "[harness]     --judge_arm grounded --judge_max_prefixes 50 \\"
  echo "[harness]     --judge_prefixes $PREFIX_FILE \\"
  echo "[harness]     --judge_candidates $CAND_FILE"
  echo "[harness] then READ the dump before touching the rubric."
fi

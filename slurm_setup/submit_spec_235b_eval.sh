#!/usr/bin/env bash
# GPT-5.4 grounded evaluation. Defaults to the original paired sweep, preflight only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/paths.sh"
ACTION="${1:---dry_run}"
if { [ "$#" -gt 1 ] && [ "$#" -ne 3 ]; } || [[ "${ACTION}" != --dry_run && "${ACTION}" != --submit ]]; then
  echo "Usage: bash $0 [--dry_run|--submit] [TRAIN_EXP base,100,200,...]" >&2
  exit 1
fi

# Pin the shared evaluation protocol, independent of the training launch environment.
export MODEL_PATH=""
export TEMPERATURES=0.7 N_SAMPLES=1 MAX_SAVED_CONVOS=1000 SEED=0
export TOP_P=0.95 TOP_K=-1 ROLLOUT_TP=1 SOLVER_ENABLE_THINKING=false
export MAX_ASSISTANT_TURNS=10 MAX_CODE_PROPOSALS=2
export MAX_NEW_TOKENS_PER_TURN=1024 MAX_RESPONSE_LENGTH=14336 MAX_PROMPT_LENGTH=2048
export SIM_MAX_TRIES=8 SIM_MAX_TOKENS=256 SIM_ENABLE_THINKING=false
export SIM_TEMPERATURE=0.7 SIM_TOP_P=0.8 SIM_TOP_K=20 SIM_MIN_P=0.0
export GROUNDED_SIM=True
# The offline validator applies a0_strict code rejection and the early-termination
# guard directly. EARLY_TERM_GUARD from the training environment is not consulted.
VAL="${DATA_ROOT}/colbench_spec/test_small.gpt-5.4.parquet"
RUNS=(qwen3_4b_spec_simsft_s2 qwen3_4b_spec_baseline)
# Baseline steps 100--500 no longer contain actor weights.
STEPS=(base,100,200,300,400 base,600,700,800,900,1000)
if [ "$#" -eq 3 ]; then
  RUNS=("$2")
  STEPS=("$3")
fi

launch() {
  local index="$1"
  shift
  bash "${HERE}/launch_eval_slurm.sh" \
    --mode eval --model Qwen/Qwen3-4B \
    --exp_name "${RUNS[index]}_235b_grounded_gpt54_first1k" \
    --train_project colbench_mt --train_exp "${RUNS[index]}" \
    --global_step "${STEPS[index]}" \
    --sim_model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --sim_tp 4 --sim_context_length 32768 --sim_max_running_requests 8 \
    --val_file "${VAL}" --max_problems 1000 \
    --partition kempner_h100 --time 24:00:00 "$@"
}

echo 'Protocol: first 1000 problems; 1000 saved conversations/checkpoint; solver temperature 0.7'
# Preflight every selected run before submitting any.
for index in "${!RUNS[@]}"; do launch "${index}" --dry_run; done
if [ "${ACTION}" = --submit ]; then
  for index in "${!RUNS[@]}"; do launch "${index}"; done
else
  echo 'Preflight only. Pass --submit to queue the selected jobs.'
fi

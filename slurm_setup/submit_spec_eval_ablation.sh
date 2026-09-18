#!/usr/bin/env bash
# Four-way checkpoint sweep: two solver runs x two simulator/temperature settings.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/paths.sh"

ACTION="${1:---dry_run}"
if [ "$#" -gt 1 ] || [[ "${ACTION}" != --dry_run && "${ACTION}" != --submit ]]; then
  echo "Usage: bash $0 [--dry_run|--submit]" >&2
  exit 1
fi

export MODEL_PATH="" N_SAMPLES=1 MAX_SAVED_CONVOS=2000 SEED=0
export TOP_P=0.95 TOP_K=-1 ROLLOUT_TP=1 SOLVER_ENABLE_THINKING=false
export MAX_ASSISTANT_TURNS=10 MAX_CODE_PROPOSALS=2
export MAX_NEW_TOKENS_PER_TURN=1024 MAX_RESPONSE_LENGTH=14336 MAX_PROMPT_LENGTH=2048
export SIM_MAX_TRIES=8 SIM_MAX_TOKENS=256 SIM_ENABLE_THINKING=false
export SIM_TEMPERATURE=0.7 SIM_TOP_P=0.8 SIM_TOP_K=20 SIM_MIN_P=0.0

VAL="${DATA_ROOT}/colbench_spec/test_small.gpt-5.4.parquet"
RUNS=(qwen3_4b_spec_simsft_s2 qwen3_4b_spec_rej8_ent003)
STEPS=(base,50,100,150,200,300,400 base,800,900,950,1000,1100,1200)

launch() {
  local index="$1" conditioning="$2" temperature="$3" dry_flag="$4"
  local grounded=False condition_tag=spec
  if [ "${conditioning}" = grounded ]; then
    grounded=True
    condition_tag=grounded
  fi
  GROUNDED_SIM="${grounded}" TEMPERATURES="${temperature}" \
    bash "${HERE}/launch_eval_slurm.sh" \
      --mode eval --model Qwen/Qwen3-4B \
      --exp_name "${RUNS[index]}_235b_${condition_tag}_gpt54_all_t${temperature/./}" \
      --train_project colbench_mt --train_exp "${RUNS[index]}" \
      --global_step "${STEPS[index]}" \
      --sim_model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
      --sim_tp 4 --sim_context_length 32768 --sim_max_running_requests 8 \
      --val_file "${VAL}" --max_problems 2000 \
      --partition kempner_h100 --time 24:00:00 ${dry_flag}
}

# Validate all four combinations before any submission.
for index in "${!RUNS[@]}"; do
  launch "${index}" grounded 0.6 --dry_run
  launch "${index}" spec 0.7 --dry_run
done

if [ "${ACTION}" = --submit ]; then
  for index in "${!RUNS[@]}"; do
    launch "${index}" grounded 0.6 ""
    launch "${index}" spec 0.7 ""
  done
else
  echo 'Preflight only. Pass --submit to queue all four jobs.'
fi

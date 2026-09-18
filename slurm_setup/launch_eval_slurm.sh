#!/usr/bin/env bash
# Spec evaluation / large-simulator serving, separate from the RL launcher.
# Reuses train_colbench.sbatch ONLY for allocation, snapshots and step cleanup.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/paths.sh"
die() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  echo 'Usage: bash slurm_setup/launch_eval_slurm.sh --mode smoke|serve|eval|judge --exp_name NAME [options]'
  echo '  --sim_model HF_ID          default: Qwen/Qwen3-235B-A22B-Instruct-2507-FP8'
  echo '  --model HF_ID              default solver: Qwen/Qwen3-4B'
  echo '  --model_path /merged/HF    optional solver checkpoint (not FSDP shards)'
  echo '  --train_exp NAME           training EXPERIMENT_NAME whose checkpoints to eval'
  echo '  --global_step 250,500      steps under that run; "base" = the unfinetuned model.'
  echo '                             FSDP shards are merged to HF inside the job (cached).'
  echo '  --val_file PATH            default: test_small.gpt-5.4.parquet'
  echo '  --partition kempner_h100|kempner_h200  --time HH:MM:SS (default 02:00:00)'
  echo '  --begin 2026-09-14T09:00  hold the job until then, THEN queue it. For a'
  echo '                            serve session you want at a particular hour:'
  echo '                            walltime does not affect the start time, so a'
  echo '                            job queued now simply runs (and expires) tonight.'
  echo '  --max_problems N           default 4; 0 explicitly selects full eval'
  echo '  --sim_tp N --sim_context_length N --sim_max_running_requests N'
  echo '  --dry_run                  validate and print; no writes or sbatch'
  echo 'judge mode (score simtrain candidates with the large model instead of a hosted API):'
  echo '  --judge_prefixes FILE      simtrain prefixes.*.jsonl'
  echo '  --judge_candidates FILE    simtrain candidates.*.jsonl (same prefix ids)'
  echo '  --judge_out FILE           default: judged.<tag>.<rubric>.<judge>.jsonl beside them'
  echo '  --judge_arm gt|grounded    default gt (rubric r6). grounded = rubric g1:'
  echo '                             both vetoes programmatic, ONE ranking call.'
  echo '  --judge_max_prefixes N     default 50 (the pilot); 0 = every prefix'
  echo '  --judge_concurrency N      default 16 in-flight judge calls'
  echo 'smoke/serve/judge: one node x 4 GPUs. eval: two nodes x 4 GPUs (solver + sim).'
}
MODE=smoke
EXP_NAME=""
MODEL=Qwen/Qwen3-4B
MODEL_PATH="${MODEL_PATH:-}"
TRAIN_EXP="${TRAIN_EXP:-}"
GLOBAL_STEP="${GLOBAL_STEP:-}"
# Merged HF checkpoints are CACHED here across eval jobs: merging is deterministic, so
# re-evaluating a step (new sim, new val set) must not pay for it twice.
MERGED_ROOT="${MERGED_ROOT:-${SCRATCH_ROOT}/verl_merged}"
TRAIN_PROJECT="${TRAIN_PROJECT:-colbench_mt}"
SIM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
SIM_TP="${SIM_TP:-}"
SIM_CONTEXT_LENGTH="${SIM_CONTEXT_LENGTH:-32768}"
SIM_MAX_RUNNING_REQUESTS="${SIM_MAX_RUNNING_REQUESTS:-8}"
SIM_MEM_FRACTION="${SIM_MEM_FRACTION:-0.90}"
SIM_CUDA_GRAPH_MAX_BS="${SIM_CUDA_GRAPH_MAX_BS:-8}"
SIM_CHUNKED_PREFILL_SIZE="${SIM_CHUNKED_PREFILL_SIZE:-2048}"
VAL_FILE="${VAL_FILE:-${DATA_ROOT}/colbench_spec/test_small.gpt-5.4.parquet}"
JUDGE_PREFIXES="${JUDGE_PREFIXES:-}"
JUDGE_CANDIDATES="${JUDGE_CANDIDATES:-}"
JUDGE_OUT="${JUDGE_OUT:-}"
# 50 mirrors run_judge_slurm.sh's pilot default, for the same reason: read the dump
# before spending the full pass. Here the cost is GPU-hours rather than dollars.
JUDGE_ARM="${JUDGE_ARM:-gt}"
JUDGE_MAX_PREFIXES="${JUDGE_MAX_PREFIXES:-50}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
MAX_PROBLEMS="${MAX_PROBLEMS:-4}"
PARTITION=kempner_h100
ACCOUNT=kempner_dam_lab
TIME=02:00:00
CPUS_PER_TASK=32
MEM=600G
BEGIN="${BEGIN:-}"
DRY_RUN=False
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry_run) DRY_RUN=True; shift; continue ;;
    --*=*) key="${1%%=*}"; value="${1#*=}"; shift ;;
    --*) key="$1"; [ $# -ge 2 ] || die "$1 needs a value"; value="$2"; shift 2 ;;
    *) die "unexpected argument $1" ;;
  esac
  case "${key}" in
    --mode|--exp_name|--model|--model_path|--sim_model|--sim_tp|--val_file|\
    --max_problems|--partition|--account|--time|--cpus_per_task|--mem|\
    --sim_context_length|--sim_max_running_requests|--train_exp|--global_step|\
    --train_project|--judge_prefixes|--judge_candidates|--judge_out|\
    --judge_max_prefixes|--judge_concurrency|--judge_arm|--begin)
      key="${key#--}"; printf -v "${key^^}" '%s' "${value}" ;;
    *) die "unknown option ${key}" ;;
  esac
done
# judge co-locates the judging client with the server on ONE node: the client is
# pure HTTP against 127.0.0.1, so a second node would buy nothing and would add a
# cross-node (here, cross-partition) reachability failure mode for no reason.
case "${MODE}" in smoke|serve|judge) NODES=1 ;; eval) NODES=2 ;; *) die 'invalid --mode' ;; esac
case "${PARTITION}" in kempner_h100|kempner_h200) ;; *) die 'use kempner_h100 or kempner_h200' ;; esac
[[ "${EXP_NAME}" =~ ^[A-Za-z0-9_-]+$ ]] || die '--exp_name must use letters, digits, _ or -'
[[ "${MAX_PROBLEMS}" =~ ^[0-9]+$ ]] || die '--max_problems must be >= 0'
[[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || die 'invalid CPU count'
if [ "${PARTITION}" = kempner_h200 ] && [ "${CPUS_PER_TASK}" -ge 64 ]; then
  die 'H200 requires fewer than 16 CPUs/GPU; use at most 63 CPUs for 4 GPUs'
fi
meta="$(resolve_model_meta "${SIM_MODEL}")"
[ -n "${meta}" ] || die "sim model not in paths.sh: ${SIM_MODEL}"
read -r SIM_WEIGHTS auto_tp sim_thinking _ _ <<< "${meta}"
SIM_TP="${SIM_TP:-${auto_tp}}"
[[ "${SIM_TP}" =~ ^[124]$ ]] || die '--sim_tp must be 1, 2 or 4 on a four-GPU node'
for key in SIM_CONTEXT_LENGTH SIM_MAX_RUNNING_REQUESTS SIM_CUDA_GRAPH_MAX_BS SIM_CHUNKED_PREFILL_SIZE; do
  [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || die "${key} must be a positive integer"
done
export SIM_ENABLE_THINKING="${SIM_ENABLE_THINKING:-}"
[ "${sim_thinking}" != off ] || SIM_ENABLE_THINKING="${SIM_ENABLE_THINKING:-false}"
# Always resolve the BASE solver weights: they are the --global_step "base" target and
# the default when no checkpoint is requested.
BASE_MODEL_PATH=""
EVAL_STEPS=""
CKPT_ROOT=""
if [ "${MODE}" = eval ]; then
  meta="$(resolve_model_meta "${MODEL}")"
  if [ -n "${meta}" ]; then
    read -r BASE_MODEL_PATH _ solver_thinking _ _ <<< "${meta}"
    if [ "${solver_thinking}" = off ]; then
      export SOLVER_ENABLE_THINKING="${SOLVER_ENABLE_THINKING:-false}"
    fi
  fi
  if [ -n "${GLOBAL_STEP}" ]; then
    # Checkpoint-sweep mode. FSDP shards are merged to HF INSIDE the job, so nothing
    # has to be merged by hand -- but the shards are checked HERE, on the login node,
    # so a typo fails in a second instead of after an 8-GPU allocation and a ~17 min
    # simulator load.
    [ -z "${MODEL_PATH}" ] || die '--global_step and --model_path are mutually exclusive'
    [ -n "${TRAIN_EXP}" ] || die '--global_step requires --train_exp (the training run name)'
    [[ "${TRAIN_EXP}" =~ ^[A-Za-z0-9_-]+$ ]] || die '--train_exp must use letters, digits, _ or -'
    CKPT_ROOT="${RUN_ROOT_BASE}/${TRAIN_PROJECT}/${TRAIN_EXP}/checkpoints"
    seen=""
    for step in ${GLOBAL_STEP//,/ }; do
      if [ "${step}" = base ]; then
        [ -n "${BASE_MODEL_PATH}" ] || die "step 'base' needs a registry solver; ${MODEL} is unknown"
        python3 "${_HERE}/serving_probe.py" weights "${BASE_MODEL_PATH}" >/dev/null \
          || die "base solver weights incomplete: ${BASE_MODEL_PATH}"
      else
        [[ "${step}" =~ ^[0-9]+$ ]] || die "--global_step takes integers or 'base', got '${step}'"
        actor="${CKPT_ROOT}/global_step_${step}/actor"
        [ -d "${actor}" ] || die "no checkpoint at ${actor}"
        # An in-progress checkpoint write leaves the dir present but shard-less.
        ls "${actor}"/model_world_size_*_rank_*.pt >/dev/null 2>&1 \
          || die "no FSDP shards in ${actor} (still being written?)"
      fi
      case ",${seen}" in *",${step},"*) die "duplicate --global_step ${step}" ;; esac
      seen="${seen}${step},"
    done
    # COMMA-separated, never space-separated. Singularity's --env-file is not a shell:
    # it does no unescaping, so the `%q` quoting used to write the env file below
    # survives VERBATIM and a space-separated list arrives as the literal `base\ 250`,
    # which then word-splits into `base\` and `250` (job 46266724).
    EVAL_STEPS="${seen%,}"
  else
    MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH}}"
    [ -n "${MODEL_PATH}" ] || die "unknown solver ${MODEL}; supply --model_path or --global_step"
  fi
  [ -f "${VAL_FILE}" ] || die "missing spec parquet: ${VAL_FILE}"
fi
# Config alone can exist during an interrupted HF download: require every shard.
python3 "${_HERE}/serving_probe.py" weights "${SIM_WEIGHTS}" || die "stage sim: bash slurm_setup/stage_model.sh ${SIM_MODEL}"
if [ "${MODE}" = eval ] && [ -z "${EVAL_STEPS}" ]; then
  python3 "${_HERE}/serving_probe.py" weights "${MODEL_PATH}" || die 'solver needs a complete merged HF checkpoint'
fi
# Output-name tags. The SIM identity and the val set are part of every eval filename so
# that the SAME checkpoint evaluated under a DIFFERENT simulator or test set lands in a
# DIFFERENT file. xcloud did this and the first Slurm port dropped it, which would have
# made a 4B-sim run and a 235B-sim run indistinguishable on disk -- the JSON's own
# `sim_model` field only ever records the served alias (`colbench-sim`).
SIM_TAG="$(model_shorthand "${SIM_MODEL}")"
VAL_TAG="$(basename "${VAL_FILE}" .parquet | tr '.' '_')"
[ "${GROUNDED_SIM:-False}" != True ] || VAL_TAG="${VAL_TAG}-grounded"
# ── judge mode: resolve and VALIDATE the judging job on the login node ────────
# Every check here is one that would otherwise surface after a four-GPU allocation
# and a ~15-minute 235B load, i.e. cost real GPU-hours to discover.
# `colbench-sim` is the alias the EVAL path needs: the solver node reads it as
# MULTITURN_MODEL_NAME. Nothing reads it in serve/judge mode -- those exist for
# external clients -- and the served alias is what a judging client records as
# `judge_model`, so both advertise the model's real shorthand instead. A generic
# alias there is the same ambiguity already fixed for the eval JSON's sim
# identity: two different judges would be indistinguishable on disk.
SIM_SERVED_NAME=colbench-sim
[ "${MODE}" = serve ] && SIM_SERVED_NAME="${SIM_TAG}"
if [ "${MODE}" = judge ]; then
  # The served alias becomes the `judge_model` recorded in EVERY judged row and is
  # what the resume guard compares against, so it must name the actual model. The
  # generic `colbench-sim` alias is exactly the ambiguity that made a 4B-sim and a
  # 235B-sim eval indistinguishable in the eval JSON; do not repeat it here.
  SIM_SERVED_NAME="${SIM_TAG}"
  [ -n "${JUDGE_PREFIXES}" ]   || die '--mode judge needs --judge_prefixes'
  [ -n "${JUDGE_CANDIDATES}" ] || die '--mode judge needs --judge_candidates'
  [ -s "${JUDGE_PREFIXES}" ]   || die "no prefixes at ${JUDGE_PREFIXES}"
  [ -s "${JUDGE_CANDIDATES}" ] || die "no candidates at ${JUDGE_CANDIDATES}"
  [[ "${JUDGE_MAX_PREFIXES}" =~ ^[0-9]+$ ]] || die '--judge_max_prefixes must be >= 0'
  [[ "${JUDGE_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || die '--judge_concurrency must be >= 1'
  # The rubric version comes from the source of truth rather than being retyped --
  # retyping it is how two scoring standards end up in one file.
  case "${JUDGE_ARM}" in gt|grounded) ;; *) die '--judge_arm must be gt or grounded' ;; esac
  JUDGE_RUBRIC="$(cd "${VERL_REPO}" && ARM="${JUDGE_ARM}" python3 -c \
    'import os
from colbench.simtrain.judge_candidates import rubric_version_for
print(rubric_version_for(os.environ["ARM"]))')" \
    || die 'cannot resolve the rubric version'
  if [ -z "${JUDGE_OUT}" ]; then
    _cand_base="$(basename "${JUDGE_CANDIDATES}" .jsonl)"
    # candidates.<tag>  ->  judged.<tag>.<rubric>.<judge>[.pilotN]
    _cand_tag="${_cand_base#candidates.}"
    _pilot=""
    [ "${JUDGE_MAX_PREFIXES}" -eq 0 ] || _pilot=".pilot${JUDGE_MAX_PREFIXES}"
    JUDGE_OUT="$(dirname "${JUDGE_CANDIDATES}")/judged.${_cand_tag}.${JUDGE_RUBRIC}.${SIM_SERVED_NAME}${_pilot}.jsonl"
  fi
  # Resume matches on prefix_id ALONE, so a file written by another judge would be
  # either silently skipped (looking like perfect agreement) or silently topped up
  # into a two-judge mixture. judge_candidates refuses this too; doing it here as
  # well means the mistake costs a second rather than an allocation.
  if [ -s "${JUDGE_OUT}" ]; then
    (cd "${VERL_REPO}" && python3 -c '
import json, sys
path, want = sys.argv[1], sys.argv[2]
seen = set()
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            seen.add(json.loads(line).get("judge_model", ""))
if seen and seen != {want}:
    sys.exit("%s was judged by %s, not %s" % (path, sorted(seen), want))
' "${JUDGE_OUT}" "${SIM_SERVED_NAME}") \
      || die "refusing to mix judges in one file.
  Pass a different --judge_out, or --allow_mixed_judge deliberately."
  fi
  echo "judge: ${JUDGE_PREFIXES}"
  echo "       ${JUDGE_CANDIDATES}"
  echo "    -> ${JUDGE_OUT}"
  echo "       arm=${JUDGE_ARM} rubric=${JUDGE_RUBRIC} judge=${SIM_SERVED_NAME} max_prefixes=${JUDGE_MAX_PREFIXES} (0=all)"
fi
[ -d "${SANDBOX}" ] || die "container missing: ${SANDBOX}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
[[ "${ROLLOUT_TP}" =~ ^[124]$ ]] || die 'ROLLOUT_TP must be 1, 2 or 4'
export SIM_DP=$((4 / SIM_TP))
echo "${MODE}: ${NODES} node(s) x 4 GPUs on ${PARTITION}; sim TP=${SIM_TP}, DP=${SIM_DP}"
echo "sim=${SIM_MODEL} weights=${SIM_WEIGHTS} context=${SIM_CONTEXT_LENGTH} max_running=${SIM_MAX_RUNNING_REQUESTS}"
if [ -n "${EVAL_STEPS}" ]; then
  echo "solver=${TRAIN_EXP} steps=[${EVAL_STEPS}] (merged to HF in-job, cached under ${MERGED_ROOT})"
else
  echo "solver=${MODEL_PATH:-<none>}"
fi
echo "val=${VAL_FILE} max_problems=${MAX_PROBLEMS} out_tag=sim-${SIM_TAG}_${VAL_TAG}"
[ "${DRY_RUN}" != True ] || exit 0

# Unique, never overwrite an earlier launch. Capture resolved knobs in the env
# file and code in the allocation's existing per-job snapshot.
export RUN_ROOT="${RUN_ROOT_BASE}/colbench_spec_eval/${EXP_NAME}/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "${RUN_ROOT}/launch" "${SLURM_LOG_DIR}"
export ENV_FILE="${RUN_ROOT}/launch/eval.env"
export ENTRYPOINT=slurm_setup/entrypoint_eval_colbench_slurm.sh
export SLURM_SETUP_DIR="${_HERE}"
export WANDB_MODE=disabled
export SIM_SMOKE=False
unset SIM_SERVER_ONLY
[ "${MODE}" = eval ] || SIM_SMOKE=True
export EVAL_MODE="${MODE}"
export EXPERIMENT_NAME="${EXP_NAME}"
export PROJECT_NAME=colbench_spec_eval
export OUT="${RUN_ROOT}/eval.json"
export CODECONTEST_ALLOW_INPROCESS=0
export CODECONTEST_EXEC_CONCURRENCY="${CODECONTEST_EXEC_CONCURRENCY:-8}"
export CODECONTEST_EXEC_MEM_GB="${CODECONTEST_EXEC_MEM_GB:-2}"
export OPENAI_API_KEY=EMPTY
# Deliberately no hosted API credentials. Defaults match the standalone spec
# runner; explicit entries prevent inherited training config changing the eval.
export N_SAMPLES="${N_SAMPLES:-1}" MAX_SAVED_CONVOS="${MAX_SAVED_CONVOS:-1000}"
export TEMPERATURES="${TEMPERATURES:-0.6}" TOP_P="${TOP_P:-0.95}" TOP_K="${TOP_K:--1}" SEED="${SEED:-0}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-10}" MAX_CODE_PROPOSALS="${MAX_CODE_PROPOSALS:-2}"
export MAX_NEW_TOKENS_PER_TURN="${MAX_NEW_TOKENS_PER_TURN:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-14336}" MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export REWARD_TIME_LIMIT="${REWARD_TIME_LIMIT:-6}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
export SIM_MAX_TRIES="${SIM_MAX_TRIES:-8}" SIM_MAX_TOKENS="${SIM_MAX_TOKENS:-256}"
export SIM_TEMPERATURE="${SIM_TEMPERATURE:-0.7}" SIM_TOP_P="${SIM_TOP_P:-0.8}"
export SIM_TOP_K="${SIM_TOP_K:-20}" SIM_MIN_P="${SIM_MIN_P:-0.0}"
export GROUNDED_SIM="${GROUNDED_SIM:-False}"
export SIM_STARTUP_TIMEOUT="${SIM_STARTUP_TIMEOUT:-2400}"
export SOLVER_ENABLE_THINKING="${SOLVER_ENABLE_THINKING:-}"
export EVAL_STEPS CKPT_ROOT MERGED_ROOT TRAIN_EXP BASE_MODEL_PATH SIM_TAG VAL_TAG
export SIM_SERVED_NAME JUDGE_PREFIXES JUDGE_CANDIDATES JUDGE_OUT
export JUDGE_MAX_PREFIXES JUDGE_CONCURRENCY JUDGE_ARM
keys=(RUN_ROOT EVAL_MODE EXPERIMENT_NAME PROJECT_NAME OUT WANDB_MODE OPENAI_API_KEY
  MODEL_PATH SIM_MODEL SIM_WEIGHTS SIM_TP SIM_DP SIM_ENABLE_THINKING SOLVER_ENABLE_THINKING
  SIM_CONTEXT_LENGTH SIM_MAX_RUNNING_REQUESTS SIM_MEM_FRACTION SIM_CUDA_GRAPH_MAX_BS
  SIM_CHUNKED_PREFILL_SIZE SIM_STARTUP_TIMEOUT VAL_FILE MAX_PROBLEMS ROLLOUT_TP
  N_SAMPLES MAX_SAVED_CONVOS TEMPERATURES TOP_P TOP_K SEED MAX_ASSISTANT_TURNS
  MAX_CODE_PROPOSALS MAX_NEW_TOKENS_PER_TURN MAX_RESPONSE_LENGTH MAX_PROMPT_LENGTH
  REWARD_TIME_LIMIT GPU_MEM_UTIL SIM_MAX_TRIES SIM_MAX_TOKENS SIM_TEMPERATURE SIM_TOP_P
  SIM_TOP_K SIM_MIN_P GROUNDED_SIM CODECONTEST_ALLOW_INPROCESS
  EVAL_STEPS CKPT_ROOT MERGED_ROOT TRAIN_EXP BASE_MODEL_PATH SIM_TAG VAL_TAG
  SIM_SERVED_NAME JUDGE_PREFIXES JUDGE_CANDIDATES JUDGE_OUT JUDGE_MAX_PREFIXES
  JUDGE_CONCURRENCY JUDGE_ARM
  CODECONTEST_EXEC_CONCURRENCY CODECONTEST_EXEC_MEM_GB)
# TEMPERATURES stays SPACE-separated on the command line (as the runner documents) but
# travels comma-separated, because the env file cannot carry a space; the entrypoint
# converts it back. Without this a multi-temperature sweep silently reached the
# validator as a single malformed argument.
TEMPERATURES="${TEMPERATURES// /,}"
# Singularity's --env-file is NOT a shell and does NO unescaping, so `printf %q` is
# always WRONG here: its backslashes survive into the container verbatim. Job 46266724
# died this way (`base 250` -> literal `base\ 250` -> steps `base\` and `250`), and %q
# escapes commas too, so a comma list is not automatically safe either.
# Write plain values, and REFUSE at submit time anything this channel cannot carry --
# a corrupt value would otherwise surface as a confusing failure inside the job, or
# worse, as a silently wrong configuration.
for key in "${keys[@]}"; do
  value="${!key}"
  case "${value}" in
    *[!A-Za-z0-9_,.:/=+-]*)
      die "${key} contains a character that Singularity's --env-file cannot carry
  (spaces, quotes and backslashes are unescapable here): '${value}'
  Use a comma-separated list instead of a space-separated one." ;;
  esac
  printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
done
# An empty array, not an inline conditional: an unset --begin must pass NO flag,
# and word-splitting a conditionally-empty string under `set -u` is how that
# breaks silently.
BEGIN_ARG=()
[ -z "${BEGIN}" ] || BEGIN_ARG=(--begin="${BEGIN}")
sbatch --parsable "${BEGIN_ARG[@]}" --job-name="spec_${MODE}_${EXP_NAME}" --partition="${PARTITION}" \
  --account="${ACCOUNT}" --nodes="${NODES}" --gres=gpu:4 --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" --time="${TIME}" --no-requeue --export=ALL \
  "${_HERE}/train_colbench.sbatch" | tee "${RUN_ROOT}/launch/job_id.txt"
echo "Run artifacts: ${RUN_ROOT}"

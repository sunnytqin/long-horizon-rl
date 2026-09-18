#!/bin/bash
# ============================================================================
# ColBench RL launcher for SLURM. The replacement for xcloud_setup/launch.py.
#
# Same flag names, same defaults, same cross-flag validation -- so a command you ran on
# XManager reads the same here. What it no longer does: build a Docker image, create an
# XManager experiment, attach TensorBoard-corp, or construct a two-job graph.
# What it does instead:
#
#   1. validates the flags (incl. checks launch.py could not do: are the weights and the
#      parquets actually staged on this cluster?)
#   2. resolves the STABLE experiment identity {model}_{exp_name} -- unchanged, and still
#      the single source of truth for every storage path
#   3. writes the whole run configuration to an --env-file next to the checkpoints
#   4. picks the node count from the sim regime and sbatch's train_colbench.sbatch
#
# The env file is the part worth knowing about. launch.py passed ~48 env vars into the
# job's spec; here they are written to $RUN_ROOT/launch/<ts>.env and handed to
# `singularity exec --env-file`. That is deterministic (no reliance on what does or does
# not survive srun and singularity's env inheritance) and it leaves the exact
# configuration of a run in a file beside its checkpoints.
#
# Examples
#   # GT-code path, frozen-base sim on its own node (the default)
#   bash slurm_setup/launch_slurm.sh --model=Qwen/Qwen3-4B --exp_name=gt_slurm_smoke \
#        --train_script=colbench/run_colbench_grpo.sh
#
#   # SPEC path, gpt-5.4 specs, validating on the golden set
#   VAL_FILE=$DATA_ROOT/colbench_spec/test_small.gpt-5.4.parquet \
#   bash slurm_setup/launch_slurm.sh --model=Qwen/Qwen3-4B --exp_name=spec_gpt54 \
#        --train_script=colbench/run_colbench_grpo_spec.sh --spec_author=gpt-5.4
#
#   # live-weights self-play (one node, no sim server), 3 chained jobs for a long run
#   bash slurm_setup/launch_slurm.sh --model=Qwen/Qwen3-4B --exp_name=selfplay \
#        --train_script=colbench/run_colbench_grpo.sh --sim_live --chain=3
# ============================================================================
set -uo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${_HERE}/paths.sh"

die() { echo "❌ $*" >&2; exit 1; }

# The literal argv, snapshotted BEFORE the parse loop below `shift`s it away. Used to write
# a re-runnable record of this launch to $RUN_ROOT/launch/<ts>.cmd -- the env file beside it
# holds the RESOLVED config, which is not the same thing: it cannot tell a default apart
# from a value that was typed, and it silently omits MAX_CKPT_KEEP / LOGGERS (read straight
# from the inherited environment inside the container, never written here).
_ORIG_ARGV=("$@")

# Env-only knobs (the ${X:-} block near the end of the env file, plus the two that are NOT
# in it). Recorded as VAR=value prefixes on the reproduce command so the record is complete
# even where the env file is not. Keep in sync with README section 5.5.
_ENV_ONLY_KNOBS=(
  SIM_MAX_TOKENS ENV_STEP_TIMEOUT SIM_CHAR_LIMIT TRAIN_FILE VAL_FILE ROLLOUT_TP
  TRAIN_BATCH_SIZE SAVE_FREQ TEST_FREQ TOTAL_EPOCHS ROLLOUT_N MAX_CKPT_KEEP LOGGERS
  ENTROPY_COEFF CLIP_RATIO_LOW CLIP_RATIO_HIGH ACTOR_LR RESUME_FROM_PATH
  COLBENCH_DEBUG_CONVO COLBENCH_DEBUG_CONVO_N COLBENCH_DEBUG_CONVO_PREVIEW
  COLBENCH_DEBUG_SIM
)

# ── Defaults (mirroring launch.py's flag defaults) ─────────────────────────────────────
MODEL=""
EXP_NAME=""
EXPERIMENT_NAME=""
PROJECT_NAME="colbench_mt"
TRAIN_SCRIPT=""
EVAL_ONLY="False"

# sim regime. NOTE the changed default: launch.py had sim_remote=False (co-host the sim on
# the one 8-GPU node). Here it is TRUE, because a node has 4 GPUs and co-hosting would cost
# half of them -- see the node-split comment in train_colbench.sbatch.
SIM_REMOTE="True"
SIM_LIVE="False"
SIM_SMOKE="False"
SIM_MODEL=""
SIM_PROTOCOL=""
SIM_PROMPT=""
SIM_CODE_LEAK_DETECTOR="auto"
# User-sim rejection-sampling budget, one meaning on BOTH paths: 0 = sampler OFF (one
# unscreened draw, injected as-is -- the no-guardrail baseline), N = up to N resamples and
# the episode ends if all N leak. EMPTY here on purpose: each run script then falls back to
# its OWN documented default (`:-0` on the GT path, `:-8` on the spec path), which is why
# this is not spelled "0". Before 2026-09 the launcher default WAS "0" and the spec agent
# coerced 0 -> 8, so "off" was not reachable on the spec path at all; that coercion is gone.
SIM_REJECT_MAX_TRIES=""

# train knobs (empty => the train script's own default, same convention as launch.py)
TRAIN_TURNS="all"
MAX_ASSISTANT_TURNS=""
MAX_CODE_PROPOSALS=""
KL_LOSS_COEF=""
LOSS_AGG_MODE=""
LENGTH_PENALTY_COEF=""
LENGTH_SOFT_CAP=""
TERMINATE_ON_ALLPASS="False"
BINARY_REWARD="False"
GROUNDED_SIM="False"
EARLY_TERM_GUARD="True"
SPEC_AUTHOR=""
WARM_START_CKPT_DIR=""

# slurm placement
# online | offline | disabled. See configure_wandb() in entrypoint_common_slurm.sh.
WANDB="${WANDB_MODE:-online}"
WANDB_ENTITY_FLAG=""

PARTITION="kempner_h100"
ACCOUNT="kempner_dam_lab"
TIME="2-00:00:00"
GPUS_PER_NODE="4"
# 88 of the node's 96. Taking all 4 GPUs already means no other GPU job can share the
# node, and GPU dominates billing (TRESBillingWeights: gpu=2648.8 vs cpu=1.1, so 88 cores
# add ~1% to the cost of 4 H100s). The exec sidecar runs up to 180 concurrent subprocesses
# -- xcloud gave it 180 vCPU -- so CPU is not a place to economize here. Left short of 96
# so a node with a stray CPU-only job can still take us; lower it if the job pends.
CPUS_PER_TASK="88"
# Explicit, because kempner_h100 refuses --mem=0 ("not permited to request all the memory
# on the node ... please specify your request"). ~1.2T of the node's ~1.47T.
MEM="1200G"
CHAIN="1"
DRY_RUN="False"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0; }

# --flag=value | --flag value | --flag (boolean true) | --noflag (boolean false)
_BOOLS=" sim_remote sim_live sim_smoke terminate_on_allpass binary_reward grounded_sim early_term_guard dry_run "
while [ $# -gt 0 ]; do
  _arg="$1"; shift
  case "${_arg}" in
    -h|--help) usage ;;
    --*)
      _k="${_arg#--}"; _v=""
      if [[ "${_k}" == *=* ]]; then _v="${_k#*=}"; _k="${_k%%=*}"
      elif [[ " ${_BOOLS} " == *" ${_k} "* ]]; then _v="True"
      elif [[ "${_k}" == no* ]] && [[ " ${_BOOLS} " == *" ${_k#no} "* ]]; then _k="${_k#no}"; _v="False"
      else _v="${1:-}"; [ $# -gt 0 ] && shift || die "flag --${_k} needs a value"
      fi
      # normalize true/false spellings so --sim_remote=false works like --nosim_remote
      if [[ " ${_BOOLS} " == *" ${_k} "* ]]; then
        case "$(echo "${_v}" | tr '[:upper:]' '[:lower:]')" in
          true|1|yes)  _v="True" ;;
          false|0|no)  _v="False" ;;
        esac
      fi
      _K="$(echo "${_k}" | tr '[:lower:]' '[:upper:]')"
      case "${_K}" in
        MODEL|EXP_NAME|EXPERIMENT_NAME|PROJECT_NAME|TRAIN_SCRIPT|EVAL_ONLY|\
        SIM_REMOTE|SIM_LIVE|SIM_SMOKE|SIM_MODEL|SIM_PROTOCOL|SIM_PROMPT|\
        SIM_CODE_LEAK_DETECTOR|SIM_REJECT_MAX_TRIES|TRAIN_TURNS|MAX_ASSISTANT_TURNS|\
        MAX_CODE_PROPOSALS|KL_LOSS_COEF|LOSS_AGG_MODE|LENGTH_PENALTY_COEF|LENGTH_SOFT_CAP|\
        TERMINATE_ON_ALLPASS|BINARY_REWARD|GROUNDED_SIM|EARLY_TERM_GUARD|SPEC_AUTHOR|\
        WARM_START_CKPT_DIR|PARTITION|ACCOUNT|TIME|GPUS_PER_NODE|CPUS_PER_TASK|MEM|CHAIN|DRY_RUN|\
        WANDB|WANDB_ENTITY_FLAG)
            printf -v "${_K}" '%s' "${_v}" 2>/dev/null || eval "${_K}=\${_v}" ;;
        *)  die "unknown flag --${_k} (see --help)" ;;
      esac
      ;;
    *) die "unexpected argument '${_arg}'" ;;
  esac
done

# ── Validation (ported from launch.py, plus cluster-local checks it could not do) ──────
[ -n "${MODEL}" ]        || die "--model is required"
[ -n "${TRAIN_SCRIPT}" ] || die "--train_script is required"

if [ "${EVAL_ONLY}" = "True" ]; then
  die "This launcher is training-only. For spec eval use: bash slurm_setup/launch_eval_slurm.sh --mode eval --exp_name NAME"
fi

# The colbench --model allowlist. The hybrid Qwen3 models are ColBench-only on purpose:
# codecontest has no solver-thinking plumbing and no FSDP_PROFILE plumbing (see the long
# comment in launch.py). Kept as a check because a model outside it will fail in the
# entrypoint anyway -- better to fail in milliseconds on the login node.
case "${MODEL}" in
  Qwen/Qwen3-4B|Qwen/Qwen2.5-14B|Qwen/Qwen3-14B|Qwen/Qwen3-32B) ;;
  *) die "unsupported --model '${MODEL}' for the colbench stack. Add it to paths.sh::resolve_model_meta AND to this allowlist." ;;
esac

# --sim_live means "no separate simulator exists at all", so any flag that serves one is a
# contradiction. Checked here as well as in the entrypoint.
if [ "${SIM_LIVE}" = "True" ]; then
  if [ -n "${SIM_MODEL}" ] || [ "${SIM_SMOKE}" = "True" ]; then
    die "--sim_live cannot be combined with --sim_model / --sim_smoke, which serve a SEPARATE frozen sim."
  fi
  # sim_remote defaults to True here, so silently override rather than nagging about a
  # default the user never typed.
  SIM_REMOTE="False"
fi

case "${WANDB}" in
  online|offline|disabled) ;;
  *) die "--wandb must be online, offline or disabled; got '${WANDB}'." ;;
esac
# Fail on the login node, where it is fixable in one command, rather than letting the job
# discover it and silently downgrade to offline 10 minutes in.
if [ "${WANDB}" = "online" ] && ! grep -q 'api.wandb.ai' "${HOME}/.netrc" 2>/dev/null; then
  die "--wandb=online needs wandb credentials. Run \`wandb login\` on this login node (writes \$HOME/.netrc), or pass --wandb=offline."
fi

case "${SIM_PROTOCOL}" in
  ""|assistant|userlm) ;;
  *) die "--sim_protocol must be '' (auto-derive from --sim_model), 'assistant' or 'userlm'; got '${SIM_PROTOCOL}'." ;;
esac
if [ "${SIM_PROTOCOL}" = "userlm" ] && [ "${SIM_LIVE}" = "True" ]; then
  die "--sim_protocol=userlm cannot be combined with --sim_live: the live sim IS the training policy, which is a code-writing assistant, not a user LM. Serve the user LM as a frozen sim (--sim_model microsoft/UserLM-8b)."
fi

# THE stable experiment identity. Normally {model}_{exp_name} -- no script name, no step,
# no job id -- so a run and its (future) eval share it and the paths always line up.
_shorthand() {
  case "$1" in
    Qwen/Qwen2.5-14B) echo "qwen2_5_14b" ;;
    Qwen/Qwen3-4B)    echo "qwen3_4b" ;;
    # `tr -- ` is REQUIRED: the set '-.' starts with a dash, so without it tr parses it as
    # an option ("tr: invalid option -- '.'"), prints nothing, and the shorthand comes back
    # EMPTY -- which silently keys the run's storage as "_<exp_name>", colliding across
    # models. Matches launch.py's .replace("-","_").replace(".","_").
    *) echo "${1##*/}" | tr '[:upper:]' '[:lower:]' | tr -- '-.' '__' ;;
  esac
}
if [ -z "${EXPERIMENT_NAME}" ]; then
  [ -n "${EXP_NAME}" ] || die "set --exp_name (forms {model}_{exp_name}) or --experiment_name (verbatim override)."
  EXPERIMENT_NAME="$(_shorthand "${MODEL}")_${EXP_NAME}"
fi

# Dataset path: same derivation as the entrypoint (a *spec* train script -> the spec
# parquets). Duplicated here ONLY to validate early; the entrypoint remains authoritative.
case "${TRAIN_SCRIPT}" in
  *spec*) _SUBDIR="colbench_spec" ;;
  *)      _SUBDIR="colbench" ;;
esac
[ -f "${VERL_REPO}/${TRAIN_SCRIPT}" ] || die "--train_script '${TRAIN_SCRIPT}' not found under ${VERL_REPO}"
[ -d "${DATA_ROOT}/${_SUBDIR}" ] || die "dataset ${DATA_ROOT}/${_SUBDIR} not staged. Run (on a LOGIN node): bash ${_HERE}/stage_data.sh"

if [ -n "${SPEC_AUTHOR}" ]; then
  [ "${_SUBDIR}" = "colbench_spec" ] || die "--spec_author is SPEC-path only; use --train_script=colbench/run_colbench_grpo_spec.sh"
  for _f in "train.${SPEC_AUTHOR}.parquet" "test_small.${SPEC_AUTHOR}.parquet"; do
    [ -f "${DATA_ROOT}/${_SUBDIR}/${_f}" ] || die "--spec_author=${SPEC_AUTHOR} needs ${DATA_ROOT}/${_SUBDIR}/${_f}. Available: $(ls -1 "${DATA_ROOT}/${_SUBDIR}" | tr '\n' ' ')"
  done
fi

# Are the weights on this cluster? This is the check launch.py never needed (GCS always
# had them) and the one most likely to bite -- 3 of the 7 registry models are not staged.
_check_weights() {  # _check_weights <identity> <role>
  local meta path; meta="$(resolve_model_meta "$1")"
  [ -n "${meta}" ] || die "'$1' is not in paths.sh::resolve_model_meta"
  read -r path _ <<< "${meta}"
  [ "${path}" != "MISSING" ] && [ -d "${path}" ] \
    || die "$2 model '$1' is not staged on netscratch. Run (on a LOGIN node): bash ${_HERE}/stage_model.sh $(model_hf_repo "$1")"
  echo "${path}"
}
_solver_path="$(_check_weights "${MODEL}" "solver")" || exit 1
if [ "${SIM_LIVE}" != "True" ]; then
  _sim_identity="${SIM_MODEL:-${MODEL}}"
  _sim_path="$(_check_weights "${_sim_identity}" "sim")" || exit 1
fi

# ── Node count follows the sim regime ─────────────────────────────────────────────────
#   sim_smoke : 1 node, sim ONLY (a standalone throughput/sizing run, no training)
#   sim_live  : 1 node (the user turn is generated by the training rollout engine)
#   co-hosted : 1 node (debug; the entrypoint warns about the 4-GPU cost)
#   default   : 2 nodes -- train on node 0, frozen sim on node 1
if [ "${SIM_SMOKE}" = "True" ]; then       NODES=1
elif [ "${SIM_LIVE}" = "True" ]; then      NODES=1
elif [ "${SIM_REMOTE}" = "True" ]; then    NODES=2
else                                       NODES=1
fi

# Per-partition CPU cap. kempner_h200 rejects a submit outright with "You must request
# less that 16 cores per gpu for the kempner_h200 partition", and kempner_h100 rejects
# --mem=0 the same way -- both AT SUBMIT TIME, so a bad default costs a round trip rather
# than failing in the job. Clamp instead of erroring: the CPU count is a throughput knob
# for the exec sidecar, not something worth blocking a launch over.
# Node shapes for reference (both 4 GPU, both ~1.47T RAM):
#     kempner_h100  96 CPU  4x H100 80GB   (no per-GPU CPU cap; --mem=0 rejected)
#     kempner_h200  64 CPU  4x H200        (<16 CPU/GPU enforced -> 60 max here)
_cpu_cap_per_gpu=""
case "${PARTITION}" in
  kempner_h200) _cpu_cap_per_gpu=15 ;;   # "less that 16", so 15 is the safe integer
esac
if [ -n "${_cpu_cap_per_gpu}" ]; then
  _max_cpus=$(( _cpu_cap_per_gpu * GPUS_PER_NODE ))
  if [ "${CPUS_PER_TASK}" -gt "${_max_cpus}" ]; then
    echo "note: ${PARTITION} caps CPUs at ${_cpu_cap_per_gpu}/GPU -> lowering --cpus_per_task ${CPUS_PER_TASK} -> ${_max_cpus}"
    CPUS_PER_TASK="${_max_cpus}"
  fi
fi

export RUN_ROOT="${RUN_ROOT:-${RUN_ROOT_BASE}/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "${RUN_ROOT}/launch" "${SLURM_LOG_DIR}" || die "cannot create ${RUN_ROOT}"
_TS="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="${RUN_ROOT}/launch/${_TS}.env"
CMD_FILE="${RUN_ROOT}/launch/${_TS}.cmd"

# ── Provenance and the reproduce command ──────────────────────────────────────────────
# WHY a second record next to the env file: the env file is the RESOLVED config, which is
# what the job needs but not what a human needs six weeks later. It cannot distinguish a
# default from a typed value, it omits the two env-only knobs the container inherits
# directly (MAX_CKPT_KEEP, LOGGERS), and it says nothing about which code ran.
_GIT_COMMIT="$(git -C "${VERL_REPO}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
_GIT_BRANCH="$(git -C "${VERL_REPO}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
_GIT_DIRTY="$(git -C "${VERL_REPO}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
[ "${_GIT_DIRTY}" = "0" ] && _GIT_DIRTY=""

# Only the knobs actually SET, so the record shows what was typed rather than restating
# every default. `${!_k}` is an indirect expansion -- the value of the variable NAMED by
# _k. printf %q makes each piece paste-safe (a VAL_FILE path with a space, say).
_repro_prefix=""
_notes_prefix=""
for _k in "${_ENV_ONLY_KNOBS[@]}"; do
  if [ -n "${!_k:-}" ]; then
    _repro_prefix+="$(printf '%s=%q ' "${_k}" "${!_k}")"
    _notes_prefix+="${_k}=${!_k} "
  fi
done
_repro_args=""
_notes_args=""
for _a in ${_ORIG_ARGV+"${_ORIG_ARGV[@]}"}; do
  # --dry_run is dropped: the whole point of this file is that `bash <it>` LAUNCHES. That
  # this particular invocation was a dry run is recorded by the footer line instead, so
  # nothing is lost and the command stays usable.
  case "${_a}" in --dry_run|--dry_run=*|--nodry_run) continue ;; esac
  _repro_args+="$(printf ' %q' "${_a}")"
  _notes_args+=" ${_a}"
done
_REPRO_CMD="${_repro_prefix}bash slurm_setup/launch_slurm.sh${_repro_args}"
# The PLAIN spelling is what goes into the env file and into wandb: %q output can carry
# backslashes, and the env file is `KEY="value"` lines re-parsed by singularity, where a
# stray backslash or quote would be re-interpreted. Any `"` is dropped for the same reason.
# CMD_FILE keeps the properly quoted form and is the authoritative one to re-run.
_LAUNCH_CMD_PLAIN="$(printf '%s' "${_notes_prefix}bash slurm_setup/launch_slurm.sh${_notes_args}" | tr -d '"\\')"
_WANDB_NOTES="${_LAUNCH_CMD_PLAIN}  ||  code: ${_GIT_BRANCH}@${_GIT_COMMIT}${_GIT_DIRTY:+ +${_GIT_DIRTY}dirty}  ||  cfg: ${_TS}.env"

# Filterable arm labels. Only NON-default facts, so a tag list reads as "what is unusual
# about this run" rather than a restatement of the defaults.
_tags="$(_shorthand "${MODEL}")"
case "${TRAIN_SCRIPT}" in *spec*) _tags+=",spec" ;; *) _tags+=",gt" ;; esac
[ "${SIM_LIVE}"  = "True" ] && _tags+=",sim_live"
[ "${SIM_SMOKE}" = "True" ] && _tags+=",sim_smoke"
[ "${SIM_REMOTE}" != "True" ] && [ "${SIM_LIVE}" != "True" ] && _tags+=",sim_cohost"
[ "${GROUNDED_SIM}" = "True" ] && _tags+=",grounded"
[ "${TERMINATE_ON_ALLPASS}" = "True" ] && _tags+=",term_on_allpass"
[ "${BINARY_REWARD}" = "True" ] && _tags+=",binary_reward"
[ "${EARLY_TERM_GUARD}" != "True" ] && _tags+=",no_term_guard"
# Rejection sampling, tagged in BOTH directions: an arm with the sampler on must be
# identifiable by a tag it HAS, not by one it lacks -- otherwise the on/off comparison
# reads as reject_off vs nothing in the wandb tag column.
[ "${SIM_REJECT_MAX_TRIES}" = "0" ] && _tags+=",reject_off"
[ -n "${SIM_REJECT_MAX_TRIES}" ] && [ "${SIM_REJECT_MAX_TRIES}" != "0" ] \
  && _tags+=",reject${SIM_REJECT_MAX_TRIES}"
[ "${SIM_CODE_LEAK_DETECTOR}" != "auto" ] && _tags+=",detector_${SIM_CODE_LEAK_DETECTOR}"
[ -n "${ENTROPY_COEFF:-}" ] && [ "${ENTROPY_COEFF:-0}" != "0" ] && _tags+=",entcoef${ENTROPY_COEFF}"
[ -n "${CLIP_RATIO_HIGH:-}" ] && [ "${CLIP_RATIO_HIGH}" != "0.28" ] && _tags+=",cliphi${CLIP_RATIO_HIGH}"
[ "${TRAIN_TURNS}" != "all" ] && _tags+=",turns_${TRAIN_TURNS}"
[ -n "${SIM_PROMPT}" ] && _tags+=",prompt_${SIM_PROMPT}"
[ -n "${SPEC_AUTHOR}" ] && _tags+=",author_${SPEC_AUTHOR}"
[ -n "${WARM_START_CKPT_DIR}" ] && _tags+=",warm_start"
# A replay re-enters an OLD run's checkpoint under a new name; without a tag its curve
# starting at step 700 looks like a fresh run that mysteriously skipped 700 steps.
[ -n "${RESUME_FROM_PATH:-}" ] && _tags+=",replay$(basename "${RESUME_FROM_PATH}" | sed 's/^global_step_//')"
_WANDB_TAGS="${_tags}"

# ── Write the env file ────────────────────────────────────────────────────────────────
# One KEY="value" per line. Empty values are written explicitly and mean "use the reader's
# default" -- the same semantics launch.py had when it passed "" for an unset flag, since
# every reader uses ${X:-default}.
q() { printf '%s="%s"\n' "$1" "${2//\"/\\\"}"; }
{
  echo "# ColBench slurm run config -- written by launch_slurm.sh $(date -Is)"
  echo "# host=$(hostname -s) user=${USER}"
  # storage identity (single source of truth for every path)
  q PROJECT_NAME     "${PROJECT_NAME}"
  q EXPERIMENT_NAME  "${EXPERIMENT_NAME}"
  q EXP_NAME         "${EXP_NAME}"
  q RUN_ROOT         "${RUN_ROOT}"
  # run selection
  q TRAIN_SCRIPT     "${TRAIN_SCRIPT}"
  q MODEL            "${MODEL}"
  q WARM_START_CKPT_DIR "${WARM_START_CKPT_DIR}"
  # sim regime
  q SIM_REMOTE       "${SIM_REMOTE}"
  q SIM_LIVE         "${SIM_LIVE}"
  q SIM_SMOKE        "${SIM_SMOKE}"
  q SIM_MODEL        "${SIM_MODEL}"
  q SIM_PROTOCOL     "${SIM_PROTOCOL}"
  q SIM_PROMPT       "${SIM_PROMPT}"
  q SIM_CODE_LEAK_DETECTOR "${SIM_CODE_LEAK_DETECTOR}"
  q SIM_REJECT_MAX_TRIES   "${SIM_REJECT_MAX_TRIES}"
  # train knobs
  q TRAIN_TURNS          "${TRAIN_TURNS}"
  q MAX_ASSISTANT_TURNS  "${MAX_ASSISTANT_TURNS}"
  q MAX_CODE_PROPOSALS   "${MAX_CODE_PROPOSALS}"
  q KL_LOSS_COEF         "${KL_LOSS_COEF}"
  q LOSS_AGG_MODE        "${LOSS_AGG_MODE}"
  q LENGTH_PENALTY_COEF  "${LENGTH_PENALTY_COEF}"
  q LENGTH_SOFT_CAP      "${LENGTH_SOFT_CAP}"
  q TERMINATE_ON_ALLPASS "${TERMINATE_ON_ALLPASS}"
  q BINARY_REWARD        "${BINARY_REWARD}"
  q GROUNDED_SIM         "${GROUNDED_SIM}"
  q EARLY_TERM_GUARD     "${EARLY_TERM_GUARD}"
  q SPEC_AUTHOR          "${SPEC_AUTHOR}"
  # wandb: mode + entity only. NO credential -- see configure_wandb()/node_launch.sh; the
  # key stays in $HOME/.netrc because this file lives on group-readable netscratch.
  q WANDB_MODE           "${WANDB}"
  q WANDB_ENTITY         "${WANDB_ENTITY_FLAG:-${WANDB_ENTITY_DEFAULT}}"
  q HOME_REAL            "${HOME}"
  # Env-only overrides, deliberately NOT flags (keeps the experiment CLI comparable --
  # the same convention as launch.py). Pass them on the launch shell:
  #   SIM_MAX_TOKENS=768 bash launch_slurm.sh ...
  q SIM_MAX_TOKENS   "${SIM_MAX_TOKENS:-}"
  q ENV_STEP_TIMEOUT "${ENV_STEP_TIMEOUT:-}"
  q SIM_CHAR_LIMIT   "${SIM_CHAR_LIMIT:-}"
  q TRAIN_FILE       "${TRAIN_FILE:-}"
  q VAL_FILE         "${VAL_FILE:-}"
  q ROLLOUT_TP       "${ROLLOUT_TP:-}"
  q TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE:-}"
  # Run-length / cadence knobs the train script already reads. Env-only, same convention:
  # they change how LONG a run is, not what it measures, so they stay off the experiment
  # CLI. A first smoke uses SAVE_FREQ=2 TEST_FREQ=-1 TOTAL_EPOCHS=1 to reach a checkpoint
  # in minutes instead of the production defaults (save 60 / test 20 / 15 epochs).
  q SAVE_FREQ        "${SAVE_FREQ:-}"
  q TEST_FREQ        "${TEST_FREQ:-}"
  q TOTAL_EPOCHS     "${TOTAL_EPOCHS:-}"
  q ROLLOUT_N        "${ROLLOUT_N:-}"
  # Optimizer-side knobs. Env-only (same convention), but written HERE rather than left to
  # host inheritance because they define an arm: an entropy/clip run whose coefficients are
  # absent from its own run record is not reproducible. Empty => the run script's default.
  q ENTROPY_COEFF    "${ENTROPY_COEFF:-}"
  q CLIP_RATIO_LOW   "${CLIP_RATIO_LOW:-}"
  q CLIP_RATIO_HIGH  "${CLIP_RATIO_HIGH:-}"
  q ACTOR_LR         "${ACTOR_LR:-}"
  q RESUME_FROM_PATH "${RESUME_FROM_PATH:-}"
  # rollout INSPECTION (read via os.getenv inside the agent loops, so they must be here)
  q COLBENCH_DEBUG_CONVO         "${COLBENCH_DEBUG_CONVO:-}"
  q COLBENCH_DEBUG_CONVO_N       "${COLBENCH_DEBUG_CONVO_N:-}"
  q COLBENCH_DEBUG_CONVO_PREVIEW "${COLBENCH_DEBUG_CONVO_PREVIEW:-}"
  q COLBENCH_DEBUG_SIM           "${COLBENCH_DEBUG_SIM:-}"
  # Provenance, so a run says which code produced it without reading the snapshot dir.
  # WANDB_NOTES / WANDB_TAGS are plain wandb env vars and verl's wandb.init() passes
  # neither explicitly (it passes project/name/entity/config), so wandb picks these up
  # from the environment -- the launch command lands in the run's Notes field and the arm
  # labels become filterable tags, with no change to verl.
  q GIT_COMMIT       "${_GIT_COMMIT}"
  q GIT_BRANCH       "${_GIT_BRANCH}"
  q GIT_DIRTY        "${_GIT_DIRTY}"
  q LAUNCH_CMD       "${_LAUNCH_CMD_PLAIN}"
  q WANDB_NOTES      "${_WANDB_NOTES}"
  q WANDB_TAGS       "${_WANDB_TAGS}"
} > "${ENV_FILE}"

# The re-runnable record. `bash <this file>` reproduces the launch, env-only knobs and all.
{
  echo "#!/bin/bash"
  echo "# Reproduce this launch. Written by launch_slurm.sh $(date -Is)."
  echo "# experiment ${EXPERIMENT_NAME}   project ${PROJECT_NAME}"
  echo "# code: ${_GIT_BRANCH} @ ${_GIT_COMMIT}${_GIT_DIRTY:+  (+${_GIT_DIRTY} uncommitted)}"
  echo "# resolved config: $(basename "${ENV_FILE}")"
  echo "# NB reproducing is not the same as RESUMING -- this experiment name resolves to the"
  echo "#    same \$RUN_ROOT, so re-running it resumes from its checkpoints (resume_mode=auto)."
  echo "#    Change --exp_name for an independent repeat."
  echo "set -euo pipefail"
  echo "cd ${VERL_REPO}"
  printf '%s\n' "${_REPRO_CMD}"
} > "${CMD_FILE}"

# ── Report, then submit ───────────────────────────────────────────────────────────────
_JOB_NAME="cb_${EXP_NAME:-${EXPERIMENT_NAME}}"
cat <<REPORT
==========================================================
 experiment : ${EXPERIMENT_NAME}   (project ${PROJECT_NAME})
 model      : ${MODEL}
              -> ${_solver_path}
 sim        : $( [ "${SIM_LIVE}" = "True" ] && echo "LIVE (training policy, no server)" \
              || { [ "${SIM_SMOKE}" = "True" ] && echo "SIM-ONLY sizing run (no training): ${_sim_identity} -> ${_sim_path}" \
              || { [ "${SIM_REMOTE}" = "True" ] && echo "REMOTE node: ${_sim_identity} -> ${_sim_path}" \
              || echo "CO-HOSTED: ${_sim_identity} -> ${_sim_path}"; }; } )
 script     : ${TRAIN_SCRIPT}   (dataset ${DATA_ROOT}/${_SUBDIR})
 run root   : ${RUN_ROOT}
 env file   : ${ENV_FILE}
 reproduce  : bash ${CMD_FILE}
 code       : ${_GIT_BRANCH} @ ${_GIT_COMMIT}${_GIT_DIRTY:+  (+${_GIT_DIRTY} uncommitted)}
 wandb      : ${WANDB}  entity=${WANDB_ENTITY_FLAG:-${WANDB_ENTITY_DEFAULT}}  project=${PROJECT_NAME}  run=${EXPERIMENT_NAME}
 slurm      : ${NODES} node(s) x ${GPUS_PER_NODE} GPU, ${CPUS_PER_TASK} cpu, ${MEM} on ${PARTITION} (${ACCOUNT}), ${TIME}, chain=${CHAIN}
==========================================================
REPORT

if [ "${DRY_RUN}" = "True" ]; then
  echo "--dry_run: not submitting. Config at ${ENV_FILE}, reproduce cmd at ${CMD_FILE}."
  # No "submitted:" line is appended below, which is what tells a dry-run record apart
  # from a real one -- otherwise the launch dir fills with files that all look submitted.
  echo "# NOT SUBMITTED (--dry_run)" >> "${CMD_FILE}"
  exit 0
fi

export ENV_FILE SIM_SMOKE
export SLURM_SETUP_DIR="${_HERE}"

# --chain N submits N jobs, each depending on the previous one FINISHING (afterany, not
# afterok -- a job that dies on the 2-day wall or a preemption must still be followed).
# Each one resumes from ${RUN_ROOT}/checkpoints via verl's resume_mode=auto. This is the
# Slurm stand-in for XManager auto-restarting a job that exited non-zero, and it is what
# makes a run longer than the partition's wall limit possible at all.
_dep=""
for _i in $(seq 1 "${CHAIN}"); do
  _out="$(sbatch --parsable \
      ${_dep} \
      -J "${_JOB_NAME}" \
      -p "${PARTITION}" \
      --account="${ACCOUNT}" \
      --nodes="${NODES}" \
      --gres="gpu:${GPUS_PER_NODE}" \
      -c "${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      -t "${TIME}" \
      -o "${SLURM_LOG_DIR}/${_JOB_NAME}_%j.out" \
      -e "${SLURM_LOG_DIR}/${_JOB_NAME}_%j.err" \
      "${_HERE}/train_colbench.sbatch")" || die "sbatch failed"
  echo "submitted job ${_out}  ($( [ ${_i} -eq 1 ] && echo "head" || echo "resume ${_i}/${CHAIN}, after ${_dep##*:}" ))"
  # Append the job id to the record. This is the ONLY thing that ties a launch to its job
  # -- the env/cmd files are timestamped while the code snapshot dir is named by job id,
  # so without this line there is nothing mapping `code.<jobid>/` back to the flags that
  # produced it, and a chain's links are indistinguishable.
  echo "# submitted: ${_out}  $(date -Is)  ${_i}/${CHAIN}" >> "${CMD_FILE}"
  _dep="--dependency=afterany:${_out}"
done
echo
echo "logs:  tail -f ${SLURM_LOG_DIR}/${_JOB_NAME}_<jobid>.out"
if [ "${WANDB}" = "disabled" ]; then
  echo "wandb: disabled -- console logs only"
else
  echo "wandb: https://wandb.ai/${WANDB_ENTITY_FLAG:-${WANDB_ENTITY_DEFAULT}}/${PROJECT_NAME}/runs/$(printf '%s' "${EXPERIMENT_NAME}" | tr -c 'A-Za-z0-9_-' '_' | cut -c1-60)"
fi

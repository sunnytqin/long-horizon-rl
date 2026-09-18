#!/bin/bash
# ============================================================================
# Stage 5 allocation layer: SFT the ColBench user-simulator. ONE node, 4 GPUs.
#
# Deliberately ONE file, not the launch/sbatch/entrypoint trio the GRPO path
# uses. That trio exists to solve problems this run does not have: a two-node
# train/sim split, a readiness handshake between them, an exec sidecar, and a
# rollout config with two dozen knobs. SFT here is `torchrun --standalone` over
# two parquet files -- there is no second role to coordinate with, so the extra
# indirection would be structure without content.
#
# Submit:
#   SFT_DIR=$SIMTRAIN_ROOT/base_partner/sft \
#   sbatch slurm_setup/sft_simtrain.sbatch
#
# Everything else is an env var with a default; see colbench/run_sim_sft.sh,
# which documents the verl defaults that fail silently on this dataset.
#
# WARNING -- DEFAULTS ARE DUPLICATED ACROSS TWO LAYERS. Every `--env X="${X:-...}"`
# below passes a value INTO the container, which OVERRIDES the `X=${X:-...}` default
# in run_sim_sft.sh. Changing the default in run_sim_sft.sh alone therefore does
# NOTHING when launched through this file. That cost a wasted run on 2026-09-10: the
# inner default was corrected to EPOCHS=1 but this wrapper still injected 3, so the
# job silently trained 3 epochs and overfit (val 0.315 at the epoch-1 boundary ->
# 0.373 by step 140). If you change a default, change it in BOTH places or in neither.
# ============================================================================

#SBATCH -J sim_sft
# PARTITION: override on the command line (`sbatch -p kempner_h200 ...`), which
# takes precedence over this directive. Neither partition is required on
# capability grounds -- a 4B full finetune under FSDP is ~8 GB params + 8 GB
# grads + 32 GB Adam states sharded over 4 cards, so H100's 80 GB is ample and
# H200's 141 GB buys nothing. It is purely a QUEUE question, and the two swing
# a lot: on 2026-09-10 a 4-GPU job estimated 22:44 on h100 against 15:45 on
# h200. Check before submitting:
#   sbatch --test-only -p kempner_h200 --account=kempner_dam_lab --nodes=1 \
#     --gres=gpu:4 -c 32 --mem=400G -t 0-06:00:00 --wrap true
#SBATCH -p kempner_h100
#SBATCH --account=kempner_dam_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH --mem=800G
#SBATCH -t 0-06:00:00
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/barak_lab/Lab/sqin/slurm_logs/%x_%j.out
#SBATCH -e /n/netscratch/barak_lab/Lab/sqin/slurm_logs/%x_%j.err

set -euo pipefail

REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
# shellcheck source=paths.sh
source "$REPO/slurm_setup/paths.sh"

: "${SFT_DIR:?set SFT_DIR to the build_sft_parquet --out_dir}"
MODEL_PATH=${MODEL_PATH:-$(resolve_hf_snapshot "$MODEL_ROOT/models--Qwen--Qwen3-4B-Instruct-2507")}
[ -n "$MODEL_PATH" ] || { echo "[sim_sft] ERROR: no base simulator snapshot"; exit 1; }

EXPERIMENT_NAME=${EXPERIMENT_NAME:-sim_sft_$(python3 -c "
import sys; sys.path.insert(0,'$REPO')
from colbench import prompts; print(prompts.JUDGE_RUBRIC_VERSION)" 2>/dev/null || echo r6)}
RUN_ROOT=${RUN_ROOT:-$SIMTRAIN_ROOT/sft_runs/$EXPERIMENT_NAME}
mkdir -p "$RUN_ROOT" /n/netscratch/barak_lab/Lab/sqin/slurm_logs

# ── Container. Same validated recipe as node_launch.sh, inlined because this
# ── job has one role and no env-file to distribute. The three flags that break
# ── it silently, restated so nobody has to go and read the other file:
#   * NO --cleanenv: it drops the image's PATH and you get a python without verl.
#   * NO `bash -lc`: a login shell sources ~/.bashrc and puts the host conda
#     (3.10) first; verl needs >=3.11 and dies on an import that looks like a
#     verl bug.
#   * KEEP --nv, or torch.cuda.device_count() is 0 despite the allocation.
NODE_TMP="/scratch/${USER}_${SLURM_JOB_ID:-manual}"
mkdir -p "$NODE_TMP"
FAKE_HOME="$RUN_ROOT/home"
mkdir -p "$FAKE_HOME"

# The wandb key stays in $HOME/.netrc and is bind-mounted READ-ONLY. Never an
# env var on a command line: $RUN_ROOT is group-readable netscratch and
# WANDB_API_KEY would show up in `ps`.
NETRC_BIND=()
if [ -f "${HOME}/.netrc" ]; then
  : > "$FAKE_HOME/.netrc" 2>/dev/null || true
  NETRC_BIND=(--bind "${HOME}/.netrc:${FAKE_HOME}/.netrc:ro")
fi

echo "[sim_sft] node=$(hostname -s)  job=${SLURM_JOB_ID:-manual}"
echo "[sim_sft] sandbox=$SANDBOX"
echo "[sim_sft] sft_dir=$SFT_DIR"
echo "[sim_sft] model=$MODEL_PATH"
echo "[sim_sft] experiment=$EXPERIMENT_NAME  run_root=$RUN_ROOT"

singularity exec --nv \
  --pwd "$REPO" \
  --bind "$REPO" \
  --bind /n/netscratch \
  --bind "$NODE_TMP" \
  --home "$FAKE_HOME" \
  "${NETRC_BIND[@]}" \
  --env PYTHONPATH="$REPO:$PYDEPS_DIR" \
  --env TRITON_CACHE_DIR="$NODE_TMP/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="$NODE_TMP/inductor" \
  --env XDG_CACHE_HOME="$NODE_TMP/xdg" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env SFT_DIR="$SFT_DIR" \
  --env MODEL_PATH="$MODEL_PATH" \
  --env NGPUS="${NGPUS:-4}" \
  --env EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  --env PROJECT_NAME="${PROJECT_NAME:-colbench_simtrain}" \
  --env SAVE_DIR="${SAVE_DIR:-$RUN_ROOT/ckpt}" \
  --env TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}" \
  --env MAX_LENGTH="${MAX_LENGTH:-4096}" \
  --env LR="${LR:-1e-5}" \
  --env EPOCHS="${EPOCHS:-1}" \
  --env SAVE_FREQ="${SAVE_FREQ:-20}" \
  --env WANDB_MODE="${WANDB_MODE:-online}" \
  --env WANDB_ENTITY="${WANDB_ENTITY:-harvardml}" \
  --env REPO="$REPO" \
  "$SANDBOX" bash "$REPO/colbench/run_sim_sft.sh"

echo "[sim_sft] DONE. checkpoints under ${SAVE_DIR:-$RUN_ROOT/ckpt}"

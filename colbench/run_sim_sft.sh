#!/bin/bash
# Stage 5: SFT the ColBench user-simulator on judge-selected turns.
#
# One node, `torchrun --standalone`. No sim server, no exec sidecar, no rollout:
# this is ordinary supervised fine-tuning on a three-message conversation
# (system / materialized sim user message / selected reply), which is
# MultiTurnSFTDataset's native shape.
#
# Build the parquets first (Stage 4) and PRE-SCAN them:
#   python -m colbench.simtrain.build_sft_parquet --prefixes ... --judged ... --out_dir DIR
#   python -m colbench.simtrain.prescan_sft --parquet DIR/sim_sft_train.parquet --model $BASE
# The pre-scan is the only check that catches "trained on the wrong tokens":
# a loss mask covering the prompt does not crash and does not look wrong in the
# loss curve.
#
# EVERY OVERRIDE BELOW IS LOAD-BEARING. verl's sft_trainer_engine.yaml defaults
# are tuned for a 0.5B gsm8k demo with a 7k-row dataset, and five of them fail
# SILENTLY or near-silently on a ~2k-row 4B run. Each is annotated with what it
# does if left alone. (5) are verl defaults that fail silently; (6)-(7) are
# defaults of OURS that were wrong, corrected from a measured run.

set -euo pipefail

REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
cd "$REPO"

# ── Data ─────────────────────────────────────────────────────────────────────
SFT_DIR=${SFT_DIR:?set SFT_DIR to the build_sft_parquet --out_dir}
TRAIN_FILE=${TRAIN_FILE:-$SFT_DIR/sim_sft_train.parquet}
VAL_FILE=${VAL_FILE:-$SFT_DIR/sim_sft_val.parquet}

# The SIMULATOR's base weights -- the same Qwen3-4B-Instruct-2507 the frozen
# sim is served from, because this run is producing S_1 from S_0.
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the base simulator snapshot}

NGPUS=${NGPUS:-4}
SP_SIZE=${SP_SIZE:-1}

PROJECT_NAME=${PROJECT_NAME:-colbench_simtrain}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sim_sft_r6}
SAVE_DIR=${SAVE_DIR:-$SFT_DIR/ckpt/$EXPERIMENT_NAME}

# ── The five silent verl defaults, then two of our own ───────────────────────
# (1) train_batch_size 256 > the dataset, so len(dataloader) == 0 and the run
#     dies on a ZeroDivisionError that says nothing about batch size. 32 also
#     has to divide the dp world size.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
# (2) max_length 1024. Measured token lengths on this dataset: p50 735,
#     max 988 for turn-0 rows, and later turns carry a longer dialogue, so
#     1024 raises on `truncation: error` almost immediately. Set from the
#     pre-scan's reported max, not from a guess.
MAX_LENGTH=${MAX_LENGTH:-4096}
# (3) max_token_len_per_gpu 8192 must be >= MAX_LENGTH or a single long row
#     trips an assert at whatever step it happens to land on.
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$((MAX_LENGTH * 2))}
# (4) optim.lr 1e-3. That is ~100x too high for a 4B model and it does NOT
#     look wrong over the 20-60 steps this dataset gives -- the loss falls,
#     then the model is damaged.
LR=${LR:-1e-5}
# (5) enable_thinking_default is the literal STRING "none" in the YAML, not a
#     null. Qwen3-Instruct has no thinking mode; leaving the string in place
#     feeds "none" into the chat template.
#     -> passed as `null` below.
# And one that is merely annoying rather than silent:
#     save_contents lacks "hf_model", so huggingface/ gets a config and a
#     tokenizer and NO WEIGHTS, and the merge step later has nothing to read.
# (8) save_contents is "hf_model" ONLY, which makes checkpoints NON-RESUMABLE
#     on purpose. This run is ~8 min on 4 GPUs, so re-running costs less than
#     storing the state needed to resume: dropping "model"/"optimizer"/"extra"
#     takes a checkpoint from ~65 GB (17.6 shards + 32.2 optim + 15.7 hf) to
#     ~16 GB, and job 45889174 spent 231 GB on four saves it never resumed from.
#     Safe because should_save_hf_model gathers the FULL state dict from the
#     live model itself (fsdp_checkpoint_manager.py ~L341), independent of
#     should_save_model -- each content is its own `if`. Add "model","optimizer",
#     "extra" back if you ever need mid-run resume.
# (6) MEASURED 2026-09-10, job 45889174 (2,057 rows, 4B, lr 1e-5, 64 steps/epoch):
#     3 epochs OVERFITS HARD and the loss curve alone will not tell you, because
#     TRAIN loss keeps falling the whole way -- 0.30 (ep1) -> 0.17 (ep2) -> 0.09
#     (ep3), a 4x drop that reads like success. val/loss goes the other way:
#       ep1  0.360 0.307 0.325 0.332 0.315 0.334
#       ep2  0.347 0.358 0.351 0.364 0.376 0.374
#       ep3  0.398 0.460 0.431 0.432 0.437 0.442 0.453 0.447
#     Best is step 20 (0.307); the FINAL checkpoint is 46% worse. So: ONE epoch.
#     NB val/loss is NLL on judge-selected targets, and these targets are the
#     base model's OWN samples (BoN over K draws), so initial loss is already
#     ~0.1 and a falling train loss mostly means memorisation, not learning.
#     Judge behaviour with eval_sim_sft.py -- never with this curve.
EPOCHS=${EPOCHS:-1}
MICRO_BSZ=${MICRO_BSZ:-4}
# (7) save_freq 50 straddled the val optimum at step 20 in that run: the best
#     checkpoint was never written. 20 also keeps each save cheap enough to
#     pick from afterwards (~65 GB apiece with optimizer state; 231 GB for the
#     four saves of job 45889174).
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-'["console","wandb"]'}

echo "[sim_sft] train=$TRAIN_FILE"
echo "[sim_sft] val  =$VAL_FILE"
echo "[sim_sft] model=$MODEL_PATH"
echo "[sim_sft] ngpus=$NGPUS bsz=$TRAIN_BATCH_SIZE max_length=$MAX_LENGTH lr=$LR epochs=$EPOCHS"
echo "[sim_sft] save =$SAVE_DIR"

[ -s "$TRAIN_FILE" ] || { echo "[sim_sft] ERROR: no train parquet at $TRAIN_FILE"; exit 1; }
[ -s "$VAL_FILE" ]   || { echo "[sim_sft] ERROR: no val parquet at $VAL_FILE"; exit 1; }

# `val/loss = nan` whenever the val set is smaller than one batch, which reads
# as a broken run. build_sft_parquet enforces --val_min, but this run may be
# pointed at hand-made files, so check here too.
NVAL=$(python -c "import pandas as pd,sys; print(len(pd.read_parquet(sys.argv[1])))" "$VAL_FILE")
if [ "$NVAL" -lt "$TRAIN_BATCH_SIZE" ]; then
  echo "[sim_sft] ERROR: val has $NVAL rows < train_batch_size $TRAIN_BATCH_SIZE"
  echo "[sim_sft]        verl reports val/loss = nan for that; raise --val_frac."
  exit 1
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$NGPUS" \
  -m verl.trainer.sft_trainer \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.messages_key=messages \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BSZ" \
  data.max_length="$MAX_LENGTH" \
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  data.truncation=error \
  data.enable_thinking_default=null \
  model.path="$MODEL_PATH" \
  model.use_remove_padding=true \
  engine=fsdp \
  engine.ulysses_sequence_parallel_size="$SP_SIZE" \
  optim.lr="$LR" \
  optim.lr_warmup_steps_ratio=0.03 \
  'checkpoint.save_contents=["hf_model"]' \
  trainer.default_local_dir="$SAVE_DIR" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$LOGGER" \
  trainer.total_epochs="$EPOCHS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  "$@"

echo "[sim_sft] DONE -> $SAVE_DIR"
echo "[sim_sft] the hf_model/ subdir under the last global_step_* is the servable S_1."

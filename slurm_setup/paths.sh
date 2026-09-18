#!/bin/bash
# Storage roots + model registry for the SLURM/Singularity workflow.
#
# This is the ONE file that knows where things live on FASRC. It replaces the two
# things xcloud_setup got from GCS:
#   1. `gs://xcloud-shared/sunnyqin/...`  -> local filesystem roots (below)
#   2. resolve_model_meta's <GCS_WEIGHTS_DIR> field -> a LOCAL weights dir
#
# SOURCED by both sides of the fence, and they MUST agree:
#   - launch_slurm.sh, on the LOGIN node   (pre-submit validation: does this model exist?)
#   - entrypoint_*_slurm.sh, INSIDE the container
# That only works because /n/netscratch is bind-mounted at the same absolute path
# inside the container as outside it. Do NOT introduce a path that is only valid on
# one side.
#
# Everything here is overridable from the environment (`${X:-default}`) so a one-off
# run can repoint a root without editing this file.

# ── Repo (bind-mounted into the container at the same path; also the container --pwd) ──
VERL_REPO="${VERL_REPO:-/n/home05/sqin/long-horizon-RL/verl}"

# ── Scratch root. Everything WRITTEN by a run goes under here. ────────────────────────
# netscratch, not $HOME: home is 95G with ~18G free, and one 4B FSDP checkpoint
# (params + optimizer state) is tens of GB. netscratch has ~700T free.
# Retention is per-FILE atime (~90d), so a run's own resume window is never at risk;
# see verl/slurm_setup/README.md for the keep-alive touch if you need a run to survive
# for months.
#
# WHY barak_lab AND NOT dam_lab: netscratch quotas are per-GROUP and charged by each
# file's gid. dam_lab hit its 50 TB ceiling on 2026-09-16 -- and a FULL quota does not
# just stop checkpointing, it breaks the run before it starts: the writability probe in
# entrypoint_common_slurm.sh cannot mkdir ${SCRATCH_ROOT}/.wtest, and Slurm cannot even
# create the job's .out, so jobs FAIL in ~4 s with no log at all (see
# [[reference-netscratch-quota-checkpoints]]). barak_lab was at 36.2/50 TB, so the whole
# tree moved there. Because both live on the SAME Lustre mount, that move was a rename
# plus `chgrp -R barak_lab` -- the chgrp is the part that actually shifts the quota
# charge, since a rename leaves the gid (and therefore the accounting) untouched.
# ⚠️ It is effectively ONE-WAY while dam_lab is full: `chgrp dam_lab` back fails with
# EDQUOT because the destination group has no room to accept the charge.
SCRATCH_ROOT="${SCRATCH_ROOT:-/n/netscratch/barak_lab/Lab/sqin}"

# ── The container ─────────────────────────────────────────────────────────────────────
# ONE pin drives everything: the upstream tag, the sandbox directory, and the pydeps dir.
# Switching images is therefore a single edit (or `VERL_IMAGE_TAG=... sbatch ...`).
#
# WHY sgl059.latest AND NOT sgl0512.dev2:
#   colbench/Dockerfile pins `FROM verlai/verl:sgl059.latest` -- that is the image every
#   xcloud ColBench run actually trained on. An earlier session pinned sgl0512.dev2 in
#   [[reference-verl-sglang-container]] for CPU-side inspection, and this port inherited
#   that choice. It gets you a working container and passes every CPU test, but TRAINING
#   dies on the first optimizer step (job 44444989, after a clean step-0 validation):
#       verl/utils/fsdp_utils.py:183 in offload_fsdp_model_to_cpu
#         assert flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
#       AssertionError
#   sgl0512 ships torch 2.11.0+cu130, whose FSDP1 internals no longer satisfy that
#   invariant. The assert is unconditional on the FSDP1 path, and the train script
#   hardcodes ref.fsdp_config.param_offload=True, so it cannot be configured around
#   without editing shared files. Matching the Dockerfile's image is the fix that changes
#   nothing else -- and keeps results comparable with the xcloud runs.
#
# Sandbox rather than .sif because the sandbox build is MEASURED at ~45 min here and a
# .sif build is not. It is ~500k small files, so it must live on netscratch (a build on
# lab_storage NFS ran at 2.8% CPU and never finished -- job 44400009).
VERL_IMAGE_TAG="${VERL_IMAGE_TAG:-sgl059.latest}"
SANDBOX_IMAGE="${SANDBOX_IMAGE:-docker://verlai/verl:${VERL_IMAGE_TAG}}"
# dots -> dashes so the directory name is shell- and eye-friendly
_IMG_SLUG="$(echo "${VERL_IMAGE_TAG}" | tr '.' '-')"
SANDBOX="${SANDBOX:-${SCRATCH_ROOT}/docker_images/verl-${_IMG_SLUG}}"

# ── Data root: the staged mirror of the old GCS data/ tree ────────────────────────────
# stage_data.sh pulls the HF dataset repo `sunnytqin/colbench-spec-data`, which was
# built to MIRROR the GCS layout exactly:
#     ${DATA_ROOT}/colbench/{train,test,test_small}.fence.parquet
#     ${DATA_ROOT}/colbench_spec/{train,test_small}[.<author>].parquet
# Because the layout matches, the entrypoint's dataset/SPEC_AUTHOR logic is unchanged
# from xcloud_setup -- only the prefix moved from gs:// to a directory.
DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/verl_data}"

# ── Model root: an ordinary HuggingFace cache dir (models--<org>--<name>/snapshots/<sha>) ──
MODEL_ROOT="${MODEL_ROOT:-${SCRATCH_ROOT}/models/qwen}"
# Non-Qwen weights (Llama, UserLM) land in their own cache dir so MODEL_ROOT stays the
# Qwen cache it already is on disk.
MODEL_ROOT_OTHER="${MODEL_ROOT_OTHER:-${SCRATCH_ROOT}/models/hf}"

# ── Python packages the base container does NOT ship (the Dockerfile's pip layer) ─────
# Delivered via `pip install --target` + PYTHONPATH because the container rootfs is
# read-only. Populated by stage_pydeps.sh; currently just TransferQueue, which the v1
# trainer imports unconditionally. Kept MINIMAL on purpose -- PYTHONPATH is searched before
# the container's site-packages, so anything here shadows the container's copy.
# PER-IMAGE, because these are installed wheels: a dir built against one container's
# python/ABI must never be put on another's PYTHONPATH. Re-run stage_pydeps.sh after
# changing VERL_IMAGE_TAG.
PYDEPS_DIR="${PYDEPS_DIR:-${SCRATCH_ROOT}/verl_pydeps/${_IMG_SLUG}}"

# ── Weights & Biases ──────────────────────────────────────────────────────────────────
# WANDB_ENTITY is the org/team the runs land in. WANDB_MODE:
#   online   (default) -- logs live. Verified possible here: FASRC compute nodes DO have
#              outbound internet (api.wandb.ai and huggingface.co both reachable from a
#              compute node), which contradicts an assumption made earlier in this port.
#   offline  -- writes to $RUN_ROOT/wandb and uploads later via sync_wandb.sh. Use it if a
#              partition turns out to be firewalled, or to keep a run fully air-gapped.
#   disabled -- console only. Tensorboard is RETIRED: wandb is the sole metrics backend,
#              so nothing writes tfevents any more.
WANDB_ENTITY_DEFAULT="${WANDB_ENTITY_DEFAULT:-harvardml}"
WANDB_MODE_DEFAULT="${WANDB_MODE_DEFAULT:-online}"

# ── Per-run outputs (checkpoints, wandb, sim sentinel, logs) ──────────────────────────
RUN_ROOT_BASE="${RUN_ROOT_BASE:-${SCRATCH_ROOT}/verl_runs}"
# Outputs of the user-simulator TRAINING pipeline (colbench/simtrain): collected prefixes,
# candidate draws, judged records, SFT parquets. A sibling of RUN_ROOT_BASE rather than a
# subdir of a training run, because these artifacts outlive the run that produced them --
# the prefixes collected from one partner checkpoint are the input to every later rubric.
SIMTRAIN_ROOT="${SIMTRAIN_ROOT:-${RUN_ROOT_BASE}/colbench_simtrain}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${SCRATCH_ROOT}/slurm_logs}"


# Resolve an HF-cache repo dir to its single snapshot dir.
# We store the REPO dir in the registry, not the sha, because the sha changes whenever
# the weights are re-pulled and a stale sha in a table is a silent "model not found".
# Hard-fails loudly if there is no snapshot (i.e. the model was never staged) or if
# there is more than one (ambiguous -- pin it by passing the snapshot path directly).
resolve_hf_snapshot() {
    local repo_dir="$1"
    # Not an HF cache dir at all (a plain weights directory) -> use as-is.
    if [ -f "${repo_dir}/config.json" ]; then
        echo "${repo_dir}"; return 0
    fi
    local snaps=()
    while IFS= read -r d; do snaps+=("$d"); done < <(find "${repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
    if [ "${#snaps[@]}" -eq 0 ]; then
        echo ""; return 0
    fi
    if [ "${#snaps[@]}" -gt 1 ]; then
        echo "resolve_hf_snapshot: ${repo_dir} has ${#snaps[@]} snapshots; pin one explicitly." >&2
        echo ""; return 0
    fi
    echo "${snaps[0]}"
}


# resolve_hf_snapshot, but never EMPTY: an unstaged model yields the literal token
# "MISSING" so the registry line keeps all 5 fields. This matters -- an empty first field
# would make `read -r PATH TP THINK MEM FSDP` SHIFT every value left (TP would read
# "none", THINKING "0.4"...) and the run would proceed with silently wrong settings
# instead of failing. Callers test `[ "$path" = "MISSING" ]`; require_model_weights()
# in entrypoint_common_slurm.sh does it for them.
_snap_or_missing() {
    local s; s="$(resolve_hf_snapshot "$1")"
    [ -n "${s}" ] && echo "${s}" || echo "MISSING"
}


# Model registry. Maps a model NAME (the SAME identity used for --model and --sim_model)
# to "<LOCAL_WEIGHTS_DIR> <SIM_TP> <THINKING> <ROLLOUT_MEM_UTIL> <FSDP_PROFILE>".
#
# Ported from xcloud_setup/entrypoint_colbench.sh::resolve_model_meta with the leading
# <GCS_WEIGHTS_DIR> field DROPPED (there is no remote to pull from -- the weights are
# already on netscratch). The other four fields and their meaning are UNCHANGED; read
# the long comment block above resolve_model_meta in that file for what they do and why
# (SIM_TP sizing, THINKING off-vs-none for hybrid Qwen3, per-model mem_fraction_static,
# and the fsdp2_offload profile 32B needs to survive its first weight sync).
#
# NOTE the 5-field arity: every `read -r ... <<< "$(resolve_model_meta X)"` in the slurm
# entrypoints unpacks 5, not 6. Keep this table in sync with the xcloud one on the four
# shared fields -- they are the same experimental facts, not two independent choices.
resolve_model_meta() {
  local repo_dir=""
  case "$1" in
      "Qwen/Qwen2.5-14B")
          repo_dir="${MODEL_ROOT}/models--Qwen--Qwen2.5-14B"
          echo "$(_snap_or_missing "${repo_dir}") 1 none 0.4 default" ;;
      "Qwen/Qwen3-4B")
          # Identity is "Qwen3-4B" but the weights are the -Instruct-2507 checkpoint,
          # exactly as on xcloud (gs://.../models/Qwen3-4B-Instruct-2507/).
          repo_dir="${MODEL_ROOT}/models--Qwen--Qwen3-4B-Instruct-2507"
          echo "$(_snap_or_missing "${repo_dir}") 1 none 0.4 default" ;;
      "Qwen/Qwen3-14B")
          repo_dir="${MODEL_ROOT}/models--Qwen--Qwen3-14B"
          echo "$(_snap_or_missing "${repo_dir}") 1 off 0.4 default" ;;
      "Qwen/Qwen3-32B")
          repo_dir="${MODEL_ROOT}/models--Qwen--Qwen3-32B"
          echo "$(_snap_or_missing "${repo_dir}") 2 off 0.70 fsdp2_offload" ;;
      # ── LOCALLY TRAINED SIMS (simtrain SFT output; --sim_model only) ──
      # These are PLAIN HF directories (a verl checkpoint's huggingface/ subdir),
      # NOT an HF-cache repo with snapshots/, so they echo the path DIRECTLY
      # instead of going through _snap_or_missing.
      #
      # S_1' -- the r6-judged BoN-SFT sim, 3,198 rows from 7,825 judged prefixes,
      # ONE epoch. Validated 2026-09-10 on 614 held-out prefixes, judge-free, on
      # the TRUE average draw: code leak 0.090 -> 0.045 (p<0.0001) and GT tokens
      # per draw 1.16 -> 0.77 (p=0.0001) vs the base sim.
      # NB step 100 is the epoch-1 boundary. Do NOT use 120/140 (overfit), and do
      # NOT trust latest_checkpointed_iteration.txt, which reads 140.
      "local/sim-sft-r6-7825")
          repo_dir="${SIMTRAIN_ROOT}/sft_runs/sim_sft_r6_7825_1ep/ckpt/global_step_100/huggingface"
          echo "$([ -d "${repo_dir}" ] && echo "${repo_dir}" || echo MISSING) 1 none 0.90 default" ;;
      # GROUNDED-arm sim, SFT'd on judged.g3 tied-at-max targets (sft_g3_third:
      # 12,927 rows / 3,076 tasks from task_index %% 3 == 0). Step 403 IS the
      # epoch-1 boundary -- the run was 1 epoch, so there is no later step to
      # confuse it with, but name it explicitly anyway per the note above.
      # CAUTION when reading its curves: 73%% of rows share a prompt with other
      # rows carrying a DIFFERENT target (mean 3.81 tied targets per such
      # prompt), so NLL cannot fall below the entropy of that target set and
      # both train and val loss RISE through the epoch by construction
      # (train 0.15 -> 0.43, val 0.32 -> 0.46). That is not overfitting and is
      # not a checkpoint-selection signal. Judge behaviour is the only readout.
      "local/sim-sft-g3-third")
          repo_dir="${SIMTRAIN_ROOT}/sft_runs/sim_sft_g3_third_1ep/ckpt/global_step_403/huggingface"
          echo "$([ -d "${repo_dir}" ] && echo "${repo_dir}" || echo MISSING) 1 none 0.90 default" ;;
      # S_2 -- the SAME judged g3 data as sim-sft-g3-third, rebuilt with
      # --no-all_tied AND --min_dims gt_adherence=2 (5,134 rows vs 12,927, mean
      # target score 10.98 vs 9.90). Built because S_1 traded ~15pp of hard
      # failures for +3.9pp P(gt_adherence<=1) per served draw, which makes
      # episodes unwinnable rather than merely unhelpful. Step 160 = epoch-1 end.
      # Its val rise is 0.326 -> 0.389 against S_1's 0.32 -> 0.46, so the
      # conflicting-target entropy floor explained most but NOT all of S_1's.
      "local/sim-sft-g3-single")
          repo_dir="${SIMTRAIN_ROOT}/sft_runs/sim_sft_g3_single_1ep/ckpt/global_step_160/huggingface"
          echo "$([ -d "${repo_dir}" ] && echo "${repo_dir}" || echo MISSING) 1 none 0.90 default" ;;
      # S_1 -- the earlier 2,057-row sim, kept for the data-scale comparison.
      "local/sim-sft-r6-5k")
          repo_dir="${SIMTRAIN_ROOT}/sft_runs/sim_sft_r6_5k_1ep/ckpt/global_step_64/huggingface"
          echo "$([ -d "${repo_dir}" ] && echo "${repo_dir}" || echo MISSING) 1 none 0.90 default" ;;
      # ── SIM-ONLY large models (used ONLY as --sim_model on the sim node) ──
      # NOT staged on netscratch yet -- resolve_hf_snapshot returns "" and the caller
      # prints the `stage_model.sh` command. Last two fields are placeholders (the sim
      # path reads only <PATH> <SIM_TP> <THINKING>; the sim server pins 0.90).
      "Qwen/Qwen3-235B-A22B-Instruct-2507")
          repo_dir="${MODEL_ROOT}/models--Qwen--Qwen3-235B-A22B-Instruct-2507"
          echo "$(_snap_or_missing "${repo_dir}") 8 none 0.90 default" ;;
      "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
          # Official block-FP8 checkpoint: four-GPU serving candidate. Keep the
          # BF16 identity/default above distinct; TP=4 is not a measured fit claim.
          repo_dir="${MODEL_ROOT}/models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8"
          echo "$(_snap_or_missing "${repo_dir}") 4 none 0.90 default" ;;
      "meta-llama/Llama-3.3-70B-Instruct")
          repo_dir="${MODEL_ROOT_OTHER}/models--meta-llama--Llama-3.3-70B-Instruct"
          echo "$(_snap_or_missing "${repo_dir}") 4 none 0.90 default" ;;
      "microsoft/UserLM-8b")
          repo_dir="${MODEL_ROOT_OTHER}/models--microsoft--UserLM-8b"
          echo "$(_snap_or_missing "${repo_dir}") 1 none 0.90 default" ;;
      *)
          echo "" ;;
  esac
}


# Short, filename-safe tag for a model identity. Ported VERBATIM in behaviour from
# xcloud_setup/entrypoint_eval_colbench.sh::model_shorthand so eval filenames stay
# comparable with the xcloud-era results. Used for the SIM tag in eval output names:
# evals of the SAME checkpoint under DIFFERENT user-simulators must land in different
# files rather than silently overwrite each other.
model_shorthand() {
  case "$1" in
      "Qwen/Qwen2.5-14B") echo "qwen2_5_14b" ;;
      "Qwen/Qwen3-4B")    echo "qwen3_4b" ;;
      *) echo "${1##*/}" | tr 'A-Z' 'a-z' | tr '.-' '__' ;;
  esac
}


# The HF repo id a model identity is staged FROM. Only used to print an actionable
# error ("run stage_model.sh <repo>") when the weights are missing.
model_hf_repo() {
  case "$1" in
      "Qwen/Qwen3-4B") echo "Qwen/Qwen3-4B-Instruct-2507" ;;
      *)               echo "$1" ;;
  esac
}

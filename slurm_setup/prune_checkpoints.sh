#!/bin/bash
# ============================================================================
# Prune colbench RL checkpoints down to the ones that carry information.
#
# WHY THIS EXISTS: one Qwen3-4B GRPO checkpoint is ~47 GB (16.8 GB sharded
# weights + 30 GB Adam moments), MAX_CKPT_KEEP defaults to 10, and a
# netscratch quota is a GROUP 50 TB shared with the whole lab. Twelve colbench
# runs at ten checkpoints each is ~5.6 TB of which almost all is dead weight.
# When the quota fills, Slurm cannot even create a job's .out file, so jobs
# FAIL in ~4 s with no log and a nonsense signal (seen 2026-09-15, job
# 46634434) -- which looks like a mystery cancellation, not a disk problem.
# Check `quota /n/netscratch/barak_lab` first when a job dies instantly.
#
# NB the tree moved dam_lab -> barak_lab on 2026-09-16 because dam_lab hit
# 50.0/50.0 TB. Quota is charged per-file by gid, so the move was a rename plus
# `chgrp -R barak_lab`; pruning here now relieves barak_lab, not dam_lab.
#
# THE POLICY: thin each run to effectively SAVE_FREQ=100 -- keep every step that
# is a multiple of 100, drop the intermediate 50s -- and ALWAYS keep the
# checkpoint that carries the run's val peak even when it is not a multiple of
# 100. That halves the footprint without narrowing the window a peak can be
# found in, which keeping only 1-3 per run would have done. Two runs are
# dropped entirely: gt_slurm_smoke (a plumbing check, never a model) and
# nonspec_role_restraint_rej32 (cancelled at step 93 as uninformative).
#
# NB for a collapsed run the peak checkpoint is the last one at or BEFORE the
# detonation step, not the one nearest the peak val step -- val is measured
# every 20 steps and checkpoints every 50, so for rej1 (peak val @480,
# detonation @479) step 500 is already post-detonation and 450 is the usable
# policy. Where the two bracket the transition, both are kept.
#
# USAGE (prints the plan and changes NOTHING without --yes):
#   bash slurm_setup/prune_checkpoints.sh              # dry run
#   bash slurm_setup/prune_checkpoints.sh --yes        # actually delete
#
# Edit KEEP below when a run's peak moves or a new run lands. A run with NO
# KEEP entry is skipped entirely, never emptied -- silence means "leave alone",
# so a new run can never be deleted by forgetting to list it.
# ============================================================================
set -euo pipefail

# Derive from paths.sh rather than hardcoding: SCRATCH_ROOT moved from dam_lab to
# barak_lab on 2026-09-16 when the dam_lab group quota filled, and a hardcoded root
# here would have silently pruned nothing (an absent dir just prints "skip").
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${_HERE}/paths.sh" >/dev/null 2>&1 || true
ROOT="${ROOT:-${RUN_ROOT_BASE:-/n/netscratch/barak_lab/Lab/sqin/verl_runs}/colbench_mt}"
APPLY="no"
[ "${1:-}" = "--yes" ] && APPLY="yes"

# run <TAB> space-separated steps to KEEP.  Multiples of 100, plus the peak.
# Verified val peaks (wandb val-core/*/reward/mean@1) and detonation steps
# (first actor/entropy > 1.0):
#
#   run                              val peak     H>1 at  extra kept beyond the 100s
#   nonspec_baseline                 0.9316@200   238     -- (200 is a 100)
#   nonspec_role_restraint           0.8753@260   240     250  (peak straddles detonation)
#   ..._rej1                         0.6498@480   479     450  (500 is POST-detonation)
#   ..._rej8                         0.6104@520   515     -- (500 is a 100)
#   ..._simsft_s1p                   0.6138@300   348     -- (300 is a 100)
#   ..._rr_rej1_cliphi02_ent001      0.7010@600   605     -- (600 is a 100)
#   spec_baseline                    0.9404@860   never   850  (the reward-HACKING arm)
#   spec_rej8                        0.7460@700   655     650  (peak straddles detonation)
#   spec_rej8_ent003                 0.7438@980   1015    950  (+900 = replay source, a 100)
#   spec_rej8_n16                    0.7546@280   301     250  (peak straddles detonation)
#   spec_ent003_replay900_diag       --           --      NONE -- diagnostic replay; its own
#                                                            checkpoints are all in/past the
#                                                            detonation window, and the rerun
#                                                            uses SAVE_FREQ=-1 so writes none
#   gt_slurm_smoke                   --           --      NONE -- plumbing smoke test
#   ..._rej32                        --           --      NONE -- cancelled at step 93
KEEP=$(cat <<'EOF'
qwen3_4b_nonspec_baseline	100 200 300
qwen3_4b_nonspec_role_restraint	100 200 250
qwen3_4b_nonspec_role_restraint_rej1	100 200 300 400 450 500 600
qwen3_4b_nonspec_role_restraint_rej8	100 200 300 400 500 600 700
qwen3_4b_nonspec_role_restraint_simsft_s1p	100 200 300
qwen3_4b_nonspec_rr_rej1_cliphi02_ent001	100 200 300 400 500 600
qwen3_4b_spec_baseline	100 200 300 400 500 600 700 800 850 900 1000
qwen3_4b_spec_rej8	100 200 300 400 500 600 650 700
qwen3_4b_spec_rej8_ent003	100 200 300 400 500 600 700 800 900 950 1000 1100 1200
qwen3_4b_spec_rej8_n16	100 200 250 300
qwen3_4b_spec_ent003_replay900_diag	NONE
qwen3_4b_gt_slurm_smoke	NONE
qwen3_4b_nonspec_role_restraint_rej32	NONE
EOF
)

CKPT_GB=47   # measured: du -sh on one global_step_* dir
n_del=0; n_keep=0

while IFS=$'\t' read -r run keeps; do
  [ -n "${run}" ] || continue
  d="${ROOT}/${run}/checkpoints"
  [ -d "${d}" ] || { echo "skip (no checkpoints dir): ${run}"; continue; }
  [ "${keeps}" = "NONE" ] && keeps=""
  surviving=""
  for p in "${d}"/global_step_*; do
    [ -d "${p}" ] || continue
    step="${p##*global_step_}"
    if [[ " ${keeps} " == *" ${step} "* ]]; then
      n_keep=$((n_keep+1)); surviving="${surviving} ${step}"
      continue
    fi
    # Belt and braces: never act on a path that is not a checkpoint dir.
    case "${p}" in
      */verl_runs/colbench_mt/*/checkpoints/global_step_*) ;;
      *) echo "REFUSING unexpected path: ${p}"; continue ;;
    esac
    n_del=$((n_del+1))
    if [ "${APPLY}" = "yes" ]; then
      rm -rf "${p}"; echo "deleted  ${run}/global_step_${step}"
    else
      echo "would delete  ${run}/global_step_${step}"
    fi
  done
  # A stale latest_checkpointed_iteration.txt pointing at a deleted directory
  # makes any later resume_mode=auto relaunch of this exp_name fail in
  # find_latest_ckpt_path rather than starting clean, so repoint it at the
  # highest surviving step (or remove it when nothing survives).
  latest="${d}/latest_checkpointed_iteration.txt"
  if [ "${APPLY}" = "yes" ] && [ -f "${latest}" ]; then
    hi=$(echo ${surviving} | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
    if [ -n "${hi}" ]; then
      echo "${hi}" > "${latest}"; echo "  repointed latest -> ${hi}"
    else
      rm -f "${latest}"; echo "  removed stale latest pointer"
    fi
  fi
done <<< "${KEEP}"

echo
echo "checkpoints to delete: ${n_del}  (~$((n_del*CKPT_GB)) GB)"
echo "checkpoints kept:      ${n_keep}  (~$((n_keep*CKPT_GB)) GB)"
[ "${APPLY}" = "yes" ] || echo "DRY RUN -- nothing changed. Re-run with --yes to apply."

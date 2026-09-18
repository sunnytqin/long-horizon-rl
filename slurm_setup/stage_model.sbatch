#!/usr/bin/env bash
# CPU-only, node-local HF download followed by an ordinary rsync to netscratch.
# Usage: HF_REVISION=<commit> sbatch slurm_setup/stage_model.sbatch <HF_REPO>
# Default is the pinned four-GPU 235B FP8 serving candidate.
#SBATCH --job-name=stage_qwen235
#SBATCH --partition=test
#SBATCH --account=dam_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/n/netscratch/barak_lab/Lab/sqin/slurm_logs/stage_qwen235_%j.out
#SBATCH --error=/n/netscratch/barak_lab/Lab/sqin/slurm_logs/stage_qwen235_%j.err

set -euo pipefail
SLURM_SETUP_DIR="${SLURM_SETUP_DIR:-/n/home05/sqin/long-horizon-RL/verl/slurm_setup}"
source "${SLURM_SETUP_DIR}/paths.sh"
REPO_ID="${1:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
[[ "${REPO_ID}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid HF repository'; exit 1; }
if [ "${REPO_ID}" = Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 ]; then
  export HF_REVISION="${HF_REVISION:-b939f17ff56df5c734de72b3d06cb51164ac0489}"
fi
case "${REPO_ID}" in Qwen/*) DEST_ROOT="${MODEL_ROOT}" ;; *) DEST_ROOT="${MODEL_ROOT_OTHER}" ;; esac
CACHE_NAME="models--${REPO_ID//\//--}"
# Do not use a caller's TMPDIR: it may point to home or NFS. test nodes
# normally have local /tmp; prefer /scratch where it is available/writable.
LOCAL_PARENT=/tmp
if [ -d /scratch ] && [ -w /scratch ]; then LOCAL_PARENT=/scratch; fi
available_kb=$(df -Pk "${LOCAL_PARENT}" | awk 'NR==2 {print $4}')
if [ "${available_kb}" -lt 314572800 ]; then
  echo "Need at least 300 GiB free on ${LOCAL_PARENT}; available=${available_kb} KiB"
  exit 1
fi
STAGE=$(mktemp -d "${LOCAL_PARENT}/${USER}_hf_${SLURM_JOB_ID}_XXXXXX")
echo "job=${SLURM_JOB_ID} node=$(hostname -s) repo=${REPO_ID} revision=${HF_REVISION:-main}"
echo "node-local stage=${STAGE}; final cache=${DEST_ROOT}/${CACHE_NAME}"
mkdir -p "${STAGE}/models/${CACHE_NAME}" "${DEST_ROOT}/${CACHE_NAME}"
# Snapshot code, too: queued/running jobs must not consume edits mid-script.
cp "${SLURM_SETUP_DIR}/stage_model.sh" "${SLURM_SETUP_DIR}/paths.sh" \
   "${SLURM_SETUP_DIR}/serving_probe.py" "${STAGE}/"
if [ -d "${DEST_ROOT}/${CACHE_NAME}" ]; then
  echo 'Seeding completed cached shards (no locks or partial downloads)...'
  rsync -a --exclude='*.incomplete' --exclude='*.lock' \
    "${DEST_ROOT}/${CACHE_NAME}/" "${STAGE}/models/${CACHE_NAME}/"
fi
export MODEL_ROOT="${STAGE}/models" MODEL_ROOT_OTHER="${STAGE}/models"
export MODEL_STAGE_CACHE="${STAGE}/cache" MODEL_STAGE_TMP="${STAGE}/tmp"
export HF_HUB_DISABLE_PROGRESS_BARS=1
rc=0
bash "${STAGE}/stage_model.sh" "${REPO_ID}" || rc=$?
# Persist completed shards even if the download fails; never delete/overwrite
# existing cached blobs. Partial .incomplete files stay node-local.
echo "Download exited ${rc}; syncing completed files to ${DEST_ROOT}/${CACHE_NAME}"
rsync -a --ignore-existing --exclude='*.incomplete' --exclude='*.lock' \
  "${STAGE}/models/${CACHE_NAME}/" "${DEST_ROOT}/${CACHE_NAME}/"
if [ "${rc}" -ne 0 ]; then
  echo "Download failed; completed files preserved on netscratch; local stage retained: ${STAGE}"
  exit "${rc}"
fi
snapshot="$(resolve_hf_snapshot "${DEST_ROOT}/${CACHE_NAME}")"
python3 "${STAGE}/serving_probe.py" weights "${snapshot}"
echo "SUCCESS: complete model at ${snapshot}"
# Only this job's mktemp directory; final model/cache on netscratch are untouched.
case "${STAGE}" in "${LOCAL_PARENT}/${USER}_hf_${SLURM_JOB_ID}_"*) rm -rf -- "${STAGE}" ;; esac
echo 'Removed the job-local staging copy after verifying the final model.'

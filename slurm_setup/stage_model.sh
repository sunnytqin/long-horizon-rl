#!/bin/bash
# Stage model weights into the configured cache. Prefer stage_model.sbatch for
# large downloads: it avoids Hugging Face file locks on netscratch/NFS by
# downloading to node-local disk and rsyncing completed files back.
#
# Verify the result independently of how it was downloaded -- a file count is not
# enough, since an interrupted shard can be present but truncated:
#   python3 slurm_setup/serving_probe.py weights <snapshot-dir>
#
#   bash slurm_setup/stage_model.sh Qwen/Qwen3-4B-Instruct-2507
#   bash slurm_setup/stage_model.sh microsoft/UserLM-8b
#
# Downloads into an ordinary HF cache layout (models--<org>--<name>/snapshots/<sha>), which
# is what paths.sh::resolve_hf_snapshot expects -- it resolves the sha at run time rather
# than pinning it in the registry, so a re-download does not silently break the table.
#
# Already staged (nothing to do): Qwen3-4B-Instruct-2507, Qwen3-14B, Qwen3-32B, and
# Qwen3-235B-A22B-Instruct-2507-FP8 (220 GiB, verified 2026-09-13 -- the four-GPU
# serving candidate).
# Not staged, needed only as large remote sims: Qwen3-235B-A22B-Instruct-2507 in BF16
# (~470G, does NOT fit on a four-GPU node), Llama-3.3-70B-Instruct (~140G),
# UserLM-8b (~32G, fp32 -- served with --dtype bfloat16).
#
# Gated repos (Llama) need a token: export HF_TOKEN=hf_... before running.
# Optional HF_REVISION pins a commit (record the resolved snapshot in the run).
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/paths.sh"

REPO_ID="${1:-}"
[ -n "${REPO_ID}" ] || { echo "usage: bash stage_model.sh <hf-repo-id>"; exit 1; }

# Qwen models join the existing Qwen cache; everything else goes to MODEL_ROOT_OTHER, so
# the two roots match what resolve_model_meta already points at.
case "${REPO_ID}" in
  Qwen/*) TARGET_ROOT="${MODEL_ROOT}" ;;
  *)      TARGET_ROOT="${MODEL_ROOT_OTHER}" ;;
esac
# Pin every cache explicitly; do not inherit a login-shell HF/XDG cache in HOME.
STAGE_CACHE="${MODEL_STAGE_CACHE:-${SCRATCH_ROOT}/cache}"
STAGE_TMP="${MODEL_STAGE_TMP:-${SCRATCH_ROOT}/tmp/model_stage_${SLURM_JOB_ID:-$$}}"
mkdir -p "${TARGET_ROOT}/.home" "${STAGE_CACHE}/xet" "${STAGE_CACHE}/xdg" "${STAGE_TMP}"
export SINGULARITY_CACHEDIR="${STAGE_CACHE}/singularity"
export SINGULARITY_TMPDIR="${STAGE_TMP}"

[ -d "${SANDBOX}" ] || { echo "❌ container sandbox missing at ${SANDBOX}. Run: sbatch ${_HERE}/build_sandbox.sbatch"; exit 1; }

echo "==> ${REPO_ID}  ->  ${TARGET_ROOT}"
echo "    (large models take a while; run this under tmux/screen)"

# HF_HUB_CACHE (not HF_HOME) so the cache root IS the target dir and we get
# models--<org>--<name>/ directly inside it. --home / SSL_CERT_FILE: see node_launch.sh.
singularity exec \
  --bind /n/netscratch \
  --bind "${TARGET_ROOT}" --bind "${STAGE_CACHE}" --bind "${STAGE_TMP}" \
  --home "${TARGET_ROOT}/.home" \
  --env HF_HOME="${STAGE_CACHE}" \
  --env HF_HUB_CACHE="${TARGET_ROOT}" \
  --env HF_XET_CACHE="${STAGE_CACHE}/xet" \
  --env XDG_CACHE_HOME="${STAGE_CACHE}/xdg" \
  --env TMPDIR="${STAGE_TMP}" \
  --env HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}" \
  --env PYTHONUNBUFFERED=1 \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env REPO_ID="${REPO_ID}" \
  --env HF_REVISION="${HF_REVISION:-main}" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "${SANDBOX}" python3 -c "
import os
from huggingface_hub import snapshot_download
print('Download storage:', {k: os.environ.get(k) for k in
      ('HOME', 'HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'XDG_CACHE_HOME', 'TMPDIR')}, flush=True)
p = snapshot_download(
    repo_id=os.environ['REPO_ID'],
    revision=os.environ['HF_REVISION'],
    # Weights + configs + tokenizer only. Skips the .pth/.gguf/.msgpack/.h5 mirrors some
    # repos ship, which sglang never reads and which can double or triple the download.
    allow_patterns=['*.safetensors', '*.safetensors.index.json', '*.json', '*.txt',
                    '*.model', 'tokenizer*', '*.py'],
    max_workers=8,
)
print('SNAPSHOT', p)
" 2>&1 | sed -u 's/^/   /'

echo
echo "==> registry check:"
for _id in "${REPO_ID}" "${REPO_ID%-Instruct-2507}"; do
  _meta="$(resolve_model_meta "${_id}" 2>/dev/null || true)"
  [ -n "${_meta}" ] && { read -r _p _ <<< "${_meta}"; echo "   ${_id} -> ${_p}"; }
done
echo "   (if nothing printed, add ${REPO_ID} to paths.sh::resolve_model_meta -- the download"
echo "    is fine, the registry just does not know the identity yet)"

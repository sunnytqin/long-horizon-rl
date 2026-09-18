#!/bin/bash
# Stage the ColBench parquets onto netscratch. Run ONCE, on a LOGIN node (compute nodes
# have no internet). Idempotent -- re-running only fetches what changed.
#
# This is the FASRC replacement for `gcloud storage cp -r gs://.../data/...`. The bridge is
# the HF dataset repo sunnytqin/colbench-spec-data, which was built (by
# ~/upload_colbench_specs.sh) to MIRROR the GCS layout precisely:
#
#     data/colbench/{train,test,test_small}.fence.parquet          <- GT path (the DEFAULT)
#     data/colbench_spec/{train,test_small}.parquet                <- spec path, 4B self-play
#     data/colbench_spec/{train,test_small}.<author>.parquet        <- spec arms
#
# Because the layout matches, the entrypoint's dataset switch, --spec_author suffix
# resolution and the *.fence.* defaults in run_colbench_grpo.sh all work unchanged. That
# mirroring is the single reason this port needed no dataset-path surgery.
#
# The download runs INSIDE the container rather than in a conda env: the container already
# has huggingface_hub, and a login node shares its network namespace with it, so there is
# one fewer environment to keep working.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/paths.sh"

HF_DATASET_REPO="${HF_DATASET_REPO:-sunnytqin/colbench-spec-data}"

# Download to LOCAL disk, then rsync onto netscratch. This is not an optimization -- it is
# required. huggingface_hub coordinates its parallel downloads with `filelock`, and
# fcntl/flock locking on this NFS deadlocks: the process ends up holding several fds on
# .cache/huggingface/.gitignore.lock and never advances (observed: 6 minutes, 24 KB
# written, 0 bytes of payload). /tmp here is the node's local LVM volume with 3.4 T free,
# so the whole dataset lands there in seconds and the NFS write is one sequential rsync.
STAGE="${HF_STAGE_DIR:-${TMPDIR:-/tmp}/${USER}_colbench_hf_stage}"

[ -d "${SANDBOX}" ] || { echo "❌ container sandbox missing at ${SANDBOX}. Run: sbatch ${_HERE}/build_sandbox.sbatch"; exit 1; }

mkdir -p "${STAGE}/.home" "${DATA_ROOT}"
echo "==> downloading ${HF_DATASET_REPO} -> ${STAGE}"

# --home: a writable HOME for the hf cache. --env HOME= does NOT work (Singularity refuses
# to override HOME that way and only prints a warning).
# SSL_CERT_FILE/CURL_CA_BUNDLE: the RHEL8 host exports a CA path that does not exist inside
# the Ubuntu container, and the inherited value makes every TLS call fail with
# FileNotFoundError. See node_launch.sh for the same fix.
singularity exec \
  --bind /n/netscratch \
  --bind "${STAGE}" \
  --home "${STAGE}/.home" \
  --env HF_HOME="${STAGE}/.hf" \
  --env REPO="${HF_DATASET_REPO}" \
  --env STAGE="${STAGE}" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "${SANDBOX}" python3 -c "
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id=os.environ['REPO'], repo_type='dataset',
    local_dir=os.environ['STAGE'] + '/snapshot',
)
print('downloaded to', p)
" 2>&1 | sed 's/^/   /' || { echo "❌ download failed"; exit 1; }

# The repo nests everything under data/, so flatten one level into DATA_ROOT.
echo "==> installing into ${DATA_ROOT}"
rsync -a --delete-after "${STAGE}/snapshot/data/" "${DATA_ROOT}/"

# Local-only extras. A few parquets exist in ~/data but were never uploaded (the augmented
# spec train set, and the pre-2026-07-31 unsuffixed marker-protocol GT parquets that older
# runs used via TRAIN_FILE). Copy them alongside so an old run stays reproducible; skip
# quietly if they are gone.
_extra() {  # _extra <src> <dest-relative>
  [ -f "$1" ] || return 0
  mkdir -p "$(dirname "${DATA_ROOT}/$2")"
  rsync -a "$1" "${DATA_ROOT}/$2"
  echo "   + (local-only) $2"
}
_extra "$HOME/data/colbench_spec/train.selfplay_aug.parquet"      colbench_spec/train.selfplay_aug.parquet
_extra "$HOME/data/colbench_spec/test_small.selfplay_aug.parquet" colbench_spec/test_small.selfplay_aug.parquet

echo
echo "==> staged:"
find "${DATA_ROOT}" -maxdepth 2 -name '*.parquet' -printf '   %-56p %10s bytes\n' | sort
echo
echo "DATA_ROOT=${DATA_ROOT}"
echo "Local stage: ${STAGE} (safe to delete; keeping it makes a re-run incremental)."

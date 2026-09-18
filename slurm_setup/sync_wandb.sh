#!/bin/bash
# Upload OFFLINE wandb runs to the server. Run on a LOGIN node.
#
#   bash slurm_setup/sync_wandb.sh                          # every offline run found
#   bash slurm_setup/sync_wandb.sh qwen3_4b_gt_slurm_smoke  # one experiment
#
# Only needed for --wandb=offline. The default is ONLINE, which works because FASRC
# compute nodes have outbound internet (verified: api.wandb.ai and huggingface.co are both
# reachable from a compute node). Keep this for the cases where online is not an option --
# a firewalled partition, or a deliberately air-gapped run.
#
# Credentials: this runs on a login node with your REAL $HOME, so wandb picks up
# ~/.netrc as usual. Nothing about the offline run directory contains a key.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/paths.sh"

FILTER="${1:-}"
SEARCH="${RUN_ROOT_BASE}"

# maxdepth 6, not 4: the expected path is
# verl_runs/<project>/<experiment>/wandb/offline-run-* (4 levels), but leave slack so a
# layout change cannot silently make offline runs invisible to this script.
mapfile -t RUNS < <(find "${SEARCH}" -maxdepth 6 -type d -name 'offline-run-*' 2>/dev/null | sort)
if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "No offline wandb runs under ${SEARCH}."
    echo "(Runs launched with the default --wandb=online logged directly and need no sync.)"
    exit 0
fi

echo "==> found ${#RUNS[@]} offline run dir(s) under ${SEARCH}"
_n=0
for r in "${RUNS[@]}"; do
    if [ -n "${FILTER}" ] && [[ "${r}" != *"${FILTER}"* ]]; then continue; fi
    echo "--> ${r}"
    # Run wandb from inside the container (it is installed there, and this is the same
    # version that wrote the run). --home $HOME so ~/.netrc resolves; the run dirs live on
    # netscratch so that bind is needed too.
    singularity exec \
      --bind /n/netscratch \
      --home "${HOME}" \
      --env WANDB_ENTITY="${WANDB_ENTITY:-${WANDB_ENTITY_DEFAULT}}" \
      --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
      --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
      "${SANDBOX}" wandb sync "${r}" 2>&1 | sed 's/^/     /'
    _n=$((_n + 1))
done
echo
echo "synced ${_n} run(s)."
[ "${_n}" -eq 0 ] && [ -n "${FILTER}" ] && echo "(nothing matched '${FILTER}')"
exit 0

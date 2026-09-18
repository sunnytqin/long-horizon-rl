#!/bin/bash
# Install the python packages verl needs that the base container does NOT ship.
# Run ONCE on a LOGIN node (compute nodes have no internet). Idempotent.
#
# THIS IS THE PORT OF THE DOCKERFILE'S PIP LAYER. colbench/Dockerfile step 3 pip-installed
# a pile of packages on top of verlai/verl:*, and the slurm port initially used the raw base
# image -- so training died at `ModuleNotFoundError: No module named 'transfer_queue'`
# (job 44440881), several minutes in, after Ray was already up.
#
# Why transfer_queue is NOT optional: verl ships a mock in
# verl/utils/transferqueue_utils.py and `transfer_queue.enable` defaults to False, which
# makes it LOOK optional. But TaskRunnerV1.run() does a bare `import transfer_queue as tq`
# and then sets config.transfer_queue.enable = True itself, so the v1 trainer always needs
# the real package. Installing it (rather than switching to the v0 trainer) is deliberate:
# it is what the xcloud image did, so results stay comparable across the migration.
#
# What we deliberately do NOT install from that Dockerfile layer:
#   google-cloud-cli, gcsfs, google-cloud-storage, tensorflow*, tf-keras  -- all GCS/TB-on-
#       GCS plumbing, and there is no GCS here. Tensorboard is RETIRED entirely (wandb is
#       the only metrics backend), so its absence does not matter either way.
#   jupyterlab, ipdb, matplotlib  -- interactive dev conveniences, not run dependencies.
#   nltk, math-verify, latex2sympy2-extended  -- math/codecontest reward helpers. ColBench
#       grades by functional equivalence through the exec sidecar and never imports them.
#       Add them here if a codecontest port needs them.
#   `pip install --no-deps -e .` of the repo -- replaced by PYTHONPATH, which is what makes
#       the code snapshot possible at all.
#
# HOW it is delivered: the container rootfs is READ-ONLY, so nothing can be installed into
# it. Instead `pip install --target` puts the packages in a netscratch dir that
# node_launch.sh appends to PYTHONPATH. The pip runs INSIDE the container so the wheels
# match its python 3.12 exactly.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/paths.sh"

PKGS=("TransferQueue")

[ -d "${SANDBOX}" ] || { echo "❌ container sandbox missing at ${SANDBOX}"; exit 1; }
mkdir -p "${PYDEPS_DIR}" "${PYDEPS_DIR}/.home"

echo "==> installing into ${PYDEPS_DIR}: ${PKGS[*]}"

# --no-deps ON PURPOSE. Without it pip resolves each package's full dependency tree into the
# target dir, and because PYTHONPATH is searched BEFORE the container's site-packages, a
# stray copy of torch/numpy/ray there would SHADOW the container's -- silently swapping the
# versions the whole stack was built against. Keep this dir minimal; if a package genuinely
# needs something the container lacks, add that thing to PKGS explicitly so the choice is
# visible in this file.
singularity exec \
  --bind /n/netscratch \
  --home "${PYDEPS_DIR}/.home" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "${SANDBOX}" pip install --no-cache-dir --no-deps --upgrade \
      --target "${PYDEPS_DIR}" "${PKGS[@]}" 2>&1 | sed 's/^/   /'

# Prune top-level dirs that are NOT importable package payload. The TransferQueue wheel
# ships its own repo's recipe/ scripts/ tests/ tutorial/ at top level, and verl has
# recipe/ scripts/ tests/ of its OWN at repo root -- so leaving them here would let
# `import recipe` / `import tests` resolve to TransferQueue's copies. That is a silent,
# miserable failure mode, so it is removed at install time rather than worked around by
# PYTHONPATH ordering (node_launch.sh puts the repo first as a second line of defence).
for _d in recipe scripts tests tutorial docs examples; do
    if [ -d "${PYDEPS_DIR}/${_d}" ]; then
        echo "   pruning stray top-level ${_d}/ (would shadow verl's own)"
        rm -rf "${PYDEPS_DIR}/${_d}"
    fi
done

echo
echo "==> verifying inside the container (repo first on PYTHONPATH, then pydeps):"
singularity exec \
  --bind /n/netscratch --bind "${VERL_REPO}" \
  --pwd "${VERL_REPO}" --home "${PYDEPS_DIR}/.home" \
  --env PYTHONPATH="${VERL_REPO}:${PYDEPS_DIR}" \
  "${SANDBOX}" python3 -c "
import transfer_queue as tq, torch, numpy, recipe, tests
assert '/usr/local/lib' in torch.__file__, f'torch SHADOWED: {torch.__file__}'
assert 'verl_pydeps' in tq.__file__, f'transfer_queue not from pydeps: {tq.__file__}'
assert 'verl_pydeps' not in recipe.__path__[0], f'recipe SHADOWED: {recipe.__path__[0]}'
assert 'verl_pydeps' not in tests.__path__[0], f'tests SHADOWED: {tests.__path__[0]}'
from verl.trainer.main_ppo import TaskRunnerV1
print('   transfer_queue', tq.__version__, '(pydeps)')
print('   torch         ', torch.__version__, '(container)')
print('   numpy         ', numpy.__version__, '(container)')
print('   TaskRunnerV1 imports OK')
" 2>&1 | grep -E '^   |Error|assert' || { echo "❌ verification failed"; exit 1; }

echo
echo "PYDEPS_DIR=${PYDEPS_DIR}  (node_launch.sh appends it to PYTHONPATH)"

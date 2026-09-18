#!/bin/bash
# Runs ON a compute node, OUTSIDE the container, as the body of one srun step.
# Its only job is to set up node-local scratch and then exec `singularity exec` with the
# validated recipe. Kept in its own file so train_colbench.sbatch stays readable and so
# the exec recipe exists in exactly ONE place.
#
# Role (train vs sim) comes in through SIM_SERVER_ONLY, set per-step by the sbatch script.
#
# THE EXEC RECIPE -- every flag here was validated by slurm_setup/gpu_smoke.sbatch
# (job 44421994, 13/13 checks). The three easy ways to break it:
#
#   * DO NOT add --cleanenv. It drops the IMAGE's environment, so PATH loses the container
#     python and you get /usr/bin/python without sglang -- or worse, the host's.
#   * DO NOT invoke the entrypoint through `bash -lc`. A login shell sources ~/.bashrc,
#     which puts the host conda (python 3.10) first on PATH. verl needs >=3.11
#     (enum.StrEnum), so it dies on an import and the traceback looks like a verl bug.
#     Plain `bash <script>` is fine; that is what we do.
#   * DO NOT drop --nv. It is Singularity's equivalent of docker --gpus all; without it
#     torch.cuda.device_count() is 0 even though Slurm allocated the GPUs.
#
# SSL_CERT_FILE / CURL_CA_BUNDLE are overridden for a subtler reason: the HOST is RHEL8 and
# exports SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt, but the container is Ubuntu, where
# that file does not exist (its bundle is ca-certificates.crt). Because Singularity inherits
# the host environment, every TLS call inside the container dies on
# `FileNotFoundError` from ssl.create_default_context. HF_HUB_OFFLINE means training should
# never make one, but "should never" is not a reason to leave a landmine in the env.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=paths.sh
source "${_HERE}/paths.sh"

: "${ENV_FILE:?ENV_FILE must be set by the sbatch script}"
: "${ENTRYPOINT:?ENTRYPOINT must be set by the sbatch script (repo-relative)}"

# Node-local scratch on /scratch: a real local disk (LVM, ~840G), NOT part of the 95G home
# quota, and wiped when the job ends. Everything that is hot, per-process and disposable
# goes here rather than onto NFS: Ray's spill/session dirs, and the Triton / Inductor /
# outlines compile caches. Those caches are the ones that make an NFS home miserable --
# thousands of tiny files written under lock contention at engine startup.
NODE_TMP="/scratch/${USER}_${SLURM_JOB_ID:-manual}"
mkdir -p "${NODE_TMP}"/{ray,triton,inductor,xdg,outlines}

# A per-run HOME on netscratch. The container inherits the host env, so without this every
# library that writes to ~/.cache or ~/.config lands in the real $HOME -- which has ~18G
# free and is the wrong place for a training run's droppings. Also keeps runs isolated.
#
# It MUST be done with --home, not --env HOME=. Singularity refuses to let HOME be set that
# way ("Overriding HOME environment variable with SINGULARITYENV_HOME is not permitted") --
# it prints a WARNING and carries on with the real $HOME, so the mistake is silent unless
# you read the log. --home <path> both sets HOME and bind-mounts it.
FAKE_HOME="${RUN_ROOT:-${SCRATCH_ROOT}/verl_runs/_scratch}/home"
mkdir -p "${FAKE_HOME}"

ROLE_LABEL=$([ "${SIM_SERVER_ONLY:-}" = "True" ] && echo SIM || echo TRAIN)
# wandb credentials, when logging online. ~/.netrc is exposed READ-ONLY at the fake
# HOME's .netrc, so wandb finds it exactly where it expects. Deliberately a bind and not a
# copy, and deliberately not an env var: $RUN_ROOT is on netscratch and group-readable
# (drwxr-sr-x, dam_lab), and WANDB_API_KEY on a command line shows up in `ps`. This way the
# key never leaves $HOME. The bind target must exist first, hence the touch.
NETRC_BIND=()
if [ "${WANDB_MODE:-online}" = "online" ] && [ -s "${HOME_REAL:-$HOME}/.netrc" ]; then
    : > "${FAKE_HOME}/.netrc" 2>/dev/null || true
    NETRC_BIND=(--bind "${HOME_REAL:-$HOME}/.netrc:${FAKE_HOME}/.netrc:ro")
fi

echo "[node_launch] node=$(hostname -s) role=${ROLE_LABEL} gpus=${CUDA_VISIBLE_DEVICES:-<all allocated>}"
echo "[node_launch] sandbox=${SANDBOX}"
echo "[node_launch] node_tmp=${NODE_TMP}  fake_home=${FAKE_HOME}"

exec singularity exec --nv \
  --pwd "${VERL_REPO}" \
  --bind "${VERL_REPO}" \
  --bind /n/netscratch \
  --bind /n/lab_storage \
  --bind "${NODE_TMP}" \
  --home "${FAKE_HOME}" \
  "${NETRC_BIND[@]}" \
  --env-file "${ENV_FILE}" \
  --env SIM_SERVER_ONLY="${SIM_SERVER_ONLY:-}" \
  --env SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
  --env PYTHONPATH="${VERL_REPO}:${PYDEPS_DIR}" \
  --env RAY_TMPDIR="${NODE_TMP}/ray" \
  --env TRITON_CACHE_DIR="${NODE_TMP}/triton" \
  --env TORCHINDUCTOR_CACHE_DIR="${NODE_TMP}/inductor" \
  --env XDG_CACHE_HOME="${NODE_TMP}/xdg" \
  --env OUTLINES_CACHE_DIR="${NODE_TMP}/outlines" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "${SANDBOX}" bash "${ENTRYPOINT}"

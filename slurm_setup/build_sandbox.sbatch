#!/bin/bash
# ============================================================================
# Build the verl SGLang container as a Singularity SANDBOX on netscratch.
#
# Run this once, and again whenever the sandbox has decayed (netscratch reclaims by
# per-FILE atime, so a sandbox can "exist" while /usr/bin/python has been swept out from
# under it -- that is how we lost one twice). Measured cost: ~45 min. Verify with
# gpu_smoke.sbatch afterwards.
#
# WHY A SANDBOX, AND WHY HERE -- these are two separate decisions, both measured:
#
#   FORMAT: sandbox (an extracted rootfs directory, ~500k files) vs .sif (one compressed
#   squashfs file). `singularity build` exposes no compressor or level flag, so a .sif's
#   gzip cost is not tunable; the sandbox skips compression entirely. Sandbox-on-netscratch
#   is MEASURED at ~45 min. A .sif build here is UNMEASURED, so it stays unchosen rather
#   than assumed better. A .sif would be more durable (atomic, cannot half-rot) -- worth
#   revisiting, but only with a number.
#
#   LOCATION: netscratch, not lab_storage. A sandbox build on lab_storage NFS ran at 2.8%
#   CPU (2:40 of CPU in 95:18 of wall) and never finished -- job 44400009, cancelled.
#   It is 500k small file creates; Isilon NFS will not do that. holylabs is not an option
#   either: /n/holylabs/LABS/dam_lab/Lab/sqin is ACL-locked to dam_lab_admin (job 44399496
#   died on a mkdir there in 2 seconds).
#
# Runs on a CPU partition on purpose: the build needs ZERO GPU, and the first version of
# this held an H100 idle for 45 minutes of untarring.
# ============================================================================

#SBATCH -J verl_build_sandbox
#SBATCH -p seas_compute
#SBATCH --account=dam_lab
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /n/netscratch/barak_lab/Lab/sqin/slurm_logs/verl_build_sandbox_%j.out
#SBATCH -e /n/netscratch/barak_lab/Lab/sqin/slurm_logs/verl_build_sandbox_%j.err

set -uo pipefail

SLURM_SETUP_DIR="${SLURM_SETUP_DIR:-/n/home05/sqin/long-horizon-RL/verl/slurm_setup}"
source "${SLURM_SETUP_DIR}/paths.sh"

CACHE="${SINGULARITY_CACHEDIR:-${SCRATCH_ROOT}/docker_images/sing_cache}"
mkdir -p "$(dirname "${SANDBOX}")" "${CACHE}" "${SLURM_LOG_DIR}"

echo "=========================================================="
echo " sandbox : ${SANDBOX}"
echo " image   : ${SANDBOX_IMAGE}"
echo " cache   : ${CACHE}   ($(du -sh "${CACHE}" 2>/dev/null | cut -f1 || echo empty))"
echo " node    : $(hostname -s)   job ${SLURM_JOB_ID}"
echo "=========================================================="

if [ "${FORCE_REBUILD:-False}" != "True" ] \
   && [ -x "${SANDBOX}/usr/bin/python" ] && [ -x "${SANDBOX}/bin/bash" ]; then
    echo "complete sandbox already present -- nothing to do (FORCE_REBUILD=True to rebuild)"
    exit 0
fi

export SINGULARITY_DISABLE_CACHE=false
export SINGULARITY_CACHEDIR="${CACHE}"   # holds the pulled layers -> a rebuild re-extracts
                                         # from disk instead of re-downloading 16G
export SINGULARITY_TMPDIR=/scratch       # node-local; unpacking 37G through NFS tmp is the
                                         # same mistake as building the sandbox on NFS

echo "building (expect ~45 min; progress is invisible because singularity does not report"
echo "extraction -- watch bytes with: cat /proc/\$(pgrep -u \$USER -f 'singularity build')/io)"
time singularity build --force --sandbox "${SANDBOX}" "${SANDBOX_IMAGE}"
_rc=$?

if [ "${_rc}" -ne 0 ]; then
    echo "❌ build failed (rc=${_rc})"; exit "${_rc}"
fi
if [ ! -x "${SANDBOX}/usr/bin/python" ]; then
    echo "❌ build reported success but ${SANDBOX}/usr/bin/python is missing"; exit 1
fi

echo
echo "✅ sandbox built at ${SANDBOX}"
echo "   next: sbatch ${SLURM_SETUP_DIR}/gpu_smoke.sbatch    (verifies GPU + sglang + agent loops)"

#!/bin/bash
# Unattended CPU-only validation of the CodeContests multi-turn agent loop inside
# the verl SGLang container. Robust to interactive-node loss (this is a batch job).
#
# It (1) builds a Singularity sandbox from the *cached* image (no re-download), then
# (2) runs the verl-dependent + pure-code CPU tests with the validated exec recipe
# (container env kept, PYTHONPATH via --env, NO --cleanenv, NO `bash -lc`).
#
# Submit:  sbatch codecontest/slurm_cpu_test.sbatch
# Logs:    /n/netscratch/dam_lab/Lab/sqin/docker_images/cc_cpu_test_<jobid>.{out,err}
# Adjust --partition / --account below if needed.

#SBATCH -J cc_cpu_test
#SBATCH -p seas_compute
#SBATCH --account=dam_lab
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /n/netscratch/dam_lab/Lab/sqin/docker_images/cc_cpu_test_%j.out
#SBATCH -e /n/netscratch/dam_lab/Lab/sqin/docker_images/cc_cpu_test_%j.err

set -uo pipefail
set -x

REPO=/n/home05/sqin/long-horizon-RL/verl
DIR=/n/netscratch/dam_lab/Lab/sqin/docker_images
SANDBOX=$DIR/verl-sgl0512-dev2

export SINGULARITY_DISABLE_CACHE=false
export SINGULARITY_CACHEDIR=$DIR/sing_cache       # already holds the 16G image -> no download
export SINGULARITY_TMPDIR=/scratch                # node-local: fast build scratch

# ---------------------------------------------------------------------------
# 1) Build the sandbox from cache, unless a complete one already exists.
# ---------------------------------------------------------------------------
if [ -x "$SANDBOX/usr/bin/python" ]; then
    echo "[build] sandbox already present at $SANDBOX -- skipping build"
else
    echo "[build] building sandbox from cached image (download-free) ..."
    singularity build --force --sandbox "$SANDBOX" docker://verlai/verl:sgl0512.dev2
    echo "[build] exit=$?"
fi

# Validated exec recipe: keep the image's env (sglang lives there), inject PYTHONPATH,
# bind the repo. Do NOT use --cleanenv (drops sglang env) or `bash -lc` (leaks host conda).
run() { singularity exec --bind "$REPO" --env PYTHONPATH="$REPO" "$SANDBOX" "$@"; }

echo "================ [A] versions + verl import + agent registration ================"
run python -c "import sys,sglang,torch,ray,verl; \
from codecontest.code_refine_agent import CodeRefineAgentLoop; \
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry; \
print('py',sys.version.split()[0],'| sglang',sglang.__version__,'| torch',torch.__version__,'| ray',ray.__version__); \
print('AGENT_REGISTERED:', 'code_refine_agent' in _agent_loop_registry)"

echo "================ [B] agent-loop control-flow / masking / reward ================"
run python "$REPO/codecontest/tests/test_code_refine_agent.py"

echo "================ [C] pure-code tests (local_exec, env) ================"
run python "$REPO/codecontest/tests/test_local_exec.py"
run python "$REPO/codecontest/tests/test_env.py"

echo "================ DONE ================"

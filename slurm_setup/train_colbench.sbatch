#!/bin/bash
# ============================================================================
# ColBench multi-turn GRPO -- the SLURM allocation layer.
#
# This replaces xcloud_setup/launch.py's xm job GRAPH (and nothing else): it asks Slurm
# for the nodes, decides which node plays which role, and runs one srun step per role.
# All the run configuration arrives as an --env-file written by launch_slurm.sh.
#
# THE NODE SPLIT, and why it is the default here:
#   node 0 = TRAIN  (all 4 GPUs -> the solver policy)
#   node 1 = SIM    (all 4 GPUs -> the frozen user-simulator, SGLang tp x dp)
#
# On xcloud a single 8-GPU node co-hosted both: 1-2 GPUs for the sim, 6 for training. A
# kempner_h100/h200 node has FOUR GPUs, so co-hosting would cost half the node's training
# capacity (SIM_TP=1 -> 2 train GPUs + 1 idle). Splitting across two nodes gives 4 train
# GPUs AND a 4x-data-parallel sim, so it is better on both sides at the same total cost.
#
# ONE JOB, NOT TWO. On XManager the sim was a second job that queued independently, which
# is why the training side polled a GCS sentinel for up to two hours and why a stale
# sentinel from a prior run was a live hazard. Here both nodes come out of one allocation:
# they start together, the sentinel is a readiness handshake with a 30-minute budget, and
# its path carries $SLURM_JOB_ID so staleness is structurally impossible.
#
# Submit through launch_slurm.sh (it validates flags, writes the env file, and picks the
# node count). Direct sbatch works too if ENV_FILE / RUN_ROOT are already exported.
# ============================================================================

#SBATCH -J colbench
#SBATCH -p kempner_h100
#SBATCH --account=kempner_dam_lab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -c 88
#SBATCH --mem=1200G
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /n/netscratch/barak_lab/Lab/sqin/slurm_logs/%x_%j.out
#SBATCH -e /n/netscratch/barak_lab/Lab/sqin/slurm_logs/%x_%j.err

# -c 88 (of 96) and --mem=1200G (of ~1.47T): taking all 4 GPUs already means no other GPU
# job can share the node, and GPU dominates billing ~2400:1 over CPU. The exec sidecar runs
# up to CODECONTEST_EXEC_CONCURRENCY=180 subprocesses and host-RAM OOM there has bitten this
# stack before, so there is no reason to be stingy on a node we already hold.
# NOT --mem=0: kempner_h100 REFUSES it outright ("You are not permited to request all the
# memory on the node for the kempner_h100 partition, please specify your request"), so the
# request has to be an explicit number. launch_slurm.sh --mem / --cpus_per_task lower them
# if a job pends behind a stray CPU-only job.
# --requeue + --open-mode=append: if the job is preempted or hits the 2-day wall, Slurm can
# put it back and the log continues. Resume itself is verl's resume_mode=auto reading
# $CKPT_LOCAL_DIR, which never left the shared filesystem.

set -uo pipefail

SLURM_SETUP_DIR="${SLURM_SETUP_DIR:-/n/home05/sqin/long-horizon-RL/verl/slurm_setup}"
# shellcheck source=paths.sh
source "${SLURM_SETUP_DIR}/paths.sh"

: "${ENV_FILE:?ENV_FILE not set -- submit via launch_slurm.sh}"
export ENTRYPOINT="${ENTRYPOINT:-slurm_setup/entrypoint_colbench_slurm.sh}"

if [ ! -f "${ENV_FILE}" ]; then
    echo "❌ ENV_FILE ${ENV_FILE} does not exist."; exit 1
fi

echo "=========================================================="
echo " job ${SLURM_JOB_ID}  (${SLURM_JOB_NAME})"
echo " partition ${SLURM_JOB_PARTITION}  nodes ${SLURM_JOB_NUM_NODES}  nodelist ${SLURM_JOB_NODELIST}"
echo " restart count: ${SLURM_RESTART_COUNT:-0}"
echo " env file: ${ENV_FILE}"
echo "=========================================================="
sed 's/^/   cfg | /' "${ENV_FILE}"
echo "=========================================================="

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")

# Role assignment. Three shapes, all decided by launch_slurm.sh's node count plus SIM_SMOKE:
#   2 nodes            -> node 0 trains, node 1 serves the frozen sim  (the default)
#   1 node             -> node 0 trains; the sim is either LIVE (no server) or co-hosted
#   1 node + SIM_SMOKE -> node 0 serves the sim and self-benchmarks; NO training at all
# SIM_SMOKE is read from the submitting environment (launch_slurm.sh exports it), not from
# the env file -- the env file is only opened inside the container.
if [ "${SIM_SMOKE:-False}" = "True" ]; then
    TRAIN_NODE=""
    SIM_NODE="${NODES[0]}"
    echo "roles: SIM-ONLY sizing run on ${SIM_NODE} (no training step)"
else
    TRAIN_NODE="${NODES[0]}"
    SIM_NODE="${NODES[1]:-}"
    if [ -z "${SIM_NODE}" ]; then
        echo "roles: TRAIN=${TRAIN_NODE}, no sim node (expect SIM_LIVE=True or a co-hosted sim)"
    else
        echo "roles: TRAIN=${TRAIN_NODE}  SIM=${SIM_NODE}"
    fi
fi

# ── Preflight: does each node actually see its GPUs in an srun STEP? ────────────────────
# A step inherits the job's --gres by default, but "by default" is exactly the kind of
# assumption that turns into a 40-minute failure at the first weight sync. This costs
# seconds and converts that into an immediate, obvious error.
for _n in "${NODES[@]}"; do
    _seen=$(srun --nodes=1 --ntasks=1 --nodelist="${_n}" --overlap \
                 bash -c 'nvidia-smi -L 2>/dev/null | wc -l' 2>/dev/null || echo 0)
    echo "preflight: ${_n} sees ${_seen} GPU(s)"
    if [ "${_seen}" -lt 1 ]; then
        echo "❌ ${_n} sees no GPUs inside an srun step. Check --gres/--gpus-per-node on this partition."
        exit 1
    fi
done

# ── Immutable per-job CODE SNAPSHOT ───────────────────────────────────────────────────
# The entrypoint and the train script are read from the repo, not baked into the image the
# way the Dockerfile used to COPY them in. That is convenient (an edit takes effect on the
# next sbatch, no 45-minute rebuild) but it has a sharp edge: bash reads a script
# INCREMENTALLY, so editing a file while a job is running moves the bytes under the running
# interpreter and it resumes at a stale offset -- mid-word. Observed exactly once, in job
# 44436175: a one-line cosmetic edit five minutes into the run produced
# `line 244: ify_environment: command not found` (i.e. "verify_environment" minus "ver")
# and took the job down. Python is safer (imports read whole files, once) but not immune,
# and hydra/verl read config YAML at startup.
#
# So each job copies the code (23 MB, ~1500 files, a couple of seconds) to
# $RUN_ROOT/code.$SLURM_JOB_ID and runs from THERE. That restores the immutability the
# Docker image gave us, keeps the no-rebuild edit loop, and leaves a record of exactly
# which code a run used sitting next to its checkpoints. A --chain job snapshots
# separately, so a fix lands on the next link -- which is what you want.
# Set CODE_SNAPSHOT=False to run straight from the checkout (then do not touch the repo).
SNAPSHOT="${RUN_ROOT:-${SCRATCH_ROOT}/verl_runs/_scratch}/code.${SLURM_JOB_ID}"
if [ "${CODE_SNAPSHOT:-True}" = "True" ]; then
    echo "code snapshot: ${VERL_REPO} -> ${SNAPSHOT}"
    mkdir -p "${SNAPSHOT}"
    # NOTE the LEADING SLASHES on /data and /checkpoints. An rsync exclude without one
    # matches a path component at ANY depth, so plain 'data' also ate
    # verl/trainer/config/data/ -- and hydra then died with
    # "In 'ppo_trainer': Could not find 'data/legacy_data'" (job 44437409). A leading
    # slash anchors the pattern to the transfer root, which is what we actually meant:
    # skip the repo-root data/ and checkpoints/ dirs, nothing else.
    rsync -a --delete \
          --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
          --exclude='/data' --exclude='/checkpoints' \
          --exclude='/colbench/slurm_out' --exclude='/colbench/runs' \
          "${VERL_REPO}/" "${SNAPSHOT}/" \
      || { echo "❌ code snapshot failed"; exit 1; }
    export VERL_REPO="${SNAPSHOT}"
    SLURM_SETUP_DIR="${SNAPSHOT}/slurm_setup"
    echo "   running from the snapshot; the checkout is now safe to edit mid-run"
else
    echo "⚠️  CODE_SNAPSHOT=False -- running straight from ${VERL_REPO}. Do NOT edit it while this job runs."
fi

SIM_LOG="${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.sim.log"
mkdir -p "${SLURM_LOG_DIR}"

# Tear down BOTH steps whichever way we leave. Without this, a training crash leaves the
# sim step holding 4 H100s for the rest of the wall clock.
SIM_PID=""
TRAIN_PID=""
cleanup() {
    [ -n "${SIM_PID}" ]   && kill "${SIM_PID}"   2>/dev/null
    [ -n "${TRAIN_PID}" ] && kill "${TRAIN_PID}" 2>/dev/null
    wait 2>/dev/null
    return 0
}
trap cleanup EXIT

# ── Sim node ───────────────────────────────────────────────────────────────────────────
# Its own log file: the sim server is chatty (shard loading, cudagraph capture, per-request
# lines) and interleaving it with the trainer's stdout makes both unreadable.
if [ -n "${SIM_NODE}" ]; then
    echo "starting sim step on ${SIM_NODE} -> ${SIM_LOG}"
    srun --nodes=1 --ntasks=1 --nodelist="${SIM_NODE}" \
         --export=ALL,SIM_SERVER_ONLY=True \
         --output="${SIM_LOG}" --error="${SIM_LOG}" \
         bash "${SLURM_SETUP_DIR}/node_launch.sh" &
    SIM_PID=$!
fi

# ── Train node (stdout goes to the job log) ────────────────────────────────────────────
if [ -n "${TRAIN_NODE}" ]; then
    echo "starting train step on ${TRAIN_NODE}"
    # Plain --export=ALL: SIM_SERVER_ONLY is simply absent from the batch environment, so
    # the train step inherits no role flag and node_launch.sh passes an empty one through.
    # (`--export=ALL,SIM_SERVER_ONLY=` would be the symmetric spelling, but an empty value
    # in --export is not portably parsed, and absence says the same thing.)
    srun --nodes=1 --ntasks=1 --nodelist="${TRAIN_NODE}" \
         --export=ALL \
         bash "${SLURM_SETUP_DIR}/node_launch.sh" &
    TRAIN_PID=$!
fi

# ── Wait, coupling the two steps' fates ────────────────────────────────────────────────
# Poll rather than `wait -n` so we know WHICH side finished (bash's `wait -n` reports a
# status but not a pid until 5.1's -p). 20s granularity is irrelevant against a run of
# hours and costs nothing.
RC=0
if [ -z "${TRAIN_PID}" ]; then
    # SIM-ONLY sizing run: the sim step IS the job. It prints its throughput table and then
    # keeps serving, so this job ends when the wall clock runs out or you scancel it.
    echo "sim-only run: waiting on the sim step (scancel ${SLURM_JOB_ID} when done)"
    wait "${SIM_PID}"; RC=$?
elif [ -n "${SIM_PID}" ]; then
    while kill -0 "${TRAIN_PID}" 2>/dev/null && kill -0 "${SIM_PID}" 2>/dev/null; do
        sleep 20
    done
    if kill -0 "${TRAIN_PID}" 2>/dev/null; then
        # The SIM step died first. Training is either still polling the sentinel or already
        # running against a dead endpoint, and neither recovers -- the solver reads
        # OPENAI_BASE_URL once at startup and a restarted sim would come back elsewhere.
        # Kill training so the job exits and a chained/requeued job can resume cleanly from
        # the last checkpoint (this is the same intent as the in-entrypoint sim watchdog,
        # for the window before training is up).
        echo "❌ sim step exited before training -> tearing down the train step for a clean resume"
        kill -TERM "${TRAIN_PID}" 2>/dev/null
        wait "${TRAIN_PID}" 2>/dev/null
        RC=1
    else
        wait "${TRAIN_PID}"; RC=$?
        kill "${SIM_PID}" 2>/dev/null
    fi
else
    wait "${TRAIN_PID}"; RC=$?
fi

echo "=========================================================="
echo " exit code: ${RC}"
echo " checkpoints: ${RUN_ROOT:-<see entrypoint log>}/checkpoints"
echo " wandb:       ${WANDB_ENTITY:-harvardml}/${PROJECT_NAME:-colbench_mt} run ${EXPERIMENT_NAME:-<see entrypoint log>}"
[ -n "${SIM_NODE}" ] && echo " sim log:     ${SIM_LOG}"
echo "=========================================================="
exit "${RC}"

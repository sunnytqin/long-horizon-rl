#!/usr/bin/env bash
# Dedicated spec eval / serving entrypoint. No RL or checkpoint mutation.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/entrypoint_common_slurm.sh"
: "${RUN_ROOT:?}" "${EVAL_MODE:?}" "${SIM_WEIGHTS:?}"
# Output-name tags normally arrive from launch_eval_slurm.sh. Derive them here as well
# so a MANUAL entrypoint run still names its output correctly instead of dying under
# `set -u` on an unbound SIM_TAG.
SIM_TAG="${SIM_TAG:-$(model_shorthand "${SIM_MODEL:-unknown-sim}")}"
VAL_TAG="${VAL_TAG:-$(basename "${VAL_FILE:-unknown-val}" .parquet | tr '.' '_')}"
# The OpenAI-API alias this node serves under. `colbench-sim` for the eval path (the
# solver reads it as MULTITURN_MODEL_NAME, and the eval JSON records that alias). In
# judge mode the launcher overrides it with the model's real shorthand, because the
# alias is written into every judged row as `judge_model` and a generic one would make
# two different judges indistinguishable on disk.
SIM_SERVED_NAME="${SIM_SERVED_NAME:-colbench-sim}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export SIM_PORT="${SIM_PORT:-$((30000 + SLURM_JOB_ID % 20000))}"
export SIM_SENTINEL="${RUN_ROOT}/sim_url.${SLURM_JOB_ID}.txt"
SIM_PID=""
EXEC_SERVER_PID=""
cleanup() {
  [ -z "${SIM_PID}" ] || kill "${SIM_PID}" 2>/dev/null || true
  [ -z "${EXEC_SERVER_PID}" ] || stop_exec_sidecar
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
verify_environment
if [ "${SIM_SERVER_ONLY:-}" = True ]; then
  # A deterministic job-derived port can collide with an orphaned server or another
  # user's process on the assigned node. Resolve the first free port before paying
  # the ~15-minute 235B startup cost. The solver discovers the selected port through
  # SIM_SENTINEL, so it does not need to predict it.
  SIM_PORT="$(python3 - "${SIM_PORT}" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for offset in range(1000):
  port = 30000 + ((start - 30000 + offset) % 20000)
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
      sock.bind(("0.0.0.0", port))
    except OSError:
      continue
  print(port)
  break
else:
  raise SystemExit("no free simulator port found in 1000 candidates")
PY
)"
  export SIM_PORT
  echo "Selected simulator port: ${SIM_PORT}"
  sentinel_clear "${SIM_SENTINEL}"
  python3 "${_HERE}/serving_probe.py" weights "${SIM_WEIGHTS}"
  nvidia-smi > "${RUN_ROOT}/sim_gpu_before.txt"
  args=(python3 -m sglang.launch_server --model-path "${SIM_WEIGHTS}"
    --served-model-name "${SIM_SERVED_NAME}" --host 0.0.0.0 --port "${SIM_PORT}"
    --tp "${SIM_TP}" --dp-size "${SIM_DP}" --mem-fraction-static "${SIM_MEM_FRACTION}"
    --context-length "${SIM_CONTEXT_LENGTH}" --max-running-requests "${SIM_MAX_RUNNING_REQUESTS}"
    --cuda-graph-max-bs "${SIM_CUDA_GRAPH_MAX_BS}" --chunked-prefill-size "${SIM_CHUNKED_PREFILL_SIZE}")
  printf '%q ' "${args[@]}" > "${RUN_ROOT}/sim_server.cmd"
  printf '\n' >> "${RUN_ROOT}/sim_server.cmd"
  "${args[@]}" &
  SIM_PID=$!
  deadline=$((SECONDS + SIM_STARTUP_TIMEOUT))
  ready=False
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    kill -0 "${SIM_PID}" 2>/dev/null || { echo 'sim server exited during startup'; exit 1; }
    if curl --max-time 5 -fsS "http://127.0.0.1:${SIM_PORT}/health" >/dev/null 2>&1; then
      ready=True; break
    fi
    sleep 5
  done
  [ "${ready}" = True ] || { echo 'sim server startup timed out'; exit 1; }
  # Actual inference (including JSON judge-shaped output), not just /health.
  python3 "${_HERE}/serving_probe.py" chat \
    --base_url "http://127.0.0.1:${SIM_PORT}/v1" --model "${SIM_SERVED_NAME}" \
    --out "${RUN_ROOT}/serving_probe.json"
  nvidia-smi > "${RUN_ROOT}/sim_gpu_after.txt"
  if [ "${EVAL_MODE}" = smoke ]; then
    echo 'Serving smoke passed; terminating server and releasing allocation.'
    exit 0
  fi
  if [ "${EVAL_MODE}" = judge ]; then
    # Judging is pure HTTP, so it runs HERE, beside the server, rather than on a
    # second node: 127.0.0.1 removes a cross-node reachability failure mode and
    # halves the GPUs the comparison costs.
    : "${JUDGE_PREFIXES:?}" "${JUDGE_CANDIDATES:?}" "${JUDGE_OUT:?}"
    mkdir -p "$(dirname "${JUDGE_OUT}")"
    limit=()
    [ "${JUDGE_MAX_PREFIXES:-0}" -eq 0 ] || limit=(--max_prefixes "${JUDGE_MAX_PREFIXES}")
    # No --price_in/--price_out/--max_cost: those cap DOLLARS on a metered API. The
    # cap on a local judge is the wall clock, which Slurm already enforces.
    # --judge_model is the served alias, so it lands in every row as the judge's
    # identity and the resume guard has something meaningful to compare.
    python3 -m colbench.simtrain.judge_candidates \
      --prefixes "${JUDGE_PREFIXES}" --candidates "${JUDGE_CANDIDATES}" \
      --out "${JUDGE_OUT}" "${limit[@]}" \
      --judge_model "${SIM_SERVED_NAME}" \
      --judge_base_url "http://127.0.0.1:${SIM_PORT}/v1" \
      --judge_vendor vllm --judge_api_key EMPTY \
      --arm "${JUDGE_ARM:-gt}" \
      --temperature "${JUDGE_TEMPERATURE:-0.0}" \
      --concurrency "${JUDGE_CONCURRENCY:-16}" \
      --timeout "${JUDGE_TIMEOUT:-900}"
    # The readable artifact -- the actual deliverable of a pilot pass.
    python3 -m colbench.simtrain.dump_judged \
      --prefixes "${JUDGE_PREFIXES}" --judged "${JUDGE_OUT}" \
      --out "${JUDGE_OUT%.jsonl}.dump.txt"
    echo "[judge] judged -> ${JUDGE_OUT}"
    echo "[judge] dump   -> ${JUDGE_OUT%.jsonl}.dump.txt"
    exit 0
  fi
  sentinel_publish "http://$(hostname -s):${SIM_PORT}/v1" "${SIM_SENTINEL}"
  echo "Sim endpoint: $(sentinel_read "${SIM_SENTINEL}")"
  wait "${SIM_PID}"
else
  [ "${EVAL_MODE}" = eval ] || { echo 'solver role requires EVAL_MODE=eval'; exit 1; }
  # Only the single-checkpoint path has a MODEL_PATH to verify up front. In the
  # --global_step path it is deliberately EMPTY (each step resolves its own merged
  # directory in eval_one_step, which probes it there), so probing it here would
  # read a relative 'config.json' and abort the whole job.
  if [ -z "${EVAL_STEPS:-}" ]; then
    python3 "${_HERE}/serving_probe.py" weights "${MODEL_PATH}"
  fi
  deadline=$((SECONDS + SIM_STARTUP_TIMEOUT + 600))
  export OPENAI_BASE_URL=""
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    # Missing sentinel is expected while a 235B server loads. Do not let set -e
    # turn the first unsuccessful readiness poll into an eval failure.
    url="$(sentinel_read "${SIM_SENTINEL}" || true)"
    if [ -n "${url}" ] && curl --max-time 5 -fsS "${url%/v1}/health" >/dev/null 2>&1; then
      OPENAI_BASE_URL="${url}"; break
    fi
    sleep 5
  done
  [ -n "${OPENAI_BASE_URL}" ] || { echo 'remote sim readiness timed out'; exit 1; }
  export MULTITURN_MODEL_NAME="${SIM_SERVED_NAME}"
  # Exercise network reachability and inference from the solver node too.
  python3 "${_HERE}/serving_probe.py" chat --base_url "${OPENAI_BASE_URL}" \
    --model "${MULTITURN_MODEL_NAME}" --out "${RUN_ROOT}/remote_probe.json"
  # Decode the comma-separated transport form back to the space-separated list the
  # runner expects (launch_eval_slurm.sh encodes it; --env-file cannot carry spaces).
  [ -z "${TEMPERATURES:-}" ] || export TEMPERATURES="${TEMPERATURES//,/ }"
  export CODECONTEST_EXEC_PORT=$((10000 + SLURM_JOB_ID % 10000))
  export CODECONTEST_ALLOW_INPROCESS=0
  start_exec_sidecar
  wait_exec_healthy
  sha256sum "${VAL_FILE}" > "${RUN_ROOT}/val_file.sha256"

  # Merge-then-eval for ONE checkpoint step. Ported from
  # xcloud_setup/entrypoint_eval_colbench.sh::eval_one_step, minus the GCS
  # download/upload (the checkpoints are already on netscratch).
  eval_one_step() {
    local step="$1" merged
    if [ "${step}" = base ]; then
      merged="${BASE_MODEL_PATH}"
      echo "[eval] step=base -> ${merged} (no FSDP merge)"
    else
      merged="${MERGED_ROOT}/${TRAIN_EXP}/global_step_${step}"
      if python3 "${_HERE}/serving_probe.py" weights "${merged}" >/dev/null 2>&1; then
        echo "[eval] step=${step}: reusing cached merge at ${merged}"
      else
        echo "[eval] step=${step}: merging FSDP shards -> HF at ${merged}"
        mkdir -p "$(dirname "${merged}")"
        rm -rf "${merged}.partial"
        python3 -m verl.model_merger merge --backend fsdp \
          --local_dir "${CKPT_ROOT}/global_step_${step}/actor" \
          --target_dir "${merged}.partial"
        # Publish only a merge that VERIFIES. Renaming an unchecked directory into
        # place would let a half-written merge be read as a cache hit by the next
        # job, which then evaluates truncated weights and reports a real-looking
        # number. The rename is the commit point.
        python3 "${_HERE}/serving_probe.py" weights "${merged}.partial"
        mv "${merged}.partial" "${merged}"
      fi
    fi
    # run_validate_colbench_spec.sh defaults MODEL_PATH to Qwen2.5-14B-Instruct when
    # it is empty, so an unresolved path would SILENTLY evaluate a different model and
    # report real-looking numbers. Fail loudly instead -- a crash is recoverable, a
    # wrong number in a results file is not.
    [ -n "${merged}" ] && [ -d "${merged}" ] \
      || { echo "step ${step}: unresolved solver dir '${merged}'"; exit 1; }
    # Sim identity + val set are in the stem: the same checkpoint evaluated under a
    # different simulator or test set must not overwrite this file.
    MODEL_PATH="${merged}" \
    OUT="${RUN_ROOT}/step${step}_sim-${SIM_TAG}_${VAL_TAG}.json" \
      bash colbench/run_validate_colbench_spec.sh
    echo "[eval] step=${step} done"
  }

  if [ -n "${EVAL_STEPS:-}" ]; then
    echo "[eval] steps: ${EVAL_STEPS}"
    # Comma-separated (see launch_eval_slurm.sh): --env-file does no unescaping, so a
    # space-separated list would arrive with literal backslashes still in it.
    for _step in ${EVAL_STEPS//,/ }; do eval_one_step "${_step}"; done
  else
    OUT="${RUN_ROOT}/eval_sim-${SIM_TAG}_${VAL_TAG}.json" \
      bash colbench/run_validate_colbench_spec.sh
  fi
fi

#!/usr/bin/env bash
# Offline multi-turn VALIDATION / inspection for a trained ColBench solver.
#
# Runs the same solver<->frozen-simulator conversation as training (see
# colbench/colbench_agent.py) on the test set, with FREELY TUNABLE inference hparams, and
# dumps a random sample of trajectories in human-readable "conversation" form to JSON so you
# can manually examine what the checkpoint is doing.
#
# This is NOT the RL trainer -- it is a standalone SGLang offline batch job (same inference
# backend as training, so it runs in the same SGLang container -- no vLLM) that reuses the
# training env/templates/exec-grading. Point --model at a MERGED HF checkpoint (see the
# reference-codecontest-gcs-checkpoint-sync memory for how to merge a VERL FSDP checkpoint to
# HF), not a raw FSDP shard dir.
#
# REQUIRES a reachable frozen user-simulator server: the harness reaches it over
#   OPENAI_BASE_URL (e.g. http://127.0.0.1:30000/v1) + MULTITURN_MODEL_NAME.
# entrypoint_eval_colbench.sh brings this up (same as training); for a manual run start a
# sim SGLang OpenAI server yourself and export those two vars first.
#
# Env overrides: MODEL_PATH, VAL_FILE, OUT, MAX_PROBLEMS, N_SAMPLES, MAX_SAVED_CONVOS,
#   TEMPERATURES, TOP_P, TOP_K, SEED, MAX_ASSISTANT_TURNS, MAX_NEW_TOKENS_PER_TURN,
#   MAX_RESPONSE_LENGTH, MAX_PROMPT_LENGTH, REWARD_TIME_LIMIT, ROLLOUT_TP, GPU_MEM_UTIL,
#   CODECONTEST_EXEC_URL, CODECONTEST_ALLOW_INPROCESS, CODECONTEST_EXEC_CONCURRENCY,
#   OPENAI_BASE_URL, MULTITURN_MODEL_NAME,
#   SIM_REJECT_MAX_TRIES (0=off; 32 to enable), SIM_REJECT_NGRAM_N, SIM_REJECT_MIN_OPS.

set -xeuo pipefail

# --- code-exec backend (identical semantics to training) ---
# Preferred: a running sandbox sidecar. Otherwise fall back to the in-process executor (fine
# for a single-container eval; the eval job is read-only).
export CODECONTEST_EXEC_URL=${CODECONTEST_EXEC_URL:-}
if [ -z "${CODECONTEST_EXEC_URL}" ]; then
  export CODECONTEST_ALLOW_INPROCESS=${CODECONTEST_ALLOW_INPROCESS:-1}
fi
export CODECONTEST_EXEC_MEM_GB=${CODECONTEST_EXEC_MEM_GB:-2}
export CODECONTEST_EXEC_CONCURRENCY=${CODECONTEST_EXEC_CONCURRENCY:-32}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}
VAL_FILE=${VAL_FILE:-$HOME/data/colbench/test.parquet}
OUT=${OUT:-runs/validate_$(date +%m%d_%H%M).json}

# Eval scope. Default: the FULL test set (MAX_PROBLEMS unset -> all rows), but only
# MAX_SAVED_CONVOS conversations are written (metrics still cover every trajectory).
MAX_PROBLEMS=${MAX_PROBLEMS:-}
N_SAMPLES=${N_SAMPLES:-1}
MAX_SAVED_CONVOS=${MAX_SAVED_CONVOS:-1000}

# User-simulator rejection sampling (prevents the frozen sim from leaking code). Off by
# default (0 tries); set SIM_REJECT_MAX_TRIES=32 to enable. On exhaustion the conversation is
# a "simulation failure" (a third outcome, excluded from the pass-rate denominator). Each saved
# conversation records per-turn rejection stats (sim_reject_events). Detectors: (A) def-regex +
# (B) ```python fence always; (D) operator-gated n-gram is OFF by default (SIM_REJECT_NGRAM_N=0,
# a future consideration) -- set SIM_REJECT_NGRAM_N=10 to enable it.
SIM_REJECT_MAX_TRIES=${SIM_REJECT_MAX_TRIES:-0}
SIM_REJECT_NGRAM_N=${SIM_REJECT_NGRAM_N:-0}
SIM_REJECT_MIN_OPS=${SIM_REJECT_MIN_OPS:-2}

# Inference hparams. TEMPERATURES (space-separated, e.g. "0.0 0.6") sweeps several temps in
# ONE run: the engine is loaded once and each temp writes its own tagged JSON. Defaults match
# run_colbench_grpo.sh so a checkpoint is evaluated with the budgets it trained under.
TEMPERATURES=${TEMPERATURES:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
SEED=${SEED:-0}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-10}
MAX_NEW_TOKENS_PER_TURN=${MAX_NEW_TOKENS_PER_TURN:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-14336}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
REWARD_TIME_LIMIT=${REWARD_TIME_LIMIT:-6}

# SGLang engine (solver). The frozen sim server runs on its own reserved GPUs.
ROLLOUT_TP=${ROLLOUT_TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}   # SGLang mem_fraction_static

# MAX_PROBLEMS is optional; pass the flag only for a positive integer (empty / 0 / non-numeric
# => all problems). Guards against MAX_PROBLEMS=0 slicing the val set to empty.
MAX_PROBLEMS_ARG=()
if [[ "${MAX_PROBLEMS}" =~ ^[0-9]+$ && "${MAX_PROBLEMS}" -gt 0 ]]; then
  MAX_PROBLEMS_ARG=(--max_problems "${MAX_PROBLEMS}")
fi

python3 colbench/validate_colbench.py \
    --model "${MODEL_PATH}" \
    --val_file "${VAL_FILE}" \
    --out "${OUT}" \
    "${MAX_PROBLEMS_ARG[@]}" \
    --n_samples ${N_SAMPLES} \
    --max_saved_convos ${MAX_SAVED_CONVOS} \
    --temperatures ${TEMPERATURES} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --seed ${SEED} \
    --max_assistant_turns ${MAX_ASSISTANT_TURNS} \
    --max_new_tokens_per_turn ${MAX_NEW_TOKENS_PER_TURN} \
    --max_response_length ${MAX_RESPONSE_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --reward_time_limit ${REWARD_TIME_LIMIT} \
    --sim_reject_max_tries ${SIM_REJECT_MAX_TRIES} \
    --sim_reject_ngram_n ${SIM_REJECT_NGRAM_N} \
    --sim_reject_min_ops ${SIM_REJECT_MIN_OPS} \
    --tensor_parallel_size ${ROLLOUT_TP} \
    --gpu_memory_utilization ${GPU_MEM_UTIL} \
    "$@"

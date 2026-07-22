#!/usr/bin/env bash
# Offline multi-turn VALIDATION / inspection for a trained ColBench solver -- SPEC PATH.
#
# Sibling of colbench/run_validate_colbench.sh (the GT runner). Same idea -- run the training
# conversation on the held-out test set with FREELY TUNABLE inference hparams and dump a random
# sample of trajectories to JSON -- but for the SPEC loop: the frozen user-simulator conditions
# on the NL spec (never the GT source), so termination is USER-DRIVEN ([TERMINATE]) and there is
# no code-leak surface (that's why the GT-only SIM_REJECT_* reject-sampling knobs are DROPPED
# here; the spec path instead rejects the sim WRITING code via SIM_MAX_TRIES).
#
# This is NOT the RL trainer -- it is a standalone SGLang offline batch job (same inference
# backend as training) that reuses the spec env/templates/exec-grading. Point --model at a MERGED
# HF checkpoint (see reference-codecontest-gcs-checkpoint-sync), not a raw FSDP shard dir. It
# drives colbench/validate_colbench_spec.py with --solver_backend sglang (the solver is an
# offline sgl.Engine).
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
#   MAX_CODE_PROPOSALS (spec code-proposal cap), SIM_MAX_TRIES (sim code-rejection tries),
#   SIM_MAX_TOKENS (per-turn sim token bound).

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
VAL_FILE=${VAL_FILE:-$HOME/data/colbench_spec/test_small.parquet}
OUT=${OUT:-runs/validate_spec_$(date +%m%d_%H%M).json}

# Eval scope. Default: the FULL test set (MAX_PROBLEMS unset -> all rows), but only
# MAX_SAVED_CONVOS conversations are written (metrics still cover every trajectory).
MAX_PROBLEMS=${MAX_PROBLEMS:-}
N_SAMPLES=${N_SAMPLES:-1}
MAX_SAVED_CONVOS=${MAX_SAVED_CONVOS:-1000}

# Spec-path guardrails (replace the GT path's SIM_REJECT_* leak knobs).
#   MAX_CODE_PROPOSALS: force-terminate after this many solver code proposals (default 2, matches
#     run_colbench_grpo_spec.sh).
#   SIM_MAX_TRIES: rejection-sampling tries when the sim WRITES code; on exhaustion the
#     conversation is aborted (terminated_by 'sim_code_reject') for inspection (default 8).
MAX_CODE_PROPOSALS=${MAX_CODE_PROPOSALS:-2}
export SIM_MAX_TRIES=${SIM_MAX_TRIES:-8}
# Generation-time token bound on each user (sim) turn (read by env_spec.openai_sim_backend too).
export SIM_MAX_TOKENS=${SIM_MAX_TOKENS:-256}

# Inference hparams. TEMPERATURES (space-separated, e.g. "0.0 0.6") sweeps several temps in
# ONE run: the engine is loaded once and each temp writes its own tagged JSON. Defaults match
# run_colbench_grpo_spec.sh so a checkpoint is evaluated with the budgets it trained under.
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

# --solver_backend sglang: the solver is an offline sgl.Engine (the merged checkpoint). The sim
# stays on the default 'local' backend, which reaches OPENAI_BASE_URL / MULTITURN_MODEL_NAME (the
# frozen SGLang server the entrypoint brought up -- there is NO vLLM in this container) -- so we
# pass NO --sim_backend here.
python3 colbench/validate_colbench_spec.py \
    --solver_backend sglang \
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
    --max_code_proposals ${MAX_CODE_PROPOSALS} \
    --max_new_tokens_per_turn ${MAX_NEW_TOKENS_PER_TURN} \
    --max_response_length ${MAX_RESPONSE_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --reward_time_limit ${REWARD_TIME_LIMIT} \
    --sim_max_tries ${SIM_MAX_TRIES} \
    --sim_max_tokens ${SIM_MAX_TOKENS} \
    --tensor_parallel_size ${ROLLOUT_TP} \
    --gpu_memory_utilization ${GPU_MEM_UTIL} \
    "$@"

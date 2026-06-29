#!/usr/bin/env bash
# Offline multi-turn VALIDATION / inspection for a trained CodeContests solver.
#
# Runs the same oracle code-refinement conversation as training (see
# codecontest/code_refine_agent.py) on the test set, with FREELY TUNABLE inference
# hparams, and dumps every trajectory in human-readable "conversation" form to JSON
# so you can manually examine what the checkpoint is doing.
#
# This is NOT the RL trainer -- it is a standalone SGLang offline batch job (same
# inference backend as training, so it runs in the same SGLang container -- no vLLM)
# that reuses the training env/templates/exec-grading. Point --model at a MERGED HF
# checkpoint
# (see reference-codecontest-gcs-checkpoint-sync memory for how to merge a VERL
# FSDP checkpoint to HF), not a raw FSDP shard dir.
#
# Run INSIDE the verl container, e.g. on FASRC:
#   singularity exec --nv \
#     --bind /n/home05/sqin/long-horizon-RL/verl:/workspace/verl \
#     --bind /n/netscratch/dam_lab/Lab/sqin:/data \
#     <verl.sif> \
#     bash -c 'cd /workspace/verl && PYTHONPATH=/workspace/verl codecontest/run_validate_codecontest.sh'
#
# Env overrides: MODEL_PATH, VAL_FILE, OUT, MAX_PROBLEMS, N_SAMPLES, TEMPERATURE,
#   TOP_P, TOP_K, SEED, MAX_ASSISTANT_TURNS, MAX_NEW_TOKENS_PER_TURN,
#   MAX_RESPONSE_LENGTH, MAX_PROMPT_LENGTH, MAX_GT_TEST, MAX_FAILURES_SHOWN,
#   MAX_FEEDBACK_CHARS, ROLLOUT_TP, GPU_MEM_UTIL, CODECONTEST_EXEC_URL,
#   CODECONTEST_ALLOW_INPROCESS, CODECONTEST_EXEC_CONCURRENCY.

set -xeuo pipefail

# --- code-exec backend (identical semantics to training) ---
# Preferred: a running sandbox sidecar (codecontest/run_sandbox.sh).
#   export CODECONTEST_EXEC_URL=http://cc-sandbox:8088
# Otherwise fall back to the in-process executor (fine for a single-container eval;
# the eval job is read-only so the "strictly worse" caveat that matters in training
# -- a bad solution killing a rollout worker -- is much less of a concern here).
export CODECONTEST_EXEC_URL=${CODECONTEST_EXEC_URL:-}
if [ -z "${CODECONTEST_EXEC_URL}" ]; then
  export CODECONTEST_ALLOW_INPROCESS=${CODECONTEST_ALLOW_INPROCESS:-1}
fi
export CODECONTEST_EXEC_MEM_GB=${CODECONTEST_EXEC_MEM_GB:-2}
export CODECONTEST_EXEC_CONCURRENCY=${CODECONTEST_EXEC_CONCURRENCY:-32}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}
VAL_FILE=${VAL_FILE:-$HOME/data/codecontests/test.parquet}
OUT=${OUT:-runs/validate_$(date +%m%d_%H%M).json}

# Eval scope
MAX_PROBLEMS=${MAX_PROBLEMS:-64}
N_SAMPLES=${N_SAMPLES:-4}

# Inference hparams (tune these to probe the model)
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:--1}
SEED=${SEED:-0}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-4}
MAX_NEW_TOKENS_PER_TURN=${MAX_NEW_TOKENS_PER_TURN:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}

# Oracle/feedback knobs -- keep matched to training to reproduce its grading
MAX_GT_TEST=${MAX_GT_TEST:-20}
MAX_FAILURES_SHOWN=${MAX_FAILURES_SHOWN:-3}
MAX_FEEDBACK_CHARS=${MAX_FEEDBACK_CHARS:-0}

# SGLang engine
ROLLOUT_TP=${ROLLOUT_TP:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}   # SGLang mem_fraction_static

python3 codecontest/validate_codecontest.py \
    --model "${MODEL_PATH}" \
    --val_file "${VAL_FILE}" \
    --out "${OUT}" \
    --max_problems ${MAX_PROBLEMS} \
    --n_samples ${N_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --seed ${SEED} \
    --max_assistant_turns ${MAX_ASSISTANT_TURNS} \
    --max_new_tokens_per_turn ${MAX_NEW_TOKENS_PER_TURN} \
    --max_response_length ${MAX_RESPONSE_LENGTH} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --max_gt_test ${MAX_GT_TEST} \
    --max_failures_shown ${MAX_FAILURES_SHOWN} \
    --max_feedback_chars ${MAX_FEEDBACK_CHARS} \
    --tensor_parallel_size ${ROLLOUT_TP} \
    --gpu_memory_utilization ${GPU_MEM_UTIL} \
    "$@"

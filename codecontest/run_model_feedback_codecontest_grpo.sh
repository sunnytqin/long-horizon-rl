#!/usr/bin/env bash
# GRPO | Qwen2.5-14B-Instruct | FSDP | multi-turn MODEL-FEEDBACK code-refinement on CodeContests
#
# Sibling of run_oracle_codecontest_grpo.sh. Identical hyperparameters; the ONLY change is the
# agent loop: `model_feedback_agent` instead of `code_refine_agent`. In this loop the between-turns
# user turn is written by the SAME policy run as a "user model" (a second, masked inference call
# that diagnoses the failing tests), instead of injecting the raw failing cases for the solver to
# reflect on itself. Training is still solver-only (the diagnosis turn is mask=0). The feedback call
# reuses the solver's sampling params; two dedicated knobs govern it: MAX_FEEDBACK_CHARS (failing
# cases fed INTO the user model) and MAX_FEEDBACK_TOKENS (diagnosis length OUT of it).
#
# Prereq: python codecontest/preprocess_codecontests.py --local_dir ~/data/codecontests
# Run from the repo root (so `codecontest` is importable, like `recipe`).
#
# Env overrides: MODEL_PATH, INFER_BACKEND(sglang|vllm), NGPUS_PER_NODE, ROLLOUT_N,
#   MAX_ASSISTANT_TURNS, TRAIN_BATCH_SIZE, MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH,
#   MAX_NEW_TOKENS_PER_TURN, MAX_FAILURES_SHOWN, MAX_GT_TEST,
#   MAX_FEEDBACK_CHARS, MAX_FEEDBACK_TOKENS,
#   CODECONTEST_EXEC_MEM_GB, CODECONTEST_EXEC_CONCURRENCY, ENV_STEP_TIMEOUT,
#   ROLLOUT_GPU_MEM_UTIL, MULTI_STAGE_WAKE_UP, ULYSSES_SP, PARAM_OFFLOAD, OPT_OFFLOAD.


set -xeuo pipefail


# Exp configs. Distinct EXPERIMENT_NAME default from the oracle run so checkpoints/tensorboard/eval
# paths never collide. Cloud launch.py/entrypoint still override both names verbatim.
PROJECT_NAME=${PROJECT_NAME:-codecontest_mt}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-local_modelfb_${EXP_NAME:-debug}_$(date +%m%d_%H%M)}

AGENTLOOP_CONFIG_PATH=${AGENTLOOP_CONFIG_PATH:-codecontest/config/agent_loop_config.yaml}

INFER_BACKEND=${INFER_BACKEND:-sglang}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# Model and dataset
DATA_DIR=${DATA_DIR:-$HOME/data/codecontests}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}


# Data hparams
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096} # cap the initial prompt (coding queston itself) len
max_response_length=${MAX_RESPONSE_LENGTH:-8192} # episode TAIL: all assistant turns + injected feedback (full seq = prompt + this)
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}


# Rollout hparams
rollout_n=${ROLLOUT_N:-16}
rollout_tp=${ROLLOUT_TP:-2}                      # SGLang inference TP (helps ROLLOUT-phase GPU mem)
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
multi_stage_wake_up=${MULTI_STAGE_WAKE_UP:-False}  # SGLang: stage engine wake-up to cut rollout->train peak mem


# Multi-turn / oracle knob.
max_assistant_turns=${MAX_ASSISTANT_TURNS:-4}    # total solver attempts (1=single turn RL)
max_new_tokens_per_turn=${MAX_NEW_TOKENS_PER_TURN:-4096} # solver generation len (the user-model feedback cap is MAX_FEEDBACK_TOKENS)
max_failures_shown=${MAX_FAILURES_SHOWN:-3}
max_gt_test=${MAX_GT_TEST:-20}   # GT cases graded per turn -- DON'T shrink: fewer => false-positive rewards
# Char budget for the failing cases in the USER-MODEL PROMPT (problem+code+failures). These
# live only in the user model's throwaway single-turn prompt (NOT the solver's cumulative
# conversation) and that prompt is tokenized UNCAPPED -- its only ceiling is the engine
# context (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), so we set it generously at 8000
# (~2700 tokens, leaves ample room alongside the problem+code). Set 0 to auto-derive
# (~0.5 * prompt_length chars) instead.
max_feedback_chars=${MAX_FEEDBACK_CHARS:-8000}
# Max NEW tokens the user model may generate for its diagnosis -- the exact hard bound on
# the injected solver turn (skeleton + <= this). The diagnosis lands in the solver's
# response tail, so it SHARES MAX_RESPONSE_LENGTH with all solver code turns + all feedback
# turns: at 2048 with up to 3 feedback turns that reserves ~6144 of 8192 for feedback. Push
# higher only if you also raise MAX_RESPONSE_LENGTH, else the solver starves -> overflow
# (unsolved => reward 0). Check feedback_resp_len_mean for actual usage before bumping.
max_feedback_tokens=${MAX_FEEDBACK_TOKENS:-2048}
on_overflow=${ON_OVERFLOW:-end_zero_reward}
rollout_temp=${ROLLOUT_TEMP:-0.6}
rollout_top_p=${ROLLOUT_TOP_P:-0.95}
env_step_timeout=${ENV_STEP_TIMEOUT:-180}        # hard wall on one code-grading step (sec)


# Code-exec sandbox.
export CODECONTEST_EXEC_URL=${CODECONTEST_EXEC_URL:-}
export CODECONTEST_EXEC_MEM_GB=${CODECONTEST_EXEC_MEM_GB:-2}        # per-process addr-space headroom cap (GB)
export CODECONTEST_EXEC_CONCURRENCY=${CODECONTEST_EXEC_CONCURRENCY:-32}  # max concurrent child executions


# Training hparams
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.02}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}


# ===== Rollout<->training mismatch correction (TIS) ===== (see run_oracle_codecontest_grpo.sh)
rollout_is=${ROLLOUT_IS:-token}                     # token | sequence | null (null => metrics-only)
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}   # TIS upper bound on the IS weight


# ===== PPO clip range (clip-higher / DAPO) ===== (see run_oracle_codecontest_grpo.sh)
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}


# ===== GPU-OOM playbook ===== (see run_oracle_codecontest_grpo.sh)
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}
[ -n "${PYTORCH_CUDA_ALLOC_CONF}" ] && export PYTORCH_CUDA_ALLOC_CONF
ulysses_sp=${ULYSSES_SP:-1}
param_offload=${PARAM_OFFLOAD:-True}
optimizer_offload=${OPT_OFFLOAD:-True}


python3 -m verl.trainer.main_ppo \
   algorithm.adv_estimator=grpo \
   algorithm.use_kl_in_reward=False \
   algorithm.rollout_correction.rollout_is=${rollout_is} \
   algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
   data.train_files="['${DATA_DIR}/train.parquet']" \
   data.val_files="['${DATA_DIR}/test.parquet']" \
   data.train_batch_size=${train_batch_size} \
   data.max_prompt_length=${max_prompt_length} \
   data.max_response_length=${max_response_length} \
   data.return_raw_chat=True \
   data.filter_overlong_prompts=True \
   data.truncation='error' \
   actor_rollout_ref.model.path="${MODEL_PATH}" \
   actor_rollout_ref.model.use_remove_padding=True \
   actor_rollout_ref.model.enable_gradient_checkpointing=True \
   actor_rollout_ref.actor.optim.lr=${actor_lr} \
   actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
   actor_rollout_ref.actor.use_dynamic_bsz=True \
   actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
   actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sp} \
   actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
   actor_rollout_ref.actor.use_kl_loss=True \
   actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
   actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
   actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
   actor_rollout_ref.actor.entropy_coeff=0 \
   actor_rollout_ref.rollout.temperature=${rollout_temp} \
   actor_rollout_ref.rollout.top_p=${rollout_top_p} \
   actor_rollout_ref.rollout.name=${INFER_BACKEND} \
   actor_rollout_ref.rollout.mode=async \
   actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
   actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
   actor_rollout_ref.rollout.multi_stage_wake_up=${multi_stage_wake_up} \
   actor_rollout_ref.rollout.calculate_log_probs=True \
   actor_rollout_ref.rollout.n=${rollout_n} \
   actor_rollout_ref.rollout.multi_turn.enable=True \
   actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
   actor_rollout_ref.rollout.multi_turn.format=hermes \
   actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENTLOOP_CONFIG_PATH} \
   actor_rollout_ref.rollout.agent.default_agent_loop=model_feedback_agent \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   reward_model.reward_manager=naive \
   +codecontest.max_new_tokens_per_turn=${max_new_tokens_per_turn} \
   +codecontest.max_failures_shown=${max_failures_shown} \
   +codecontest.max_gt_test=${max_gt_test} \
   +codecontest.max_feedback_chars=${max_feedback_chars} \
   +codecontest.max_feedback_tokens=${max_feedback_tokens} \
   +codecontest.on_overflow=${on_overflow} \
   +codecontest.env_step_timeout=${env_step_timeout} \
   trainer.balance_batch=True \
   trainer.logger='["console","tensorboard"]' \
   trainer.project_name=${PROJECT_NAME} \
   trainer.experiment_name=${EXPERIMENT_NAME} \
   trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
   trainer.nnodes=${NNODES} \
   trainer.save_freq=${save_freq} \
   trainer.test_freq=${test_freq} \
   trainer.total_epochs=${total_epochs} \
   "$@"

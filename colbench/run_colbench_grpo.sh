#!/usr/bin/env bash
# GRPO | Qwen2.5-14B-Instruct (or Qwen3-4B-Instruct) | FSDP | multi-turn ColBench
#
# Trains a SOLVER with the custom `colbench_agent` loop: the solver asks a FROZEN user
# simulator clarification questions to extract hidden requirements, then submits a Python
# function ("I WANT TO ANSWER:"). The submission is graded ONCE at the end for functional
# equivalence against a hidden ground-truth function; the trajectory reward is the FRACTIONAL
# GT pass-rate in [0,1]. Reward is produced inside the agent loop
# (AgentLoopOutput.reward_score), so the default `naive` reward manager passes it through.
#
# The user simulator is a SEPARATE frozen SGLang OpenAI server (same base model); the agent
# loop reaches it over OPENAI_BASE_URL / MULTITURN_MODEL_NAME (exported by
# colbench/entrypoint_colbench.sh). The GT function is passed ONLY to the simulator prompt --
# it never enters the solver's trajectory. Grading reuses the codecontest exec sidecar.
#
# Prereq: python colbench/preprocess_colbench.py --src_dir InfoPO/data/colbench_code \
#           --local_dir ~/data/colbench
# Run from the repo root (so `colbench` and `codecontest` are importable).
#
# In-training validation runs on a LIGHT subsample (test_small.parquet, ~2k, written by
# preprocess_colbench.py) -- the full 10k test.parquet is reserved for the offline eval loop
# (colbench/validate_colbench.py). Override the val set with VAL_FILE=.../test.parquet.
#
# Env overrides: MODEL_PATH, VAL_FILE, INFER_BACKEND(sglang|vllm), NGPUS_PER_NODE, ROLLOUT_N,
#   MAX_ASSISTANT_TURNS, TRAIN_BATCH_SIZE, MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH,
#   MAX_NEW_TOKENS_PER_TURN, TRAIN_TURNS, REWARD_TIME_LIMIT, ENV_STEP_TIMEOUT,
#   CODECONTEST_EXEC_MEM_GB, CODECONTEST_EXEC_CONCURRENCY, ROLLOUT_GPU_MEM_UTIL,
#   KL_LOSS_COEF, PARAM_OFFLOAD, OPT_OFFLOAD.


set -xeuo pipefail


# Exp configs. In cloud runs launch.py/entrypoint set PROJECT_NAME + the STABLE
# EXPERIMENT_NAME (`{model}_{exp_name}`), consumed verbatim -- the single source of truth for
# checkpoint/tensorboard/eval paths. The timestamped fallback only fires for ad-hoc LOCAL runs.
PROJECT_NAME=${PROJECT_NAME:-colbench_mt}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-local_${EXP_NAME:-debug}_$(date +%m%d_%H%M)}

AGENTLOOP_CONFIG_PATH=${AGENTLOOP_CONFIG_PATH:-colbench/config/agent_loop_config.yaml}

INFER_BACKEND=${INFER_BACKEND:-sglang}
NNODES=${NNODES:-1}
# Training GPU count. The entrypoint reserves the LAST GPU for the frozen sim server and
# exports NGPUS_PER_NODE (default 6 on an 8-GPU node: GPUs 0-5 train, GPU 7 = sim, GPU 6 idle).
# MUST be divisible by rollout_tp below (6 % 2 == 0); 7 would crash (7 % 2 != 0). For a small
# model you can run NGPUS_PER_NODE=7 with ROLLOUT_TP=1 to avoid the idle GPU.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}


# Model and dataset
DATA_DIR=${DATA_DIR:-$HOME/data/colbench}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}


# Data hparams
# MUST be divisible by NGPUS_PER_NODE (6) and by the rollout dp size (NGPUS_PER_NODE/rollout_tp
# = 3), else verl's data dispatch asserts. 120 = 6*20 satisfies both; if you change the GPU
# count, keep train/mini divisible by it. (128 % 6 != 0 -- the old default would crash.)
train_batch_size=${TRAIN_BATCH_SIZE:-120}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-120}
# Budgets from InfoPO colbench_trainer.yaml + our stability finding.
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}      # initial (public) problem prompt cap
# Episode TAIL: all solver turns + injected <=400-char user replies. total seq = prompt+this.
# max_model_len (total context) ~= 16384 -> response = 16384 - 2048 = 14336.
max_response_length=${MAX_RESPONSE_LENGTH:-14336}
# Actor TRAINING dynamic-bsz token budget per GPU. HARD FLOOR = max_prompt+max_response.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}


# Rollout hparams
rollout_n=${ROLLOUT_N:-4}                          # GRPO group size (fractional reward is denser than binary)
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
multi_stage_wake_up=${MULTI_STAGE_WAKE_UP:-False}


# Multi-turn / ColBench knobs.
max_assistant_turns=${MAX_ASSISTANT_TURNS:-10}     # total solver turns (clarify + submit)
max_new_tokens_per_turn=${MAX_NEW_TOKENS_PER_TURN:-1024}  # per-turn solver generation cap
reward_time_limit=${REWARD_TIME_LIMIT:-6}          # per-case GT exec timeout (sec)
env_step_timeout=${ENV_STEP_TIMEOUT:-180}          # hard wall on one blocking env call (sim turn or grading)
# SET 2 gradient-masking arm (shared with codecontest): all | final_only.
train_turns=${TRAIN_TURNS:-all}
# Solver sampling. Training temperature 0.6 (InfoPO); simulator uses temp 0 (env.py).
rollout_temp=${ROLLOUT_TEMP:-0.6}
rollout_top_p=${ROLLOUT_TOP_P:-0.95}


# Code-exec sandbox (reused from codecontest, UNCHANGED). The sidecar grades ColBench
# functional equivalence via a stdin-driven comparison harness (colbench/reward.py). The
# entrypoint brings the sidecar up and exports CODECONTEST_EXEC_URL.
export CODECONTEST_EXEC_URL=${CODECONTEST_EXEC_URL:-}
export CODECONTEST_EXEC_MEM_GB=${CODECONTEST_EXEC_MEM_GB:-2}
export CODECONTEST_EXEC_CONCURRENCY=${CODECONTEST_EXEC_CONCURRENCY:-32}


# Training hparams
actor_lr=${ACTOR_LR:-1e-6}
# KL = 0.01: the proven stability fix for this multi-turn stack (entropy explosion, not a
# rollout mismatch -- see project-codecontest-rl-stability-plan). Do NOT drop to 0.
kl_loss_coef=${KL_LOSS_COEF:-0.01}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}


# Rollout<->training mismatch correction (TIS) + clip-higher. Proven INERT for this stack
# (KL fixed the instability, not these) but kept ON for comparability with the codecontest
# runs. Set ROLLOUT_IS=null for a metrics-only control.
rollout_is=${ROLLOUT_IS:-sequence}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}


# FSDP CPU offload (frees GPU model/optimizer state; ~free given the node's host RAM).
param_offload=${PARAM_OFFLOAD:-True}
optimizer_offload=${OPT_OFFLOAD:-True}
ulysses_sp=${ULYSSES_SP:-1}


python3 -m verl.trainer.main_ppo \
   algorithm.adv_estimator=grpo \
   algorithm.use_kl_in_reward=False \
   algorithm.rollout_correction.rollout_is=${rollout_is} \
   algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
   data.train_files="['${DATA_DIR}/train.parquet']" \
   data.val_files="['${VAL_FILE:-${DATA_DIR}/test_small.parquet}']" \
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
   actor_rollout_ref.rollout.agent.default_agent_loop=colbench_agent \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   reward_model.reward_manager=naive \
   +colbench.max_new_tokens_per_turn=${max_new_tokens_per_turn} \
   +colbench.train_turns=${train_turns} \
   +colbench.reward_time_limit=${reward_time_limit} \
   +colbench.env_step_timeout=${env_step_timeout} \
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

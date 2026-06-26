#!/usr/bin/env bash
# GRPO | Qwen2.5-14B-Instruct | FSDP | multi-turn oracle code-refinement on CodeContests
#
# Trains a solver with the custom `code_refine_agent` loop: the model writes code,
# an oracle env runs it against ground-truth tests and feeds back failing cases for
# up to MAX_ASSISTANT_TURNS turns; the trajectory reward is binary (final code passes
# all GT tests -> 1, else 0). Reward is produced inside the agent loop
# (AgentLoopOutput.reward_score), so the default `naive` reward manager just passes it
# through -- no custom reward function needed.
#
# Prereq: python codecontest/preprocess_codecontests.py --local_dir ~/data/codecontests
# Run from the repo root (so `codecontest` is importable, like `recipe`).
#
# Env overrides: MODEL_PATH, INFER_BACKEND(sglang|vllm), NGPUS_PER_NODE, ROLLOUT_N,
#   MAX_ASSISTANT_TURNS, TRAIN_BATCH_SIZE, MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH,
#   MAX_NEW_TOKENS_PER_TURN, MAX_FAILURES_SHOWN, MAX_GT_TEST,
#   CODECONTEST_EXEC_MEM_GB, CODECONTEST_EXEC_CONCURRENCY, ENV_STEP_TIMEOUT,
#   ROLLOUT_GPU_MEM_UTIL, MULTI_STAGE_WAKE_UP, ULYSSES_SP, PARAM_OFFLOAD, OPT_OFFLOAD.


set -xeuo pipefail


# Exp configs
PROJECT_NAME=${PROJECT_NAME:-codecontest_mt}
if [ -n "${EXP_NAME:-}" ]; then
 EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_14b_grpo_oracle_${EXP_NAME}_$(date +%m%d_%H%M)}
else
 EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_14b_grpo_oracle_$(date +%m%d_%H%M)}
fi

AGENTLOOP_CONFIG_PATH=${AGENTLOOP_CONFIG_PATH:-codecontest/config/agent_loop_config.yaml}

INFER_BACKEND=${INFER_BACKEND:-sglang}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# Model and dataset
DATA_DIR=${DATA_DIR:-$HOME/data/codecontests}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}


# Data hparams
train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096} # cap the initial prompt (coding queston itself) len
max_response_length=${MAX_RESPONSE_LENGTH:-16384} # episode TAIL: all assistant turns + injected feedback (full seq = prompt + this)
# Actor TRAINING dynamic-bsz token budget per GPU (NOT the SGLang context; that auto-resolves to prompt+response).
# Also drives log_prob_max_token_len_per_gpu. HARD FLOOR = max_prompt_length + max_response_length (a single
# trajectory can't be split across micro-batches), so it must be >= 4096+16384=20480. Bigger = more memory/GPU.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}


# Rollout hparams
rollout_n=${ROLLOUT_N:-16}                       # GRPO group size
rollout_tp=${ROLLOUT_TP:-2}                      # SGLang inference TP (helps ROLLOUT-phase GPU mem)
# SGLang KV fraction; lower => more room for FSDP at the rollout->train transition.
# Lowered 0.6 -> 0.5 after a GPU-VRAM OOM at the weight-sync resume: SGLang's
# torch_memory_saver failed to re-acquire its KV cache, signature
#   [torch_memory_saver.cpp] CUresult error: 2 (out of memory) func=cu_mem_create
# inside resume_memory_occupation -> the TP scheduler died -> gloo barrier
# "Connection closed by peer" -> raylet SYSTEM_ERROR. If you hit that again, drop
# this further (0.45) and/or set PARAM_OFFLOAD=True (below).
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
# NOTE: default OFF -- enabling this crashed SGLangHttpServer during launch_servers on this
# container. Re-enable (MULTI_STAGE_WAKE_UP=True) only if your SGLang build supports it.
multi_stage_wake_up=${MULTI_STAGE_WAKE_UP:-False}  # SGLang: stage engine wake-up to cut rollout->train peak mem


# Multi-turn / oracle knob.
max_assistant_turns=${MAX_ASSISTANT_TURNS:-4}    # total solver attempts (1=single turn RL)
max_new_tokens_per_turn=${MAX_NEW_TOKENS_PER_TURN:-4096} # controls solver generation len
max_failures_shown=${MAX_FAILURES_SHOWN:-3}
max_gt_test=${MAX_GT_TEST:-20}   # GT cases graded per turn -- DON'T shrink: fewer => false-positive rewards
on_overflow=${ON_OVERFLOW:-end_zero_reward}
rollout_temp=${ROLLOUT_TEMP:-0.8}
rollout_top_p=${ROLLOUT_TOP_P:-0.95}
env_step_timeout=${ENV_STEP_TIMEOUT:-180}        # hard wall on one code-grading step (sec)


# Code-exec sandbox.
#
# Preferred: run the SLIM SANDBOX SIDECAR (codecontest/exec_server.py in its own
# container, see codecontest/run_sandbox.sh) and point the trainer at it here. The
# untrusted exec() then happens behind a container boundary -- a memory bomb / fork
# bomb / busy loop dies inside the sidecar's cgroup, never touching this trainer or
# the Ray OOM killer. Reward semantics are identical to the in-process path.
#   export CODECONTEST_EXEC_URL=http://cc-sandbox:8088   # set to enable the sidecar
export CODECONTEST_EXEC_URL=${CODECONTEST_EXEC_URL:-}
#
# The two knobs below now configure the SIDECAR (set them in run_sandbox.sh). On the
# trainer they only affect the in-process FALLBACK used when CODECONTEST_EXEC_URL is
# unset (e.g. the 1-GPU smoke run). Worst-case sandbox RAM ~= CONCURRENCY * MEM_GB.
export CODECONTEST_EXEC_MEM_GB=${CODECONTEST_EXEC_MEM_GB:-2}        # per-process addr-space headroom cap (GB)
export CODECONTEST_EXEC_CONCURRENCY=${CODECONTEST_EXEC_CONCURRENCY:-64}  # max concurrent child executions


# Training hparams
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}


# GPU-OOM mitigation (training-phase / rollout->train transition). Tuned for 14B on 8xH100.
# expandable_segments:True reduces allocator fragmentation BUT can crash SGLang's CUDA-graph
# capture during server launch -- so it's UNSET by default. Opt in only with enforce_eager,
# e.g.: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ... actor_rollout_ref.rollout.enforce_eager=True
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}
[ -n "${PYTORCH_CUDA_ALLOC_CONF}" ] && export PYTORCH_CUDA_ALLOC_CONF
# Ulysses sequence parallelism: shard the long-episode (~20k tok) activations across SP
# GPUs during the actor fwd/bwd -- the right lever for long-seq OOM (no host-RAM cost,
# unlike offload). Needs use_remove_padding=True (set below). 1 = off.
ulysses_sp=${ULYSSES_SP:-2}
# FSDP CPU offload: frees GPU model/optimizer state but ADDS host RAM. This is the
# next escalation for the rollout->train transition GPU-OOM (cu_mem_create) above:
# moving actor PARAMS off-GPU during rollout frees the VRAM SGLang needs to wake. With
# ~1 TB host RAM here the cost is basically free -- flip to True if 0.5 KV util isn't
# enough (optimizer_offload below is already commonly enabled via OPT_OFFLOAD=True).
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPT_OFFLOAD:-False}


python3 -m verl.trainer.main_ppo \
   algorithm.adv_estimator=grpo \
   algorithm.use_kl_in_reward=False \
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
   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
   actor_rollout_ref.actor.entropy_coeff=0 \
   actor_rollout_ref.rollout.temperature=${rollout_temp} \
   actor_rollout_ref.rollout.top_p=${rollout_top_p} \
   actor_rollout_ref.rollout.name=${INFER_BACKEND} \
   actor_rollout_ref.rollout.mode=async \
   actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
   actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
   actor_rollout_ref.rollout.multi_stage_wake_up=${multi_stage_wake_up} \
   actor_rollout_ref.rollout.n=${rollout_n} \
   actor_rollout_ref.rollout.multi_turn.enable=True \
   actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
   actor_rollout_ref.rollout.multi_turn.format=hermes \
   actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENTLOOP_CONFIG_PATH} \
   actor_rollout_ref.rollout.agent.default_agent_loop=code_refine_agent \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   reward_model.reward_manager=naive \
   +codecontest.max_new_tokens_per_turn=${max_new_tokens_per_turn} \
   +codecontest.max_failures_shown=${max_failures_shown} \
   +codecontest.max_gt_test=${max_gt_test} \
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

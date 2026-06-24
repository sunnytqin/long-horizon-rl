#!/usr/bin/env bash
# GRPO | Qwen2.5-7B-Instruct | FSDP | multi-turn oracle code-refinement on CodeContests
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
#   MAX_NEW_TOKENS_PER_TURN, MAX_FAILURES_SHOWN, MAX_GT_TEST.

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
INFER_BACKEND=${INFER_BACKEND:-sglang}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

DATA_DIR=${DATA_DIR:-$HOME/data/codecontests}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

rollout_n=${ROLLOUT_N:-8}                       # GRPO group size
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}

# Multi-turn / oracle knobs.
max_assistant_turns=${MAX_ASSISTANT_TURNS:-3}    # turn 0 + ~2 refinements
max_new_tokens_per_turn=${MAX_NEW_TOKENS_PER_TURN:-2048}
max_failures_shown=${MAX_FAILURES_SHOWN:-3}
max_gt_test=${MAX_GT_TEST:-20}
on_overflow=${ON_OVERFLOW:-end_zero_reward}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-codecontests_multiturn}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_7b_grpo_oracle_$(date +%Y%m%d_%H%M)}

AGENTLOOP_CONFIG_PATH=${AGENTLOOP_CONFIG_PATH:-codecontest/config/agent_loop_config.yaml}

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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.name=${INFER_BACKEND} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
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
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    "$@"

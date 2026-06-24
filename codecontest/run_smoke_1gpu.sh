#!/usr/bin/env bash
# Tiny single-GPU END-TO-END smoke for the CodeContests multi-turn agent loop.
# Goal: confirm main_ppo loads `code_refine_agent`, runs multi-turn SGLang rollouts
# that call our loop + local code-exec + binary reward, and completes a couple of
# optimizer steps -- NOT to learn anything. Use a tiny model so it fits one GPU.
#
# Run INSIDE the verl SGLang container, e.g. on FASRC:
#   singularity exec --nv \
#     --bind /n/home05/sqin/long-horizon-RL/verl:/workspace/verl \
#     --bind /n/netscratch/dam_lab/Lab/sqin:/data \
#     /n/netscratch/dam_lab/Lab/sqin/docker_images/verl-sgl0512-dev2.sif \
#     bash -c 'cd /workspace/verl && PYTHONPATH=/workspace/verl codecontest/run_smoke_1gpu.sh'
#
# Prereq: a tiny smoke parquet, e.g.
#   PYTHONPATH=$(pwd) python codecontest/preprocess_codecontests.py \
#       --local_dir /data/codecontests_smoke --max_train 24 --max_val 8

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}     # tiny: fits a small GPU
INFER_BACKEND=${INFER_BACKEND:-sglang}
DATA_DIR=${DATA_DIR:-/data/codecontests_smoke}
AGENTLOOP_CONFIG_PATH=${AGENTLOOP_CONFIG_PATH:-codecontest/config/agent_loop_config.yaml}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['${DATA_DIR}/train.parquet']" \
    data.val_files="['${DATA_DIR}/test.parquet']" \
    data.train_batch_size=8 \
    data.max_prompt_length=1536 \
    data.max_response_length=3072 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=${INFER_BACKEND} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENTLOOP_CONFIG_PATH} \
    actor_rollout_ref.rollout.agent.default_agent_loop=code_refine_agent \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    +codecontest.max_new_tokens_per_turn=768 \
    +codecontest.max_failures_shown=3 \
    +codecontest.max_gt_test=8 \
    +codecontest.on_overflow=end_zero_reward \
    trainer.logger='["console"]' \
    trainer.project_name=codecontests_smoke \
    trainer.experiment_name=smoke_1gpu \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=2 \
    "$@"

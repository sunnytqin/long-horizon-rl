lag#!/usr/bin/env bash
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


# Exp configs.
# In cloud runs launch.py/entrypoint set PROJECT_NAME + the STABLE EXPERIMENT_NAME
# (`{model}_{exp_name}`), which we consume verbatim -- it is the single source of truth
# for checkpoint/tensorboard/eval paths. The timestamped fallback below only fires for
# ad-hoc LOCAL runs, so it can't collide with a tracked cloud experiment's checkpoints.
PROJECT_NAME=${PROJECT_NAME:-codecontest_mt}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-local_${EXP_NAME:-debug}_$(date +%m%d_%H%M)}

AGENTLOOP_CONFIG_PATH=${AGENTLOOP_CONFIG_PATH:-codecontest/config/agent_loop_config.yaml}

INFER_BACKEND=${INFER_BACKEND:-sglang}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}


# Model and dataset
DATA_DIR=${DATA_DIR:-$HOME/data/codecontests}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}


# Data hparams
# Prompts per training step. trajectories/step = train_batch_size * rollout_n (drives ROLLOUT
# time + host load, NOT training-phase GPU mem under dynamic bsz). Keep both divisible by
# n_gpus (8) and train % mini == 0.
train_batch_size=${TRAIN_BATCH_SIZE:-128}
# = train_batch_size => fully ON-POLICY GRPO (one update per rollout batch). Smaller => more
# gradient steps per batch but off-policy drift on the reused rollout data.
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096} # cap the initial prompt (coding queston itself) len
max_response_length=${MAX_RESPONSE_LENGTH:-8192} # episode TAIL: all assistant turns + injected feedback (full seq = prompt + this)
# Actor TRAINING dynamic-bsz token budget per GPU (NOT the SGLang context; that auto-resolves to prompt+response).
# Also drives log_prob_max_token_len_per_gpu. HARD FLOOR = max_prompt_length + max_response_length (a single
# trajectory can't be split across micro-batches), so it must be >= 4096+16384=20480. Bigger = more memory/GPU.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}


# Rollout hparams
# GRPO group size. KEY knob for sparse binary reward: too small => many all-pass/all-fail
# groups with zero advantage (no gradient). 16 gives more mixed groups => more signal.
rollout_n=${ROLLOUT_N:-16}
rollout_tp=${ROLLOUT_TP:-2}                      # SGLang inference TP (helps ROLLOUT-phase GPU mem)
# SGLang KV fraction; lower => more room for FSDP at the rollout->train transition.
# Lowered 0.6 -> 0.5 after a GPU-VRAM OOM at the weight-sync resume: SGLang's
# torch_memory_saver failed to re-acquire its KV cache, signature
#   [torch_memory_saver.cpp] CUresult error: 2 (out of memory) func=cu_mem_create
# inside resume_memory_occupation -> the TP scheduler died -> gloo barrier
# "Connection closed by peer" -> raylet SYSTEM_ERROR. If you hit that again, drop
# this further (0.45) and/or set PARAM_OFFLOAD=True (below).
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
# NOTE: default OFF -- enabling this crashed SGLangHttpServer during launch_servers on this
# container. Re-enable (MULTI_STAGE_WAKE_UP=True) only if your SGLang build supports it.
multi_stage_wake_up=${MULTI_STAGE_WAKE_UP:-False}  # SGLang: stage engine wake-up to cut rollout->train peak mem


# Multi-turn / oracle knob.
max_assistant_turns=${MAX_ASSISTANT_TURNS:-4}    # total solver attempts (1=single turn RL)
max_new_tokens_per_turn=${MAX_NEW_TOKENS_PER_TURN:-4096} # controls solver generation len
max_failures_shown=${MAX_FAILURES_SHOWN:-3}
max_gt_test=${MAX_GT_TEST:-20}   # GT cases graded per turn -- DON'T shrink: fewer => false-positive rewards
# Combined char budget for the failing-case fields in the injected feedback turn.
# CodeContests has pathologically large test INPUTS (tens of KB); without this a single
# failing case can blow the feedback to 100k+ tokens, which the agent loop then blindly
# left-truncates (dropping the user-turn role framing). A water-filling policy clips only
# the large fields (input/output/expected). Default 0 => the agent loop AUTO-DERIVES the
# budget from max_prompt_length (the cap the feedback is actually checked against), so it
# tracks that knob automatically. Set a positive value to pin an absolute char budget.
max_feedback_chars=${MAX_FEEDBACK_CHARS:-0}
on_overflow=${ON_OVERFLOW:-end_zero_reward}
rollout_temp=${ROLLOUT_TEMP:-0.6}
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
export CODECONTEST_EXEC_CONCURRENCY=${CODECONTEST_EXEC_CONCURRENCY:-32}  # max concurrent child executions


# Training hparams
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.02}
total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}


# ===== Rollout<->training mismatch correction (TIS) =====
# SGLang SAMPLES the rollout, but FSDP RECOMPUTES the log-probs used in the loss. Even with
# identical weights the two distributions differ (bf16 vs the inference kernels, different
# backends), so the policy ratio pi_theta/pi_old is being formed against the WRONG behavior
# policy. That bias accumulates per token, so it is worse for long multi-turn sequences -- a
# leading cause of the KL-explosion / reward-collapse cycle (see docs/algo/rollout_corr.md,
# "When Speed Kills Stability").
#
# We enable truncated importance sampling in DECOUPLED mode (bypass_mode=False, the default):
# GRPO is otherwise unchanged (it still recomputes old_log_prob), we just reweight each token
# by the clamped pi_old/pi_rollout ratio. Cost is ~free -- no extra forward pass, the rollout
# already returns its logprobs. calculate_log_probs=True is REQUIRED: it makes SGLang emit the
# per-token rollout logprobs the agent loop forwards as `rollout_log_probs`.
#
# This ALSO logs the off-policy gap as `rollout_corr/*` (kl, log_ppl_abs_diff, chi2_token,
# rollout_is_max, rollout_is_eff_sample_size, ...) -- pure diagnostics, no gradient/speed cost.
# For a CONTROL run that confirms the mechanism WITHOUT touching training, set ROLLOUT_IS=null:
# the metrics still log, the correction is off.
rollout_is=${ROLLOUT_IS:-token}                     # token | sequence | null (null => metrics-only)
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}   # TIS upper bound on the IS weight


# ===== PPO clip range (clip-higher / DAPO) =====
# Asymmetric clipping: keep the lower bound tight but widen the UPPER bound so low-prob
# tokens are allowed to gain probability (preserves exploration) while the upside ratio
# that drives KL spikes is still clamped. Defaults = symmetric 0.2/0.2 (verl default,
# i.e. clip-higher OFF) so it's a no-op unless you opt in. Run B = TIS + clip-higher:
# set CLIP_RATIO_HIGH=0.28 (the DAPO value; don't exceed ~0.3 or the clip stops protecting).
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}


# ===== GPU-OOM playbook (14B / 8xH100) -- debugging lessons, so future-you skips the dead ends =====
# OOM hit at the ROLLOUT->TRAIN transition. The dominant tensor is the LM-head LOGITS
# [per-GPU tokens x ~152k vocab x 2B ~= 6.9 GB at 24576 tokens], NOT attention activations.
# WHAT ACTUALLY HELPS:
#   - lower gpu_memory_utilization (smaller SGLang KV reservation)
#   - PARAM_OFFLOAD / OPT_OFFLOAD (free GPU model+optimizer state; ~free here given ~1 TB host RAM)
#   - lower ppo_max_token_len_per_gpu toward its 20480 floor (shrinks the logits tensor)
# DEAD ENDS -- DON'T re-chase these when an OOM reappears:
#   - ULYSSES_SP=2 : launched + trained FINE (it did NOT break the pipeline) but did NOT help the
#       OOM -- it shards ATTENTION activations, not the logits tensor. Only useful if you ALSO
#       lower the per-GPU token budget. Left at 1.
#   - MULTI_STAGE_WAKE_UP=True : crashed SGLangHttpServer during launch_servers on this container.
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True : crashes SGLang's CUDA-graph capture at
#       launch (would need rollout.enforce_eager=True). UNSET by default for that reason.
# ===================================================================================================
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}
[ -n "${PYTORCH_CUDA_ALLOC_CONF}" ] && export PYTORCH_CUDA_ALLOC_CONF
# Ulysses SP: shards attention activations only (see playbook -- did NOT help the logits OOM).
# Needs use_remove_padding=True (set below). 1 = off (recommended here).
ulysses_sp=${ULYSSES_SP:-1}
# FSDP CPU offload: frees GPU model/optimizer state but ADDS host RAM. This is the
# next escalation for the rollout->train transition GPU-OOM (cu_mem_create) above:
# moving actor PARAMS off-GPU during rollout frees the VRAM SGLang needs to wake. With
# ~1 TB host RAM here the cost is basically free -- flip to True if 0.5 KV util isn't
# enough (optimizer_offload below is already commonly enabled via OPT_OFFLOAD=True).
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
   actor_rollout_ref.rollout.agent.default_agent_loop=code_refine_agent \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   reward_model.reward_manager=naive \
   +codecontest.max_new_tokens_per_turn=${max_new_tokens_per_turn} \
   +codecontest.max_failures_shown=${max_failures_shown} \
   +codecontest.max_gt_test=${max_gt_test} \
   +codecontest.max_feedback_chars=${max_feedback_chars} \
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

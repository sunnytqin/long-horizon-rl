#!/usr/bin/env bash
# GRPO | Qwen3-4B-Instruct (or Qwen2.5-14B) | FSDP | multi-turn ColBench -- SPEC path.
#
# Sibling of run_colbench_grpo.sh for the SPEC setting. The solver talks to a FROZEN user
# simulator that is conditioned on an authored natural-language SPEC (persona/scenario/
# requirements/plot) instead of the hidden GT source -- so the sim can never leak the answer.
# The solver proposes a Python function inside a ```python block (no submit marker); the USER
# ends the conversation with [TERMINATE]. The loop grades the LAST function shown for functional
# equivalence against a hidden ground-truth function; the trajectory reward is the FRACTIONAL GT
# pass-rate in [0,1]. Reward is produced inside the agent loop (AgentLoopOutput.reward_score), so
# the default `naive` reward manager passes it through.
#
# Differences vs run_colbench_grpo.sh (the GT path): (1) the spec agent loop + config, (2) the
# spec dataset, (3) the spec-path guardrails max_code_proposals + sim_max_tries instead of the
# GT-leak knob sim_reject_max_tries. Every GRPO/PPO knob (KL=0.01, TIS, clip, rollout.n, batch
# sizes, offload) is kept IDENTICAL to the GT run for a clean A/B.
#
# The user simulator is a SEPARATE frozen SGLang OpenAI server (frozen base model); the agent
# loop reaches it over OPENAI_BASE_URL / MULTITURN_MODEL_NAME (exported by
# colbench/entrypoint_colbench.sh -- unchanged for the spec path). The GT function is NEVER
# passed to the sim prompt (only the spec is). Grading reuses the codecontest exec sidecar.
#
# Prereq: python colbench/preprocess_colbench_spec.py --raw_parquet ... --specs_jsonl ... \
#           --out ~/data/colbench_spec/train.parquet    (+ a test-split test_small.parquet)
# Run from the repo root (so `colbench` and `codecontest` are importable).
#
# In-training validation runs on VAL_FILE (default test_small.parquet in the spec data dir), a
# held-out spec set. Deeper offline eval: colbench/validate_colbench_spec.py.
#
# Env overrides: MODEL_PATH, VAL_FILE, INFER_BACKEND(sglang|vllm), NGPUS_PER_NODE, ROLLOUT_N,
#   MAX_ASSISTANT_TURNS, MAX_CODE_PROPOSALS, SIM_MAX_TRIES, TRAIN_BATCH_SIZE, MAX_PROMPT_LENGTH,
#   MAX_RESPONSE_LENGTH, MAX_NEW_TOKENS_PER_TURN, TRAIN_TURNS, REWARD_TIME_LIMIT, ENV_STEP_TIMEOUT,
#   CODECONTEST_EXEC_MEM_GB, CODECONTEST_EXEC_CONCURRENCY, ROLLOUT_GPU_MEM_UTIL,
#   KL_LOSS_COEF, PARAM_OFFLOAD, OPT_OFFLOAD.


set -xeuo pipefail


# Exp configs. In cloud runs launch.py/entrypoint set PROJECT_NAME + the STABLE
# EXPERIMENT_NAME (`{model}_{exp_name}`), consumed verbatim -- the single source of truth for
# checkpoint/tensorboard/eval paths. The timestamped fallback only fires for ad-hoc LOCAL runs.
PROJECT_NAME=${PROJECT_NAME:-colbench_mt}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-local_${EXP_NAME:-debug}_$(date +%m%d_%H%M)}

# SPEC path: register + default to the spec agent loop.
AGENTLOOP_CONFIG_PATH=${AGENTLOOP_CONFIG_PATH:-colbench/config/agent_loop_config_spec.yaml}

INFER_BACKEND=${INFER_BACKEND:-sglang}
NNODES=${NNODES:-1}
# Training GPU count. The entrypoint reserves the LAST GPU for the frozen sim server and
# exports NGPUS_PER_NODE (default 6 on an 8-GPU node: GPUs 0-5 train, GPU 7 = sim, GPU 6 idle).
# MUST be divisible by rollout_tp below (6 % 2 == 0); 7 would crash (7 % 2 != 0). For a small
# model you can run NGPUS_PER_NODE=7 with ROLLOUT_TP=1 to avoid the idle GPU.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}


# Model and dataset. SPEC path: the spec parquet(s) live under colbench_spec.
DATA_DIR=${DATA_DIR:-$HOME/data/colbench_spec}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-14B-Instruct}


# Data hparams
# MUST be divisible by NGPUS_PER_NODE (6) and by the rollout dp size (NGPUS_PER_NODE/rollout_tp
# = 3), else verl's data dispatch asserts. 120 = 6*20 satisfies both; if you change the GPU
# count, keep train/mini divisible by it. (128 % 6 != 0 -- the old default would crash.)
train_batch_size=${TRAIN_BATCH_SIZE:-120}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-120}
# Budgets from InfoPO colbench_trainer.yaml + our stability finding.
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}      # initial (public) problem prompt cap
# Episode TAIL: all solver turns + injected user replies. total seq = prompt+this.
# max_model_len (total context) ~= 16384 -> response = 16384 - 2048 = 14336.
max_response_length=${MAX_RESPONSE_LENGTH:-14336}
# Actor TRAINING dynamic-bsz token budget per GPU. HARD FLOOR = max_prompt+max_response.
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}


# Rollout hparams
rollout_n=${ROLLOUT_N:-4}                          # GRPO group size (fractional reward is denser than binary)
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
multi_stage_wake_up=${MULTI_STAGE_WAKE_UP:-False}


# Multi-turn / ColBench SPEC knobs.
max_assistant_turns=${MAX_ASSISTANT_TURNS:-10}     # total solver turns (clarify + propose)
max_new_tokens_per_turn=${MAX_NEW_TOKENS_PER_TURN:-1024}  # per-turn solver generation cap
reward_time_limit=${REWARD_TIME_LIMIT:-6}          # per-case GT exec timeout (sec)
env_step_timeout=${ENV_STEP_TIMEOUT:-180}          # hard wall on one blocking env call (sim turn or grading)
# SET 2 gradient-masking arm: all | final_only. NOTE: the spec agent loop currently REJECTS
# final_only (the sim can [TERMINATE] after a non-code turn, so the last solver turn may not be
# the graded code turn); leave this at 'all' until spec last-code masking lands.
train_turns=${TRAIN_TURNS:-all}
# Spec-path guardrail: max ```python proposals before the loop force-grades the last one (default
# 2, reduced from 3 after eval). Replaces the GT path's sim_reject_max_tries.
max_code_proposals=${MAX_CODE_PROPOSALS:-2}
# Sim no-code rejection sampling: an ordinary user never pastes a function, so re-query the sim up
# to N times if its reply contains a code fence; on exhaustion the conversation aborts
# (terminated_by "sim_code_reject") and the last shown code is graded. Default 8 (matches eval).
sim_max_tries=${SIM_MAX_TRIES:-8}
# Solver sampling. Defaults = the SOLVER model's recommended generation settings; match these
# to whatever --model you train. Qwen3-4B-Instruct-2507: temp 0.7, top_p 0.8, top_k 20, min_p 0
# (min_p is verl's default 0, not a settable rollout field). Qwen3-32B (thinking): 0.6/0.95/20.
# NB: Qwen3 degrades under greedy (temp 0) -- always sample.
rollout_temp=${ROLLOUT_TEMP:-0.7}
rollout_top_p=${ROLLOUT_TOP_P:-0.8}
rollout_top_k=${ROLLOUT_TOP_K:-20}
# In-training VAL uses the SAME solver sampling as training (val_kwargs below reuse these vars),
# NOT verl's default greedy (temp 0). Two reasons: (1) val should reflect the sampling regime the
# policy is optimized under, and (2) Qwen3 degrades under greedy. The frozen user-sim is unchanged
# in val (separate server, its own _sim_sampling). val n=1 = one sampled trajectory per problem.


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
# ACTOR engine build. Defaults MATCH verl's own (fsdp / false), so a run that does not set these
# is byte-identical to before. entrypoint_colbench.sh exports fsdp2/True for models whose registry
# FSDP_PROFILE is 'fsdp2_offload' (32B): FSDP2 + CPUOffloadPolicy keeps the actor on CPU through
# the weight sync, which is the only way the 32B sync fits on one 8xH100 node. NB with fsdp2 +
# offload_policy the param_offload/optimizer_offload above become NO-OPS -- CPUOffloadPolicy owns
# CPU<->GPU placement instead. See the FSDP_PROFILE block in entrypoint_colbench.sh.
actor_fsdp_strategy=${ACTOR_FSDP_STRATEGY:-fsdp}
actor_offload_policy=${ACTOR_OFFLOAD_POLICY:-False}


# Solver chat-template kwargs. entrypoint_colbench.sh sets SOLVER_ENABLE_THINKING=false for a
# HYBRID Qwen3 solver (14B/32B, per the model registry's THINKING field) so every solver turn is
# templated with enable_thinking=False -- otherwise the <think> block eats the 1024-token
# max_new_tokens_per_turn budget and the real answer truncates. UNSET for non-hybrid models
# (Qwen2.5-14B, Qwen3-4B-Instruct-2507), whose chat templates error on the kwarg -> pass NOTHING,
# leaving those runs byte-identical to before. Reuses verl's existing
# data.apply_chat_template_kwargs (default {}), which the agent loop applies at every turn.
chat_template_args=()
if [ -n "${SOLVER_ENABLE_THINKING:-}" ]; then
    chat_template_args+=("+data.apply_chat_template_kwargs.enable_thinking=${SOLVER_ENABLE_THINKING}")
fi


python3 -m verl.trainer.main_ppo \
   ${chat_template_args[@]+"${chat_template_args[@]}"} \
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
   actor_rollout_ref.actor.fsdp_config.strategy=${actor_fsdp_strategy} \
   actor_rollout_ref.actor.fsdp_config.offload_policy=${actor_offload_policy} \
   actor_rollout_ref.actor.use_kl_loss=True \
   actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
   actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
   actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
   actor_rollout_ref.actor.entropy_coeff=0 \
   actor_rollout_ref.rollout.temperature=${rollout_temp} \
   actor_rollout_ref.rollout.top_p=${rollout_top_p} \
   actor_rollout_ref.rollout.top_k=${rollout_top_k} \
   actor_rollout_ref.rollout.name=${INFER_BACKEND} \
   actor_rollout_ref.rollout.mode=async \
   actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
   actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
   actor_rollout_ref.rollout.multi_stage_wake_up=${multi_stage_wake_up} \
   actor_rollout_ref.rollout.calculate_log_probs=True \
   actor_rollout_ref.rollout.n=${rollout_n} \
   actor_rollout_ref.rollout.val_kwargs.temperature=${rollout_temp} \
   actor_rollout_ref.rollout.val_kwargs.top_p=${rollout_top_p} \
   actor_rollout_ref.rollout.val_kwargs.top_k=${rollout_top_k} \
   actor_rollout_ref.rollout.val_kwargs.do_sample=True \
   actor_rollout_ref.rollout.val_kwargs.n=1 \
   actor_rollout_ref.rollout.multi_turn.enable=True \
   actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
   actor_rollout_ref.rollout.multi_turn.format=hermes \
   actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENTLOOP_CONFIG_PATH} \
   actor_rollout_ref.rollout.agent.default_agent_loop=colbench_spec_agent \
   actor_rollout_ref.ref.fsdp_config.param_offload=True \
   reward_model.reward_manager=naive \
   +colbench.max_new_tokens_per_turn=${max_new_tokens_per_turn} \
   +colbench.train_turns=${train_turns} \
   +colbench.max_code_proposals=${max_code_proposals} \
   +colbench.sim_max_tries=${sim_max_tries} \
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

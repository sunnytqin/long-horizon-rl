# codecontest — multi-turn RL on CodeContests

A multi-turn code-refinement agent loop for VERL. The solver writes a Python
solution, an environment grades it against ground-truth stdin/stdout tests, and
a feedback turn is injected so the solver can try again. The trajectory reward
is binary: did the final submission pass every executed GT test.

Two arms differ only in **where the feedback turn comes from**:

| Agent loop | Feedback turn | Selected by |
|---|---|---|
| `code_refine_agent` | the failing GT cases, injected verbatim; the solver is asked to reflect on them | `run_oracle_codecontest_grpo.sh` |
| `model_feedback_agent` | a second inference call — the *same* policy run as a "user model" — reads (problem, failed code, failing cases) and writes a 3-bullet diagnosis. Only the diagnosis is shown to the solver | `run_model_feedback_codecontest_grpo.sh` |

Training is **solver-only** in both. The feedback turns are injected with
mask=0, so they carry no gradient. Reward is oracle-graded either way, so the
user model can only shape the solver's *context*, never its reward value.

## Layout

```
code_refine_agent.py      oracle-feedback agent loop (AgentLoopBase)
model_feedback_agent.py   policy-written-feedback variant
env.py                    GTOracleEnv: grades a turn, builds the feedback text
templates.py              prompts + feedback formatting (water-fill char budget)
masking.py                which solver turns get gradient (shared with colbench)
local_exec.py             the sandbox: runs untrusted code in a child process
exec_server.py            HTTP wrapper around local_exec (the sidecar)
exec_client.py            client shim the trainer's env talks to
validate_codecontest.py   offline multi-turn eval / conversation dumper
preprocess_codecontests.py  HF dataset -> VERL parquet
config/agent_loop_config.yaml  registers both loops by name
```

`masking.py` is imported by `colbench/colbench_agent.py` and
`colbench/colbench_spec_agent.py`. Re-run colbench's tests if you touch it.

## Setup

**Data.** Build the parquet once; every row carries its GT tests in both
`reward_model.ground_truth` (the standard VERL reward channel) and
`extra_info.ground_truth` (read by the agent loop at rollout), so the same test
set drives mid-turn feedback and the final reward.

```bash
PYTHONPATH=$(pwd) python codecontest/preprocess_codecontests.py \
    --local_dir ~/data/codecontests
# quick slice for smoke runs:
PYTHONPATH=$(pwd) python codecontest/preprocess_codecontests.py \
    --local_dir /tmp/cc --max_train 24 --max_val 8
```

**The exec sidecar is not optional.** Untrusted model code must not `exec()`
inside a Ray rollout worker — a memory bomb or hang there takes the worker
down. `exec_client` refuses to fall back to in-process exec silently: with no
`CODECONTEST_EXEC_URL` set it grades every turn UNSOLVED and logs why. The
managed deployment starts the sidecar as a background process in the same
container (`start_exec_sidecar` in `xcloud_setup/entrypoint_common.sh`) and
blocks on `/health` before training.

For local dev only, opt in explicitly:

```bash
export CODECONTEST_ALLOW_INPROCESS=1     # dev/smoke ONLY
```

Sidecar knobs, read at **import** time so they must be set before it starts:

| Env var | Default | Meaning |
|---|---|---|
| `CODECONTEST_EXEC_URL` | unset | sidecar base URL; unset ⇒ refuse to grade |
| `CODECONTEST_EXEC_PORT` | 8088 | sidecar bind port |
| `CODECONTEST_EXEC_CONCURRENCY` | 64 (180 in the managed job) | global cap on concurrently-alive exec children |
| `CODECONTEST_EXEC_MEM_GB` | 2 (1 in the managed job) | per-exec address-space growth cap (`RLIMIT_AS`) |

## Train

Launches go through `launch.sh` (the CitC-side XManager wrapper). It builds the
image, submits the job, and runs `xcloud_setup/entrypoint.sh`, which downloads
data, resumes any existing checkpoint, starts the exec sidecar, and then execs
the training script.

```bash
launch.sh \
  --script ./codecontest/run_model_feedback_codecontest_grpo.sh \
  --model Qwen/Qwen2.5-14B \
  --exp_name model_feedback_codecontest_grpo_MT4_tis_clip_kl_finalturn \
  --max_assistant_turns 4 \
  --train_turns final_only
```

`--exp_name` is **the one thing you name per run.** With `--model` it forms the
stable identity `{model}_{exp_name}` that keys *all* storage — checkpoints,
tensorboard, eval dumps. It deliberately carries no script name, step, or
experiment id, so (a) a run and its eval resolve to the same GCS paths and
(b) relaunching the same `exp_name` **resumes** that run. There is no
train-side resume flag; resume is automatic from the latest step.

`--train_turns` selects the gradient-masking arm: `all` (baseline, every solver
turn), `final_only` (train only the last solver turn — clean credit), or
`upto_last_code`.

## Eval

Same launcher with `--eval_only True`. Pass the **same `--model` and
`--exp_name` as the training run** so the experiment identity — and hence the
checkpoint directory — resolves to the same place. The step is never inferred:
name it with `--eval_step`.

```bash
launch.sh \
  --eval_only True \
  --model Qwen/Qwen2.5-14B \
  --exp_name model_feedback_codecontest_grpo_MT4_tis_clip_kl_warmstart \
  --eval_step global_step_40 \
  --max_assistant_turns 4 \
  --eval_n_samples 8 \
  --eval_temperatures '0.0 0.8' \
  --feedback_mode model_feedback
```

`--eval_step` also accepts a space-separated list
(`'global_step_60 global_step_120'`) — all steps come from the same experiment
and run sequentially in one job, so data and the sidecar are set up once and a
failed step doesn't sink the rest. `base` is a valid step (evaluates the
unmodified base model).

`--feedback_mode` must match how the checkpoint was trained (`oracle` vs
`model_feedback`); it is tagged into the output filename so an oracle eval and
a model-feedback eval of the same checkpoint never collide. Each config writes
its own `..._turns<N>_n<K>_t<temp>_<mode>.json` conversation dump.

Reports pass@k per turn cutoff. `per_turn[0]["pass@k"]["1"]` is the
single-turn solve rate — the number that must match a matched-`n` single-turn
run's pass@1.

### Running eval directly

`validate_codecontest.py` is standalone and needs no trainer. Its wrapper reads
env knobs (`MODEL_PATH`, `VAL_FILE`, `N_SAMPLES`, `TEMPERATURES`,
`MAX_ASSISTANT_TURNS`, `FEEDBACK_MODE`, …):

```bash
MODEL_PATH=/path/to/merged_hf_checkpoint \
FEEDBACK_MODE=model_feedback N_SAMPLES=8 TEMPERATURES='0.0 0.8' \
    bash codecontest/run_validate_codecontest.sh
```

`--help` on the script itself prints copy-pasteable examples.

## Smoke test and tests

A tiny single-GPU end-to-end check that `main_ppo` loads the loop, runs
multi-turn SGLang rollouts through local exec, and completes a couple of
optimizer steps. It is not meant to learn anything — use a tiny model.

```bash
# inside the verl SGLang container, from the repo root
PYTHONPATH=$(pwd) bash codecontest/run_smoke_1gpu.sh
```

Its header carries the exact `singularity exec` line for FASRC.

Unit tests (CPU, no GPU, no Ray):

```bash
CODECONTEST_ALLOW_INPROCESS=1 python -m pytest codecontest -q \
    --ignore=codecontest/tests/test_code_refine_agent.py
```

The `--ignore` is required **outside** the container: that file imports
`verl.protocol` → `ray` at module level, so it fails at collection and takes
the whole run with it. Inside the container, drop the flag.

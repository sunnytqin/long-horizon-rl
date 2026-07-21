# ColBench spec-conditioned RL — Phase 1 (eval) done, Phase 2 (training) handoff

Phase 0 (`selfplay/`, spec authoring) and **Phase 1 (the spec-conditioned eval loop + env)** are
DONE and validated. Phase 2 = **wire the spec path into GRPO training** and run it. This doc is the
handoff. Read `selfplay/PHASE1_HANDOFF.md` first for the setting/rationale; this continues it.

---

## The setting (one paragraph)

Instead of conditioning the frozen user-simulator on the hidden GT function source (which lets it
leak the answer), condition it on a natural-language **spec** `{persona, scenario, requirements,
plot}` authored offline in Phase 0. **Grading is UNCHANGED** — objective GT code + `test_cases` via
the exec sidecar. The sim is a deliberately **imperfect** reward-shaping signal; the GT tests are
the only true reward. "Group = one spec": each task's spec is fixed and `rollout.n` rollouts share
it, so a GRPO group is N rollouts against the same authored user.

---

## What Phase 1 delivered (the foundation — do NOT rebuild)

A complete, CPU-tested, GPU-validated **spec path that runs entirely parallel to the GT path**
(GT files untouched except additive `templates.py` helpers). Files in `verl/colbench/`:

- **`templates.py`** (additive) — the shared text contract, used byte-identically by eval and
  (to-be-built) training:
  - `COLBENCH_SPEC_AGENT_SYSTEM_PROMPT` — solver prompt, **no "I WANT TO ANSWER:"**; propose the
    complete function in a ```python block; the USER ends the conversation.
  - `SPEC_SIM_SYSTEM_PROMPT` — the frozen sim's system prompt (persona/scenario/requirements/plot;
    **GT code NEVER injected**). Final tuned version: hard "code-on-the-table" gate before
    `[TERMINATE]`, plot-only correction, no-code / no-review, skill-calibrated feedback, brevity +
    answer-only-what's-asked (anti over-reveal). See "Sim-prompt state" below.
  - `build_spec_sim_messages(spec, messages) -> (system, user)`; `TERMINATE_MARKER="[TERMINATE]"`;
    `sim_terminated(reply)`; `contains_code(text)`; `extract_last_code(messages)`;
    `sim_wrote_code(reply)` (```-fence detector).
- **`env_spec.py`** — `ColBenchSpecUserSimEnv` (dataclass: problem/spec/ground_truth/test_cases/
  max_steps/reward_time_limit/sim_backend/**sim_max_tries=8**). `generate_user_turn` = spec-
  conditioned sim reply with **rejection sampling** on code (see state machine); `score` =
  `reward.grade` (identical to GT env). Exposes `last_sim_raw`, `last_sim_code_rejected`,
  `last_sim_code_reject_exhausted`. Also `make_openai_sim_backend(...)` (a REAL-OpenAI sim backend
  for comparison studies; schema-adaptive for the GPT-5 family; **no vLLM extras**). NO
  `is_answer`, NO code-leak/rejection machinery (leak impossible by construction).
- **`preprocess_colbench_spec.py`** — joins a specs JSONL to the raw InfoPO parquet by `index`,
  attaches `extra_info.spec` + `agent_name="colbench_spec_agent"` + the spec solver system prompt,
  keeps the grading payload. Only rows with a usable spec (parsed ok + non-empty requirements/plot).
- **`validate_colbench_spec.py`** + **`run_validate_spec_slurm.sh`** — the eval loop (the reference
  implementation of the termination state machine); dual solver backend (openai/sglang), dual sim
  backend (vllm/openai). Metrics: `terminated_by_counts`, `showed_code_rate`, `mean_code_proposals`,
  `false_terminate_rate`, `feedback_lift_pass_rate` (first-code vs final-code), `sim_code_rejected_*`.
  Auto-dumps `<out>.aborts.txt`.
- **`tests/test_env_spec.py`** — 16 CPU tests, incl. the inline `drive()` that PINS the termination
  state machine. `ping_openai_sim.py` — one-shot OpenAI-sim sanity ping.

### Validated result (1k, clean Qwen sim = the production sim)
`all_pass 0.543`, `mean_pass 0.670`, **70% user-terminated**, feedback lift small (+0.04→+0.10
after the brevity edits), code-writing rare (47 rejects / 3 aborts per 997). Qwen behaves like a
faithful ordinary user on termination; it is **verbose** (~4–6 sentences vs the "1–2" instruction)
— a Qwen-4B capability limit, logged as future sim work (see Open issues). A gpt-5.4-mini sim was
run for comparison (more fluent/brief but **over-persistent** — `code_cap` in ~73%, and it hit the
OpenAI 2M TPM rate limit → 18% of turns became "No response."; needs backoff if reused).

---

## THE TERMINATION STATE MACHINE (the contract the training agent loop MUST mirror)

Reference impl: `validate_colbench_spec.py::run_eval` and the pinned `tests/test_env_spec.py::drive`.
Per turn, per active trajectory:

1. **Solver generates.** If the turn contains code (`templates.contains_code`): set
   `last_code = templates.extract_last_code(dialogue)`, `code_proposals += 1`, `showed_code = True`;
   if this is the first code, also record `first_code`.
2. **Turn cap** (`turn == max_assistant_turns-1`, default 10) → stop; grade `last_code`
   (`terminated_by = "turn_cap"`, or `"no_code"` if none shown → reward 0).
3. **Code cap** (`code_proposals >= max_code_proposals`, **default 2**) → stop; grade `last_code`
   (`terminated_by = "code_cap"`).
4. **Else the sim replies** (`env.generate_user_turn`, which does the rejection sampling):
   - If the sim exhausted all `sim_max_tries` still writing code
     (`env.last_sim_code_reject_exhausted`) → stop; `terminated_by = "sim_code_reject"`; grade
     `last_code` (0 if none). Save the offending reply for inspection.
   - Elif `templates.sim_terminated(env.last_sim_raw)` → stop; grade `last_code`
     (`terminated_by = "user"`, or `"no_code"` + reward 0 if none shown).
   - Else append the sim reply (**mask=0**) and continue.

**Rejection sampling** (inside `generate_user_turn`): the sim is an ordinary user → must never paste
code. If a reply contains a ``` fence (`sim_wrote_code`), re-query up to `sim_max_tries` (8; temp>0
makes retries differ). If ALL tries write code → set the exhaustion flag (do NOT strip, do NOT
inject — stripping yields weird fragments). Track `sim_code_rejected` (discarded count).

**Reward** = GT `pass_rate` (unchanged). Grade once, at whatever stop. Reward 0 iff no code was ever
shown. Persuading the sim buys nothing → no reward hacking; the only cost of user-termination is
variance (`false_terminate_rate`).

**Masking**: solver turns mask=1, sim turns mask=0; `codecontest.masking.apply_train_turns_mask`
still applies (`train_turns` all vs final_only → final = the last solver code turn).

---

## Locked design decisions (do NOT re-litigate without reason)

- **User-driven termination** (sim emits `[TERMINATE]`), grade the last code shown, reward 0 if none.
- Termination is **plot-driven, not correctness-driven** — the sim never verifies correctness (it's
  an imperfect signal; GT tests are the only reward).
- **Two guardrails**: `max_assistant_turns=10`, **`max_code_proposals=2`** (the current default,
  reduced from 3 after eval).
- **Hard gate**: the sim may not `[TERMINATE]` until a complete ```python block has been shown.
- **No code-leak machinery** in the spec path (impossible; the sim never sees GT).
- **Rejection sampling, no strip**: code-writing sim reply → re-query ≤8×; all-fail → abort the
  conversation (`sim_code_reject`) + grade last code. Aborts are auto-tracked (`.aborts.txt`,
  `sim_code_reject_aborted`).
- **`SIM_MAX_TOKENS=256`** generation bound on each sim turn (replaced the old 400-char post-hoc
  slice that chopped mid-sentence). Read by both sim backends. **No mid-sentence truncation.**
- Solver proposes via a ```python block (no submit marker).
- Reuse existing config knobs; minimize new ones (see memory `feedback-minimize-new-configs`). The
  ONLY new knobs are `max_code_proposals` and `sim_max_tries`/`SIM_MAX_TOKENS`.

### Sim-prompt state (final, in `SPEC_SIM_SYSTEM_PROMPT`)
Curbs over-reveal (answer only what's asked, never hint at a hidden requirement list, reveal as
questions draw it out) and pushes brevity (1–2 sentences). Over-reveal dropped and feedback lift
rose after these edits; residual verbosity is a Qwen-4B limit, not a prompt bug.

---

## What Phase 2 must build (the RL wiring)

Mirror the GT path (`colbench_agent.py`, `run_colbench_grpo.sh`, `config/agent_loop_config.yaml`,
`entrypoint_colbench.sh`) with `*_spec` siblings. Keep the GT path intact.

1. **`colbench_spec_agent.py`** — `@register("colbench_spec_agent")`
   `class ColBenchSpecAgentLoop(AgentLoopBase)`. Copy `ColBenchAgentLoop`'s structure
   (budget/overflow bookkeeping, weights-version handling, response-mask, `AgentLoopOutput`,
   `apply_train_turns_mask`) and **replace the run loop with the termination state machine above**.
   - Read `extra_info.spec`; build `ColBenchSpecUserSimEnv` (sim_backend = the frozen-sim HTTP call,
     reuse `env.openai_sim_backend`).
   - New knob `+colbench.max_code_proposals` (default 2). `extra_fields`: `terminated_by`,
     `code_proposals`, `showed_code`, `sim_code_rejected`, `first_code_pass_rate` (optional).
     DROP the GT path's `sim_failed`/`sim_reject_tries`.
   - **GOTCHA**: a custom AgentLoop MUST propagate `min_global_steps`/`max_global_steps` in
     `extra_fields` or the trainer crashes — the GT `ColBenchAgentLoop` already does this
     (`colbench_agent.py:154-183,304`); copy it verbatim. See memory
     `reference-custom-agent-loop-global-steps`.
2. **`config/agent_loop_config_spec.yaml`** — register `colbench_spec_agent` → its `_target_`.
   (Or add the entry to the existing `agent_loop_config.yaml`; per-row `extra_info.agent_name`
   already routes to it. `reward_manager=naive` forwards `AgentLoopOutput.reward_score`.)
3. **`run_colbench_grpo_spec.sh`** — the train script. Reuse `run_colbench_grpo.sh` wholesale;
   change: dataset → the spec parquet, agent config → the spec one, add `max_code_proposals=2`.
   Keep GRPO/PPO knobs comparable to the GT run for A/B. `rollout.n` = specs-per-group.
4. **Training entrypoint — TARGET xmanager/GCS FIRST (user's call, 2026-07-21).** The user has
   xmanager/GCS access again and wants the RL run on GCP, mirroring the GT path's
   `entrypoint_colbench.sh` (GCS download of model/data/specs → GPU split: solver on first N GPUs,
   frozen sim on last SIM_TP GPUs → exec-sidecar startup `python -m codecontest.exec_server &`,
   `CODECONTEST_EXEC_URL=http://127.0.0.1:8088` → the train script). Add a `*_spec` entrypoint/
   launch path (the spec parquet + specs must be uploaded to GCS; the sim conditioning is the only
   change vs the GT entrypoint). A Harvard/Slurm container entrypoint (same GPU split, minus the
   GCS download since netscratch already has everything) is a SECONDARY fallback if GCP is
   unavailable — do not build it first.
5. **Spec-path eval in training** — in-training val on a spec parquet (small, e.g. the 1k or a
   held-out slice) using the same env; the offline `validate_colbench_spec.py` already exists for
   deeper eval.

---

## Data

**Grading source (unchanged):** raw InfoPO parquet
`/n/home05/sqin/long-horizon-RL/InfoPO/data/colbench_code/train.parquet` — specs join by `index`
to recover GT code + `test_cases`.

**Validated 1k (use for the FIRST RL run):** `~/data/colbench_spec/selfplay_1k.parquet` (997 rows)
+ `selfplay_cond30.parquet` (30). Built + validated in Phase 1.

**FULL 10k specs — READY, preprocessing STILL NEEDED.** The full self-play train specs now exist:
`/n/netscratch/dam_lab/Lab/sqin/colbench_specs/specs/train.selfplay.plot.jsonl` — **10000 rows,
9986 usable** (parsed ok + non-empty requirements/plot), regenerated 2026-07-20. To train on the
full set, run the (already-built) preprocessor once:

```bash
cd verl && python colbench/preprocess_colbench_spec.py \
  --raw_parquet /n/home05/sqin/long-horizon-RL/InfoPO/data/colbench_code/train.parquet \
  --specs_jsonl /n/netscratch/dam_lab/Lab/sqin/colbench_specs/specs/train.selfplay.plot.jsonl \
  --out ~/data/colbench_spec/selfplay_10k.parquet
# → ~9986 rows, VERL schema, agent_name=colbench_spec_agent
```

Plan: bring the RL loop up on the **1k** (fast, already validated), then switch the dataset to the
**10k** parquet for the real run. (A `strong.plot.jsonl`, 1000 rows / gpt-5.4-mini, exists for a
teacher-vs-selfplay A/B if ever wanted.) Test-set specs not generated yet (see Phase 1 handoff).

---

## Environment (FASRC/Kempner) — CRITICAL

- **Spec gen / eval (light):** conda env `/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf`
  (py3.10, vllm, openai). This is what `run_validate_spec_slurm.sh` uses. CANNOT import
  `verl.experimental.agent_loop` (needs py≥3.11).
- **RL TRAINING (heavy): needs the VERL/SGLang CONTAINER** (`verlai/verl:sgl0512.dev2`, py3.12),
  NOT the conda env. Singularity sandbox rebuild recipe + exact exec rules (no `--cleanenv`, no
  `bash -lc`) are in `selfplay/PHASE1_HANDOFF.md` and memory `reference-verl-sglang-container`.
- Model = `Qwen/Qwen3-4B-Instruct-2507` at
  `/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen3-4B-Instruct-2507`.
- Exec sidecar host-RAM OOM is a known trap — tune RLIMIT/concurrency, don't cut `max_gt_test`
  (memory `reference-codecontest-exec-sandbox`).
- Stability lesson from the CodeContests GRPO run: `KL=0.01` fixed an entropy explosion
  (memory `project-codecontest-rl-stability-plan`); start the spec run with the same guardrail.

---

## Open issues / future work (not blockers for the first run)

- **Sim verbosity (Qwen-4B limit).** The sim writes ~4–6 sentences vs the "1–2" instruction and is
  somewhat over-revealing; the prompt edits reduced but did not eliminate this. The real fix is a
  stronger/finetuned sim — and this is where the intended **co-evolution** (improve both solver AND
  user-sim during RL) comes in. Track it; don't block on it.
- **No scored fidelity metric yet** — fidelity is eyeballed + mechanical proxies (rejections,
  termination mix, lift). An LLM-judge rubric (plot-adherence / persona / no-code / realistic
  termination) was deliberately deferred until co-evolution is designed (use an INDEPENDENT judge,
  not the sim's own family).
- **GPT sim rate-limit** — `make_openai_sim_backend` fails fast to "No response." on a 429 (no
  backoff). If GPT is ever used as a reference sim at scale, add retry-with-backoff + concurrency
  throttle. Detection is easy: count `"No response."` turns + grep the slurm log for `RateLimit`.

---

## Code map (spec path, `verl/colbench/`)
- `templates.py` (additive spec helpers) · `env_spec.py` · `preprocess_colbench_spec.py` ·
  `validate_colbench_spec.py` · `run_validate_spec_slurm.sh` · `ping_openai_sim.py` ·
  `tests/test_env_spec.py` (16 tests: `CODECONTEST_ALLOW_INPROCESS=1 pytest colbench/tests/test_env_spec.py`).
- GT path to mirror: `colbench_agent.py` · `run_colbench_grpo.sh` · `validate_colbench.py` ·
  `config/agent_loop_config.yaml` · `entrypoint_colbench.sh` / `entrypoint_eval_colbench.sh`.

## First steps for the Phase-2 session
1. Preprocess the 10k (command above) — or start on the validated 1k.
2. Write `colbench_spec_agent.py` (copy `ColBenchAgentLoop`, swap in the termination state machine;
   propagate global_steps). Add its `config/agent_loop_config_spec.yaml` entry.
3. `run_colbench_grpo_spec.sh` + the **xmanager/GCS** entrypoint (mirror `entrypoint_colbench.sh`;
   upload the spec parquet + specs to GCS). Harvard/Slurm container entrypoint only as fallback.
4. Smoke on 1k (few steps, KL=0.01), confirm reward/masking/metrics flow, then scale to 10k.

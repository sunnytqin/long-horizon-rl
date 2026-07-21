# ColBench self-play specs — Phase 0 done, Phase 1 handoff

This subpackage (`verl/colbench/selfplay/`) implements **Phase 0** of the new ColBench setting:
generate, per task, a **self-authored spec** that a frozen user-simulator will later be
conditioned on — instead of leaking the hidden GT code to the simulator. Phase 0 is complete
and validated. Phase 1 (the spec-conditioned simulator + RL training) is a **separate task**;
this doc is the handoff.

---

## The setting (why this exists)

Today's ColBench RL conditions the frozen user-simulator on the hidden **GT function source**,
so the sim can leak the answer (`detect_code_leak`/`sim_reject` are band-aids). New setting:
the sim is conditioned on a natural-language **spec** authored offline. Grading is UNCHANGED —
still the objective GT code + `test_cases` (functional-equivalence via the exec sidecar). The
spec only replaces *what the sim conditions on*.

"Group = one spec": each task's spec is fixed, and `rollout.n` rollouts share it, so a GRPO
group is N rollouts against the same authored user.

---

## Phase 0 outcome (DONE, validated)

**Design (locked after several iterations — see "Decisions locked" below):** each spec is

```
persona   {who, domain, python_skill(non-coder|hobbyist|analyst|engineer), communication_style}
scenario  2-4 sentence situation
requirements   the COMPLETE intent the sim must eventually convey (exhaustive, exact constants,
               words not code), written in the THIRD PERSON as the user's needs ("The user needs
               the function to...", "The user wants..."). The sim SEES this.
plot      a short, tailored DIRECTION (1-3 sentences, NOT a script) written as an INSTRUCTION TO
          THE USER-SIMULATOR ("the user would...", "the user first...") for how THIS conversation
          naturally isn't clear up front: which single detail is initially missing/wrong/vague,
          and — CONDITIONAL on the assistant (never assume the assistant asks/does anything) —
          how it surfaces: "IF the assistant asks X, the user would...; if not, it stays hidden"
          / "IF the assistant shows code doing Y, the user would correct...". Mechanisms may mix:
          reveal-only-if-asked (the gated, highest-training-value case), correct-on-seeing-code,
          or volunteer-unprompted. The user NEVER runs/tests code. The sim IMPROVISES actual
          turns from requirements+plot+persona — we do NOT script turns.
```

**Validation (1000-task train slice; `requirements` full-spec solve rate = does a solver
reconstruct GT from the requirements alone, graded on the SAME local Qwen3-4B solver; the plot
does NOT affect this — it's graded on requirements only):**

| author | requirements faithfulness | plot mechanism diversity | parsed ok |
|---|---|---|---|
| **self-play Qwen3-4B** | **0.777** (mean_pass 0.862) | WIDER (react + ask + gated) | 997/1000 |
| **strong gpt-5.4-mini** | **0.841** (mean_pass 0.873) | NARROWER (mostly "ask") | 997/1000 |

- FINAL numbers, with the locked prompt (conditional plots, third-person requirements). Teacher
  is ~6pts more faithful (tighter formulas, fewer garbles) — real, not noise. But self-play
  plots are more mechanism-diverse (the teacher makes that WORSE), and 0.777 is a solid floor.
- **Decision: use SELF-PLAY (no teacher, no fallback)** — for the research principle (the model
  authors its own environment) and plot diversity, NOT because the numbers are equal. Both 1k
  datasets are saved, so Phase 1 can A/B self-play vs teacher specs on the same tasks if desired.
- Plots are CONDITIONAL on the assistant (never assume it asks) and written as sim instructions
  ("the user would..."); the simulator can never run/test code. Self-play naturally produces the
  gated "won't reveal unless asked" behavior (the highest-training-value case); strong funnels
  more toward "if the assistant asks".

**Data (Phase-0 deliverable)** in `/n/netscratch/dam_lab/Lab/sqin/colbench_specs/specs/`:
- `train.selfplay.plot.jsonl` — **FULL TRAIN SET, 10000 rows** (indices 0-9999, 9986 parsed ok),
  self-play, locked prompt. **This is the Phase-1 training data.** (Eval pipeline already
  validated on it.)
- `test_small.selfplay.plot.jsonl` — **2000 rows** (indices 0-1999, 1998 parsed ok), self-play,
  eval-set faithfulness 0.779. Authored on `InfoPO/data/colbench_code/test_small.parquet` (=
  first 2000 rows of test.parquet, the deterministic `--val_small 2000` slice), so its `index`
  aligns with the in-training val set. **This is the Phase-1 spec-conditioned eval set.**
- `train.strong.plot.jsonl` — **1000 rows**, gpt-5.4-mini (comparison only; NOT full set).
- To add if needed: full-test specs (`test.parquet`, 10000 rows) — same command with
  `--data_file .../test.parquet` (no --max_rows). Not generated yet.
Each line is one JSON object with keys
`index, backend, mode, problem_description, persona{who,domain,python_skill,communication_style},
scenario, requirements, plot, ok, raw`. `index` = row position in the source parquet
`/n/home05/sqin/long-horizon-RL/InfoPO/data/colbench_code/train.parquet` (raw InfoPO schema) —
join on it to recover GT code + `test_cases` for reward. Earlier prompt iterations are archived
as `*.r1/r2/r3.jsonl` and `*.v1_episode/v2/v3.jsonl` (ignore; superseded).

---

## What Phase 1 must build

The spec-conditioned simulator + the training/eval wiring. Keep the current GT-code path intact;
add the spec path alongside (clean split, as in Phase 0).

1. **Spec-conditioned sim prompt/env.** New sim system prompt conditioned on
   `persona + scenario + requirements + plot` (NOT GT code). The sim must: (a) stay in persona,
   (b) reveal `requirements` gradually, (c) follow the `plot` — critically, for the
   "withhold until asked" plots, NOT volunteer the gated detail unless the solver asks. Reuse
   `env.ColBenchUserSimEnv.generate_user_turn` seam; swap the conditioning source. The sim never
   runs code (it has no executor) — plots already respect this.
2. **Spec-augmented preprocess.** `preprocess_colbench.py` (or a sibling) must attach the spec to
   each row's `extra_info` so the agent loop/env can read it, while `reward_model`/`test_cases`
   stay for grading. Only rows with a spec (the 1k) are used for the spec-path run.
3. **Agent-loop / run-script / eval wiring.** `colbench_agent.py`, `run_colbench_grpo.sh`,
   `validate_colbench.py`/`entrypoint_eval_colbench.sh` — add a flag to select the spec path.
   The dialogue-rollout solve rate then comes "for free" and gives the extraction-through-
   -dialogue gap vs the 0.67 full-spec ceiling.
4. **(Optional) K>1 specs per GT** for larger/among-spec diversity; not needed for the first run.

The 1k specs are the test set for bringing Phase 1 up.

---

## Running on the Harvard (FASRC/Kempner) cluster — IMPORTANT for future sessions

Historically RL training ran on **xmanager/GCP** (`entrypoint.py`). The user is temporarily
logged out of xmanager and is running on Harvard/Slurm; we will likely do **some training on
Harvard too**. Two very different environments:

### Spec generation / eval (Phase 0, light) — conda env, NO container
- Env: `/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf` (py3.10; vllm 0.10.1, openai, pandas,
  transformers). Durable (holylabs, not scratch).
- Harness: `verl/colbench/selfplay/run_specs_slurm.sh` — one sbatch (`-p kempner_h100`,
  `--account kempner_dam_lab`, 1 GPU): `conda activate openrlhf` → serve the model with
  `vllm.entrypoints.openai.api_server` → author (`generate_specs`) → diagnose (`diagnose_specs`,
  in-process grading via `CODECONTEST_ALLOW_INPROCESS=1`). Config via env: MODE(static|plot),
  BACKEND(selfplay|strong), MAX_ROWS, GEN_VENDOR(vllm|openai), etc.
- vllm vs sglang is irrelevant here (plain OpenAI-API client calls). Model =
  `Qwen/Qwen3-4B-Instruct-2507` (the training model), weights at
  `/n/netscratch/dam_lab/Lab/sqin/models/qwen/models--Qwen--Qwen3-4B-Instruct-2507`.
- OpenAI teacher (if ever needed): `GEN_VENDOR=openai`, key file `~/.openai_key` (chmod 600).
  Compute nodes may be firewalled from the internet → author on the LOGIN node (has internet),
  then the GPU job only diagnoses (generate_specs is resumable and skips the done authoring).

### RL TRAINING (Phase 1, heavy) — needs the VERL/SGLang CONTAINER, not the conda env
- The `openrlhf` conda env is py3.10 and CANNOT import `verl.experimental.agent_loop`
  (VERL needs py≥3.11 / `enum.StrEnum`). Training must run in the **VERL/SGLang container**
  (`verlai/verl:sgl0512.dev2`, py3.12).
- The Singularity sandbox at `/n/netscratch/dam_lab/Lab/sqin/docker_images/verl-sgl0512-dev2`
  was **PURGED by netscratch retention** (dangling symlinks, missing `/usr/bin/python`). The
  download cache `sing_cache` (16G) survived, so a **download-free rebuild** works:
  `SINGULARITY_DISABLE_CACHE=false SINGULARITY_CACHEDIR=.../sing_cache SINGULARITY_TMPDIR=/scratch
  singularity build --force --sandbox <dest> docker://verlai/verl:sgl0512.dev2` — do it in an
  **sbatch** job (extracting the 37G rootfs onto netscratch NFS is slow, ~40 min).
  Exec recipe: `singularity exec --nv --bind <repo> --env PYTHONPATH=<repo> <sandbox> python ...`
  — NEVER `--cleanenv`, NEVER `bash -lc` (see memory `reference-verl-sglang-container`).
- Multi-GPU single-node sbatch on `kempner_h100`/`kempner` (4 GPUs): the container runs the
  ColBench GRPO train script; the solver serves on the first N GPUs, the frozen sim on the last
  SIM_TP GPUs (same split as `entrypoint_colbench.sh`). Reward exec: start
  `python -m codecontest.exec_server &` inside the container and export
  `CODECONTEST_EXEC_URL=http://127.0.0.1:8088` (as `entrypoint.sh` does), or fall back to
  in-process (`CODECONTEST_ALLOW_INPROCESS=1`) for small jobs.
- The xmanager `entrypoint.sh` downloads model+data from GCS then runs the train script. On
  Harvard, model+data+specs are already on netscratch — so a Slurm entrypoint is simpler: just
  bind them and run the train script in the container. A Harvard training entrypoint does NOT
  exist yet — Phase 1 should add a thin sbatch wrapper mirroring `entrypoint_colbench.sh`'s GPU
  split + sidecar startup, minus the GCS download.

---

## Decisions locked (do not re-litigate in Phase 1 without reason)

- Spec = `requirements` + `plot` (+ persona/scenario). NOT a verbatim turn-by-turn script — a
  turn script over-controls the sim and kills rollout diversity.
- The sim SEES `requirements` (the substance) and follows the `plot` (the arc). The plot is a
  DIRECTION, not dialogue.
- Reward stays GT code + `test_cases`, untouched. The spec only changes sim conditioning.
- Self-play only (frozen base authors). NO teacher, NO fallback (violates self-play).
- The user-simulator CANNOT run/test code — plots must surface gaps via conversation only
  (asking / reacting to proposed code / remembering / withholding-until-asked), never testing.
- `requirements` must stay exhaustive (every exact constant/formula, params-as-params) — this is
  what keeps the full-spec faithfulness at ~0.67; do not soften it for the sake of the plot.

## Code map (`verl/colbench/selfplay/`)
- `spec_templates.py` — PLOT_AUTHOR_*/SPEC_AUTHOR_* prompts, FULL_SPEC_SOLVER_*, parsers, builders.
- `generate_specs.py` — CLI, `--mode static|plot`, `--backend selfplay|strong`, `--gen_vendor
  vllm|openai`, resumable.
- `diagnose_specs.py` — CLI, full-spec (requirements) solve-rate diagnostic.
- `llm_client.py` — `ChatEndpoint` (vllm + vanilla-OpenAI vendors).
- `dataio.py` — parquet row → task (both raw InfoPO + preprocessed schemas), JSONL cache helpers.
- `run_specs_slurm.sh` — the FASRC sbatch harness.
- `tests/` — 24 CPU tests (`CODECONTEST_ALLOW_INPROCESS=1 pytest colbench/selfplay/tests`).

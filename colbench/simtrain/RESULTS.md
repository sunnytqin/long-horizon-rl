# Training the ColBench user-simulator: results

**Status: the PoC answered its question.** Training the user-simulator with
LLM-judge-selected SFT **closes the code-leak channel in RL almost entirely**
(peak leak 0.459 -> 0.004, a ~100x reduction) at the cost of a lower reward
ceiling, and **delays the entropy collapse by ~100 steps without preventing it**.

Dates: built 2026-09-09, evaluated and run 2026-09-10/11.
Plan: `~/.claude/plans/let-s-make-a-plan-iterative-kettle.md`.

---

## 1. The question

On the ColBench GT (non-spec) path the frozen user-simulator sees the hidden
ground-truth function, and the solver learns to extract it instead of solving the
task. Three arms had already shown this is an **equilibrium, not a prompting
defect** -- the solver receives gradient and a frozen sim never does -- so the
fix has to change the sim, not its prompt.

This PoC asks: does BoN-SFT on judge-selected simulator turns produce a
measurably better simulator, and does training an assistant against it reduce
leakage?

## 2. The headline: the GRPO comparison

Two runs, identical config and base init, differing **only** in the frozen sim.

| | baseline (frozen S_0) | S_1' (SFT sim) |
|---|---|---|
| job | 45772513 | 45916752 |
| wandb | `qwen3_4b_nonspec_role_restraint` | `qwen3_4b_nonspec_role_restraint_simsft_s1p` |

```
        BASELINE (frozen S_0)      S_1' SFT sim
step   reward   leak  turns |  reward   leak  turns
   0   0.458  0.022  1.272 |  0.419  0.003  1.395
  60   0.567  0.092  2.308 |  0.471  0.002  1.954
 100   0.648  0.246  2.333 |  0.533  0.002  3.296
 140   0.746  0.408  2.378 |  0.560  0.002  3.042
 180   0.833  0.453  2.142 |  0.522  0.000  2.110
 220   0.865  0.459  2.208 |  0.590  0.000  3.101
 260   0.875  0.389  2.960 |  0.576  0.001  3.934
 280   0.002  0.094  8.985 |  0.584  0.002  3.084   <- baseline COLLAPSE
 300     -      -      -   |  0.614  0.002  5.094   <- S_1' peak reward
 360     -      -      -   |  0.600  0.000  3.878
 380     -      -      -   |  0.006  0.004  7.535   <- S_1' COLLAPSE
```

**Three findings.**

1. **The leak channel is closed, not merely narrowed.** The baseline's leak rises
   monotonically with reward to a peak of **0.459**. Against S_1' it never exceeds
   **0.004** at any step. At *matched reward* (~0.60) the baseline was leaking
   0.143 (step 80) against S_1''s 0.002 (step 300) -- a ~70x difference that is
   not a matter of degree.

2. **The reward ceiling drops, 0.875 -> 0.614.** This is the expected cost and it
   is the point: reward is oracle-graded, so the assistant genuinely solves fewer
   tasks once it can no longer extract the answer. `answered_at_turn` also rises
   (2.1-2.9 -> 3.0-5.1), i.e. the assistant asks more questions -- the intended
   behaviour against a simulator that will not just hand over the spec.

3. **The collapse is delayed ~100 steps but the mechanism is unchanged.** Both
   runs die the same way -- entropy explosion, then `group_reward_std/zero_frac`
   -> 1:

   ```
   step   baseline entropy | S_1' entropy
    200      0.409         |   0.304
    220      0.579         |   0.360
    240      1.012         |   0.320
    260      5.894         |   0.504
    280      5.043 (dead)  |   0.451
    340        -           |   0.513
    360        -           |   2.046
    380        -           |   8.716 (dead)
   ```

   Entropy also grows *more slowly* against S_1' (0.360 vs 0.579 at step 220) and
   `zero_frac` stays lower and flatter (0.39-0.51 vs 0.41-0.70 drifting up). So
   the better sim buys headroom, but **the collapse is a separate defect** --
   consistent with the prior finding that both GT arms collapse at ~270-280 and
   KL=0.01 does not prevent it. Do not attribute it to the sim.

**Caveat on interpretation.** A lower reward ceiling is only good news if the
assistant is failing *honestly* rather than being starved of information it
legitimately needs. The static eval below says S_1' releases less ground truth
per draw AND is no less truthful, which supports the honest reading -- but
"is the remaining 0.61 reward earned by real problem-solving?" is not directly
measured here.

## 3. The simulator itself (static eval, 614 held-out prefixes)

Held-out = 500 tasks of `test_small.fence.parquet`, **zero** ground-truth overlap
with the 10k training tasks (`task_id` is file-local -- see gotchas). Same base
partner, same `role_restraint` prompt, byte-identical prompts to every arm, K=8.

**Measured on the TRUE AVERAGE DRAW** (weighted by `dup_counts`; n = 614 x 8 =
4,912 per arm). See gotchas for why this matters.

| axis | S_0 base | S_1 (2,057 rows) | S_1' (3,198 rows) | verdict |
|---|---|---|---|---|
| code leak (programmatic) | 0.0904 | 0.0397 | 0.0454 | **improved**, p<0.0001 |
| over-release (judge-free) | 0.3730 | 0.3402 | 0.3154 | **improved**, p=0.0001 |
| new GT tokens / draw | 1.16 | 0.94 | 0.77 | **improved** |
| untrue (wrong+unsure) | 0.4483 | 0.4395 | not yet judged | **null**, p=0.47 |
| calibration (mean) | 2.796 | 2.663 | not yet judged | **null**, p=0.74 |
| in_character (mean) | 3.701 | 3.779 | not yet judged | slightly up |
| uniques per 8 draws | 5.07 | 7.23 | 6.76 | more diverse |

Paired per-prefix sign tests, S_1' vs S_0: code leak 77 better / 32 worse
(p<0.0001); over-release 141 / 83 (p=0.0001). S_1' vs S_1: code leak 39/45
(p=0.59, saturated) but over-release 134/102 (**p=0.043**) -- the extra data
helped on release only.

**Truthfulness is unchanged, and this was checked twice.** gpt-5.4-mini weighted
gives p=0.47; an independent **blind** hand-judge of 50 turn-0 pairs (scored
before the arm key was revealed; 8 better / 10 worse / 32 tied) gives p=0.81,
with both arms wrong ~20%. `all_untrue` remains the largest single drop reason
(28% of prefixes).

**Why the static 2x became a 100x in RL.** The static number is the leak rate of
a single unadversarial draw. The RL number is the outcome of an *optimiser*
hunting for the channel. Halving the per-draw rate -- and specifically deleting
the sim's **modal** behaviour of transcribing the code verbatim -- removed the
thing the assistant was finding and amplifying. That gap between a 2x static
improvement and a ~100x equilibrium improvement is the most important result
here, and it is why static sim metrics should be read as a *lower bound* on the
RL effect.

## 4. What the pipeline is

`colbench/simtrain/`, sibling of `colbench/selfplay/`. Turn-level (contextual
bandit), because `templates.str_dialogue_history` renders the whole dialogue into
ONE user message under a constant system prompt -- so every sim turn is an
independent single-turn generation and a `(rendered prompt, reply)` pair is the
complete state. Every stage after collection is a pure JSONL->JSONL function.

| stage | file | role |
|---|---|---|
| 1 | `collect_prefixes.py` | partner-vs-sim episodes; **materialises** the rendered sim prompt per user turn |
| 2 | `collect_candidates.py` | K draws per prefix, dedup (multiplicity kept in `dup_counts`) |
| 3 | `judge_candidates.py` | the three-stage judge (below) |
| 4 | `build_sft_parquet.py` + `prescan_sft.py` | `[system, sim_user, assistant]` rows; pre-scan asserts the loss mask |
| 5 | `../run_sim_sft.sh` + `slurm_setup/sft_simtrain.sbatch` | one node, `torchrun --standalone` |
| 6 | `eval_sim_sft.py` | paired S_0-vs-S_1 report |

**The judge is three stages, and the shape matters** (rubric r6 / harness h2):

1. **code veto -- programmatic** (`templates.detect_code_leak`, no API call).
   Judge and regex agreed 217/217 under r2, so paying a model for it was pointless.
2. **truth veto -- its own focused call** with evidence required (quote the reply
   span + the line of code). Fails CLOSED on a parse error. 9 numbered rules,
   each written from a real observed failure.
3. **rank** the survivors on volunteering / calibration / in_character.

`fidelity` was **deleted** from the ranker: a 4 there would outvote a stage-2
veto, which was exactly the r3 failure.

**The key conceptual rule (user-supplied), which the rubric is built around:**

- **DERIVABLE** content -- logic or structure a competent programmer could work
  out. Restraint applies here.
- **ARBITRARY** content -- facts that exist only in the hidden information (a
  hardcoded list, a magic threshold, exact key names). **If the agent asks for it
  specifically it MUST be supplied, and supplied in full** -- naming three of
  fourteen platforms is not a small answer, it is a *wrong* answer.

## 5. Gotchas (each cost real time)

**Measurement**

- **Weight sim metrics by `dup_counts`.** `candidates` is post-dedup; averaging
  over *distinct* replies silently reweights toward the sim's RARE outputs. This
  produced two wrong conclusions before it was caught: it understated the
  code-leak win (S_0 0.0796 -> 0.0904 once weighted) and *manufactured* a
  truthfulness win that vanishes under weighting (S_0 wrong 0.3465 -> 0.3060).
  Both because **S_0's modal reply is a verbatim transcription of the hidden
  code, which is simultaneously its leakiest and its most accurate behaviour.**
- **Never average each arm over its own kept set.** `eval_sim_sft.py` pairs on
  prefixes kept in BOTH arms, because a keep-rate change alone manufactures a
  difference. The BoN-selected over-release number moves *opposite* to the
  pool number (0.311 -> 0.363 selected vs 0.373 -> 0.340 pooled) -- selection
  flatters S_0 precisely because S_0 needs more filtering.
- **`task_id` is FILE-LOCAL.** Checking train/eval overlap by it reports 485 of
  487 "contaminated" when ground-truth overlap is **0**. Use `ground_truth`.
- **`volunteering` is miscalibrated -- do not use it, do not floor it.** It
  scores how much information a reply CONTAINS, not unrequested content:
  `3248-0-0` answers exactly the two questions asked and nothing else, and scored
  **0**. A `vol>=3` floor would delete 449 mostly-good multi-part answers and
  select for evasion -- the r2 failure mode, whose optimum was a polite refusal.
  Fourth dimension found miscalibrated, after `responsiveness` (scored 4 on 100%),
  `fidelity` (never executed) and `information_release` (licensed full dumps).
- **`sim_leak_frac` under-counts by ~4.9%.** `detect_code_leak` detector (C) --
  expressions over the GT's own identifiers, e.g. a body handed over as a
  backticked expression -- is OPT-IN (`expr_over_gt_names=True`), used by
  simtrain stage 1 only so training-run numbers stay comparable. Every historical
  leak curve is a lower bound.

**Operational**

- **SFT defaults are duplicated across two layers.** `sft_simtrain.sbatch` passes
  `--env X="${X:-...}"`, which OVERRIDES the default in `run_sim_sft.sh`. Fixing
  only the inner file does nothing; that silently re-ran 3 epochs after the
  default had "been" changed to 1.
- **3 epochs overfits and the TRAIN curve hides it.** Train loss falls 0.30 ->
  0.17 -> 0.09 across epochs while val rises 0.31 -> 0.36 -> 0.45. Use ONE epoch.
  Within epoch 1 val is pure noise (0.26-0.35, no trend, and two runs on
  different data sizes converge) -- so take the epoch-1 END, never a val argmin.
- **The loss curve cannot tell you whether the SFT worked.** Targets are the base
  model's OWN BoN samples, so initial loss is already ~0.11 (ppl 1.12). Verified
  not a masking bug: verbatim-in-prompt on only 19/2057 rows, though ~81% of
  target *tokens* appear in the prompt (unavoidable -- the sim answers questions
  about code in its own context). Judge behaviour, never NLL.
- **`latest_checkpointed_iteration.txt` points at the LAST save, not the best.**
  Name the step explicitly.
- **A verl job can report a nonzero exit having trained fine** -- check for
  `Final validation metrics` first. (One "FAILED" run only failed because the
  script was edited mid-run; bash reads scripts by byte offset.)
- **Two eval arms cannot share a `RUN_TAG`.** `PREFIX_FILE`/`CAND_FILE` in
  `run_collect_slurm.sh` are plain assignments, NOT overridable, so arm 2
  silently overwrites arm 1's candidates. Give each arm its own `RUN_TAG` and
  verify with `sha256sum` that both arms' prefixes are byte-identical.

## 6. Artefacts

```
$SIMTRAIN_ROOT = /n/netscratch/barak_lab/Lab/sqin/verl_runs/colbench_simtrain

base_partner/                     12,819 prefixes + candidates (10k tasks, BASE partner)
  judged.train.fence.c1.r6.jsonl  7,825 judged  (4,994 remain; ~$8, resumable)
  sft_7825/                       3,440 rows -> 3,198 train / 242 val
eval_heldout/                     614 held-out prefixes + S_0 candidates + judged
eval_heldout_s1/                  S_1  candidates + judged   (same prefixes)
eval_heldout_s1p/                 S_1' candidates (judged: PENDING, ~$1.50)
sft_runs/sim_sft_r6_5k_1ep/       S_1  -> ckpt/global_step_64/huggingface
sft_runs/sim_sft_r6_7825_1ep/     S_1' -> ckpt/global_step_100/huggingface
```

Both sims are registered in `slurm_setup/paths.sh` as `local/sim-sft-r6-5k` and
`local/sim-sft-r6-7825`, usable directly as `--sim_model`.

The **1,286 hacker prefixes** (collected under the trained `global_step_200`
partner before the reframe) are kept as a held-out robustness probe: does S_1
resist a trap it never trained on? Not yet run.

## 7. Why the partner is the BASE model

Training the sim against a partner already trained to hack it is ill-posed as the
FIRST step of alternating optimisation: it wants S_1 = BR(A_0), then A_1 = BR(S_1).
Measured, and the switch cost almost nothing: **90%** of turn-1 agent turns under
the trained partner contain its learned confirm-then-offer trap vs **0.9%** under
the base partner, and the sim's code-leak rate falls 0.404 -> 0.034 on the
partner swap alone. Turn-0 prefixes are near partner-invariant and the turn-1
hacker prefixes yielded **zero** usable targets, so no training signal was lost.

## 8. Open

- **Judge S_1' under gpt-5.4-mini** on the 614 eval prefixes (~$1.50) so its
  truthfulness/calibration sit in the same table as S_0 and S_1. Expectation
  after the S_1 result: no gain.
- **Truthfulness is untouched by this pipeline and probably cannot be fixed by
  it.** BoN can only amplify behaviour already in the base pool, and the truth
  veto kills ~46% of candidates, so the pool is thin on truthful-and-restrained
  replies. A sim that confidently confirms a wrong spec makes an episode
  unwinnable exactly as a leak does. Needs a different intervention.
- **The entropy collapse** (~280 baseline, ~380 against S_1') is unexplained and
  unrelated to the sim. Suspects on record: rejection sampling off, and the
  fenced submit protocol.
- **The confirmation channel.** The rubric scores confirming the agent's own
  correct guess as 4 ("the agent earned it"). Under the base partner the agent
  covers only 8.3% of GT tokens in its own turn, so this is not currently
  dominant -- but it was inherited rather than decided, and matters more as A_k
  improves.
- **Two stage-2 rules under-fire** (~11% of rows): partial arbitrary data
  (`123-0-0`) and counter-questions the GT settles (`113-0-0`).
- **Second judge on the ranking stage.** Only the truth axis has had an
  independent check.

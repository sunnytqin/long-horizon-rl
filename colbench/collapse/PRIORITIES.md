# Entropy-collapse investigation priorities (2026-09-18)

The next questions are (1) whether collapse persists when training against the
235B simulator, and (2) how the entropy coefficient changes learning with the
SFT simulator. Defer covariance repair, sampler–actor mismatch, and the
long-conversation hypothesis until these are established. This supersedes the
ordering of next steps in RESULTS.md §9.

## Existing entropy-coefficient comparison

Read from the exact run logs with `parse_runs.parse_run` on 2026-09-18.
The zero-coefficient run has metrics through 477; ent003 through 766.
The first zero-coefficient job (46631406) produced no step metrics; the usable
jobs are 46651436 and 46729600, with the later job winning overlapping steps.
Ent003 is job 46946121.

| Measurement | `qwen3_4b_spec_simsft_s2` | `qwen3_4b_spec_simsft_s2_ent003` |
|---|---:|---:|
| Entropy coefficient | 0 | -0.003 |
| Mean validation reward, steps 101–200 | 0.6200 | 0.6215 |
| Mean validation reward, steps 201–300 | 0.6683 | 0.6291 |
| Mean validation reward, steps 301–400 | 0.7254 | 0.6461 |
| Mean training reward, steps 301–400 | 0.7040 | 0.6353 |
| Mean actor entropy, steps 301–400 | 0.5915 | 0.1713 |
| First evaluated step with reward ≥0.70 | 320 | 500 |
| Peak validation reward | 0.7341 @ 400 | 0.7180 @ 640 |
| Best five-consecutive-evaluation mean | 0.7289 (340–420) | 0.7145 (560–640) |
| Detonation step (RESULTS.md definition) | 464 | 747 |

Validation-window means average evaluations every 20 steps; training/entropy
window means average every available training step. Best-window scores are
descriptive, selected after observing the runs, not unbiased performance estimates.

The implementation uses `policy_loss -= entropy_coeff * entropy_loss`
(`verl/workers/utils/losses.py`), so -0.003 ADDS 0.003 H to the minimized
loss: this is an entropy penalty, not an exploration bonus.

The clearest result is slower middle-stage learning alongside lower entropy,
followed by delayed collapse. Early learning is not uniformly worse. The
penalized run eventually approaches the unpenalized peak; its peak is lower
by 0.0162 and its best five-evaluation mean by 0.0144. These single trajectories
do not establish a lower attainable ceiling or statistical significance.
Suppression of useful exploration is a plausible interpretation, not a measured
mechanism. The same-step reward gap is much larger than the best-score gap.

Launch records match on solver, SFT simulator (`local/sim-sft-g3-single`),
grounded simulator conditioning, GPT-5.4 specs, rejection disabled, early
termination guard disabled, terminate-on-allpass enabled, `upto_last_code`
training mask, and two code proposals. Checkpoint cadence, restart history,
and code snapshots differ, so this is not a replicated controlled seed study.
These scores are in-training `val-core/colbench_spec_local/reward/mean@1`;
do not describe them as the independent 235B evaluation scores.

Follow-up: evaluate retained checkpoints from both arms under one identical
235B evaluation protocol, compare matched-step and best-checkpoint performance,
then test an intermediate penalty (proposed -0.001) with otherwise fixed
settings. Repeated seeds are needed to distinguish ceiling changes from run
variability. Do not select an entropy coefficient solely for delaying collapse.

## Larger-simulator training experiment

Start a fresh Qwen3-4B solver run with coefficient 0, replacing only the SFT
simulator with `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`, the model identity used
by the existing 235B evaluation wrapper. Preserve the SFT s2 training protocol
above, including grounded conditioning and disabled rejection. Importing the
evaluation wrapper's guardrails would change multiple factors at once.

Proposed training invocation, from the verl root (not submitted):

```bash
ENTROPY_COEFF=0 TEST_FREQ=20 SAVE_FREQ=50 TOTAL_EPOCHS=15 MAX_CKPT_KEEP=10 \
bash slurm_setup/launch_slurm.sh \
  --model=Qwen/Qwen3-4B --exp_name=spec_sim235b_s2_ent0 \
  --train_script=colbench/run_colbench_grpo_spec.sh --spec_author=gpt-5.4 \
  --grounded_sim --sim_model=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
  --sim_reject_max_tries=0 --noearly_term_guard --terminate_on_allpass \
  --train_turns=upto_last_code --max_code_proposals=2 \
  --partition=kempner_h200
```

The registry assigns TP=4 to this simulator; the default remote-simulator
layout uses separate training and simulator nodes. This invocation is a draft,
not a serving/throughput preflight. Before submission, verify the large-model
server's context/concurrency settings and training-scale throughput; the
evaluation wrapper uses context 32768 and max-running-requests 8, while the
training server does not explicitly pin those settings. Checkpoint retention
also needs to preserve the pre-collapse peak as the run extends.

Log mean and turn-bucket entropy from the first update, their token fractions,
reward, turns, lengths, and simulator errors/timeouts. Compare trajectories by
updates and generated tokens as well as wall time. Observe beyond the previous
collapse horizons; surviving a short pilot does not establish stability.
Report a finite non-collapsing run as "no collapse through step N", not a cure.

If collapse reproduces, it extends the finding beyond the small SFT simulator
but not beyond multi-turn simulated-user RL generally. If it does not, simulator
behavior/distribution becomes a stronger lead. Model size alone is not isolated
by this replacement: model family variant, training, and FP8 serving also differ.

## Deferred questions

1. Correct the covariance proxy and/or measure before/after-update entropy on
   fixed recorded contexts.
2. Measure sampler–actor mismatch by turn.
3. Determine whether late-turn entropy is concentrated in unusually long or
   failed conversations. Turn3+ pools different conversation lengths and uses
   token weighting, so current results do not settle this. Eventually compare
   entropy within joint turn/length/reward strata and with equal episode weights
   before testing exclusion. Excluding long episodes now would change the
   training distribution before showing that length identifies harmful data.

"""ColBench user-SIMULATOR training (the sim side of the loop).

Everything so far trains the SOLVER against a frozen simulator. That simulator
gets hacked: on the GT (non-spec) path it is shown the hidden ground-truth
function, and a solver trained against it learns to extract that function --
measured as ``sim_leak_frac`` rising 0.251 -> 0.635 over
``qwen3_4b_nonspec_baseline``. Rejection sampling
(``templates.detect_code_leak``) filters only SYNTACTIC leaks; a prose
transcription of the algorithm passes it untouched.

This subpackage trains the simulator instead of filtering it. It is a
proof-of-concept answering exactly two questions:

  1. Does an LLM-judge rubric give MEANINGFUL, SEPARABLE scores to simulator
     turns?
  2. Does SFT on judge-selected turns MEASURABLY improve the simulator?

THE STRUCTURAL FACT THE WHOLE PIPELINE RESTS ON: the GT sim prompt is not a
chat transcript. ``templates.str_dialogue_history`` re-renders the entire
dialogue into ONE user message under a constant system prompt, so each
simulator turn is an INDEPENDENT single-turn generation. The training
formulation is therefore turn-level (a contextual bandit), a
``(rendered sim prompt, reply)`` pair is the complete state description, and
every stage after collection is a pure JSONL->JSONL function: no env, no GPU,
CPU-unit-testable.

The stages, each a script in this package:

  * ``collect_prefixes``   -- run a HACKING partner assistant against the frozen
    sim and MATERIALIZE the rendered sim prompt at every user turn.
  * ``collect_candidates`` -- K draws per prefix from the base sim.
  * ``judge_candidates``   -- score each draw against the rubric
    (``judge_rubric``, prompt text in ``colbench.prompts``).
  * ``build_sft_parquet``  -- best-of-N selection -> a verl MultiTurnSFT parquet.
  * ``eval_sim_sft``       -- paired, JUDGE-FREE offline evaluation on held-out
    tasks.

Nothing here touches the solver training path or ``validate_colbench.py`` (the
GT-arm yardstick).
"""

# Bumped when the COLLECTOR's record schema or episode semantics change, so a
# prefix file from an older collector is never silently mixed with a newer one.
# Stamped into prefix filenames AND into every record, mirroring
# validate_colbench's EVAL_HARNESS_VERSION.
COLLECTOR_VERSION = "c1"

# Bumped when the judge MECHANICS change (K-in-one-call shape, permutation,
# parser, aggregation) as opposed to the rubric TEXT, whose version is
# ``prompts.JUDGE_RUBRIC_VERSION``. Two versions because they fail differently:
# a rubric edit changes what "good" means and invalidates comparisons, while a
# harness edit changes only how the same rubric is administered.
# h1 -> h2 (2026-09-10): stage 1 now screens with
# ``detect_code_leak(expr_over_gt_names=True)``, which catches a function body
# handed over as a BACKTICKED EXPRESSION rather than as a def or a fence
# ("the function should return `(num_teams * (cost_a - cost_b)) * years`").
# Measured: 4.9% of base-partner candidates, all of which the judge's own
# code_leak also scored clean. Same rubric text, a wider veto -- so the
# HARNESS version moves, not the rubric version.
JUDGE_HARNESS_VERSION = "h2"

"""ColBench self-play spec setting (Phase 0).

Phase 0 is offline spec authoring plus the diagnostic.

A CLEAN, SELF-CONTAINED alternative to the current GT-code-conditioned
simulator. Instead of handing the frozen user simulator the hidden ground-truth
*code* (which leaks), we author a natural-language **spec** (persona + scenario
+ complete requirements) offline and will later condition the simulator on that
prose. This subpackage builds ONLY the Phase-0 pieces:

  * ``generate_specs``  -- author one spec per task from (public problem +
    hidden GT code), with a pluggable backend: ``strong`` (external teacher) or
    ``selfplay`` (the trained model's frozen base checkpoint; no external model
    in the loop).
  * ``diagnose_specs``  -- the full-spec solve-rate diagnostic: hand a solver
    the ENTIRE spec (single turn, no dialogue), grade against the UNCHANGED GT
    ``test_cases``, and report the solve rate. Run it on strong-gen and self-gen
    specs to see whether self-authored specs are good enough before committing
    to Phase-1 training plumbing.

Grading and reward stay exactly as in the main package (``colbench.reward`` ->
the codecontest exec sidecar); the GT code is the reward, the spec only governs
conditioning. Nothing here touches the existing GT-code-conditioned training
path.
"""

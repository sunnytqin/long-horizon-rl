# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gradient-masking policy for the multi-turn refinement study (SET 2).

Single source of truth shared by the oracle (code_refine_agent) and model-feedback
(model_feedback_agent) loops so the two arms are provably identical.

The policy selects WHICH solver turns contribute to the training loss. It does NOT
touch the rolled-out sequence: masked tokens stay in `prompt_ids`/`response_ids` and
in context, so later turns still attend to them -- they are only zeroed in
`response_mask` and thus excluded from the gradient. Feedback / user turns are always
mask=0 and are not represented here; this governs solver turns only.

Modes (see project-codecontest-rl-stability-plan, SET 2):
  "all"        train every solver turn (baseline; prior behavior).
  "final_only" train ONLY the last solver turn. Because the rollout loop breaks on
               solve, the last solver turn's own pass/fail always equals the trajectory
               reward, so its per-token advantage sign is aligned with what that turn
               actually did -- clean credit assignment, no reinforcement of failed
               intermediate attempts. Still teaches refinement (the final turn is
               conditioned on all prior feedback).
  "upto_last_code" (ColBench spec path, Intervention 2) train turns [0 .. last_code_idx]
               and ZERO every turn AFTER the last code proposal. `last_code_idx` is the
               solver-turn ordinal that produced the graded code (passed by the caller).
               Motivation: on the spec path the frozen sim can keep talking after the
               solver's final code, and the solver learned to emit reward-irrelevant
               trailing prose (capitulation / gibberish) that still soaked up the positive
               trajectory advantage -- a "free ride" that drove the ~step-300 collapse.
               Dropping post-code turns removes that pathway while KEEPING clarification
               turns (the ColBench skill) in the gradient. If last_code_idx is None (no
               code was ever shown) this falls back to "all" so the negative advantage on a
               no-code ramble is preserved. NOTE: turns 0..last_code_idx still include any
               EARLIER (failed) code attempt, which under the flat trajectory reward gets
               the same advantage sign as the final code -- the inter-attempt credit
               confound. That is deliberately OUT OF SCOPE here (Int 2 targets trailing
               ramble only); see the FUTURE note below on shaped between-attempt credit.

DROPPED -- "refinement_only" (mask turn 0, train turns 1..N): removed on purpose. Under
the FLAT trajectory outcome reward the single advantage is broadcast to every trained
token, so training turns 1..N reinforces FAILED intermediate attempts with the same
positive advantage as the solving turn, while excluding the equally-failed turn 0 -- an
asymmetric confound. Its "cleanly train the corrector" rationale only holds under a
PER-TURN shaped reward (SCoRe), which we don't use. Do not re-add without that.

FUTURE -- a "final_refinement" hybrid could get BOTH clean credit and front-loading
suppression: train only the final turn, but zero-gradient the whole trajectory when
solved_at_turn==0 (turn-0 solves stay in the batch for the GRPO group baseline but
contribute no gradient). Consider once the final_only vs all comparison is in.

FUTURE -- shaped between-attempt credit (the "upto_last_code" residual, deferred): under
the flat trajectory reward, when a trajectory has multiple code attempts every kept
attempt gets the same advantage sign, so a failed first attempt that the solver later
FIXED still receives positive credit. A per-attempt shaped reward (e.g. credit a code
turn only when it improves pass-rate over the previous attempt, wrong->right) would fix
this. Requires a reward rewrite (per-turn signal), so it is out of scope for Int 2.
"""

TRAIN_TURNS_MODES = ("all", "final_only", "upto_last_code")


def apply_train_turns_mask(response_mask, solver_spans, mode, last_code_idx=None):
    """Zero out the solver-turn spans that `mode` excludes from training, IN PLACE.

    Args:
        response_mask: the full 0/1 loss mask being built (solver turns 1, feedback
            turns 0). Mutated in place.
        solver_spans: list of (start, end) half-open index ranges into
            `response_mask`, one per solver turn, in emission order.
        mode: one of TRAIN_TURNS_MODES -- "all" (no-op), "final_only", or "upto_last_code".
        last_code_idx: for "upto_last_code" only, the index into `solver_spans` of the
            solver turn that produced the graded code. None => no code shown => fall back
            to "all" (keep everything). Ignored by the other modes.
    """
    if mode not in TRAIN_TURNS_MODES:
        raise ValueError(f"train_turns must be one of {TRAIN_TURNS_MODES}, got {mode!r}")
    if mode == "all" or not solver_spans:
        return
    if mode == "upto_last_code":
        # Keep [0 .. last_code_idx], zero every turn AFTER the last code proposal. No code
        # shown (idx None) -> keep all, so a no-code ramble keeps its negative advantage.
        if last_code_idx is None:
            return
        for i, (start, end) in enumerate(solver_spans):
            if i > last_code_idx:
                for j in range(start, end):
                    response_mask[j] = 0
        return
    # final_only: keep only the last solver turn.
    keep = {len(solver_spans) - 1}
    for i, (start, end) in enumerate(solver_spans):
        if i not in keep:
            for j in range(start, end):
                response_mask[j] = 0

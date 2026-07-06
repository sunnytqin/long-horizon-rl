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
"""

TRAIN_TURNS_MODES = ("all", "final_only")


def apply_train_turns_mask(response_mask, solver_spans, mode):
    """Zero out the solver-turn spans that `mode` excludes from training, IN PLACE.

    Args:
        response_mask: the full 0/1 loss mask being built (solver turns 1, feedback
            turns 0). Mutated in place.
        solver_spans: list of (start, end) half-open index ranges into
            `response_mask`, one per solver turn, in emission order.
        mode: one of TRAIN_TURNS_MODES -- "all" (no-op) or "final_only".
    """
    if mode not in TRAIN_TURNS_MODES:
        raise ValueError(f"train_turns must be one of {TRAIN_TURNS_MODES}, got {mode!r}")
    if mode == "all" or not solver_spans:
        return
    # final_only: keep only the last solver turn.
    keep = {len(solver_spans) - 1}
    for i, (start, end) in enumerate(solver_spans):
        if i not in keep:
            for j in range(start, end):
                response_mask[j] = 0

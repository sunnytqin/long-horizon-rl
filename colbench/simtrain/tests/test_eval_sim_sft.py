# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Stage 6 tests: the pairing rules, the judge-free metric, and the sign test.

The pairing rules are what this file is really for. An evaluation that averages
each arm over whatever prefixes it happened to keep will report a difference
whenever the arms keep DIFFERENT prefixes -- which is exactly what a keep-rate
change means -- so it can manufacture an improvement out of nothing. Every test
below either pins a denominator or pins a refusal.
"""

import json
import os
import tempfile

import pytest

from colbench import prompts
from colbench.simtrain import eval_sim_sft as E


GT = (
    "def score(intelligence, feelings):\n"
    "    threshold = 0.5\n"
    "    return 'yes' if (intelligence * 0.4 + feelings * 0.3) / 10 >= threshold else 'no'\n"
)


def _prefix(pid, task=1):
  return {
      "prefix_id": pid,
      "task_id": task,
      "task_index": task,
      "turn_idx": 0,
      "ground_truth": GT,
      "problem_description": "I want a function that scores an entity.",
      "sim_dialogue": [
          {"role": "user", "content": "I want a function that scores an entity."},
          {"role": "assistant", "content": "What scale do the inputs use?"},
      ],
      "partner_reply": "What scale do the inputs use?",
      "sim_user": f"rendered dialogue for {pid}",
      "sim_system": prompts.SIM_ROLE_RESTRAINT_SYSTEM_PROMPT,
  }


def _judged(pid, candidates, totals, truth=None, code_vetoed=None, cals=None):
  cals = cals if cals is not None else [4] * len(totals)
  scores = []
  for t, cal in zip(totals, cals):
    scores.append(
        None
        if t is None
        else {
            "code_leak": 1,
            "calibration": cal,
            "in_character": 4,
            "volunteering": max(0, t - 8),
            "total": t,
        }
    )
  return {
      "prefix_id": pid,
      "ok": True,
      "candidates": candidates,
      "scores": scores,
      "margin": 1,
      "rubric_version": prompts.JUDGE_RUBRIC_VERSION,
      "truth_flags": truth if truth is not None else ["ok"] * len(candidates),
      "code_vetoed": code_vetoed or [],
  }


def test_judge_free_metric_subtracts_what_the_agent_already_had():
  """The subtraction is the whole metric.

  Without it this is a length proxy: any substantive reply mentions GT
  identifiers. With it, CONFIRMING what the agent itself proposed scores zero,
  which is the behaviour the rubric deliberately permits.
  """
  pre = _prefix("p1")
  # `threshold` and `0.5` are in the GT and in neither the problem statement
  # nor the dialogue -> newly released.
  assert E.new_gt_tokens(pre, "the threshold is 0.5") >= {"threshold", "0.5"}
  # Nothing from the GT at all.
  assert not E.new_gt_tokens(pre, "I would rather not say")
  # Already in the agent's own turn -> not a release.
  pre2 = dict(pre, partner_reply="Is the threshold 0.5?")
  assert not E.new_gt_tokens(pre2, "that is right, the threshold is 0.5")


def test_judge_free_metric_is_approximate_and_overcounts_return_strings():
  """A known limitation, pinned so it is not rediscovered as a bug.

  The fixture GT RETURNS the literals 'yes'/'no', so a reply that merely agrees
  ("yes, ...") shares a token with the ground truth and counts as a release.
  The metric is a judge-free approximation and cannot tell an English "yes"
  from a returned "yes"; it is used for DIRECTION on aggregates, never as a
  per-reply verdict, which is why this is tolerable.
  """
  pre = _prefix("p1")
  assert "yes" in E.released_tokens(GT)
  assert E.new_gt_tokens(pre, "yes") == {"yes"}


def test_single_digits_are_excluded_from_the_metric():
  """A bare 0/1/2 occurs in nearly any sentence about code. Including them
  inflated the measured rate 0.52 -> 0.58 with every extra hit noise."""
  pre = _prefix("p1")
  assert not E.new_gt_tokens(pre, "it returns 0 or 1 depending on the input")


def test_sign_test_excludes_ties_and_is_two_sided():
  """Ties dominate here by construction -- after the vetoes the survivors score
  alike -- so an unconditional test would never move off p=1."""
  assert E.sign_test_p(0, 0) == 1.0
  assert E.sign_test_p(5, 5) == 1.0
  assert E.sign_test_p(10, 0) < 0.01
  assert E.sign_test_p(0, 10) == pytest.approx(E.sign_test_p(10, 0))
  # 8 of 10 one way is suggestive but not significant; 100 of 100 is.
  assert 0.01 < E.sign_test_p(8, 2) < 0.15
  assert E.sign_test_p(100, 0) < 1e-20


def test_target_metrics_are_averaged_only_over_prefixes_KEPT_IN_BOTH_arms():
  """The trap this evaluation exists to avoid.

  S_1 keeps one extra prefix, and that prefix's target is terrible. If the
  report averaged each arm over its own kept set, S_1 would look WORSE for
  having salvaged a prefix; if it averaged over the union it would compare
  different tasks. Only the intersection is a paired comparison, and the
  report must say how many rows that is.
  """
  prefixes = {f"p{i}": _prefix(f"p{i}", task=i) for i in range(3)}
  base = [
      _judged("p0", ["a"], [12]),
      _judged("p1", ["b"], [10]),
      # calibration 0 -> below the floor -> S_0 drops this prefix entirely.
      _judged("p2", ["dump: the threshold is 0.5"], [4], cals=[0]),
  ]
  sft = [
      _judged("p0", ["a"], [12]),
      _judged("p1", ["b better"], [12]),
      # S_1 salvages p2, but its target is the worst in the set.
      _judged("p2", ["dump: the threshold is 0.5"], [10]),
  ]
  floors = {"calibration": 3, "in_character": 3}
  rep = E.compare(
      E.arm_stats(prefixes, base, floors), E.arm_stats(prefixes, sft, floors)
  )
  assert "prefixes judged in BOTH arms   3" in rep
  # The paired rows are p0 and p1 only -- p2 is kept by S_1 alone.
  assert "(both arms kept -- rows below)            2" in rep
  # S_1 tied p0 and improved p1 by 2, so the paired total delta is +1.00.
  # Crucially it is NOT dragged down by p2's bad target, which is not paired.
  assert "+1.00" in rep
  # p2 shows up as a keep-rate gain instead: 2/3 -> 3/3.
  assert "0.667" in rep and "1.000" in rep
  # and the reason S_0 lost it is attributed.
  assert "below_floor" in rep


def test_arm_stats_counts_drops_and_failure_modes_per_arm():
  prefixes = {"p0": _prefix("p0"), "p1": _prefix("p1", task=2)}
  judged = [
      _judged("p0", ["x", "y"], [None, 12], truth=["wrong", "ok"]),
      _judged("p1", ["z"], [None], truth=["unsure"], code_vetoed=[]),
  ]
  st = E.arm_stats(prefixes, judged, {"calibration": 3, "in_character": 3})
  assert st["counts"]["truth_wrong"] == 1
  assert st["counts"]["truth_unsure"] == 1
  assert st["counts"]["truth_checked"] == 3
  assert st["counts"]["candidates"] == 3
  assert st["per_prefix"]["p0"]["kept"] is True
  assert st["per_prefix"]["p1"]["kept"] is False


def test_a_judged_row_without_its_prefix_is_counted_not_crashed():
  st = E.arm_stats({}, [_judged("ghost", ["x"], [12])], None)
  assert st["counts"]["missing_prefix"] == 1 and not st["per_prefix"]


def test_no_shared_prefixes_says_so_rather_than_reporting_zeros():
  """Pointing the two arms at different prefix files is an easy mistake and
  produces a report full of plausible zeros unless it is called out."""
  prefixes = {"a": _prefix("a"), "b": _prefix("b", task=2)}
  rep = E.compare(
      E.arm_stats(prefixes, [_judged("a", ["x"], [12])], None),
      E.arm_stats(prefixes, [_judged("b", ["x"], [12])], None),
  )
  assert "no shared prefixes" in rep


def test_cli_refuses_arms_judged_under_different_rubrics():
  """Otherwise the report measures the rubric rewrite, not the SFT -- and this
  project rewrote the rubric five times, so it is a live hazard."""
  with tempfile.TemporaryDirectory() as d:
    pf = os.path.join(d, "pre.jsonl")
    with open(pf, "w") as f:
      f.write(json.dumps(_prefix("p0")) + "\n")
    bf = os.path.join(d, "base.jsonl")
    with open(bf, "w") as f:
      f.write(json.dumps(_judged("p0", ["x"], [12])) + "\n")
    sf = os.path.join(d, "sft.jsonl")
    with open(sf, "w") as f:
      rec = _judged("p0", ["x"], [12])
      rec["rubric_version"] = "r2"
      f.write(json.dumps(rec) + "\n")
    with pytest.raises(SystemExit, match="different rubrics"):
      E.main(["--prefixes", pf, "--base", bf, "--sft", sf])


# ── the SERVING metric: expected draw, not best-of-K ─────────────────────────

def test_expected_draw_section_is_dup_weighted_not_deduped():
  """The candidate list is DEDUPED, so an unweighted mean over it reweights
  toward the sim's RARE outputs. Here arm S_0's BAD reply was drawn 7 times and
  its good one once; a deduped mean would call that a 2.0 average, while what a
  served model actually emits is 7/8 bad.

  Both arms have the SAME best-of-K target (total 12), so the best-of-K rows
  cannot distinguish them at all -- only the expected-draw rows can.
  """
  prefixes = {"p0": _prefix("p0")}
  base = _judged("p0", ["bad reply", "good reply"], [4, 12], cals=[0, 4])
  base["dup_counts"] = [7, 1]
  sft = _judged("p0", ["bad reply", "good reply"], [4, 12], cals=[0, 4])
  sft["dup_counts"] = [1, 7]

  rep = E.compare(
      E.arm_stats(prefixes, [base], None),
      E.arm_stats(prefixes, [sft], None),
  )
  assert "EXPECTED DRAW" in rep
  # E[calibration] = (7*0 + 1*4)/8 = 0.50 for S_0, (1*0 + 7*4)/8 = 3.50 for S_1.
  line = next(l for l in rep.splitlines() if "E[calibration" in l)
  assert "0.50" in line and "3.50" in line, line
  # ...while the best-of-K judge-total row sees no difference whatsoever.
  bok = next(l for l in rep.splitlines() if l.strip().startswith("judge total"))
  assert "+0.00" in bok, bok


def test_expected_draw_reports_veto_probability_per_kind():
  """A served model that writes code 1 turn in 8 is the failure this pipeline
  exists to remove, so the probability has to be reported unconditionally --
  dimension means are conditional on surviving the vetoes and cannot show it."""
  prefixes = {"p0": _prefix("p0")}
  base = _judged("p0", ["code", "clean"], [None, 12], code_vetoed=[0])
  base["dup_counts"] = [2, 6]
  base["form_vetoed"] = []
  sft = _judged("p0", ["code", "clean"], [None, 12], code_vetoed=[0])
  sft["dup_counts"] = [1, 7]
  sft["form_vetoed"] = []

  rep = E.compare(
      E.arm_stats(prefixes, [base], None),
      E.arm_stats(prefixes, [sft], None),
  )
  line = next(l for l in rep.splitlines() if "writes code" in l)
  assert "0.250" in line and "0.125" in line, line

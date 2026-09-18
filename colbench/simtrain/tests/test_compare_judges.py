"""CPU tests for the two-judge comparison.

The failure modes worth pinning are the ones that would let a BAD judge look
acceptable: a comparison that silently runs over different candidates, a
comparison that averages two scoring standards together, and an agreement
statistic inflated by a dominant label.

Run: pytest colbench/simtrain/tests/test_compare_judges.py
"""

import copy

import pytest

from colbench.simtrain import compare_judges
from colbench.simtrain import judge_rubric

GT = "def f(x):\n    return x * 3 if x > 7 else 0\n"
PREFIX = {
    "prefix_id": "1-0-0",
    "task_id": 1,
    "turn_idx": 0,
    "problem_description": "Write f(x).",
    "ground_truth": GT,
    "partner_reply": "What is the cutoff?",
    "sim_reply": "Seven.",
    "sim_dialogue": [],
}


def _judged(judge="gpt-5.4-mini", truth=("ok", "ok"), cal=(4, 3), pid="1-0-0"):
  return {
      "prefix_id": pid,
      "task_id": 1,
      "turn_idx": 0,
      "candidates": ["Above seven it triples.", "Seven is the cutoff."],
      "dup_counts": [5, 1],
      "rubric_version": "r6",
      "harness_version": "h2",
      "judge_model": judge,
      "code_vetoed": [],
      "ok": True,
      "truth_ok": True,
      "truth_flags": list(truth),
      "truth_checks": {},
      "scores": [
          {"code_leak": 1, "volunteering": 4, "calibration": cal[0],
           "in_character": 4, "total": 8 + cal[0]},
          {"code_leak": 1, "volunteering": 4, "calibration": cal[1],
           "in_character": 4, "total": 8 + cal[1]},
      ],
  }


def _run(a, b, prefixes=None):
  prefixes = prefixes or {"1-0-0": PREFIX}
  return compare_judges.compare(prefixes, {"1-0-0": a}, {"1-0-0": b})


def test_identical_files_show_no_disagreement():
  """The 'does not manufacture signal' control, as eval_sim_sft does it.

  Two truth labels, not one, so kappa is DEFINED here and a perfect score is
  meaningful; the single-label case is covered by the kappa test below.
  """
  row = _judged(truth=("ok", "wrong"))
  report, disagreements = _run(row, copy.deepcopy(row))
  assert not disagreements
  assert "agreement=1.000" in report
  assert "kappa= 1.000" in report
  assert "DISAGREEING PREFIXES: 0/1" in report


def test_different_candidate_texts_are_refused():
  """Otherwise a SIM comparison masquerades as a JUDGE comparison."""
  a = _judged()
  b = _judged()
  b["candidates"] = ["something the other sim said", "and another"]
  with pytest.raises(SystemExit, match="DIFFERENT candidate texts"):
    _run(a, b)


def test_differing_programmatic_veto_is_refused():
  """Stage 1 is a regex; a difference means a harness mismatch."""
  a = _judged()
  b = _judged()
  b["code_vetoed"] = [1]
  with pytest.raises(SystemExit, match="programmatic"):
    _run(a, b)


def test_mixed_stamp_file_is_refused():
  rows = [_judged(pid="1-0-0"), _judged(judge="other", pid="2-0-0")]
  import json, tempfile, os
  fd, path = tempfile.mkstemp(suffix=".jsonl")
  with os.fdopen(fd, "w") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  with pytest.raises(SystemExit, match="mixes scoring standards"):
    compare_judges.load_judged(path)
  os.unlink(path)


def test_truth_disagreement_is_reported_and_dumped():
  a = _judged(truth=("ok", "ok"))
  b = _judged(judge="qwen", truth=("wrong", "ok"))
  report, disagreements = _run(a, b)
  assert len(disagreements) == 1
  assert any("truth[0]" in r for r in disagreements[0]["reasons"])
  assert "agreement=0.500" in report
  dump = compare_judges.render_disagreements(
      {"1-0-0": PREFIX}, {"1-0-0": a}, {"1-0-0": b}, disagreements
  )
  assert "DISAGREE" in dump
  assert "Above seven it triples." in dump


def test_selection_flip_is_detected():
  """The headline: a different PICK changes the SFT set."""
  a = _judged(cal=(4, 3))
  b = _judged(judge="qwen", cal=(3, 4))
  _, disagreements = _run(a, b)
  assert disagreements
  assert disagreements[0]["pick_a"] == 0
  assert disagreements[0]["pick_b"] == 1


def test_keep_to_drop_flip_is_detected():
  a = _judged(truth=("ok", "ok"))
  b = _judged(judge="qwen", truth=("wrong", "wrong"))
  report, disagreements = _run(a, b)
  assert disagreements[0]["pick_b"] is None
  assert disagreements[0]["why_b"] == judge_rubric.DROP_ALL_UNTRUE
  assert "keep rate      A 1.000" in report
  assert "B 0.000" in report


def test_kappa_discounts_a_dominant_label():
  """Two judges that both almost always say 'ok' agree ~0.9 by chance alone."""
  pairs = [("ok", "ok")] * 90 + [("ok", "wrong")] * 5 + [("wrong", "ok")] * 5
  k = compare_judges.kappa(pairs)
  assert k is not None
  assert k < 0.1  # 0.90 raw agreement, near-zero real agreement
  assert compare_judges.kappa([]) is None
  assert compare_judges.kappa([("ok", "ok")] * 10) is None  # undefined, not 1.0


def test_no_shared_prefixes_is_refused():
  with pytest.raises(SystemExit, match="share no prefix ids"):
    compare_judges.compare({}, {"1-0-0": _judged()}, {"9-0-0": _judged(pid="9-0-0")})

"""CPU tests for the judge's RESUME identity guard.

Resume matches on ``prefix_id`` alone. That is correct for re-running the SAME
judge (never re-pay for a row already bought) and catastrophic for a different
one, in two opposite directions that both look like success:

  * a COMPLETE file -- every row reads as done, the new judge is never called,
    and the two judges appear to agree perfectly;
  * a PARTIAL file -- the normal state, since a bulk pass stops on the spend cap
    -- the new judge fills the gaps and the file becomes an undeclared mixture
    that downstream stages treat as one scoring standard.

Run: pytest colbench/simtrain/tests/test_judge_resume.py
"""

import json

import pytest

from colbench import prompts
from colbench.simtrain import JUDGE_HARNESS_VERSION
from colbench.simtrain.judge_candidates import existing_prefix_ids


def _judged(tmp_path, rows):
  path = tmp_path / "judged.jsonl"
  with open(path, "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  return str(path)


def _row(pid, judge="gpt-5.4-mini", rubric=None, harness=None):
  return {
      "prefix_id": pid,
      "judge_model": judge,
      "rubric_version": rubric or prompts.JUDGE_RUBRIC_VERSION,
      "harness_version": harness or JUDGE_HARNESS_VERSION,
  }


def test_missing_file_reads_as_empty(tmp_path):
  assert existing_prefix_ids(str(tmp_path / "nope.jsonl"), "any-judge") == set()


def test_same_judge_resumes(tmp_path):
  path = _judged(tmp_path, [_row("1-0-0"), _row("2-0-0")])
  assert existing_prefix_ids(path, "gpt-5.4-mini") == {"1-0-0", "2-0-0"}


def test_different_judge_is_refused(tmp_path):
  """The complete-file case: without this, the swap is a silent no-op."""
  path = _judged(tmp_path, [_row("1-0-0"), _row("2-0-0")])
  with pytest.raises(SystemExit) as excinfo:
    existing_prefix_ids(path, "colbench-sim")
  assert "gpt-5.4-mini" in str(excinfo.value)
  assert "colbench-sim" in str(excinfo.value)


def test_different_rubric_or_harness_is_refused(tmp_path):
  """Same failure mode one level down; h1 -> h2 really happened."""
  for bad in ({"rubric": "r5"}, {"harness": "h1"}):
    path = _judged(tmp_path, [_row("1-0-0", **bad)])
    with pytest.raises(SystemExit):
      existing_prefix_ids(path, "gpt-5.4-mini")


def test_partial_file_with_two_judges_is_refused(tmp_path):
  """The corrupting case: a spend-capped file topped up by a second judge."""
  path = _judged(
      tmp_path, [_row("1-0-0"), _row("2-0-0", judge="colbench-sim")]
  )
  with pytest.raises(SystemExit):
    existing_prefix_ids(path, "gpt-5.4-mini")


def test_allow_mixed_overrides(tmp_path):
  path = _judged(tmp_path, [_row("1-0-0"), _row("2-0-0")])
  assert existing_prefix_ids(
      path, "colbench-sim", allow_mixed=True
  ) == {"1-0-0", "2-0-0"}


def test_empty_judge_model_skips_the_identity_check(tmp_path):
  """Callers that do not name a judge still get plain resume behaviour."""
  path = _judged(tmp_path, [_row("1-0-0")])
  assert existing_prefix_ids(path, "") == {"1-0-0"}

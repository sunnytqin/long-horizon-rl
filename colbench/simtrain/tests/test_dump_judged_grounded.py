"""The dump must render BOTH rubrics. It is the deliverable of a pilot pass.

r6 verdicts carry `code_leak`; g1 verdicts do not (g1 vetoes code
programmatically), so any hardcoded reference to that field crashes on a g1
file -- and a rubric with a different number of dimensions must not be scored
out of r6's denominator.

Run: pytest colbench/simtrain/tests/test_dump_judged_grounded.py
"""

from colbench.simtrain.dump_judged import _abbrev
from colbench.simtrain.dump_judged import render_prefix

GT = "def total_fee(base_rate, hours_worked):\n    return base_rate * hours_worked\n"
PREFIX = {
    "prefix_id": "3-0-1",
    "problem_description": "Write total_fee(base_rate, hours_worked).",
    "ground_truth": GT,
    "partner_reply": "How is the fee computed?",
    "sim_reply": "Base rate times hours.",
    "sim_reply_leak_reason": None,
    "sim_reply_leak_reason_strict": None,
    "sim_dialogue": [],
}
JUDGED = {
    "prefix_id": "3-0-1",
    "ok": True,
    "margin": 3,
    "judge_incoherent": False,
    "terminate_allowed": False,
    "candidates": [
        "It's the base rate multiplied by the hours worked.",
        "It should be `base_rate * hours_worked`.",
        "[TERMINATE]",
        "I'm not sure, whatever you think.",
        "Looks right, thanks! [TERMINATE]",
    ],
    "dup_counts": [3, 2, 1, 2, 1],
    "code_vetoed": [1],
    "term_vetoed": [2],
    "form_vetoed": [4],
    "scores": [
        {"gt_adherence": 4, "plot_adherence": 4, "not_overhelpful": 4,
         "note": "answers", "total": 12},
        None,
        None,
        {"gt_adherence": 1, "plot_adherence": 4, "not_overhelpful": 4,
         "note": "withholds", "total": 9},
        None,
    ],
}


def test_g1_page_renders_without_code_leak():
  page = render_prefix(PREFIX, JUDGED)
  assert "gt_a=4" in page and "plot=4" in page and "not=4" in page
  # r*'s ranker leak column must be absent, not printed as None.
  assert "leak=" not in page.split("CANDIDATES")[1]


def test_the_form_veto_is_named_distinctly_from_the_premature_one():
  """Two different rules about the sentinel; a reader has to be able to tell
  "ended too early" from "spoke and ended in one reply"."""
  page = render_prefix(PREFIX, JUDGED)
  assert "TERMINATION-FORM veto" in page
  assert "PREMATURE-TERMINATION veto" in page


def test_both_programmatic_vetoes_are_named_in_the_dump():
  page = render_prefix(PREFIX, JUDGED)
  assert "code veto" in page
  assert "PREMATURE-TERMINATION veto" in page


def test_per_prefix_annotations_are_in_the_header():
  """Without terminate_allowed, a premature-termination veto below reads as
  unexplained."""
  page = render_prefix(PREFIX, JUDGED)
  header = page.split("CANDIDATES")[1].splitlines()[1]
  assert "terminate_allowed=False" in header


def test_denominator_follows_the_rubrics_dimension_count():
  two_dims = dict(JUDGED, scores=[
      {"gt_adherence": 4, "not_overhelpful": 3, "note": "x", "total": 7},
      None, None, None, None,
  ])
  assert "total  7/8" in render_prefix(PREFIX, two_dims)


def test_abbreviations_stay_unambiguous_and_stable():
  r6 = ("calibration", "in_character", "volunteering")
  assert [_abbrev(d, r6) for d in r6] == ["cali", "in_c", "volu"]
  g1 = ("gt_adherence", "not_overhelpful", "plot_adherence")
  assert [_abbrev(d, g1) for d in g1] == ["gt_a", "not", "plot"]
  # Widens rather than colliding.
  clash = ("plot_adherence", "plot_timing")
  assert _abbrev("plot_adherence", clash) != _abbrev("plot_timing", clash)

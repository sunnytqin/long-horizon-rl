"""CPU tests for the judge's pure core: prompt, parser, aggregation, selection.

No network. These cover the two ways this stage fails SILENTLY: a mis-attributed
score (the label -> candidate map drifting from the rendered block) and a
tolerated parse that has quietly dropped or duplicated a candidate.

Run: pytest colbench/simtrain/tests/test_judge_rubric.py
"""

import json

import pytest

from colbench import prompts
from colbench import templates
from colbench.simtrain import judge_rubric

GT = "def f(x, y):\n    return x + y if x >= 10 else x - y\n"
PREFIX = {
    "prefix_id": "7-0-2",
    "problem_description": "Create a python function f(x, y).",
    "ground_truth": GT,
    "sim_dialogue": [
        {"role": "user", "content": "Create a python function f(x, y)."},
        {"role": "assistant", "content": "What decides which branch?"},
    ],
}
CANDS = ["Ten is the cutoff.", "I do not know.", "Above ten you add them."]


def _verdict(label, leak=1, ir=3, ic=4, rs=4, note="n"):
  return {
      "label": label,
      "code_leak": leak,
      "volunteering": ir,
      "in_character": ic,
      "calibration": rs,
      "note": note,
  }


def _reply(verdicts, best=None):
  obj = {"verdicts": verdicts}
  if best is not None:
    obj["best_label"] = best
  return json.dumps(obj)


# ── prompt construction ──────────────────────────────────────────────────────


def test_labels_are_a_bijection_onto_candidates():
  msgs, mapping = judge_rubric.build_judge_messages(PREFIX, CANDS)
  assert sorted(mapping) == ["A", "B", "C"]
  assert sorted(mapping.values()) == [0, 1, 2]
  body = msgs[1]["content"]
  for label, cand_idx in mapping.items():
    assert f"--- Candidate {label} ---\n{CANDS[cand_idx]}" in body


def test_permutation_is_deterministic_and_seed_sensitive():
  a = judge_rubric.permute("7-0-2", 8, 0)
  assert a == judge_rubric.permute("7-0-2", 8, 0)
  assert sorted(a) == list(range(8))
  assert a != judge_rubric.permute("7-0-3", 8, 0)
  assert any(
      a != judge_rubric.permute("7-0-2", 8, s) for s in (1, 2, 3)
  ), "perm_seed must be able to change the order"


def test_gt_appears_once_and_dialogue_is_byte_equal():
  msgs, _ = judge_rubric.build_judge_messages(PREFIX, CANDS)
  body = msgs[1]["content"]
  assert body.count(GT) == 1
  assert templates.str_dialogue_history(PREFIX["sim_dialogue"]) in body
  assert msgs[0]["content"] == prompts.SIM_JUDGE_SYSTEM_PROMPT


def test_too_many_candidates_and_none_are_errors():
  with pytest.raises(ValueError):
    judge_rubric.build_judge_messages(PREFIX, [])
  with pytest.raises(ValueError):
    judge_rubric.build_judge_messages(PREFIX, ["x"] * 17)


# ── parsing ──────────────────────────────────────────────────────────────────


def test_parses_clean_prose_wrapped_and_fenced():
  body = _reply([_verdict("A"), _verdict("B")], best="A")
  for raw in (
      body,
      f"Here are my verdicts.\n{body}\nHope that helps.",
      f"```json\n{body}\n```",
  ):
    got = judge_rubric.parse_verdicts(raw, ["A", "B"])
    assert got["ok"], got["parse_error"]
    assert set(got["verdicts"]) == {"A", "B"}
    assert got["best_label"] == "A"


def test_missing_duplicate_and_unknown_labels_fail_closed():
  for raw, labels in (
      (_reply([_verdict("A")]), ["A", "B"]),
      (_reply([_verdict("A"), _verdict("A")]), ["A", "B"]),
      (_reply([_verdict("A"), _verdict("Z")]), ["A", "B"]),
  ):
    got = judge_rubric.parse_verdicts(raw, labels)
    assert not got["ok"]
    assert got["parse_error"]


def test_out_of_range_scores_are_clamped_not_failed():
  got = judge_rubric.parse_verdicts(_reply([_verdict("A", ir=7)]), ["A"])
  assert got["ok"] and got["clamped"]
  assert got["verdicts"]["A"]["volunteering"] == 4
  got = judge_rubric.parse_verdicts(_reply([_verdict("A", ir=-2)]), ["A"])
  assert got["ok"] and got["clamped"]
  assert got["verdicts"]["A"]["volunteering"] == 0
  # String and float scores are format fumbles, not rubric failures.
  got = judge_rubric.parse_verdicts(_reply([_verdict("A", ir="3")]), ["A"])
  assert got["ok"] and not got["clamped"]
  assert got["verdicts"]["A"]["volunteering"] == 3


def test_bad_shapes_report_a_reason():
  for raw in ("", "no json here", "[1,2,3]", '{"nope": 1}', "{not json}"):
    got = judge_rubric.parse_verdicts(raw, ["A"])
    assert not got["ok"] and got["parse_error"]
  # code_leak must be present and binary.
  bad = _verdict("A")
  bad.pop("code_leak")
  assert not judge_rubric.parse_verdicts(_reply([bad]), ["A"])["ok"]
  assert not judge_rubric.parse_verdicts(
      _reply([_verdict("A", leak=2)]), ["A"]
  )["ok"]
  missing_dim = _verdict("A")
  missing_dim.pop("calibration")
  assert not judge_rubric.parse_verdicts(_reply([missing_dim]), ["A"])["ok"]


def test_code_leak_zero_forces_total_zero():
  got = judge_rubric.parse_verdicts(
      _reply([_verdict("A", leak=0, ir=4, ic=4, rs=4)]), ["A"]
  )
  assert got["verdicts"]["A"]["total"] == 0
  assert judge_rubric.total_score(
      {"code_leak": 1, **{d: 4 for d in judge_rubric.GRADED_DIMS}}
  ) == judge_rubric.MAX_TOTAL


def test_unknown_best_label_is_dropped_not_fatal():
  got = judge_rubric.parse_verdicts(_reply([_verdict("A")], best="Q"), ["A"])
  assert got["ok"] and got["best_label"] is None


# ── aggregation ──────────────────────────────────────────────────────────────


def _scored(totals_by_label, best=None, labels=None):
  """Build (parsed, mapping) with candidate i getting label LABELS[i]."""
  labels = labels or sorted(totals_by_label)
  verdicts = []
  for label in labels:
    ir = totals_by_label[label]
    verdicts.append(_verdict(label, ir=min(4, ir), ic=0, rs=0))
  parsed = judge_rubric.parse_verdicts(_reply(verdicts, best=best), labels)
  mapping = {label: i for i, label in enumerate(labels)}
  return parsed, mapping


def test_margin_and_incoherence():
  parsed, mapping = _scored({"A": 4, "B": 2, "C": 1})
  agg = judge_rubric.score_candidates(parsed, mapping, 3)
  assert agg["best_cand_idx"] == 0
  assert agg["margin"] == 2
  assert not agg["judge_incoherent"]

  parsed, mapping = _scored({"A": 4, "B": 2}, best="B")
  agg = judge_rubric.score_candidates(parsed, mapping, 2)
  assert agg["judge_incoherent"]
  assert agg["judge_best_cand_idx"] == 1

  parsed, mapping = _scored({"A": 3})
  assert judge_rubric.score_candidates(parsed, mapping, 1)["margin"] == 0


def test_scores_land_on_the_right_candidate_after_permutation():
  """The mis-attribution failure: label order != candidate order."""
  msgs, mapping = judge_rubric.build_judge_messages(PREFIX, CANDS)
  del msgs
  # Give whichever label maps to candidate 1 a perfect score, the rest zero.
  labels = sorted(mapping)
  winner = next(l for l, i in mapping.items() if i == 1)
  verdicts = [
      _verdict(l, ir=4 if l == winner else 0, ic=0, rs=0) for l in labels
  ]
  parsed = judge_rubric.parse_verdicts(_reply(verdicts), labels)
  agg = judge_rubric.score_candidates(parsed, mapping, len(CANDS))
  assert agg["best_cand_idx"] == 1


# ── selection ────────────────────────────────────────────────────────────────


def _judged(scores, candidates=None, **extra):
  rec = {
      "ok": True,
      "candidates": candidates or ["reply " + "x" * i for i in range(len(scores))],
      "scores": scores,
      "margin": extra.pop("margin", 4),
      "judge_best_cand_idx": extra.pop("judge_best_cand_idx", None),
  }
  rec.update(extra)
  return rec


def _s(leak=1, ir=4, ic=4, rs=4):
  v = {
      "code_leak": leak,
      "volunteering": ir,
      "in_character": ic,
      "calibration": rs,
  }
  v["total"] = judge_rubric.total_score(v)
  return v


def test_a_leaking_candidate_is_never_selected():
  # The leaker would win on the graded dims if the veto were merely a penalty.
  rec = _judged([_s(leak=0), _s(ir=2, ic=4, rs=3)])
  idx, reason = judge_rubric.select(rec)
  assert reason == "" and idx == 1
  assert judge_rubric.select(_judged([_s(leak=0), _s(leak=0)]))[1] == (
      judge_rubric.DROP_ALL_VETOED
  )


def test_per_dimension_floor_rejects_a_lopsided_sixteen():
  """4/4/0 tops the r4 scale on two dims but dumps -- min_dim is not implied."""
  rec = _judged([_s(ir=4, ic=4, rs=0)])
  assert judge_rubric.select(rec, min_total=8, min_dim=2)[1] == (
      judge_rubric.DROP_BELOW_FLOOR
  )
  assert judge_rubric.select(rec, min_total=8, min_dim=0)[0] == 0


def test_floor_and_drop_reasons():
  assert judge_rubric.select(_judged([], candidates=[]))[1] == (
      judge_rubric.DROP_ALL_EMPTY
  )
  assert judge_rubric.select({"ok": False, "candidates": ["x"]})[1] == (
      judge_rubric.DROP_JUDGE_FAILED
  )
  low = _judged([_s(ir=2, ic=2, rs=2)])  # total 6 on the r4 scale
  assert judge_rubric.select(low, min_total=12)[1] == (
      judge_rubric.DROP_BELOW_FLOOR
  )
  assert judge_rubric.select(low, min_total=6)[0] == 0


def test_margin_gate_is_off_by_default_but_available():
  rec = _judged([_s(), _s()], margin=0)
  assert judge_rubric.select(rec)[1] == ""
  assert judge_rubric.select(rec, min_margin=3)[1] == (
      judge_rubric.DROP_BELOW_MARGIN
  )


def test_tie_break_judge_pick_then_shortest():
  rec = _judged(
      [_s(), _s(), _s()],
      candidates=["a long long reply", "short", "mid reply"],
      judge_best_cand_idx=2,
  )
  assert judge_rubric.select(rec)[0] == 2
  rec["judge_best_cand_idx"] = None
  assert judge_rubric.select(rec)[0] == 1  # shortest
  rec["candidates"] = ["same", "same", "same"]
  assert judge_rubric.select(rec)[0] == 0  # lowest index


def test_rubric_text_and_code_agree_on_the_dimension_set():
  """A rubric edit that renames a dimension in ONE of the two places is fatal.

  ``parse_verdicts`` requires every name in ``GRADED_DIMS``; the judge emits
  whatever the prompt's example JSON shows. Rename one and not the other and
  every single call fails to parse -- at full price, after the bulk run has
  started.
  """
  body = prompts.SIM_JUDGE_USER_TEMPLATE
  for dim in judge_rubric.GRADED_DIMS + ("code_leak",):
    assert f"{dim} --" in body, f"{dim} has no anchor block in the rubric"
    assert f'"{dim}"' in body, f"{dim} is missing from the example JSON"
  # And nothing retired is still being asked for.
  for gone in ("information_release", "responsiveness"):
    assert gone not in body, f"{gone} was retired but still appears in r2"


def test_selection_defaults_to_ranking_only_with_the_veto_kept():
  """r2 selects the best surviving candidate; the floors are off by default."""
  # A weak-but-clean group: every candidate would fail r1's total>=12 floor.
  rec = _judged([_s(ir=1, ic=1, rs=1), _s(ir=2, ic=1, rs=1)])
  idx, reason = judge_rubric.select(rec)
  assert reason == "" and idx == 1, "ranking-only selection must keep the best"
  # The veto still overrides any score.
  assert judge_rubric.select(_judged([_s(leak=0, ir=4, ic=4, rs=4)]))[1] == (
      judge_rubric.DROP_ALL_VETOED
  )
  # The floors remain available as an explicit measurement knob.
  assert judge_rubric.select(rec, min_total=12)[1] == (
      judge_rubric.DROP_BELOW_FLOOR
  )


def test_select_works_on_a_record_from_a_DIFFERENT_rubric_version():
  """Comparing r1 and r2 over one pilot batch is the normal case.

  A judged file written under an older rubric carries that rubric's dimension
  names. Iterating the imported ``GRADED_DIMS`` over it raises KeyError, which
  reads as a crash instead of as a version mismatch.
  """
  legacy = {
      "code_leak": 1,
      "information_release": 3,
      "fidelity": 4,
      "in_character": 4,
      "responsiveness": 4,  # r1's retired dimension
      "total": 15,
      "note": "",
  }
  assert judge_rubric.graded_dims_of(legacy) == (
      "fidelity",
      "in_character",
      "information_release",
      "responsiveness",
  )
  rec = {"ok": True, "candidates": ["a reply"], "scores": [legacy], "margin": 0}
  assert judge_rubric.select(rec) == (0, "")
  assert judge_rubric.select(rec, min_total=12, min_dim=2) == (0, "")
  legacy_low = dict(legacy, responsiveness=1)
  rec_low = {"ok": True, "candidates": ["x"], "scores": [legacy_low], "margin": 0}
  assert judge_rubric.select(rec_low, min_total=12, min_dim=2)[1] == (
      judge_rubric.DROP_BELOW_FLOOR
  )


def test_per_dimension_floor_drops_the_prefix_rather_than_taking_the_least_bad():
  """The floor's job under r2: a group where EVERY clean draw dumps is dropped.

  Ranking alone would hand back the least-bad dump and we would fine-tune the
  simulator on it. That is the case the pilot found at 22% of prefixes.
  """
  dumps = _judged([_s(ir=4, ic=4, rs=0), _s(ir=4, ic=4, rs=1)])
  # `rs` is the calibration slot in this helper (see _s).
  assert judge_rubric.select(dumps)[0] == 1, "ranking-only takes the least-bad"
  assert judge_rubric.select(dumps, min_dims={"calibration": 3})[1] == (
      judge_rubric.DROP_BELOW_FLOOR
  )
  # One good draw in the group is enough to keep it, and it must be the one
  # chosen even when a dumpier candidate scores higher overall.
  # dump totals 12, the restrained one 11 -- so ranking on total alone gets it
  # backwards, which is exactly why the floor is on the dimension, not the sum.
  mixed = _judged([_s(ir=4, ic=4, rs=0), _s(ir=2, ic=2, rs=4)])
  assert judge_rubric.select(mixed)[0] == 0, "total alone prefers the dump"
  assert judge_rubric.select(mixed, min_dims={"calibration": 3})[0] == 1


def test_per_dimension_floor_ignores_dimensions_a_record_does_not_have():
  """An r1 record must not be failed by a floor written for an r2 dimension."""
  legacy = {
      "code_leak": 1,
      "information_release": 4,
      "fidelity": 4,
      "in_character": 4,
      "responsiveness": 4,
      "total": 16,
      "note": "",
  }
  rec = {"ok": True, "candidates": ["x"], "scores": [legacy], "margin": 0}
  assert judge_rubric.select(rec, min_dims=judge_rubric.DEFAULT_MIN_DIMS) == (
      0,
      "",
  )
  # But a floor on a dimension it DOES have still applies.
  assert judge_rubric.select(rec, min_dims={"information_release": 5})[1] == (
      judge_rubric.DROP_BELOW_FLOOR
  )


def test_r4_truth_veto_removes_a_candidate_the_ranker_would_have_picked():
  """r4 stage 2, pinned to the r3 failures that forced the split.

  Under r2/r3 these were fidelity's job and it did not do it: four of twelve
  r3-selected targets contradicted the hidden code and all four scored
  fidelity=4 (115-0-0 "threshold of 5.0" against `threshold = 0.5`; 1-0-1
  "os_list is a list of OS names" against `os['partition_size']`). Truth is a
  separate call now, and its verdict is a VETO -- withholding or misstating what
  the code settles makes the episode unwinnable, which is not a milder form of
  over-releasing.

  The ranked dims are deliberately hostile here: the untrue candidate is the
  MOST restrained one, so it wins on total. Only the veto can stop it.
  """
  untrue = _s(ir=4, ic=4, rs=4)      # 12/12 -- the ranker loves it
  true_but_looser = _s(ir=4, ic=4, rs=3)
  assert untrue["total"] > true_but_looser["total"]

  rec = _judged([untrue, true_but_looser])
  assert judge_rubric.select(rec)[0] == 0, "no veto: the ranker takes it"

  for bad in (judge_rubric.TRUTH_WRONG, judge_rubric.TRUTH_UNSURE):
    vetoed = _judged(
        [untrue, true_but_looser], truth_flags=[bad, judge_rubric.TRUTH_OK]
    )
    assert judge_rubric.select(vetoed) == (1, ""), f"{bad} must veto"

  # A group with nothing true left is DROPPED, and attributed to stage 2 rather
  # than being lumped in with the code veto.
  allbad = _judged(
      [untrue, true_but_looser],
      truth_flags=[judge_rubric.TRUTH_WRONG, judge_rubric.TRUTH_UNSURE],
  )
  assert judge_rubric.select(allbad)[1] == judge_rubric.DROP_ALL_UNTRUE
  # An all-code-vetoed group still reports the code reason.
  assert judge_rubric.select(
      _judged([_s(leak=0)], code_vetoed=[0])
  )[1] == judge_rubric.DROP_ALL_VETOED


def test_r4_records_without_truth_flags_still_select():
  """r1-r3 judged files predate stage 2 and must stay readable.

  Comparing rubric versions over one pilot batch is the normal workflow, so an
  absent ``truth_flags`` means "not checked", never "failed".
  """
  rec = _judged([_s(rs=4), _s(rs=3)])
  assert "truth_flags" not in rec
  assert judge_rubric.select(rec, min_dims=judge_rubric.DEFAULT_MIN_DIMS)[0] == 0


def test_r4_pipeline_is_three_stages_and_the_ranker_no_longer_grades_truth():
  """Guards the SHAPE of r4, which is what the split bought.

  A fidelity dimension re-added to the ranking call would let a 4 there outvote
  a stage-2 veto -- the r3 failure exactly -- and nothing else in the suite
  would catch it.
  """
  from colbench import prompts

  assert prompts.JUDGE_RUBRIC_VERSION == "r6"
  assert "fidelity" not in judge_rubric.GRADED_DIMS
  assert judge_rubric.MAX_TOTAL == 12
  assert "fidelity" not in prompts.SIM_JUDGE_USER_TEMPLATE
  assert "fidelity" not in judge_rubric.DEFAULT_MIN_DIMS
  # Stage 2 must carry the four concrete failure patterns it exists to catch.
  t = prompts.SIM_TRUTH_USER_TEMPLATE
  for rule in ("threshold of 5.0", "list of DICTIONARIES", "UNREACHABLE",
               "time * 60", "yes, that's correct"):
    assert rule in t, rule
  assert "unsure" in t and "wrong" in t


def test_r5_calibration_states_that_release_is_cumulative():
  """The r4 defect: `calibration` read release PER SENTENCE.

  114-0-0's target gives the dict shape, the sort key and the direction -- the
  whole of a one-line reference implementation -- and scored 3, because it
  reads as three answers to the agent's three numbered questions. The anchors
  already said "0 = the whole specification"; what was missing was that the
  sum of the sub-answers IS the specification, and that a one-line function is
  fully released by one sentence.

  Also guards the counterweight: without "content the agent already stated is
  not a release", every truthful reply to a trivial task scores 0 while stage 2
  forbids saying nothing, which leaves nothing selectable at all.
  """
  from colbench import prompts

  cal = prompts.SIM_JUDGE_USER_TEMPLATE.split("calibration --")[1].split(
      "in_character --"
  )[0]
  assert "RELEASE IS CUMULATIVE" in cal
  assert "does not make it partial" in cal
  assert "SHORT REFERENCE IMPLEMENTATION" in cal
  assert "ALREADY stated" in cal
  assert "0 = the whole specification" in cal


def test_r6_rubric_separates_derivable_from_arbitrary_content():
  """r6's one idea, and the three r5 misjudgements it fixes.

  116-0-0: the GT hardcodes 14 platform names, the agent asked which platforms
  count, and the selected reply named THREE at calibration 4. The agent cannot
  invent the other eleven, so that reply makes the task fail -- withholding
  arbitrary data the agent asked for SPECIFICALLY is not restraint. Restraint
  is about DERIVABLE logic. Both halves have to be stated or the rubric pushes
  one way at a broad question and the wrong way at a specific one.
  """
  from colbench import prompts

  cal = prompts.SIM_JUDGE_USER_TEMPLATE.split("calibration --")[1].split(
      "in_character --"
  )[0]
  assert "DERIVABLE" in cal and "ARBITRARY" in cal
  assert "MUST BE SUPPLIED, AND SUPPLIED IN FULL" in cal
  assert "it is a WRONG answer" in cal
  # The broad-question anchors must SURVIVE the addition: r6 must not become a
  # licence to dump everything the moment a question mentions a constant.
  assert "0 = the whole specification" in cal
  assert "RELEASE IS CUMULATIVE" in cal


def test_r6_truth_stage_checks_equivalence_and_catches_self_denial():
  """The three stage-2 defects found by reading the base-partner batch.

  * arithmetic paraphrase scored `wrong`: "a 20% disadvantage" against
    `return 0.8 * force_level` (3 false positives in prefix 107-0-0 alone);
  * "I don't know what the Salter Sink method is" scored `ok` although the
    client's own opening request names it (3 of 15 selected targets, all one
    task);
  * a partial answer about arbitrary data scored `ok`, which is how naming 3 of
    14 list entries became a 4/4 target.
  """
  from colbench import prompts

  t = prompts.SIM_TRUTH_USER_TEMPLATE
  assert "EQUIVALENT PARAPHRASE" in t and "0.8 * force_level" in t
  assert "DENYING THE CLIENT'S OWN REQUEST" in t
  assert "PARTIAL ANSWER ABOUT ARBITRARY DATA" in t
  # Brevity about derivable logic must stay allowed, or this collapses into
  # "always dump everything".
  assert "Being brief about DERIVABLE logic is not" in t


def test_detector_c_separates_a_body_fragment_from_a_data_example():
  """``expr_over_gt_names``: the hole that let a backticked body through.

  The code_leak anchor has always named dict literals and indexing expressions
  as leaks, but nothing implemented them -- (A) needs a def, (B) needs a fence.
  Measured: 9.0% of base-partner candidates handed over the function body as a
  backticked expression and BOTH the regex and the judge scored them clean.

  The line is expression-vs-data, not brace-vs-no-brace, which is why 1-0-0's
  ``[{'disk': 'sda', ...}]`` must stay clean: concrete values answering "what
  shape is the input?", with key names the agent cannot guess.

  OFF by default: ``env.py`` uses this function for ``sim_leak_frac``, and
  widening the default would silently move that metric for every run.
  """
  from colbench import templates

  gt = (
      "def f(num_teams, new_plan_cost_per_team, pension_cost_per_team):\n"
      "    return {'total': num_teams * (new_plan_cost_per_team - 1)}\n"
  )
  body = "it should return `(num_teams * (new_plan_cost_per_team - 1))`"
  data = "os_list looks like [{'disk': 'sda', 'partition_size': 100}]"
  prose = "take the number of teams times the difference in cost per team"

  assert templates.detect_code_leak(body, gt, ngram_n=0) is None, "the hole"
  assert templates.detect_code_leak(
      body, gt, ngram_n=0, expr_over_gt_names=True
  ) == "expr"
  for clean in (data, prose):
    assert templates.detect_code_leak(
        clean, gt, ngram_n=0, expr_over_gt_names=True
    ) is None, clean


def test_detector_c_does_not_fire_on_hyphenated_names_or_data_examples():
  """The two false positives detector (C) shipped with, both measured.

  On prefix 116-0-0 it vetoed SIX of seven candidates: "Atari 8-bit" parsed as
  `8 - bit` (and `bit` is an identifier of the GT string 'Atari 8-bit'), and
  "TRS-80" as `TRS - 80`. Hyphenated proper nouns are ordinary content in the
  hidden data, so `-`/`+` now require whitespace beside them -- real leaks are
  spaced ("employees - 126") or use `*`/`/` ("1.2*force_level").

  Separately, "list1 is [1, 2, 2, 3]" was called a subscript because the
  pattern allowed whitespace before the bracket. A data example is not an
  index, so the bracket must now be adjacent.
  """
  from colbench import templates

  platforms = (
      "def classify_game_platform(media_type):\n"
      "    old_platforms = ['Apple II', 'Atari 8-bit', 'TRS-80 CoCo']\n"
      "    return 'Old' if media_type in old_platforms else 'New'\n"
  )
  lists = (
      "def count_elements_in_common(list1, list2):\n"
      "    return len([e for e in list1 if e in list2])\n"
  )
  for text, gt in (
      ("Old platforms include Apple II, Atari 8-bit, and TRS-80 CoCo", platforms),
      ("if list1 is [1, 2, 2, 3] and list2 is [2, 2, 3, 4] it returns 2", lists),
  ):
    assert templates.detect_code_leak(
        text, gt, ngram_n=0, expr_over_gt_names=True
    ) is None, text

  # ...while a spaced subtraction over a GT name is still caught.
  assert templates.detect_code_leak(
      "the answer is list1 - list2", lists, ngram_n=0, expr_over_gt_names=True
  ) == "expr"

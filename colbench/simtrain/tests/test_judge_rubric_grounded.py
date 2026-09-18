"""CPU tests for the GROUNDED-arm rubric (g1). No network.

g1 has no LLM veto stage, so the two things that can silently go wrong are
different from r6's. (1) Both vetoes are PROGRAMMATIC, so a candidate that
writes code or ends the conversation early must never reach the ranker -- and
`terminate_allowed` missing from an older prefix file must fail CLOSED. (2) The
scoring helpers are shared with r6, so they must read dimensions off the RECORD:
g1 verdicts have no `code_leak` field, and a helper that assumes one would score
every g1 candidate as vetoed.

Run: pytest colbench/simtrain/tests/test_judge_rubric_grounded.py
"""

from colbench import prompts
from colbench.simtrain import judge_rubric

GT = (
    "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n"
    "        return x - y\n"
)
PLOT = "The user would only mention, once they see code, that ten counts."
PREFIX = {
    "prefix_id": "7-0-2",
    "problem_description": "Create a python function f(x, y).",
    "ground_truth": GT,
    "plot": PLOT,
    "terminate_allowed": True,
    "sim_dialogue": [
        {"role": "user", "content": "Create a python function f(x, y)."},
        {"role": "assistant", "content": "What decides which branch?"},
    ],
}


def _verdict(gt_adh=4, plot_adh=4, not_over=4, note="n"):
  return {"gt_adherence": gt_adh, "plot_adherence": plot_adh,
          "not_overhelpful": not_over, "note": note,
          "total": gt_adh + plot_adh + not_over}


# ── the three programmatic vetoes ────────────────────────────────────────────

def test_code_writing_is_vetoed():
  cands = ["Ten is the cutoff.", "You want `def f(x, y): return x + y`."]
  code, term, _ = judge_rubric.grounded_vetoes(PREFIX, cands)
  assert code == [1]
  assert term == []


def test_termination_is_vetoed_only_before_code_is_shown():
  cands = ["Ten is the cutoff.", "[TERMINATE]"]
  # A complete function is on the table -> ending is permitted.
  _, term, _ = judge_rubric.grounded_vetoes(dict(PREFIX, terminate_allowed=True), cands)
  assert term == []
  # Nothing shown yet -> ending is premature.
  _, term, _ = judge_rubric.grounded_vetoes(dict(PREFIX, terminate_allowed=False), cands)
  assert term == [1]


def test_missing_terminate_allowed_fails_closed():
  """An older prefix file must not silently let every early exit through."""
  prefix = {k: v for k, v in PREFIX.items() if k != "terminate_allowed"}
  _, term, _ = judge_rubric.grounded_vetoes(prefix, ["[TERMINATE]"])
  assert term == [0]


def test_detector_c_is_on_for_the_grounded_arm():
  """The grounded prompt forbids quoting 'a variable name, or a literal value
  as code', which is exactly detector (C).

  Needs a REALISTIC ground truth: detector (C) gates on identifiers of 3+
  characters, because a one- or two-letter match fires on ordinary prose. A
  toy `f(x, y)` GT contributes no gating names at all, so nothing can fire.
  """
  gt = (
      "def total_fee(base_rate, hours_worked):\n"
      "    return base_rate * hours_worked\n"
  )
  prefix = dict(PREFIX, ground_truth=gt)
  cands = ["It should be `base_rate * hours_worked` basically."]
  code, _, _ = judge_rubric.grounded_vetoes(prefix, cands)
  assert code == [0]
  code_off, _, _ = judge_rubric.grounded_vetoes(
      prefix, cands, expr_over_gt_names=False
  )
  assert code_off == []
  # Prose describing the same behaviour is NOT a leak.
  clean, _, _ = judge_rubric.grounded_vetoes(
      prefix, ["It is the base rate times the hours worked."]
  )
  assert clean == []


def test_speaking_and_ending_in_one_reply_is_vetoed_on_form():
  """The prompt says the sentinel is a signal, not a message. At rollout time
  `sim_terminated`'s unanchored match ends the episode, so the spoken half is
  discarded -- training on it teaches a shape whose content is thrown away.

  The veto is on FORM, so it fires regardless of `terminate_allowed`: a bare
  sentinel is fine whenever ending is permitted, and a mixed reply is malformed
  even then.
  """
  cands = [
      "[TERMINATE]",                       # 0: the form the prompt asks for
      "Looks good, thanks! [TERMINATE]",   # 1: spoke AND ended
      "**[TERMINATE]**",                   # 2: emphasis still reads as bare
      "I shouldn't say [TERMINATE] yet.",  # 3: a MENTION, also forbidden
      "That threshold is ten.",            # 4: still talking, clean
  ]
  prefix = dict(PREFIX, terminate_allowed=True)
  _, term, form = judge_rubric.grounded_vetoes(prefix, cands)
  assert term == []
  assert form == [1, 3]


def test_form_veto_and_premature_veto_are_reported_separately():
  """A premature MIXED reply breaks both rules, and the dump has to be able to
  say which -- so the two lists are independent, not exclusive.
  """
  cands = ["Sure, that's right. [TERMINATE]"]
  _, term, form = judge_rubric.grounded_vetoes(
      dict(PREFIX, terminate_allowed=False), cands
  )
  assert term == [0]
  assert form == [0]


# ── the ranking call ─────────────────────────────────────────────────────────

def test_judge_messages_carry_gt_plot_and_dialogue():
  msgs, mapping = judge_rubric.build_grounded_judge_messages(
      PREFIX, ["Ten.", "Above ten you add."], perm_seed=0
  )
  user = msgs[1]["content"]
  assert GT.strip() in user
  assert PLOT in user
  assert "What decides which branch?" in user
  assert sorted(mapping) == ["A", "B"]
  assert sorted(mapping.values()) == [0, 1]


def test_empty_plot_renders_a_placeholder_not_a_blank():
  msgs, _ = judge_rubric.build_grounded_judge_messages(
      dict(PREFIX, plot=""), ["Ten."], perm_seed=0
  )
  assert "(no plot for this task)" in msgs[1]["content"]


# ── shared scoring helpers must be rubric-agnostic ───────────────────────────

def test_total_score_reads_dims_off_the_record():
  """A g1 verdict has no `code_leak`; a helper assuming one scores it 0."""
  assert judge_rubric.total_score(_verdict(4, 4, 4)) == 12
  assert judge_rubric.total_score(_verdict(2, 1, 3)) == 6
  assert judge_rubric.graded_dims_of(_verdict()) == (
      "gt_adherence", "not_overhelpful", "plot_adherence"
  )


def test_r6_verdicts_still_honour_the_code_leak_veto():
  """The absent-means-no-veto rule must not disarm r6's real veto."""
  r6 = {"code_leak": 0, "volunteering": 4, "calibration": 4,
        "in_character": 4, "note": "n"}
  assert judge_rubric.total_score(r6) == 0


def test_select_picks_the_highest_total_among_unvetoed():
  judged = {
      "candidates": ["a", "b", "c"],
      "ok": True,
      "scores": [None, _verdict(4, 4, 4), _verdict(3, 4, 4)],
  }
  pick, why = judge_rubric.select(judged)
  assert (pick, why) == (1, "")


def test_a_content_free_reply_cannot_win_on_gt_adherence():
  """THE degenerate optimum this rubric has to resist.

  A reply that says nothing is vacuously restrained, so `not_overhelpful` and
  `plot_adherence` both max out; only `gt_adherence` can stop it, which is why
  withholding is anchored at 1 rather than 4. Pin the arithmetic: a
  say-nothing reply must lose to one that answers.
  """
  says_nothing = _verdict(gt_adh=1, plot_adh=4, not_over=4)   # withholds
  answers = _verdict(gt_adh=4, plot_adh=4, not_over=3)        # answers, 1 extra
  assert judge_rubric.total_score(says_nothing) < judge_rubric.total_score(answers)
  judged = {"candidates": ["a", "b"], "ok": True,
            "scores": [says_nothing, answers]}
  pick, _ = judge_rubric.select(judged)
  assert pick == 1


def test_rubric_text_states_the_two_documented_traps():
  """These clauses are load-bearing; a reword must not quietly drop them."""
  tpl = prompts.GROUNDED_JUDGE_USER_TEMPLATE
  # the r2 / 12-0-0 trap: silence must not be rewarded
  assert "WITHHOLDS" in tpl
  # the r6 `volunteering` miscalibration: length is not the measure
  assert "LENGTH IS NOT THE MEASURE" in tpl
  # a reply must not be penalised for merely not advancing the plot
  assert "not penalised for simply not advancing the plot" in tpl
  # the rubric must NOT ask the judge to decide plot TIMING
  assert "on schedule" not in tpl and "is DUE" not in tpl
  # TERMINATION IS NOT SCORED, AND THAT IS A MEASURED DECISION, NOT AN OMISSION.
  # g2 added one sentence scoring a bare [TERMINATE] against whether the plot
  # "had been fully played out by then". Judged over the same 85 grounded
  # prefixes it flipped nine bare terminations 4 -> 0, and hand adjudication put
  # SEVEN of the nine wrong: the judge invented a confirm-before-you-leave
  # obligation that is in neither the plot nor the simulator's prompt (13-0-1,
  # whose own note concedes "the function matches the reference"), or scored 0
  # merely because the plot's conditional branch had never fired (30-0-1).
  # Terminating without checking the code is BY DESIGN correct for this
  # simulator, so no dimension may mention the sentinel.
  assert "[TERMINATE]" not in tpl
  assert "end the conversation" not in tpl


# ── end to end, with a stubbed model ─────────────────────────────────────────

def test_end_to_end_grounded_judging_with_a_stub():
  """Both vetoes fire, exactly ONE LLM call is made, and the say-nothing reply
  loses. Pins the whole g1 contract in one place."""
  import json
  from colbench.selfplay.llm_client import ChatEndpoint
  from colbench.simtrain import judge_candidates

  gt = (
      "def total_fee(base_rate, hours_worked):\n"
      "    return base_rate * hours_worked\n"
  )
  prefix = dict(
      PREFIX, ground_truth=gt, terminate_allowed=False,
      task_index=3, task_id=3, episode_idx=0, turn_idx=1,
      sim_user_sha16="abc",
      sim_dialogue=[
          {"role": "user", "content": "Write total_fee."},
          {"role": "assistant", "content": "How is the fee computed?"},
      ],
  )
  cands = {
      "prefix_id": prefix["prefix_id"],
      "candidates": [
          "It's the base rate multiplied by the hours worked.",  # answers
          "It should be `base_rate * hours_worked`.",            # code veto
          "[TERMINATE]",                                         # early-term veto
          "I'm not sure, whatever you think.",                   # withholds
      ],
      "dup_counts": [3, 2, 1, 2],
  }
  calls = []

  def fake(messages):
    calls.append(messages)
    body = messages[1]["content"]
    labels = sorted(l for l in LABELS_USED if f"--- Candidate {l} ---" in body)
    verdicts = []
    for l in labels:
      seg = body.split(f"--- Candidate {l} ---")[1].split("--- Candidate")[0]
      withholds = "not sure" in seg
      verdicts.append({
          "label": l,
          "gt_adherence": 1 if withholds else 4,
          "plot_adherence": 4,
          "not_overhelpful": 4,
          "note": "withholds" if withholds else "answers",
      })
    return json.dumps({"verdicts": verdicts, "best_label": labels[0]})

  endpoint = ChatEndpoint(
      base_url="x", model="qwen3_235b", api_key="E", backend=fake
  )
  rec = judge_candidates.judge_one_grounded(
      prefix, cands, endpoint, endpoint, perm_seed=0
  )
  assert rec["rubric_version"] == prompts.GROUNDED_JUDGE_RUBRIC_VERSION
  assert rec["code_vetoed"] == [1]
  assert rec["term_vetoed"] == [2]
  assert rec["parse_error"] == ""
  assert len(calls) == 1, "g1 must make ONE call; r6's two-stage shape leaked in"
  pick, _ = judge_rubric.select(rec)
  assert pick == 0


LABELS_USED = "ABCDEFGH"

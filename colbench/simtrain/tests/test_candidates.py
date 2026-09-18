"""CPU tests for Stage 3a (candidate draws) and the judged-dump renderer.

The transform under test that matters is ``env.finalize_sim_reply``: candidates
are drawn OUTSIDE any env, so if that function ever stops matching
``ColBenchUserSimEnv._finalize_reply`` the whole dataset is built from strings
the rollout would never have produced.

Run: pytest colbench/simtrain/tests/test_candidates.py
"""

# pylint: disable=protected-access

import json
import os
import tempfile

import pytest

from colbench import prompts
from colbench.env import ColBenchUserSimEnv
from colbench.env import finalize_sim_reply
from colbench.simtrain import collect_candidates
from colbench.simtrain import dump_judged
from colbench.simtrain import judge_rubric
from colbench.simtrain.collect_prefixes import sha16

GT = "def f(x, y):\n    return x + y if x >= 10 else x - y\n"
PREFIX = {
    "prefix_id": "3-0-1",
    "task_index": 3,
    "task_id": 103,
    "episode_idx": 0,
    "turn_idx": 1,
    "sim_system": "You are a helpful assistant.",
    "sim_user": "rendered prompt with the hidden GT",
    "problem_description": "Create a python function f(x, y).",
    "ground_truth": GT,
    "sim_dialogue": [
        {"role": "user", "content": "Create a python function f(x, y)."},
        {"role": "assistant", "content": "Which branch when x is small?"},
    ],
    "partner_reply": "Which branch when x is small?",
    "sim_reply": "Below ten you subtract.",
    "sim_reply_leak_reason": None,
    "sim_reply_leak_reason_strict": None,
    "episode_terminated_by": "turn_cap",
}
PREFIX["sim_user_sha16"] = sha16(PREFIX["sim_user"])


class _Endpoint:
  """A ChatEndpoint stand-in returning scripted replies in order."""

  def __init__(self, replies):
    self.replies = list(replies)
    self.calls = 0

  def chat(self, messages):
    self.messages = messages
    r = self.replies[min(self.calls, len(self.replies) - 1)]
    self.calls += 1
    return r


@pytest.fixture(autouse=True)
def _no_char_slice(monkeypatch):
  monkeypatch.setenv("SIM_CHAR_LIMIT", "0")


def test_finalize_sim_reply_matches_the_env_method():
  """The one function shared across the env boundary. A table, not a spot check."""
  env = ColBenchUserSimEnv(
      problem_description="p",
      ground_truth=GT,
      test_cases=[],
      sim_backend=lambda a, b: "",
  )
  cases = [
      "",
      "plain reply",
      "<think>hidden</think>visible",
      "<think>unterminated reasoning that never closes",
      "x" * 900,
      "  leading and trailing  ",
      "multi\nline\nreply",
  ]
  for limit in ("0", "400", "10"):
    os.environ["SIM_CHAR_LIMIT"] = limit
    for raw in cases:
      assert finalize_sim_reply(raw) == env._finalize_reply(raw, "u"), (
          limit,
          raw,
      )


def test_dedupe_keeps_counts_and_drops_dead_draws():
  ep = _Endpoint(["same", "same", "", "No response.", "other", "same"])
  rec = collect_candidates.draw_candidates(PREFIX, ep, k=6)
  assert rec["candidates"] == ["same", "other"]
  assert rec["dup_counts"] == [3, 1]
  assert rec["n_empty"] == 2
  assert rec["n_unique"] == 2
  assert rec["n_draws"] == 6
  assert rec["sim_user_sha16"] == PREFIX["sim_user_sha16"]
  # The prompt sent is the MATERIALIZED one, verbatim -- nothing re-rendered.
  assert ep.messages == [
      {"role": "system", "content": PREFIX["sim_system"]},
      {"role": "user", "content": PREFIX["sim_user"]},
  ]


def test_is_dead_covers_the_backend_fallback():
  assert collect_candidates.is_dead("")
  assert collect_candidates.is_dead("   ")
  assert collect_candidates.is_dead("No response.")
  assert not collect_candidates.is_dead("No.")


def test_char_limit_applies_to_candidate_draws(monkeypatch):
  monkeypatch.setenv("SIM_CHAR_LIMIT", "5")
  ep = _Endpoint(["abcdefghij"])
  rec = collect_candidates.draw_candidates(PREFIX, ep, k=1)
  assert rec["candidates"] == ["abcde"]


def test_all_draws_dead_yields_an_empty_candidate_list():
  rec = collect_candidates.draw_candidates(PREFIX, _Endpoint([""]), k=4)
  assert rec["candidates"] == []
  assert rec["n_empty"] == 4


def test_existing_prefix_ids_resume():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "candidates.jsonl")
    assert not collect_candidates.existing_prefix_ids(path)
    with open(path, "w", encoding="utf-8") as f:
      f.write(json.dumps({"prefix_id": "3-0-1"}) + "\n")
    assert collect_candidates.existing_prefix_ids(path) == {"3-0-1"}


# ── the dump ─────────────────────────────────────────────────────────────────


def _judged(candidates, verdicts, ok=True, **extra):
  scores = []
  for v in verdicts:
    v = dict(v)
    v["total"] = judge_rubric.total_score(v)
    v.setdefault("note", "")
    scores.append(v)
  rec = {
      "prefix_id": PREFIX["prefix_id"],
      "task_id": PREFIX["task_id"],
      "turn_idx": PREFIX["turn_idx"],
      "candidates": candidates,
      "dup_counts": [1] * len(candidates),
      "scores": scores,
      "ok": ok,
      "margin": 2,
      "judge_incoherent": False,
      "judge_best_cand_idx": None,
      "parse_error": "",
  }
  rec.update(extra)
  return rec


def _v(leak=1, ir=4, ic=4, rs=4):
  v = {
      "code_leak": leak,
      "volunteering": ir,
      "in_character": ic,
      "calibration": rs,
  }
  v["total"] = judge_rubric.total_score(v)
  return v


def test_page_shows_scores_candidates_and_the_hidden_gt():
  rec = _judged(["Ten is the cutoff.", "I don't know."], [_v(), _v(ir=1)])
  page = dump_judged.render_prefix(PREFIX, rec)
  assert "Ten is the cutoff." in page and "I don't know." in page
  assert "def f(x, y)" in page  # the hidden information is shown to the reader
  assert "Which branch when x is small?" in page
  # Sorted best-first: the perfect score must be rendered above the lower one.
  # Matched on the "n/MAX" form because the renderer right-pads the numerator.
  top = f"{judge_rubric.MAX_TOTAL}/{judge_rubric.MAX_TOTAL}"
  low = f"{judge_rubric.MAX_TOTAL - 3}/{judge_rubric.MAX_TOTAL}"
  assert page.index(top) < page.index(low)


def test_page_flags_a_judge_only_prose_leak():
  """The cell the whole judge exists for."""
  rec = _judged(["The rule is: add them when x is at least ten."], [_v(leak=0)])
  page = dump_judged.render_prefix(PREFIX, rec)
  assert "judge-only leak" in page


def test_page_flags_a_regex_only_leak_the_judge_missed():
  leaky = "You want ```python\ndef f(x, y):\n    return x + y\n```"
  rec = _judged([leaky], [_v(leak=1)])
  page = dump_judged.render_prefix(PREFIX, rec)
  assert "regex-only leak" in page


def test_failed_judge_page_shows_the_raw_reply():
  rec = _judged([], [], ok=False, parse_error="no JSON object found",
                raw="I cannot score these.")
  page = dump_judged.render_prefix(PREFIX, rec)
  assert "JUDGE FAILED" in page and "I cannot score these." in page


def test_summary_counts_spread_and_the_leak_agreement_table():
  rows = [
      _judged(["a good reply", "another"], [_v(), _v(ir=1)]),
      _judged(["The rule is x plus y above ten."], [_v(leak=0)]),
  ]
  rows[1]["prefix_id"] = "3-0-2"
  prefixes = {"3-0-1": PREFIX, "3-0-2": PREFIX}
  out = dump_judged.summarize(rows, prefixes)
  assert "judge_parse_fail_rate" in out
  assert "WITHIN-GROUP SPREAD" in out
  assert "JUDGE only" in out and "REGEX only" in out
  assert "selection keep rate" in out


def test_system_variant_replaces_only_the_system_message():
  """The teacher prompt must not disturb the materialized USER message.

  The whole point of the override is a candidate that answers the EXACT prompt
  the production simulator was asked, in a different voice. If the user message
  drifted too, the candidate would be answering a different question and the
  distillation target would be meaningless.
  """
  ep = _Endpoint(["a reply"])
  rec = collect_candidates.draw_candidates(PREFIX, ep, k=1, system_variant="role")
  assert ep.messages[0]["content"] == prompts.SIM_ROLE_SYSTEM_PROMPT
  assert ep.messages[0]["content"] != PREFIX["sim_system"]
  assert ep.messages[1]["content"] == PREFIX["sim_user"]
  assert rec["sim_system_variant"] == "role"

  ep = _Endpoint(["a reply"])
  rec = collect_candidates.draw_candidates(PREFIX, ep, k=1)
  assert ep.messages[0]["content"] == PREFIX["sim_system"]
  assert rec["sim_system_variant"] == "default"


def test_restraint_variant_extends_the_role_variant():
  """role_restraint must be role PLUS restraint, so the arms differ in one axis."""
  assert prompts.SIM_ROLE_RESTRAINT_SYSTEM_PROMPT.startswith(
      prompts.SIM_ROLE_SYSTEM_PROMPT
  )
  # And neither teacher mentions the hidden information -- only the user
  # message may carry it.
  for p in (prompts.SIM_ROLE_SYSTEM_PROMPT,
            prompts.SIM_ROLE_RESTRAINT_SYSTEM_PROMPT):
    assert "hidden" not in p.lower()

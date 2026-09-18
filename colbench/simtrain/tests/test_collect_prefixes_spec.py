"""CPU tests for the GROUNDED/spec prefix collector.

No server, no GPU: both roles are scripted callables.

Two things here are load-bearing and nothing downstream would notice if they
broke. (1) The materialized ``(sim_system, sim_user)`` must be byte-identical to
what the env builds -- otherwise every later stage scores and trains on bytes
the simulator was never asked. (2) ``terminate_allowed`` must record the
MINIMUM bar (a complete function is on the table) correctly, because the whole
point of grading termination timing is to compare the sim's choice against
whether ending was permitted at all -- and getting that field wrong would
silently invert the judgement.

Run: pytest colbench/simtrain/tests/test_collect_prefixes_spec.py
"""

# These tests pin the behaviour of module-private helpers.
# pylint: disable=protected-access

from colbench import templates
from colbench.simtrain import collect_prefixes

GT = (
    "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n"
    "        return x - y\n"
)
PROBLEM = "Create a python function f(x, y) that combines two numbers."
PLOT = (
    "The user would only mention, once they see the code, that the cutoff "
    "should be inclusive."
)
SUBMISSION = "```python\ndef f(x, y):\n    return x + y\n```"


def _task(task_index=0, plot=PLOT):
  return {
      "task_index": task_index,
      "task_id": 100 + task_index,
      "prompt_messages": [
          {"role": "system",
           "content": templates.COLBENCH_SPEC_AGENT_SYSTEM_PROMPT},
          {"role": "user", "content": PROBLEM},
      ],
      "problem_text": PROBLEM,
      "problem_description": PROBLEM,
      "ground_truth": GT,
      "test_cases": ["f(1, 2)", "f(20, 5)"],
      "spec": {"plot": plot},
  }


def _scripted(turns):
  def fn(*_args, **_kwargs):
    i = min(fn.calls, len(turns) - 1)
    fn.calls += 1
    return turns[i]
  fn.calls = 0
  return fn


def _run(solver_turns, sim_turns, **kw):
  return collect_prefixes.run_episode_spec(
      _task(), 0,
      _scripted(solver_turns), _scripted(sim_turns),
      max_assistant_turns=kw.pop("max_assistant_turns", 6),
      provenance={"partner_model": "test"},
      **kw,
  )


def test_materialized_prompt_is_byte_identical_to_the_env():
  """The invariant the whole pipeline rests on (asserted in-loop too)."""
  prefixes, _ = _run(
      ["What is the cutoff?", SUBMISSION],
      ["Ten.", "[TERMINATE]"],
  )
  assert prefixes
  for rec in prefixes:
    system, user = templates.build_grounded_sim_messages(
        PROBLEM, GT, PLOT, rec["sim_dialogue"]
    )
    assert rec["sim_system"] == system
    assert rec["sim_user"] == user
    assert rec["sim_user_sha16"] == collect_prefixes.sha16(user)


def test_grounded_sim_prompt_carries_the_gt_and_the_plot():
  """Grounded conditioning is GT source + plot; that is what makes code-leak
  detection against the GT meaningful on this arm."""
  prefixes, _ = _run(["What is the cutoff?"], ["Ten."])
  assert GT.strip() in prefixes[0]["sim_system"]
  assert PLOT in prefixes[0]["sim_system"]
  assert prefixes[0]["plot"] == PLOT
  assert prefixes[0]["sim_conditioning"] == "grounded"


def test_terminate_allowed_tracks_whether_code_is_on_the_table():
  """False before any complete function, True afterwards."""
  prefixes, _ = _run(
      ["What is the cutoff?", "Anything else?", SUBMISSION, "Updated?"],
      ["Ten.", "No.", "Looks right.", "Fine."],
  )
  by_turn = {r["turn_idx"]: r["terminate_allowed"] for r in prefixes}
  assert by_turn[0] is False
  assert by_turn[1] is False
  assert by_turn[2] is True   # the SUBMISSION turn itself already counts


def test_user_termination_after_code_ends_the_episode():
  prefixes, episode = _run([SUBMISSION], ["[TERMINATE]"])
  assert episode["episode_terminated_by"] == "user"
  assert episode["showed_code"] is True
  assert prefixes[-1]["sim_reply_terminated"] is True
  assert all(r["episode_terminated_by"] == "user" for r in prefixes)


def test_code_cap_stops_the_episode():
  prefixes, episode = _run(
      [SUBMISSION, SUBMISSION, SUBMISSION],
      ["Not quite.", "Still off.", "Hmm."],
      max_code_proposals=2,
  )
  assert episode["episode_terminated_by"] == "code_cap"
  assert episode["n_code_proposals"] == 2
  del prefixes


def test_turn_cap_without_code_is_no_code():
  _, episode = _run(
      ["What is the cutoff?"], ["Ten."], max_assistant_turns=3,
  )
  assert episode["episode_terminated_by"] == "no_code"
  assert episode["showed_code"] is False


def test_solver_empty_reply_ends_the_episode():
  _, episode = _run(["   "], ["Ten."])
  assert episode["episode_terminated_by"] == "solver_empty"


def test_leak_detection_is_recorded_against_the_gt():
  """The grounded sim SEES the GT, so this measures something real."""
  prefixes, _ = _run(
      ["What is the cutoff?"],
      ["You want `def f(x, y): return x + y` basically."],
  )
  assert prefixes[0]["sim_reply_leak_reason"]


def test_missing_spec_yields_an_empty_plot_rather_than_crashing():
  task = _task()
  task.pop("spec")
  prefixes, _ = collect_prefixes.run_episode_spec(
      task, 0, _scripted(["What is the cutoff?"]), _scripted(["Ten."]),
      max_assistant_turns=3, provenance={},
  )
  assert prefixes[0]["plot"] == ""


def test_grounded_prefix_rejects_a_gt_path_teacher_prompt(tmp_path):
  """role/role_restraint carry no GT and no plot; applying one to a grounded
  prefix would silently draw candidates from a blind simulator."""
  import json
  import os
  import pytest
  from colbench.simtrain import collect_candidates

  # main() asserts this before anything else; unset would truncate every reply.
  os.environ.setdefault("SIM_CHAR_LIMIT", "0")

  prefixes = tmp_path / "prefixes.jsonl"
  rec = {
      "prefix_id": "1-0-0",
      "sim_system": "grounded system",
      "sim_user": "u",
      "sim_user_sha16": collect_prefixes.sha16("u"),
      "sim_conditioning": "grounded",
  }
  prefixes.write_text(json.dumps(rec) + "\n")
  argv = ["--prefixes", str(prefixes), "--out", str(tmp_path / "c.jsonl"),
          "--sim_base_url", "u", "--sim_model", "m",
          "--sim_system", "role_restraint"]
  with pytest.raises(SystemExit, match="GT-path teacher prompt"):
    collect_candidates.main(argv)


def test_early_term_guard_is_a_policy_separate_from_the_recorded_fact():
  """`qwen3_4b_spec_rej8` ran with --noearly_term_guard, so collection must be
  able to reproduce that. The recorded FACT must not change with the policy."""
  # Guard OFF: the sim's [TERMINATE] stands even with no code shown.
  _, episode = _run(["What is the cutoff?"], ["[TERMINATE]"],
                    early_term_guard=False)
  assert episode["episode_terminated_by"] == "no_code"
  assert episode["early_term_guard"] is False

  # Guard ON is the default, and `terminate_allowed` is recorded either way.
  for guard in (True, False):
    prefixes, _ = _run(
        ["What is the cutoff?", SUBMISSION, "Updated?"],
        ["Ten.", "Looks right.", "Fine."],
        early_term_guard=guard,
    )
    by_turn = {r["turn_idx"]: r["terminate_allowed"] for r in prefixes}
    assert by_turn[0] is False, "no code shown yet"
    assert by_turn[1] is True, "a complete function is on the table"


def test_rejected_drafts_are_recorded_not_just_counted():
  """`sim_reply` is what the SAMPLER settled on; the discarded drafts are what
  the sim actually wanted to say, i.e. its modal behaviour."""
  # Draw 1 writes code (rejected), draw 2 is clean (accepted).
  draws = iter([
      "You want `def f(x, y): return x + y` basically.",
      "Ten is the cutoff.",
  ])

  def sim(*_a, **_k):
    return next(draws, "Ten is the cutoff.")

  prefixes, _ = collect_prefixes.run_episode_spec(
      _task(), 0, _scripted(["What is the cutoff?"]), sim,
      max_assistant_turns=3, provenance={}, sim_max_tries=8,
      sim_code_leak_detector="a0_strict",
  )
  rec = prefixes[0]
  assert rec["sim_reply"] == "Ten is the cutoff."
  assert rec["sim_code_rejected"] == 1
  assert rec["sim_code_reject_samples"], "the discarded draft must be KEPT"
  assert "def f" in rec["sim_code_reject_samples"][0]

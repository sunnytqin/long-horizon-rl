"""CPU tests for Stage 1 (the prefix collector).

No server, no GPU, no tokenizer: both roles are scripted callables. The
load-bearing test here is ``test_materialized_prompt_is_byte_identical`` -- if
the recorded ``(sim_system, sim_user)`` ever drifts from what
``env._build_sim_prompt`` builds, every stage downstream is scoring and training
on bytes the simulator was never asked, and nothing else in the pipeline would
notice.

Run: pytest colbench/simtrain/tests/test_collect_prefixes.py
"""

# These tests pin the behaviour of module-private helpers, so they reach for
# them directly.
# pylint: disable=protected-access

import json
import os
import tempfile

import pytest

# The module-level setup above has to run before these imports resolve, so they
# cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench import templates
from colbench.env import ColBenchUserSimEnv
from colbench.simtrain import collect_prefixes

GT = (
    "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n"
    "        return x - y\n"
)
PROBLEM = "Create a python function f(x, y) that combines two numbers."
SUBMISSION = "```python\ndef f(x, y):\n    return x + y\n```"


def _task(task_index=0):
  return {
      "task_index": task_index,
      "task_id": 100 + task_index,
      "prompt_messages": [
          {"role": "system", "content": templates.COLBENCH_AGENT_SYSTEM_PROMPT},
          {"role": "user", "content": PROBLEM},
      ],
      "problem_text": PROBLEM,
      "problem_description": PROBLEM,
      "ground_truth": GT,
      "test_cases": ["f(1, 2)", "f(20, 5)"],
  }


def _scripted_solver(turns):
  """A solver that emits ``turns`` in order, then repeats the last one."""

  def solver(messages):
    del messages
    i = min(solver.calls, len(turns) - 1)
    solver.calls += 1
    return turns[i]

  solver.calls = 0
  return solver


def _sim(reply="Sure, x should be at least ten."):
  """A sim backend that always returns ``reply``, recording its prompts."""

  def backend(system_content, user_content):
    backend.seen.append((system_content, user_content))
    return reply

  backend.seen = []
  return backend


def _run(solver_turns, sim_reply="Anything above ten counts.", max_turns=10):
  sim = _sim(sim_reply)
  prefixes, episode = collect_prefixes.run_episode(
      _task(),
      episode_idx=0,
      solver_chat=_scripted_solver(solver_turns),
      sim_backend=sim,
      max_assistant_turns=max_turns,
      provenance={"sim_model": "stub"},
  )
  return prefixes, episode, sim


@pytest.fixture(autouse=True)
def _no_char_slice(monkeypatch):
  """Match training (run_colbench_grpo.sh exports SIM_CHAR_LIMIT=0)."""
  monkeypatch.setenv("SIM_CHAR_LIMIT", "0")


def test_materialized_prompt_is_byte_identical():
  """The recorded prompt == what the env would have built. THE invariant.

  Everything downstream reads ``sim_user`` verbatim instead of re-rendering, so
  this equality is the only thing standing between the pipeline and silently
  judging/training on a prompt the simulator never saw.
  """
  prefixes, _, sim = _run(
      ["What is the threshold?", "And below it?"], max_turns=3
  )
  assert len(prefixes) == 2
  env = ColBenchUserSimEnv(
      problem_description=PROBLEM,
      ground_truth=GT,
      test_cases=[],
      sim_backend=lambda a, b: "",
  )
  for rec, (seen_system, seen_user) in zip(prefixes, sim.seen, strict=True):
    built = env._build_sim_prompt(rec["sim_dialogue"])
    assert (rec["sim_system"], rec["sim_user"]) == built
    # And the bytes the BACKEND actually received, not just the ones the
    # builder would produce from the recorded dialogue.
    assert (rec["sim_system"], rec["sim_user"]) == (seen_system, seen_user)
    assert rec["sim_user_sha16"] == collect_prefixes.sha16(rec["sim_user"])


def test_submit_at_turn_three_yields_two_prefixes():
  prefixes, episode, _ = _run(["Q1?", "Q2?", SUBMISSION])
  assert [p["turn_idx"] for p in prefixes] == [0, 1]
  assert episode["episode_terminated_by"] == "submit"
  assert all(p["episode_terminated_by"] == "submit" for p in prefixes)
  assert episode["n_assistant_turns"] == 3
  assert episode["n_prefixes"] == 2


def test_never_submits_at_cap_four_yields_three_prefixes():
  prefixes, episode, _ = _run(["Q?"], max_turns=4)
  assert [p["turn_idx"] for p in prefixes] == [0, 1, 2]
  assert episode["episode_terminated_by"] == "turn_cap"


def test_submit_on_first_turn_yields_no_prefixes():
  """Exactly why the EPISODE file, not the prefix file, is the resume key."""
  prefixes, episode, _ = _run([SUBMISSION])
  assert not prefixes
  assert episode["n_prefixes"] == 0
  assert episode["episode_terminated_by"] == "submit"


def test_empty_sides_terminate_the_episode():
  _, episode, _ = _run(["   "])
  assert episode["episode_terminated_by"] == "solver_empty"
  prefixes, episode, _ = _run(["Q?"], sim_reply="")
  # The prefix is still recorded: the prompt was real and the empty reply is
  # itself data about the sim.
  assert len(prefixes) == 1
  assert episode["episode_terminated_by"] == "sim_empty"


def test_partner_reply_raw_keeps_the_think_block():
  prefixes, _, _ = _run(
      ["<think>secret reasoning</think>What threshold?"], max_turns=2
  )
  rec = prefixes[0]
  assert "secret reasoning" in rec["partner_reply_raw"]
  assert "secret reasoning" not in rec["partner_reply"]
  assert rec["partner_reply"] == "What threshold?"


def test_leak_invariant_gt_only_in_the_sim_prompt():
  """The GT is in ``sim_user`` and in NO turn of the dialogue the solver sees."""
  prefixes, _, _ = _run(["Q1?", "Q2?"], max_turns=3)
  for rec in prefixes:
    assert GT in rec["sim_user"]
    for msg in rec["sim_dialogue"]:
      assert GT not in msg["content"]


def test_sim_reply_leak_columns():
  """A code-writing sim reply is flagged by the judge-free detector."""
  leaky = "You want:\n```python\ndef f(x, y):\n    return x + y\n```"
  prefixes, _, _ = _run(["Q?"], sim_reply=leaky)
  rec = prefixes[0]
  assert rec["sim_reply_leak_reason"] is not None
  prefixes, _, _ = _run(["Q?"], sim_reply="Ten is the cutoff.")
  assert prefixes[0]["sim_reply_leak_reason"] is None


def test_char_limit_is_applied_from_the_environment(monkeypatch):
  """SIM_CHAR_LIMIT still binds through env._finalize_reply. The silent one."""
  monkeypatch.setenv("SIM_CHAR_LIMIT", "10")
  prefixes, _, _ = _run(["Q?"], sim_reply="x" * 50)
  assert prefixes[0]["sim_reply"] == "x" * 10


def test_require_sim_char_limit_raises_when_unset(monkeypatch):
  monkeypatch.delenv("SIM_CHAR_LIMIT", raising=False)
  with pytest.raises(SystemExit):
    collect_prefixes.require_sim_char_limit()
  monkeypatch.setenv("SIM_CHAR_LIMIT", "0")
  assert collect_prefixes.require_sim_char_limit() == 0


def test_resume_key_and_prefix_dedupe():
  with tempfile.TemporaryDirectory() as d:
    ep_path = os.path.join(d, "episodes.jsonl")
    px_path = os.path.join(d, "prefixes.jsonl")
    assert not collect_prefixes.existing_episodes(ep_path)
    prefixes, episode, _ = _run(["Q1?", "Q2?", SUBMISSION])
    with open(px_path, "w", encoding="utf-8") as f:
      for r in prefixes:
        f.write(json.dumps(r) + "\n")
    with open(ep_path, "w", encoding="utf-8") as f:
      f.write(json.dumps(episode) + "\n")
    assert collect_prefixes.existing_episodes(ep_path) == {(0, 0)}
    assert len(collect_prefixes.read_prefixes(px_path)) == 2
    # A kill between the two appends duplicates the prefix block on resume.
    with open(px_path, "a", encoding="utf-8") as f:
      for r in prefixes:
        f.write(json.dumps(r) + "\n")
    deduped = collect_prefixes.read_prefixes(px_path)
    assert len(deduped) == 2
    assert [r["prefix_id"] for r in deduped] == ["0-0-0", "0-0-1"]


def test_sim_system_arm_reaches_the_recorded_prompt_and_the_backend():
  """The collector's episodes must be played by the arm we intend to study.

  Episodes generated against one simulator are off-distribution context for
  another, so a collector silently running the stock prompt while the training
  run serves `role_restraint` would poison every prefix before anything else
  ran. Nothing downstream could detect that.
  """
  for arm, marker in (
      ("", "helpful assistant"),
      ("role", "role-playing"),
      ("role_restraint", "role-playing"),
  ):
    sim = _sim("Ten is the cutoff.")
    prefixes, _ = collect_prefixes.run_episode(
        _task(),
        episode_idx=0,
        solver_chat=_scripted_solver(["What threshold?"]),
        sim_backend=sim,
        max_assistant_turns=2,
        provenance={},
        sim_prompt=arm,
    )
    rec = prefixes[0]
    # Recorded, sent to the backend, and equal to what the env would build.
    assert marker in rec["sim_system"], (arm, rec["sim_system"][:50])
    assert sim.seen[0][0] == rec["sim_system"]
    env = ColBenchUserSimEnv(
        problem_description=PROBLEM,
        ground_truth=GT,
        test_cases=[],
        sim_prompt=arm,
        sim_backend=lambda a, b: "",
    )
    assert (rec["sim_system"], rec["sim_user"]) == env._build_sim_prompt(
        rec["sim_dialogue"]
    )
    assert rec["sim_system_variant"] == (arm or "default")
    # The hidden GT rides the USER message in every arm, never the system one.
    assert GT in rec["sim_user"] and GT not in rec["sim_system"]


def test_role_restraint_is_the_runner_default():
  """A drifting default here is the whole failure mode, so pin it."""
  runner = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(collect_prefixes.__file__))),
      "simtrain",
      "run_collect_slurm.sh",
  )
  text = open(runner, encoding="utf-8").read()
  assert "SIM_SYSTEM=${SIM_SYSTEM:-role_restraint}" in text
  # Both stages must receive it; a collector/candidate mismatch is silent.
  assert text.count('--sim_system "$SIM_SYSTEM"') == 2

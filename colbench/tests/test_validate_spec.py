"""CPU tests for colbench.validate_colbench_spec.run_eval.

No GPU, no SGLang, no sim server.

Sibling of ``test_validate.py`` (which covers the GT harness) for the SPEC
harness, driven with a FAKE solver + a scripted sim backend + the in-process exec
grader. Focused on the termination accounting, which is what the offline dumps are
read for: the reply that ENDED the episode is recorded, a termination drawn before
the solver has shown any code is resampled, and if the sim insists anyway the
episode still ends -- with the guard's fingerprints in the summary and a readable
sidecar.
"""

from argparse import Namespace
import json
import os

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

# The module-level setup above (env vars, sys.path) has to run
# before these imports resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
import pandas as pd

from colbench import validate_colbench_spec as vcs

GT = (
    "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n "
    "       return x - y\n"
)
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."
SPEC = {
    "persona": {
        "who": "an analyst",
        "domain": "ops",
        "python_skill": "analyst",
        "communication_style": "brief",
    },
    "scenario": "Needs a small helper for a report.",
    "requirements": "If x is at least 10 return x+y, otherwise x-y.",
    "plot": "The user reveals the cutoff of 10 only if asked.",
}
TERMINATE = "Sounds good, I think you have it. [TERMINATE]"


def _code_turn(src):
  return "Here's my function:\n```python\n" + src + "```"


class FakeTokenizer:
  """Minimal chat tokenizer: join message contents; whitespace 'tokens'."""

  def apply_chat_template(
      self, messages, add_generation_prompt=True, tokenize=False, **kwargs
  ):
    return "\n".join(str(m["content"]) for m in messages)

  def encode(self, text, add_special_tokens=False):
    return (text or "").split()


class FakeSolver:
  """Scripted solver: turn `i`'s text for every conversation in the batch."""

  def __init__(self, turns):
    self._turns = list(turns)
    self._i = 0

  def generate(self, message_lists, tokenizer, max_model_len):
    del max_model_len  # part of the shared solver interface
    text = self._turns[min(self._i, len(self._turns) - 1)]
    self._i += 1
    return [
        {"text": text, "tokens": len(tokenizer.encode(text))}
        for _ in message_lists
    ]


def _scripted_sim(replies):
  """Sim backend returning successive canned replies (the last one repeats)."""
  seq = list(replies)
  state = {"i": 0}

  def backend(system_content, user_content):
    del system_content, user_content
    r = seq[min(state["i"], len(seq) - 1)]
    state["i"] += 1
    return r

  return backend


def _val_df():
  gt = {
      "problem_description": PROBLEM,
      "ground_truth": GT,
      "test_cases": list(CALLS),
  }
  return pd.DataFrame(
      [
          {
              "prompt": [
                  {"role": "system", "content": "sys"},
                  {"role": "user", "content": PROBLEM},
              ],
              "reward_model": {"style": "rule", "ground_truth": gt},
              "extra_info": {"ground_truth": gt, "spec": SPEC, "index": 0},
          }
      ]
  )


def _args(
    sim_max_tries=8,
    max_assistant_turns=4,
    grounded=False,
    sim_code_leak_detector="a0_strict",
):
  return Namespace(
      grounded=grounded,
      sim_code_leak_detector=sim_code_leak_detector,
      model="fake-model",
      solver_backend="openai",
      val_file="fake.parquet",
      sim_backend="local",
      sim_model="",
      max_assistant_turns=max_assistant_turns,
      max_code_proposals=2,
      reward_time_limit=6.0,
      max_response_length=4096,
      max_prompt_length=2048,
      max_new_tokens_per_turn=256,
      top_p=0.95,
      top_k=-1,
      grade_concurrency=2,
      seed=0,
      max_saved_convos=100,
      sim_max_tries=sim_max_tries,
  )


def _run(tmp_path, solver_turns, sim_replies, **arg_kwargs):
  args = _args(**arg_kwargs)
  out_path = str(tmp_path / "eval.json")
  summary = vcs.run_eval(
      FakeSolver(solver_turns),
      FakeTokenizer(),
      _val_df(),
      temperature=0.6,
      n_samples=1,
      args=args,
      out_path=out_path,
      max_model_len=args.max_prompt_length + args.max_response_length,
      sim_backend=_scripted_sim(sim_replies),
  )
  with open(out_path, encoding="utf-8") as f:
    dump = json.load(f)
  return summary, dump, out_path


def test_terminating_reply_is_recorded(tmp_path):
  # The sim ends the episode AFTER code is shown (the healthy case). The closing
  # reply must reach the dump -- it used to be dropped, leaving a transcript that
  # simply stopped with no way to tell why.
  summary, dump, _ = _run(
      tmp_path,
      solver_turns=["What's the cutoff for x?", _code_turn(GT)],
      sim_replies=["It's 10.", "Perfect, thanks! [TERMINATE]"],
  )
  (traj,) = dump["trajectories"]
  assert traj["terminated_by"] == "user"
  assert traj["reward"] == 1.0
  assert traj["terminating_reply"] == "Perfect, thanks! [TERMINATE]"
  assert traj["terminate_standalone"] is False  # trailing marker, not bare
  assert traj["messages"][-1] == {
      "role": "user",
      "content": "Perfect, thanks! [TERMINATE]",
  }
  # It is NOT counted as a user turn -- the solver never answered it.
  assert traj["num_user_turns"] == 1
  assert summary["premature_terminate_rate"] == 0.0
  assert summary["terminate_standalone_rate"] == 0.0


def test_premature_terminate_is_resampled_and_conversation_continues(tmp_path):
  # Draw 1 wants out before any code exists -> rejected; draw 2 is a real user
  # turn, so the episode proceeds to a code proposal instead of dying at turn 1
  # with reward 0.
  summary, dump, _ = _run(
      tmp_path,
      solver_turns=["What's the cutoff for x?", _code_turn(GT)],
      sim_replies=[TERMINATE, "It's 10.", "Great, thanks! [TERMINATE]"],
  )
  (traj,) = dump["trajectories"]
  assert traj["terminated_by"] == "user"
  assert traj["reward"] == 1.0
  assert traj["sim_early_term_rejected"] == 1
  assert traj["early_term_exhausted"] is False
  assert traj["early_term_rejected_replies"] == [TERMINATE]
  assert summary["sim_early_term_rejected_total"] == 1
  assert summary["premature_terminate_rate"] == 0.0
  assert summary["showed_code_rate"] == 1.0


def test_premature_terminate_exhaustion_still_ends_and_is_reported(tmp_path):
  # The sim insists on ending across the whole budget: the episode still ends as
  # 'no_code' (the guard never creates a new terminal state), but the summary now
  # says the guard was overruled and the sidecar shows the discarded draws.
  summary, dump, out_path = _run(
      tmp_path,
      solver_turns=["What's the cutoff for x?"],
      sim_replies=[TERMINATE],
      sim_max_tries=2,
  )
  (traj,) = dump["trajectories"]
  assert traj["terminated_by"] == "no_code"
  assert traj["reward"] == 0.0
  assert traj["showed_code"] is False
  assert traj["overflow"] is False
  assert traj["num_assistant_turns"] == 1  # nowhere near the turn cap
  assert traj["sim_early_term_rejected"] == 2
  assert traj["early_term_exhausted"] is True
  assert traj["terminating_reply"] == TERMINATE
  assert summary["premature_terminate_rate"] == 1.0
  assert summary["sim_early_term_exhausted"] == 1

  premature_path = os.path.splitext(out_path)[0] + ".premature_term.txt"
  text = open(premature_path, encoding="utf-8").read()
  assert "1 premature terminations" in text
  assert "What's the cutoff for x?" in text
  assert "DISCARDED EARLY-TERMINATE DRAW 1" in text


def test_clean_run_writes_an_empty_premature_sidecar(tmp_path):
  # The file is written even when nothing fired, so "no premature terminations"
  # is distinguishable from "this run never checked".
  _, _, out_path = _run(
      tmp_path,
      solver_turns=[_code_turn(GT)],
      sim_replies=["Perfect. [TERMINATE]"],
  )
  premature_path = os.path.splitext(out_path)[0] + ".premature_term.txt"
  assert (
      open(premature_path, encoding="utf-8")
      .read()
      .startswith("0 premature terminations")
  )


# ── --grounded threading (eval-side mirror of +colbench.grounded_sim) ─────────
# What the sim is CONDITIONED on is invisible in the reward numbers, so a flag
# that silently fails to reach the env would look exactly like a real result.
# These pin the wire, not the prompt text (that is test_env_spec.py's job).


def _capturing_sim(replies):
  """Scripted sim that also records the prompts it was conditioned on."""
  seq = list(replies)
  state = {"i": 0}
  seen = []

  def backend(system_content, user_content):
    seen.append((system_content, user_content))
    r = seq[min(state["i"], len(seq) - 1)]
    state["i"] += 1
    return r

  return backend, seen


def _run_capturing(tmp_path, solver_turns, sim_replies, **arg_kwargs):
  """Run one eval and return the sim's SYSTEM prompts, joined.

  The conditioning lives entirely in the system message; the user message is the
  running transcript (which contains whatever the SOLVER wrote, GT-looking code
  included), so asserting on it would prove nothing about conditioning.
  """
  args = _args(**arg_kwargs)
  backend, seen = _capturing_sim(sim_replies)
  vcs.run_eval(
      FakeSolver(solver_turns),
      FakeTokenizer(),
      _val_df(),
      temperature=0.6,
      n_samples=1,
      args=args,
      out_path=str(tmp_path / "eval.json"),
      max_model_len=args.max_prompt_length + args.max_response_length,
      sim_backend=backend,
  )
  return "\n".join(s for s, _ in seen)


def test_grounded_flag_reaches_the_sim_prompt(tmp_path):
  sys_prompts = _run_capturing(
      tmp_path,
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["It's 10.", TERMINATE],
      grounded=True,
  )
  # Grounded conditioning = hidden GT source + plot, and NOT the requirements.
  assert GT.strip() in sys_prompts
  assert SPEC["plot"] in sys_prompts
  assert SPEC["requirements"] not in sys_prompts


def test_default_eval_is_spec_conditioned(tmp_path):
  # Regression guard on the DEFAULT: the headline eval must stay spec-conditioned,
  # with the GT nowhere in the sim's conditioning.
  sys_prompts = _run_capturing(
      tmp_path,
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["It's 10.", TERMINATE],
  )
  assert GT.strip() not in sys_prompts
  assert "x >= 10" not in sys_prompts
  assert SPEC["requirements"] in sys_prompts


def test_summary_records_the_conditioning(tmp_path):
  # Two summaries are only comparable at the same conditioning, so it is recorded
  # in the JSON rather than left implicit in the filename.
  spec_dir = tmp_path / "spec"
  grounded_dir = tmp_path / "grounded"
  spec_dir.mkdir()
  grounded_dir.mkdir()
  spec_summary, _, _ = _run(
      spec_dir,
      solver_turns=[_code_turn(GT)],
      sim_replies=[TERMINATE],
  )
  grounded_summary, _, _ = _run(
      grounded_dir,
      solver_turns=[_code_turn(GT)],
      sim_replies=[TERMINATE],
      grounded=True,
  )
  assert spec_summary["sim_conditioning"] == "spec"
  assert grounded_summary["sim_conditioning"] == "grounded"


# ── STRICT code-leak screen (harness v3 default) ──────────────────────────────


def test_bare_def_is_rejected_and_resampled_by_default(tmp_path):
  # v3 DEFAULT. An unfenced `def f(...)` from the sim is a leak -- the solver
  # could read the answer straight off the user's turn -- so it must be
  # resampled, not injected. Under the pre-v3 fence-only screen this reply went
  # into the conversation verbatim.
  bare_def = "Sure: def f(x, y): return x + y"
  _, dump, _ = _run(
      tmp_path,
      solver_turns=["What's the cutoff for x?", _code_turn(GT)],
      sim_replies=[bare_def, "It's 10.", "Great, thanks! [TERMINATE]"],
  )
  (traj,) = dump["trajectories"]
  assert traj["sim_code_rejected"] == 1
  assert traj["terminated_by"] == "user"
  # The leak never reached the solver's message list.
  assert all(bare_def not in m["content"] for m in traj["messages"])


def test_bare_def_exhaustion_ends_as_sim_code_reject(tmp_path):
  # THE TRAP, pinned. When every try is a bare def the budget is exhausted, and
  # the outcome must be attributed with the SAME strict predicate that drove the
  # retries. Attributing with the fence-only detector (the 2026-08-07 training
  # bug) would call this a premature-termination exhaustion and INJECT the last
  # leaked reply instead of ending the conversation.
  bare_def = "Sure: def f(x, y): return x + y"
  _, dump, _ = _run(
      tmp_path,
      solver_turns=["What's the cutoff for x?", _code_turn(GT)],
      sim_replies=[bare_def],
      sim_max_tries=2,
  )
  (traj,) = dump["trajectories"]
  assert traj["terminated_by"] == "sim_code_reject"
  assert traj["sim_code_rejected"] == 2
  assert traj["early_term_exhausted"] is False


def test_fence_only_reproduces_the_pre_v3_screen(tmp_path):
  # The escape hatch for re-running an old eval: the same reply is injected and
  # the conversation continues, exactly as before v3.
  bare_def = "Sure: def f(x, y): return x + y"
  _, dump, _ = _run(
      tmp_path,
      solver_turns=["What's the cutoff for x?", _code_turn(GT)],
      sim_replies=[bare_def, "Great, thanks! [TERMINATE]"],
      sim_code_leak_detector="fence_only",
  )
  (traj,) = dump["trajectories"]
  assert traj["sim_code_rejected"] == 0
  assert any(bare_def in m["content"] for m in traj["messages"])


def test_summary_records_the_detector(tmp_path):
  # Comparability again: a v3 summary and a fence_only re-run are different
  # environments, so the screen is recorded next to sim_conditioning.
  summary, _, _ = _run(
      tmp_path,
      solver_turns=[_code_turn(GT)],
      sim_replies=[TERMINATE],
  )
  assert summary["sim_code_leak_detector"] == "a0_strict"
  assert summary["harness_version"] == "v3"

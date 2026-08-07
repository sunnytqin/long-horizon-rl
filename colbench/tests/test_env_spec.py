"""CPU tests for the SPEC path.

Covers colbench.env_spec and the spec templates helpers, against a mocked sim.

Covers the leak invariant (GT source NEVER enters the spec sim prompt -- only
the spec does), grading parity with the GT env, the spec-specific templates
helpers (``sim_terminated``, ``contains_code``, ``extract_last_code``,
``build_spec_sim_messages``),
and the USER-DRIVEN termination state machine via an inline driver that mirrors
the plan's loop -- so the contract is pinned before ``colbench_spec_agent`` /
``validate_colbench_spec`` implement it. Grading uses the in-process exec
fallback.
"""

import os

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

# The module-level setup above (env vars, sys.path) has to run
# before these imports resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench import templates
from colbench.env_spec import ColBenchSpecUserSimEnv

GT = (
    "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n "
    "       return x - y\n"
)
# Ignores the x<10 branch -> 0.5 pass-rate.
WRONG = "def f(x, y):\n    return x + y\n"
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
    "requirements": (
        "The user wants f(x,y): if x is at least 10 return x+y, otherwise "
        "x-y."
    ),
    "plot": (
        "The user reveals the threshold of 10 only if the assistant asks "
        "about the cutoff."
    ),
}


def _scripted_backend(replies):
  """A sim backend returning successive canned replies (last one repeats)."""
  seq = list(replies)
  state = {"i": 0}

  def backend(system_content, user_content):
    r = seq[min(state["i"], len(seq) - 1)]
    state["i"] += 1
    return r

  return backend


def _env(sim_backend=None, grounded=False, sim_prompt=""):
  return ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
      sim_backend=sim_backend,
      grounded=grounded,
      sim_prompt=sim_prompt,
  )


def _code_turn(src):
  return "Here's my function:\n```python\n" + src + "```"


# ── leak invariant: the GT source never enters the spec sim prompt ────────────


def test_spec_prompt_has_no_gt():
  captured = {}

  def backend(system_content, user_content):
    captured["sys"] = system_content
    captured["usr"] = user_content
    return "Sure, above 10 we add. "

  e = _env(sim_backend=backend)
  e.generate_user_turn(
      [
          {"role": "user", "content": PROBLEM},
          {"role": "assistant", "content": "What's the cutoff?"},
      ]
  )
  # The spec (requirements/plot/persona) IS injected; the GT code is NOT.
  assert "at least 10" in captured["sys"] and "an analyst" in captured["sys"]
  assert GT not in captured["sys"] and GT not in captured["usr"]
  assert (
      "x >= 10" not in captured["sys"] and "return x + y" not in captured["sys"]
  )


def test_generate_user_turn_no_char_truncation():
  # The old HUMAN_RESPONSE_CHARACTER_LIMIT post-hoc slice is gone: a long reply
  # is injected in full (brevity is enforced at generation via SIM_MAX_TOKENS +
  # the prompt, not by chopping).
  long_reply = (
      "x" * (templates.HUMAN_RESPONSE_CHARACTER_LIMIT + 800) + " [TERMINATE]"
  )
  e = _env(sim_backend=_scripted_backend([long_reply]))
  reply = e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert reply == e.last_sim_raw  # no post-hoc truncation
  assert len(reply) > templates.HUMAN_RESPONSE_CHARACTER_LIMIT
  assert (
      templates.sim_terminated(reply) is True
  )  # sentinel survives (nothing chopped)


# ── spec templates helpers ────────────────────────────────────────────────────


def test_sim_terminated_variants():
  assert templates.sim_terminated("All good, thanks! [TERMINATE]") is True
  assert (
      templates.sim_terminated("looks [terminate] fine") is True
  )  # case-insensitive
  assert (
      templates.sim_terminated("<think>[TERMINATE]?</think>keep going") is False
  )  # think-stripped
  assert templates.sim_terminated("could you also handle negatives?") is False


def test_sim_terminate_standalone_separates_handoff_from_mention():
  # DIAGNOSTIC only (never a termination decision): tells a real hand-off from an
  # episode killed by `sim_terminated`'s unanchored match.
  assert templates.sim_terminate_standalone("[TERMINATE]") is True
  assert templates.sim_terminate_standalone("  **[TERMINATE]**  ") is True
  assert templates.sim_terminate_standalone("[terminate].") is True
  assert (
      templates.sim_terminate_standalone("<think>hmm</think>\n[TERMINATE]")
      is True
  )
  assert templates.sim_terminate_standalone("Thanks! [TERMINATE]") is False
  assert (
      templates.sim_terminate_standalone("I shouldn't [TERMINATE] yet") is False
  )
  assert templates.sim_terminate_standalone("keep going") is False


def test_sim_prompts_name_the_sentinel_only_in_the_how_to_end_rule():
  # Regression guard for the de-mention change: every OTHER reference was reworded
  # to the ACT ("end the conversation") because the sim echoing a negated mention
  # ("you do NOT say [TERMINATE] yet") is itself a termination under the
  # unanchored match. Keep this at 1 -- adding a second mention re-opens that.
  for prompt in (
      templates.SPEC_SIM_SYSTEM_PROMPT,
      templates.GROUNDED_SIM_SYSTEM_PROMPT,
  ):
    assert prompt.count(templates.TERMINATE_MARKER) == 1
    # ...and that one mention must demand the sentinel be the whole reply.
    assert "ENTIRE reply must be exactly" in prompt


def test_contains_code_and_extract_last():
  assert templates.contains_code(_code_turn(GT)) is True
  assert templates.contains_code("def f(x, y):") is True
  assert templates.contains_code("what cutoff should I use?") is False
  dlg = [
      {"role": "assistant", "content": _code_turn(WRONG)},
      {"role": "user", "content": "no, below 10 subtract"},
      {"role": "assistant", "content": _code_turn(GT)},
  ]
  code = templates.extract_last_code(dlg)
  assert "x >= 10" in code and code.strip().startswith("def f")


def test_extract_last_code_none_when_no_code():
  assert (
      templates.extract_last_code(
          [{"role": "assistant", "content": "what cutoff?"}]
      )
      == ""
  )


# ── sim code fidelity: detection, stripping, and rejection sampling ───────────


def test_sim_wrote_code_detects_fence():
  assert templates.sim_wrote_code("Here: ```python\ndef f(): pass\n```") is True
  assert templates.sim_wrote_code("just a bare ```\nx=1\n``` block") is True
  assert (
      templates.sim_wrote_code("below 10 it should subtract, not add") is False
  )
  assert (
      templates.sim_wrote_code("set `area` to None when it's <= 0") is False
  )  # inline backticks ok


def test_generate_user_turn_rejection_samples_code():
  # First reply writes code (rejected), second is natural language -> the NL one
  # is returned.
  e = _env(
      sim_backend=_scripted_backend(
          [
              "Sure: ```python\ndef f(x, y): return x + y\n```",
              "Below 10 it should subtract instead of add.",
          ]
      )
  )
  reply = e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert "```" not in reply and "subtract" in reply
  assert e.last_sim_code_rejected == 1


def test_generate_user_turn_flags_exhaustion_when_all_tries_write_code():
  # Every try writes code -> after sim_max_tries the env flags exhaustion (no
  # strip/inject); the loop aborts the conversation for inspection. last_sim_raw
  # keeps the offending reply verbatim.
  offending = "Do this: ```python\ndef f(): return 0\n``` ok?"
  e = ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
      sim_max_tries=3,
      sim_backend=_scripted_backend([offending]),
  )
  e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert e.last_sim_code_reject_exhausted is True
  assert e.last_sim_code_rejected == 3
  assert (
      "```python" in e.last_sim_raw
  )  # offending reply kept verbatim (not stripped)


def test_generate_user_turn_no_exhaustion_when_reply_is_clean():
  e = _env(
      sim_backend=_scripted_backend(["Below 10 it should subtract, not add."])
  )
  e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert e.last_sim_code_reject_exhausted is False
  assert e.last_sim_code_rejected == 0
  assert e.last_sim_early_term_rejected == 0
  assert e.last_sim_early_term_exhausted is False


# ── PREMATURE-termination rejection (allow_terminate) ─────────────────────────
# A user cannot be satisfied by a function that does not exist yet, so before the
# solver's first code proposal a terminating draw is rejected and resampled --
# same budget as the code-writing rejection. Motivated by gold-sim evals ending
# as 'no_code' after a single clarifying question.


def test_early_terminate_rejected_before_any_code():
  # First draw wants out with no code on the table (rejected); second is a real
  # user turn -> that one is returned and the conversation continues.
  e = _env(
      sim_backend=_scripted_backend(
          [
              "Sounds good, I think you have it. [TERMINATE]",
              "Above 10 we add, below we subtract.",
          ]
      )
  )
  reply = e.generate_user_turn(
      [{"role": "user", "content": PROBLEM}], allow_terminate=False
  )
  assert templates.sim_terminated(reply) is False
  assert "subtract" in reply
  assert e.last_sim_early_term_rejected == 1
  assert e.last_sim_early_term_exhausted is False
  assert e.last_sim_code_rejected == 0
  # The discarded draw is kept for the eval dump.
  assert e.last_sim_early_term_samples == [
      "Sounds good, I think you have it. [TERMINATE]"
  ]


def test_early_terminate_also_catches_a_passing_mention():
  # `sim_terminated` cannot tell a hand-off from a reply that merely MENTIONS the
  # sentinel -- which is the most likely shape of a spurious termination. The
  # guard covers both, because it keys on the same predicate.
  mention = "I shouldn't say [TERMINATE] yet -- what format is the input?"
  e = _env(sim_backend=_scripted_backend([mention, "It's a list of ints."]))
  reply = e.generate_user_turn(
      [{"role": "user", "content": PROBLEM}], allow_terminate=False
  )
  assert reply == "It's a list of ints."
  assert e.last_sim_early_term_rejected == 1


def test_early_terminate_exhaustion_still_ends_the_episode():
  # The guard can never CREATE a new terminal state: if every draw insists on
  # ending, the last one is returned unchanged (the loop then ends the episode
  # exactly as it did before the guard existed) and the flag records that it was
  # overruled.
  e = ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
      sim_max_tries=3,
      sim_backend=_scripted_backend(["Okay, you have it. [TERMINATE]"]),
  )
  reply = e.generate_user_turn(
      [{"role": "user", "content": PROBLEM}], allow_terminate=False
  )
  assert templates.sim_terminated(reply) is True
  assert e.last_sim_early_term_rejected == 3
  assert e.last_sim_early_term_exhausted is True
  # NOT a code rejection -- the two exhaustion flags are mutually exclusive, so
  # the 'sim_code_reject' abort keeps its exact pre-existing meaning.
  assert e.last_sim_code_reject_exhausted is False


def test_code_rejection_still_owns_exhaustion_when_final_draw_writes_code():
  # Mixed grounds, code last: the returned reply contains code, so the
  # abort-don't-inject path must claim it even though a terminate was also
  # rejected along the way.
  e = ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
      sim_max_tries=2,
      sim_backend=_scripted_backend(
          [
              "You have it. [TERMINATE]",
              "Like this: ```python\ndef f(x, y): return x + y\n```",
          ]
      ),
  )
  e.generate_user_turn(
      [{"role": "user", "content": PROBLEM}], allow_terminate=False
  )
  assert e.last_sim_code_reject_exhausted is True
  assert e.last_sim_early_term_exhausted is False
  assert e.last_sim_early_term_rejected == 1
  assert e.last_sim_code_rejected == 1


def test_allow_terminate_default_is_permissive():
  # Default True = pre-existing behavior, so every pre-guard caller (and every
  # turn after the first code proposal) is untouched.
  e = _env(sim_backend=_scripted_backend(["All good, thanks! [TERMINATE]"]))
  reply = e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert templates.sim_terminated(reply) is True
  assert e.last_sim_early_term_rejected == 0
  assert e.last_sim_early_term_exhausted is False


# ── grading parity with the GT env ────────────────────────────────────────────


def test_score_full_and_partial():
  e = _env()
  assert e.score("```python\n" + GT + "```")["pass_rate"] == 1.0
  assert e.score("```python\n" + WRONG + "```")["pass_rate"] == 0.5


# ── USER-DRIVEN termination: state machine (driver mirrors the loop) ─────────


def drive(env, assistant_turns, max_turns=10, max_code_proposals=3):
  """Replicate the spec agent loop's termination state machine.

  That state machine is the pinned contract.

  ``colbench_spec_agent`` / ``validate_colbench_spec`` MUST mirror this: solver
  turn -> track last code / count proposals -> turn cap -> code cap -> else sim
  reply -> [TERMINATE]. Grades the last shown function; reward 0 (and
  terminated_by 'no_code') if none was ever shown. A terminating draw is
  inadmissible until the solver has shown code (``allow_terminate``), and the env
  resamples it out of the same try budget as a code-writing draw.
  """
  sim_dialogue = [{"role": "user", "content": env.problem_description}]
  last_code, code_proposals, terminated_by = "", 0, None
  for turn in range(max_turns):
    if turn >= len(assistant_turns):
      terminated_by = "turn_cap"
      break
    at = assistant_turns[turn]
    sim_dialogue.append({"role": "assistant", "content": at})
    if templates.contains_code(at):
      last_code = templates.extract_last_code(sim_dialogue)
      code_proposals += 1
    if turn == max_turns - 1:
      terminated_by = "turn_cap"
      break
    if code_proposals >= max_code_proposals:
      terminated_by = "code_cap"
      break
    reply = env.generate_user_turn(
        sim_dialogue, allow_terminate=bool(last_code)
    )
    if templates.sim_terminated(env.last_sim_raw):
      terminated_by = "user"
      break
    sim_dialogue.append({"role": "user", "content": reply})
  showed_code = bool(last_code)
  if showed_code:
    reward = env.score(last_code)["pass_rate"]
  else:
    reward = 0.0
    if terminated_by == "user":
      terminated_by = "no_code"
  return {
      "reward": reward,
      "terminated_by": terminated_by,
      "code_proposals": code_proposals,
      "showed_code": showed_code,
  }


def test_gated_never_asked_terminates_on_imperfect_code():
  # Solver never asks and shows WRONG code; a faithful gated sim has nothing to
  # reveal and terminates -> we grade the imperfect code (0.5). This is the
  # intended imperfect signal.
  e = _env(
      sim_backend=_scripted_backend(["Looks fine to me, thanks! [TERMINATE]"])
  )
  out = drive(e, [_code_turn(WRONG)])
  assert out["terminated_by"] == "user"
  assert out["reward"] == 0.5
  assert out["showed_code"] is True


def test_correct_on_code_then_terminate():
  # Wrong code -> one correction (no terminate) -> correct code -> terminate.
  # Grade the correct code (1.0). Sim replies: correction, then [TERMINATE].
  e = _env(
      sim_backend=_scripted_backend(
          [
              "No -- below 10 it should subtract, not add.",
              "Perfect, that's exactly it. [TERMINATE]",
          ]
      )
  )
  out = drive(e, [_code_turn(WRONG), _code_turn(GT)])
  assert out["terminated_by"] == "user"
  assert out["reward"] == 1.0
  assert out["code_proposals"] == 2


def test_code_cap_forces_terminate():
  # Sim never terminates; solver keeps proposing code -> code cap (3) fires ->
  # grade last code.
  e = _env(sim_backend=_scripted_backend(["Hmm, not quite, keep trying."]))
  out = drive(
      e,
      [_code_turn(WRONG), _code_turn(WRONG), _code_turn(GT), _code_turn(GT)],
      max_code_proposals=3,
  )
  assert out["terminated_by"] == "code_cap"
  assert out["code_proposals"] == 3
  assert (
      out["reward"] == 1.0
  )  # 3rd proposal (the code cap turn) was the correct GT


def test_turn_cap_no_code_reward_zero():
  # Solver only ever asks (no code); sim never terminates -> turn cap -> reward
  # 0.
  e = _env(
      sim_backend=_scripted_backend(["Above 10 we add, below we subtract."])
  )
  out = drive(
      e, ["what cutoff?", "and below it?", "anything else?"], max_turns=3
  )
  assert out["terminated_by"] == "turn_cap"
  assert out["reward"] == 0.0
  assert out["showed_code"] is False


def test_user_terminates_without_code_is_no_code_reward_zero():
  # Sim emits [TERMINATE] before any code was shown -> no_code, reward 0.
  e = _env(
      sim_backend=_scripted_backend(["Okay, I think you have it. [TERMINATE]"])
  )
  out = drive(e, ["Tell me your requirements?"], max_turns=5)
  assert out["terminated_by"] == "no_code"
  assert out["reward"] == 0.0
  assert out["showed_code"] is False


# ── GROUNDED sim mode (+colbench.grounded_sim) ────────────────────────────────
# The sim conditions on the hidden GT source + spec["plot"] instead of
# persona/scenario/requirements. Same env, same termination machinery -- only
# the sim's SYSTEM prompt changes. The GT is now IN the sim's prompt, so "leak
# impossible by construction" no longer holds and the episode-level invariant
# below (GT never reaches the SOLVER's message list) is what enforces it.


def _capturing_backend(reply="Above 10 we add, below we subtract."):
  """Sim backend that records every (system, user) pair it was called with."""
  seen = []

  def backend(system_content, user_content):
    seen.append((system_content, user_content))
    return reply

  return backend, seen


def test_grounded_prompt_has_gt_and_plot_not_requirements():
  backend, seen = _capturing_backend()
  e = _env(sim_backend=backend, grounded=True)
  e.generate_user_turn(
      [
          {"role": "user", "content": PROBLEM},
          {"role": "assistant", "content": "What's the cutoff?"},
      ]
  )
  sys_msg, usr_msg = seen[-1]
  # The GT source and the plot ARE injected...
  assert GT.strip() in sys_msg
  assert SPEC["plot"] in sys_msg
  assert PROBLEM in sys_msg
  # ...and the spec's persona / scenario / requirements are NOT (this arm drops
  # them, so a result is attributable to grounding rather than to persona
  # style).
  assert SPEC["requirements"] not in sys_msg
  assert SPEC["scenario"] not in sys_msg
  assert "an analyst" not in sys_msg
  # The dialogue still goes in the USER message, unchanged from the spec path.
  assert "What's the cutoff?" in usr_msg


def test_spec_mode_unchanged_when_not_grounded():
  # Regression guard for the default path: grounded=False must still put NO GT
  # in either message (pairs with test_spec_prompt_has_no_gt, which predates the
  # flag).
  backend, seen = _capturing_backend()
  e = _env(sim_backend=backend, grounded=False)
  e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  sys_msg, usr_msg = seen[-1]
  assert GT not in sys_msg and GT not in usr_msg
  assert "x >= 10" not in sys_msg
  assert SPEC["requirements"] in sys_msg  # the spec prompt is what it used


def test_grounded_still_rejects_fenced_reply():
  # Rejection sampling is the load-bearing leak defense in grounded mode (the
  # sim can SEE the GT), so it must still fire there.
  e = ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
      grounded=True,
      sim_max_tries=3,
      sim_backend=_scripted_backend(["Like this: ```python\n" + GT + "```"]),
  )
  e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert e.last_sim_code_reject_exhausted is True
  assert e.last_sim_code_rejected == 3
  # And a clean reply on a retry is accepted normally.
  e2 = _env(
      sim_backend=_scripted_backend(
          [
              "Here: ```python\ndef f(x, y): return x + y\n```",
              "Below 10 it should subtract instead.",
          ]
      ),
      grounded=True,
  )
  reply = e2.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert "```" not in reply and e2.last_sim_code_rejected == 1


def test_grounded_leak_invariant_full_episode():
  # THE test for this arm: over a whole episode the GT source (and any
  # distinctive fragment of it) must never appear in a turn injected into the
  # SOLVER's message list, even though the sim is reading it. Mirrors
  # tests/test_env.py::test_leak_invariant_full_episode.
  e = _env(
      sim_backend=_scripted_backend(
          [
              "Above a certain number we add them, otherwise we take the"
              " difference.",
              "The cutoff is ten.",
              "That's it, thanks! [TERMINATE]",
          ]
      ),
      grounded=True,
  )
  injected = []

  def _drive_capturing():
    sim_dialogue = [{"role": "user", "content": PROBLEM}]
    for at in ["what cutoff?", "and below it?", _code_turn(GT)]:
      sim_dialogue.append({"role": "assistant", "content": at})
      reply = e.generate_user_turn(sim_dialogue)
      injected.append(reply)
      if templates.sim_terminated(e.last_sim_raw):
        break
      sim_dialogue.append({"role": "user", "content": reply})
    return sim_dialogue

  dialogue = _drive_capturing()
  assert injected, "sim never spoke -- the test would be vacuous"
  for reply in injected:
    assert GT not in reply
    assert "x >= 10" not in reply and "return x + y" not in reply
    assert "def f" not in reply
    assert "```" not in reply
  # And nothing GT-shaped rode in on a USER turn of the solver-visible dialogue.
  for m in dialogue:
    if m["role"] == "user":
      assert "x >= 10" not in m["content"]


# ── MINIMAL sim modes: A1 "codeonly" / A2 "plot" ──────────────────────────────
# The ladder that goes UP from the naive arm instead of stripping the spec arm
# down. A1's whole value is being a NULL-DELTA CONTROL, which only holds if its
# sim call is byte-identical to the naive arm's -- that is what these pin.


def test_codeonly_sim_call_is_byte_identical_to_the_naive_arm():
  # THE invariant behind A1. If this drifts, an A1-vs-naive gap stops meaning
  # "residual spec-path difference" and starts meaning "we changed the prompt",
  # and the whole ladder is uninterpretable.
  backend, seen = _capturing_backend()
  e = _env(sim_backend=backend, sim_prompt="codeonly")
  dialogue = [
      {"role": "user", "content": PROBLEM},
      {"role": "assistant", "content": "What's the cutoff?"},
  ]
  e.generate_user_turn(list(dialogue))
  sys_msg, usr_msg = seen[-1]
  assert sys_msg == templates.SIM_SYSTEM_PROMPT
  assert usr_msg == templates.build_sim_user_message(
      e.problem_description, e.ground_truth, dialogue
  )


def test_codeonly_carries_gt_but_no_spec_and_no_plot():
  backend, seen = _capturing_backend()
  e = _env(sim_backend=backend, sim_prompt="codeonly")
  e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  sys_msg, usr_msg = seen[-1]
  blob = sys_msg + usr_msg
  assert GT.strip() in blob          # the sim answers off the GT ...
  assert SPEC["plot"] not in blob    # ... with NO plot (that is A2) ...
  assert SPEC["requirements"] not in blob
  assert SPEC["scenario"] not in blob
  # ... and none of the spec arm's termination apparatus.
  assert "[TERMINATE]" not in blob
  assert "role-playing" not in blob


def test_plot_mode_is_codeonly_plus_exactly_the_plot():
  backend, seen = _capturing_backend()
  e = _env(sim_backend=backend, sim_prompt="plot")
  e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  sys_msg, usr_msg = seen[-1]
  assert sys_msg == templates.SIM_SYSTEM_PROMPT
  assert SPEC["plot"] in usr_msg
  assert GT.strip() in usr_msg
  assert "[TERMINATE]" not in usr_msg
  # The dialogue still comes LAST, after the plot -- the naive prompt's shape.
  assert usr_msg.index(SPEC["plot"]) < usr_msg.index("Here is the dialogue so far")


def test_plot_mode_with_an_empty_plot_falls_back_to_the_naive_call():
  # A spec row with no plot must not silently produce a half-spliced prompt.
  backend, seen = _capturing_backend()
  e = _env(sim_backend=backend, sim_prompt="plot")
  e.spec = dict(e.spec, plot="")
  e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  _, usr_msg = seen[-1]
  assert "There is one thing you would not bring up" not in usr_msg
  assert usr_msg == templates.build_sim_user_message(
      e.problem_description, e.ground_truth, [{"role": "user", "content": PROBLEM}]
  )


def test_sim_prompt_defaults_preserve_the_legacy_grounded_flag():
  # Back-compat: +colbench.grounded_sim and validate's --grounded still work,
  # and an explicit sim_prompt wins over the bool.
  assert _env().sim_prompt == "spec"
  assert _env(grounded=True).sim_prompt == "grounded"
  e = _env(grounded=True, sim_prompt="codeonly")
  assert e.sim_prompt == "codeonly" and e.grounded is False


def test_unknown_sim_prompt_fails_loudly():
  # A typo in SIM_PROMPT must not silently fall through to the spec arm and
  # burn a run.
  # Plain try/except, not pytest.raises: this module is also run directly by the
  # __main__ block below, which has no pytest.
  try:
    _env(sim_prompt="codonly")
  except ValueError as e:
    assert "unknown sim_prompt" in str(e)
  else:
    raise AssertionError("a typo'd sim_prompt was silently accepted")


def test_codeonly_uses_the_naive_arms_leak_detector():
  # A1/A2 must reject an UNFENCED `def f(...)` exactly as the naive arm does --
  # that is the dominant leak shape, and A1 exists to reproduce naive.
  leak = "Sure: def f(x, y): return x + y"
  clean = "Above ten we add them, below we subtract."
  calls = []
  def backend(_s, _u):
    calls.append(1)
    return leak if len(calls) == 1 else clean
  e = _env(sim_backend=backend, sim_prompt="codeonly")
  reply = e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  assert e.last_sim_code_rejected == 1, "unfenced def was not rejected"
  assert reply == clean


def test_spec_and_grounded_keep_the_fence_only_detector():
  # Regression guard: tightening the detector for the ladder must NOT change
  # spec/grounded, whose completed runs were produced under fence-only.
  leak = "Sure: def f(x, y): return x + y"
  for mode in ("spec", "grounded"):
    e = _env(sim_backend=lambda _s, _u: leak, sim_prompt=mode)
    reply = e.generate_user_turn([{"role": "user", "content": PROBLEM}])
    assert e.last_sim_code_rejected == 0, mode
    assert reply == leak, mode


def test_grounded_v0_is_the_pre_guard_prompt_and_stays_frozen():
  # The ONLY job of GROUNDED_SIM_SYSTEM_PROMPT_V0 is to be byte-identical to
  # what the pre-7fb1715e grounded run used, so pin its length + a line that
  # differs from V1. If this fails, the reproduction arm is no longer a
  # reproduction. (V1 was NOT that text: 7fb1715e changed the prompt in the same
  # commit that added the termination guard.)
  from colbench import prompts  # pylint: disable=g-import-not-at-top

  assert len(prompts.GROUNDED_SIM_SYSTEM_PROMPT_V0) == 2912
  assert (
      prompts.GROUNDED_SIM_SYSTEM_PROMPT_V0
      != prompts.GROUNDED_SIM_SYSTEM_PROMPT
  )
  # V0 has no "Being unclear about what you WANT" role-boundary paragraph; that
  # wording arrived with V1.
  assert "Being unclear" not in prompts.GROUNDED_SIM_SYSTEM_PROMPT_V0


def test_grounded_v0_routes_to_the_v0_template():
  seen = {}

  def backend(system_content, _user):
    seen["system"] = system_content
    return "sure"

  for mode, template in (
      ("grounded", "GROUNDED_SIM_SYSTEM_PROMPT"),
      ("grounded_v0", "GROUNDED_SIM_SYSTEM_PROMPT_V0"),
  ):
    e = _env(sim_backend=backend, sim_prompt=mode)
    e.generate_user_turn([{"role": "user", "content": PROBLEM}])
    from colbench import prompts  # pylint: disable=g-import-not-at-top

    head = getattr(prompts, template).split("{")[0]
    assert seen["system"].startswith(head), mode
    # Both are GT-conditioned, so the legacy alias must stay truthful.
    assert e.grounded, mode


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if (
        name.startswith("test_")
        and callable(fn)
        and fn.__code__.co_argcount == 0
    ):
      fn()
      print(f"PASS {name}")
  print("done")

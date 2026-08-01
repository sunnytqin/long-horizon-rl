"""CPU tests for colbench.env.ColBenchUserSimEnv.

Mocked simulator; no GPU, no server.

Covers answer extraction / termination, the leak invariant (GT source never
enters the solver's message list across a full mocked-sim episode), grading, and
the COLBENCH_DEBUG_SIM dump. Grading uses the in-process exec fallback.
"""

# These tests pin the behaviour of module-private helpers, so they reach for
# them directly.
# pylint: disable=protected-access

import logging
import os

import pytest

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

# The module-level setup above (env vars, sys.path) has to run
# before these imports resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench import env as env_mod
from colbench import templates
from colbench.env import ColBenchUserSimEnv

GT = (
    "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n "
    "       return x - y\n"
)
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."


def _sim_stub(reply="The threshold is 10 and below it we subtract."):
  """A sim backend that records its prompt and returns a fixed reply.

  Crucially the reply contains NO ground-truth source, so if the GT ever shows
  up in a solver-visible message it must have leaked through some other path.
  """
  captured = {}

  def backend(system_content, user_content):
    captured["system"] = system_content
    captured["user"] = user_content
    return reply

  return backend, captured


def _env(**kw):
  backend, captured = _sim_stub(
      **{k: kw.pop(k) for k in list(kw) if k == "reply"}
  )
  e = ColBenchUserSimEnv(
      problem_description=PROBLEM,
      ground_truth=GT,
      test_cases=CALLS,
      max_steps=10,
      sim_backend=backend,
      **kw,
  )
  return e, captured


# ── answer extraction / termination ──────────────────────────────────────────


def test_marker_answer_extracted_midturn():
  e, _ = _env()
  text = (
      "Sure.\nI WANT TO ANSWER:\n```python\ndef f(x, y):\n    return x + y\n```"
  )
  has, ans = e.is_answer(text, episode_done=False)
  assert has is True
  assert "def f" in ans


def test_no_marker_midturn_keeps_going():
  e, _ = _env()
  has, ans = e.is_answer("What range should x cover?", episode_done=False)
  assert has is False
  assert ans == ""


def test_final_turn_code_like_fallback():
  e, _ = _env()
  # No marker, but it's the last turn and the response is code-like -> accept as
  # the answer.
  has, ans = e.is_answer("def f(x, y):\n    return x + y", episode_done=True)
  assert has is True
  assert "def f" in ans


def test_think_block_stripped_before_marker():
  e, _ = _env()
  text = (
      "<think>the user probably wants a sum</think>I WANT TO ANSWER: "
      "def f(x, y): return x + y"
  )
  has, ans = e.is_answer(text, episode_done=False)
  assert has is True
  assert "<think>" not in ans and "def f" in ans


# ── simulator turn: capping + the GT is passed ONLY to the backend ────────────


def test_user_turn_capped_and_gt_only_in_sim_prompt():
  long_reply = "x" * 999
  e, captured = _env(reply=long_reply)
  messages = [
      {"role": "user", "content": PROBLEM},
      {"role": "assistant", "content": "What is the cutoff?"},
  ]
  reply = e.generate_user_turn(messages)
  # Reply handed to the solver is capped and contains no GT.
  assert len(reply) <= templates.HUMAN_RESPONSE_CHARACTER_LIMIT
  assert GT not in reply
  # The GT source WAS passed into the sim prompt (the hidden_information seam).
  assert GT in captured["user"]
  assert e.last_sim_reply == reply


def test_sim_char_limit_env_overrides_the_slice(monkeypatch):
  """SIM_CHAR_LIMIT=0 disables the post-hoc slice.

  Aligning the GT arm's user-turn budget with the SPEC arm's (which has no slice
  at all and is bounded only by SIM_MAX_TOKENS).

  Without this the GT user delivers <=400 chars/turn while the spec user
  delivers ~700-1000, so a GT-vs-spec result would confound environment quality
  with how much the user may say. Default (unset) stays 400, so every run that
  does not opt in is byte-identical.
  """
  long_reply = "x" * 999
  messages = [{"role": "user", "content": PROBLEM}]

  monkeypatch.setenv("SIM_CHAR_LIMIT", "0")
  e, _ = _env(reply=long_reply)
  assert (
      len(e.generate_user_turn(messages)) == 999
  ), "0 must disable the slice entirely"

  monkeypatch.setenv("SIM_CHAR_LIMIT", "120")
  e, _ = _env(reply=long_reply)
  assert len(e.generate_user_turn(messages)) == 120

  monkeypatch.delenv("SIM_CHAR_LIMIT", raising=False)
  e, _ = _env(reply=long_reply)
  assert (
      len(e.generate_user_turn(messages))
      == templates.HUMAN_RESPONSE_CHARACTER_LIMIT
  )


def test_marker_answer_is_graded_as_code_by_the_spec_extractor():
  """A GT-protocol turn must yield VALID code when the SPEC path grades it.

  Arm (1)'s policy is RL'd to emit "I WANT TO ANSWER:" + code, but the golden
  spec eval has no marker convention and grades templates.extract_last_code. If
  the marker survived into the graded string the sandbox would see a SyntaxError
  and the checkpoint would score ~0 for a protocol reason -- indistinguishable
  from a genuine capability result in the cross-arm comparison.
  """
  code = "def f(x):\n    return x + 1"
  for turn in (
      f"I WANT TO ANSWER:\n{code}",
      f"I WANT TO ANSWER:\n```python\n{code}\n```",
      f"i want to answer:\n{code}",
  ):
    extracted = templates.extract_last_code(
        [{"role": "assistant", "content": turn}]
    )
    assert (
        extracted == code
    ), f"marker leaked into the graded code for: {turn!r}"
    compile(
        extracted, "<graded>", "exec"
    )  # would raise SyntaxError before the fix
  # Spec-protocol turns (no marker) are untouched.
  assert (
      templates.extract_last_code(
          [
              {
                  "role": "assistant",
                  "content": f"Here you go:\n```python\n{code}\n```",
              }
          ]
      )
      == code
  )


def test_solver_prompt_keeps_the_sweet_rl_bullets_verbatim():
  """Only bullet 3 and the trailing paragraph may diverge.

  Divergence is measured against the sweet_rl original.

  COLBENCH_AGENT_SYSTEM_PROMPT is no longer derived from _AGENT_PROMPT_RAW by
  string surgery, so nothing else stops the two drifting apart. The unchanged
  bullets are what make arm (1) still recognisably ColBench rather than a prompt
  we invented.
  """
  raw, live = (
      templates._AGENT_PROMPT_RAW,
      templates.COLBENCH_AGENT_SYSTEM_PROMPT,
  )
  for bullet in (
      "1) Note that the problem is highly personalized",
      "2) Note that you should not ask human users complicated questions",
      "4) Note that you can only interact with the human users WITHIN 10",
      "5) You should be as concise as possible",
  ):
    assert bullet in raw and bullet in live, f"bullet drifted: {bullet!r}"
  # The marker is gone from the live prompt; the fence instruction replaces it.
  assert templates.ANSWER_MARKER in raw
  assert templates.ANSWER_MARKER not in live
  assert "```python" in live
  assert "{dialogue_history}" not in live


# ── GT-path submission: a fenced function, with the marker still accepted ─────


@pytest.mark.parametrize(
    "turn,expected_def",
    [
        (
            "Here is the function:\n```python\ndef f(x):\n    return x + 1\n```",
            True,
        ),
        # Unterminated fence: the 1024-token per-turn cap cut the closing ``` off.
        ("```python\ndef f(x):\n    return x + 1", True),
        # Legacy marker protocol must still submit (old parquets / old
        # checkpoints).
        ("I WANT TO ANSWER:\ndef f(x):\n    return x + 1", True),
    ],
)
def test_final_answer_accepts_both_submit_protocols(turn, expected_def):
  has_answer, ans = templates.final_answer(turn, episode_done=False)
  assert has_answer is True
  assert ("def f" in ans) is expected_def


@pytest.mark.parametrize(
    "turn",
    [
        # A bare `def` in PROSE must not force-submit -- this is the one-shot
        # arm.
        "Sure -- something like def parse(rows), is that right?",
        # A fenced block with no function definition is not a submission either.
        "Like this:\n```python\nprint(total)\n```\nDoes that match?",
        # Plain clarification.
        "What should happen when the file is empty?",
    ],
)
def test_final_answer_does_not_submit_mid_clarification(turn):
  assert templates.final_answer(turn, episode_done=False) == (False, "")


def test_last_turn_fallback_still_submits():
  """Ran out of turns without a fence or a marker.

  The last attempt is still graded.
  """
  has_answer, ans = templates.final_answer(
      "def f(x):\n    return x", episode_done=True
  )
  assert has_answer is True and "def f" in ans


def test_fence_wins_over_a_trailing_marker():
  """A FENCE, when present, beats the marker.

  Stripping unconditionally would regress.

  "```python...``` I WANT TO ANSWER: that's it" splits at the marker to the trailing prose and
  discards the function. The marker strip is therefore gated on there being no fence at all.
  """
  code = "def f(x):\n    return x + 1"
  assert (
      templates.extract_code_answer(
          f"```python\n{code}\n```\nI WANT TO ANSWER: that's my final answer"
      )
      == code
  )
  # Marker BEFORE a fence still works (the fence is what gets extracted either
  # way).
  assert (
      templates.extract_code_answer(
          f"I WANT TO ANSWER:\n```python\n{code}\n```"
      )
      == code
  )


# ── leak invariant: GT never appears in the solver's message list ─────────────


def test_leak_invariant_full_episode():
  """Drive a full mocked episode and assert no GT leak.

  The GT must never enter a solver-visible message.

  Mirrors colbench_agent's message handling: the solver sees [system, problem,
  then alternating assistant/user-reply]; the sim's dialogue view is separate.
  The GT is only ever handed to the sim backend inside generate_user_turn.
  """
  e, _ = _env()
  solver_messages = [
      {"role": "system", "content": templates.COLBENCH_AGENT_SYSTEM_PROMPT},
      {"role": "user", "content": PROBLEM},
  ]
  sim_dialogue = [{"role": "user", "content": PROBLEM}]

  # Two clarification turns, then a final answer.
  scripted = [
      "What happens when x is small?",
      "Got it, and above the cutoff?",
      "I WANT TO ANSWER:\n```python\n" + GT + "```",
  ]
  reward_val = 0.0
  for turn, assistant_text in enumerate(scripted):
    sim_dialogue.append({"role": "assistant", "content": assistant_text})
    is_last = turn == len(scripted) - 1
    has, ans = e.is_answer(assistant_text, episode_done=is_last)
    if has:
      reward_val = e.score(ans)["pass_rate"]
      break
    reply = e.generate_user_turn(sim_dialogue)
    sim_dialogue.append({"role": "user", "content": reply})
    solver_messages.append({"role": "assistant", "content": assistant_text})
    solver_messages.append({"role": "user", "content": reply})

  # The final answer WAS the GT itself (submitted by the solver) -> full score.
  # That is the solver's OWN output, not a leak; the invariant is about the
  # HIDDEN GT reaching the solver via the environment/user turns, so we check
  # only the injected user (sim) replies.
  for m in solver_messages:
    if m["role"] == "user" and m["content"] != PROBLEM:
      assert GT not in m["content"], "GT leaked into an injected user turn"
  assert reward_val == 1.0


# ── grading through the env ───────────────────────────────────────────────────


def test_score_exact_answer_full_marks():
  e, _ = _env()
  answer = "```python\n" + GT + "```"
  assert e.score(answer)["pass_rate"] == 1.0


def test_score_partial_answer_fraction():
  e, _ = _env()
  answer = "```python\ndef f(x, y):\n    return x + y\n```"
  assert e.score(answer)["pass_rate"] == 0.5


# ── COLBENCH_DEBUG_SIM dump renders ───────────────────────────────────────────


def test_debug_sim_dump_renders(monkeypatch, caplog):
  monkeypatch.setattr(env_mod, "_DEBUG_SIM", True)
  e, _ = _env()
  with caplog.at_level(logging.WARNING):
    e.generate_user_turn([{"role": "user", "content": PROBLEM}])
  joined = "\n".join(r.getMessage() for r in caplog.records)
  assert "[COLBENCH_SIM]" in joined


# ── code-leak detection (templates.detect_code_leak) ──────────────────────────


def test_detect_def_signature():
  assert templates.detect_code_leak("Here is it: def parse(x):", GT) == "def"


def test_detect_python_fence():
  assert templates.detect_code_leak("```python\nreturn 1\n```", GT) == "fenced"


def test_detect_ngram_expression_leak():
  # An inline formula copied from the GT (operator-dense) -> caught by (D).
  gt = "def f(a, b, c):\n    return (a * b) + (c * b) - (a * c) + (b * b) - a\n"
  reply = (
      "The result is computed as (a * b) + (c * b) - (a * c) + (b * b) "
      "- a exactly."
  )
  assert templates.detect_code_leak(reply, gt) == "ngram"


def test_detect_prose_spec_not_flagged():
  # A legitimate natural-language behavior description: shares identifiers with
  # the GT but NOT a run of operators -> must NOT be flagged (the Ex4-style
  # false positive we avoid).
  gt = (
      "def check(platform, version):\n"
      "    if platform == 'Linux' and version in ['10.0', '10.1']:\n"
      "        return True\n"
  )
  reply = (
      "For Linux with versions 10.0 or 10.1 the playback is paused, otherwise "
      "it is not."
  )
  assert templates.detect_code_leak(reply, gt) is None


def test_detect_clean_reply_none():
  assert (
      templates.detect_code_leak(
          "The threshold is 10; below it we subtract.", GT
      )
      is None
  )


# ── rejection sampling (env.generate_user_turn_checked) ───────────────────────


def _scripted_backend(replies):
  """A sim backend returning successive canned replies (one per call)."""
  seq = list(replies)
  state = {"i": 0}

  def backend(system_content, user_content):
    r = seq[min(state["i"], len(seq) - 1)]
    state["i"] += 1
    return r

  return backend


def test_rejection_accepts_after_retries():
  # Two leaking samples, then a clean one -> accepted on the 3rd try.
  backend = _scripted_backend(
      [
          "def f(x, y): return x + y",
          "```python\nreturn x - y\n```",
          "The cutoff is 10 and below it we subtract.",
      ]
  )
  e = ColBenchUserSimEnv(
      problem_description=PROBLEM,
      ground_truth=GT,
      test_cases=CALLS,
      sim_backend=backend,
  )
  res = e.generate_user_turn_checked(
      [{"role": "user", "content": PROBLEM}], max_tries=8
  )
  assert res["accepted"] is True
  assert res["tries"] == 3
  assert res["reasons"] == ["def", "fenced"]
  assert "def " not in res["reply"]
  assert e.last_sim_reply == res["reply"]


def test_rejection_exhausts_to_sim_failure():
  # Every sample leaks a def -> not accepted (a "simulation failure").
  backend = _scripted_backend(["def f(x, y): return x + y"])
  e = ColBenchUserSimEnv(
      problem_description=PROBLEM,
      ground_truth=GT,
      test_cases=CALLS,
      sim_backend=backend,
  )
  res = e.generate_user_turn_checked(
      [{"role": "user", "content": PROBLEM}], max_tries=5
  )
  assert res["accepted"] is False
  assert res["reply"] is None
  assert res["tries"] == 5
  assert res["reasons"] == ["def"] * 5


# ── sim thinking-kwarg guard (SIM_ENABLE_THINKING) ────────────────────────────


def test_sim_extra_body_default_sends_nothing(monkeypatch):
  monkeypatch.delenv("SIM_ENABLE_THINKING", raising=False)
  assert env_mod._sim_extra_body() is None  # safe default for all models


def test_sim_extra_body_explicit_false(monkeypatch):
  monkeypatch.setenv("SIM_ENABLE_THINKING", "false")
  assert env_mod._sim_extra_body() == {"enable_thinking": False}


def test_sim_extra_body_explicit_true(monkeypatch):
  monkeypatch.setenv("SIM_ENABLE_THINKING", "true")
  assert env_mod._sim_extra_body() == {"enable_thinking": True}


# ── sim sampling (must NOT be greedy for Qwen3) ───────────────────────────────


def test_sim_sampling_default_is_non_greedy(monkeypatch):
  for k in ("SIM_TEMPERATURE", "SIM_TOP_P", "SIM_TOP_K", "SIM_MIN_P"):
    monkeypatch.delenv(k, raising=False)
  temperature, top_p, top_k, min_p = env_mod._sim_sampling()
  assert temperature == 0.7 and top_p == 0.8 and top_k == 20 and min_p == 0.0
  assert temperature > 0.0  # Qwen3 degrades under greedy decoding


def test_sim_sampling_env_override(monkeypatch):
  monkeypatch.setenv("SIM_TEMPERATURE", "0.6")
  monkeypatch.setenv("SIM_TOP_P", "0.95")
  monkeypatch.setenv("SIM_TOP_K", "40")
  monkeypatch.setenv("SIM_MIN_P", "0.05")
  assert env_mod._sim_sampling() == (0.6, 0.95, 40, 0.05)


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if (
        name.startswith("test_")
        and callable(fn)
        and fn.__code__.co_argcount == 0
    ):
      fn()
      print(f"PASS {name}")
  print("(run via pytest for the monkeypatch/caplog debug test)")

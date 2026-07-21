# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the SPEC path: colbench.env_spec + the spec templates helpers (mocked sim).

Covers the leak invariant (GT source NEVER enters the spec sim prompt -- only the spec does),
grading parity with the GT env, the spec-specific templates helpers
(``sim_terminated``/``contains_code``/``extract_last_code``/``build_spec_sim_messages``), and the
USER-DRIVEN termination state machine via an inline driver that mirrors the plan's loop -- so the
contract is pinned before ``colbench_spec_agent`` / ``validate_colbench_spec`` implement it.
Grading uses the in-process exec fallback.
"""

import os

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

from colbench import templates  # noqa: E402
from colbench.env_spec import ColBenchSpecUserSimEnv  # noqa: E402

GT = "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n        return x - y\n"
WRONG = "def f(x, y):\n    return x + y\n"  # ignores the x<10 branch -> 0.5 pass-rate
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."
SPEC = {
    "persona": {"who": "an analyst", "domain": "ops", "python_skill": "analyst",
                "communication_style": "brief"},
    "scenario": "Needs a small helper for a report.",
    "requirements": "The user wants f(x,y): if x is at least 10 return x+y, otherwise x-y.",
    "plot": "The user reveals the threshold of 10 only if the assistant asks about the cutoff.",
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


def _env(sim_backend=None):
    return ColBenchSpecUserSimEnv(
        problem_description=PROBLEM, spec=SPEC, ground_truth=GT, test_cases=CALLS,
        sim_backend=sim_backend,
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
    e.generate_user_turn([{"role": "user", "content": PROBLEM},
                          {"role": "assistant", "content": "What's the cutoff?"}])
    # The spec (requirements/plot/persona) IS injected; the GT code is NOT.
    assert "at least 10" in captured["sys"] and "an analyst" in captured["sys"]
    assert GT not in captured["sys"] and GT not in captured["usr"]
    assert "x >= 10" not in captured["sys"] and "return x + y" not in captured["sys"]


def test_generate_user_turn_no_char_truncation():
    # The old HUMAN_RESPONSE_CHARACTER_LIMIT post-hoc slice is gone: a long reply is injected in
    # full (brevity is enforced at generation via SIM_MAX_TOKENS + the prompt, not by chopping).
    long_reply = "x" * (templates.HUMAN_RESPONSE_CHARACTER_LIMIT + 800) + " [TERMINATE]"
    e = _env(sim_backend=_scripted_backend([long_reply]))
    reply = e.generate_user_turn([{"role": "user", "content": PROBLEM}])
    assert reply == e.last_sim_raw                              # no post-hoc truncation
    assert len(reply) > templates.HUMAN_RESPONSE_CHARACTER_LIMIT
    assert templates.sim_terminated(reply) is True             # sentinel survives (nothing chopped)


# ── spec templates helpers ────────────────────────────────────────────────────

def test_sim_terminated_variants():
    assert templates.sim_terminated("All good, thanks! [TERMINATE]") is True
    assert templates.sim_terminated("looks [terminate] fine") is True  # case-insensitive
    assert templates.sim_terminated("<think>[TERMINATE]?</think>keep going") is False  # think-stripped
    assert templates.sim_terminated("could you also handle negatives?") is False


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
    assert templates.extract_last_code([{"role": "assistant", "content": "what cutoff?"}]) == ""


# ── sim code fidelity: detection, stripping, and rejection sampling ────────────

def test_sim_wrote_code_detects_fence():
    assert templates.sim_wrote_code("Here: ```python\ndef f(): pass\n```") is True
    assert templates.sim_wrote_code("just a bare ```\nx=1\n``` block") is True
    assert templates.sim_wrote_code("below 10 it should subtract, not add") is False
    assert templates.sim_wrote_code("set `area` to None when it's <= 0") is False  # inline backticks ok


def test_generate_user_turn_rejection_samples_code():
    # First reply writes code (rejected), second is natural language -> the NL one is returned.
    e = _env(sim_backend=_scripted_backend([
        "Sure: ```python\ndef f(x, y): return x + y\n```",
        "Below 10 it should subtract instead of add.",
    ]))
    reply = e.generate_user_turn([{"role": "user", "content": PROBLEM}])
    assert "```" not in reply and "subtract" in reply
    assert e.last_sim_code_rejected == 1


def test_generate_user_turn_flags_exhaustion_when_all_tries_write_code():
    # Every try writes code -> after sim_max_tries the env flags exhaustion (no strip/inject); the
    # loop aborts the conversation for inspection. last_sim_raw keeps the offending reply verbatim.
    offending = "Do this: ```python\ndef f(): return 0\n``` ok?"
    e = ColBenchSpecUserSimEnv(
        problem_description=PROBLEM, spec=SPEC, ground_truth=GT, test_cases=CALLS, sim_max_tries=3,
        sim_backend=_scripted_backend([offending]),
    )
    e.generate_user_turn([{"role": "user", "content": PROBLEM}])
    assert e.last_sim_code_reject_exhausted is True
    assert e.last_sim_code_rejected == 3
    assert "```python" in e.last_sim_raw          # offending reply kept verbatim (not stripped)


def test_generate_user_turn_no_exhaustion_when_reply_is_clean():
    e = _env(sim_backend=_scripted_backend(["Below 10 it should subtract, not add."]))
    e.generate_user_turn([{"role": "user", "content": PROBLEM}])
    assert e.last_sim_code_reject_exhausted is False
    assert e.last_sim_code_rejected == 0


# ── grading parity with the GT env ────────────────────────────────────────────

def test_score_full_and_partial():
    e = _env()
    assert e.score("```python\n" + GT + "```")["pass_rate"] == 1.0
    assert e.score("```python\n" + WRONG + "```")["pass_rate"] == 0.5


# ── USER-DRIVEN termination state machine (inline driver mirrors the plan loop) ──

def drive(env, assistant_turns, max_turns=10, max_code_proposals=3):
    """Replicate the spec agent loop's termination state machine (the pinned contract).

    ``colbench_spec_agent`` / ``validate_colbench_spec`` MUST mirror this: solver turn -> track
    last code / count proposals -> turn cap -> code cap -> else sim reply -> [TERMINATE]. Grades
    the last shown function; reward 0 (and terminated_by 'no_code') if none was ever shown.
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
        reply = env.generate_user_turn(sim_dialogue)
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
    return {"reward": reward, "terminated_by": terminated_by,
            "code_proposals": code_proposals, "showed_code": showed_code}


def test_gated_never_asked_terminates_on_imperfect_code():
    # Solver never asks and shows WRONG code; a faithful gated sim has nothing to reveal and
    # terminates -> we grade the imperfect code (0.5). This is the intended imperfect signal.
    e = _env(sim_backend=_scripted_backend(["Looks fine to me, thanks! [TERMINATE]"]))
    out = drive(e, [_code_turn(WRONG)])
    assert out["terminated_by"] == "user"
    assert out["reward"] == 0.5
    assert out["showed_code"] is True


def test_correct_on_code_then_terminate():
    # Wrong code -> one correction (no terminate) -> correct code -> terminate. Grade the correct
    # code (1.0). Sim replies: correction, then [TERMINATE].
    e = _env(sim_backend=_scripted_backend([
        "No -- below 10 it should subtract, not add.",
        "Perfect, that's exactly it. [TERMINATE]",
    ]))
    out = drive(e, [_code_turn(WRONG), _code_turn(GT)])
    assert out["terminated_by"] == "user"
    assert out["reward"] == 1.0
    assert out["code_proposals"] == 2


def test_code_cap_forces_terminate():
    # Sim never terminates; solver keeps proposing code -> code cap (3) fires -> grade last code.
    e = _env(sim_backend=_scripted_backend(["Hmm, not quite, keep trying."]))
    out = drive(e, [_code_turn(WRONG), _code_turn(WRONG), _code_turn(GT), _code_turn(GT)],
                max_code_proposals=3)
    assert out["terminated_by"] == "code_cap"
    assert out["code_proposals"] == 3
    assert out["reward"] == 1.0  # 3rd proposal (the code cap turn) was the correct GT


def test_turn_cap_no_code_reward_zero():
    # Solver only ever asks (no code); sim never terminates -> turn cap -> reward 0.
    e = _env(sim_backend=_scripted_backend(["Above 10 we add, below we subtract."]))
    out = drive(e, ["what cutoff?", "and below it?", "anything else?"], max_turns=3)
    assert out["terminated_by"] == "turn_cap"
    assert out["reward"] == 0.0
    assert out["showed_code"] is False


def test_user_terminates_without_code_is_no_code_reward_zero():
    # Sim emits [TERMINATE] before any code was shown -> no_code, reward 0.
    e = _env(sim_backend=_scripted_backend(["Okay, I think you have it. [TERMINATE]"]))
    out = drive(e, ["Tell me your requirements?"], max_turns=5)
    assert out["terminated_by"] == "no_code"
    assert out["reward"] == 0.0
    assert out["showed_code"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and fn.__code__.co_argcount == 0:
            fn()
            print(f"PASS {name}")
    print("done")

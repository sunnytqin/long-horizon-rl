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
"""CPU tests for colbench.env.ColBenchUserSimEnv (mocked simulator; no GPU, no server).

Covers answer extraction / termination, the leak invariant (GT source never enters the
solver's message list across a full mocked-sim episode), grading, and the COLBENCH_DEBUG_SIM
dump. Grading uses the in-process exec fallback.
"""

import logging
import os

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

from colbench import env as env_mod  # noqa: E402
from colbench import templates  # noqa: E402
from colbench.env import ColBenchUserSimEnv  # noqa: E402

GT = "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n        return x - y\n"
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."


def _sim_stub(reply="The threshold is 10 and below it we subtract."):
    """A sim backend that records the prompt it received and returns a fixed short reply.

    Crucially the reply contains NO ground-truth source, so if the GT ever shows up in a
    solver-visible message it must have leaked through some other path.
    """
    captured = {}

    def backend(system_content, user_content):
        captured["system"] = system_content
        captured["user"] = user_content
        return reply

    return backend, captured


def _env(**kw):
    backend, captured = _sim_stub(**{k: kw.pop(k) for k in list(kw) if k == "reply"})
    e = ColBenchUserSimEnv(
        problem_description=PROBLEM, ground_truth=GT, test_cases=CALLS,
        max_steps=10, sim_backend=backend, **kw,
    )
    return e, captured


# ── answer extraction / termination ──────────────────────────────────────────

def test_marker_answer_extracted_midturn():
    e, _ = _env()
    text = "Sure.\nI WANT TO ANSWER:\n```python\ndef f(x, y):\n    return x + y\n```"
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
    # No marker, but it's the last turn and the response is code-like -> accept as the answer.
    has, ans = e.is_answer("def f(x, y):\n    return x + y", episode_done=True)
    assert has is True
    assert "def f" in ans


def test_think_block_stripped_before_marker():
    e, _ = _env()
    text = "<think>the user probably wants a sum</think>I WANT TO ANSWER: def f(x, y): return x + y"
    has, ans = e.is_answer(text, episode_done=False)
    assert has is True
    assert "<think>" not in ans and "def f" in ans


# ── simulator turn: capping + the GT is passed ONLY to the backend ────────────

def test_user_turn_capped_and_gt_only_in_sim_prompt():
    long_reply = "x" * 999
    e, captured = _env(reply=long_reply)
    messages = [{"role": "user", "content": PROBLEM}, {"role": "assistant", "content": "What is the cutoff?"}]
    reply = e.generate_user_turn(messages)
    # Reply handed to the solver is capped and contains no GT.
    assert len(reply) <= templates.HUMAN_RESPONSE_CHARACTER_LIMIT
    assert GT not in reply
    # The GT source WAS passed into the sim prompt (the hidden_information seam).
    assert GT in captured["user"]
    assert e.last_sim_reply == reply


# ── leak invariant: GT never appears in the solver's message list ─────────────

def test_leak_invariant_full_episode():
    """Drive a full mocked episode and assert the GT never enters solver-visible messages.

    Mirrors colbench_agent's message handling: the solver sees [system, problem, then
    alternating assistant/user-reply]; the sim's dialogue view is separate. The GT is only
    ever handed to the sim backend inside generate_user_turn.
    """
    e, captured = _env()
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

    # The final answer WAS the GT itself (submitted by the solver) -> full score. That is the
    # solver's OWN output, not a leak; the invariant is about the HIDDEN GT reaching the solver
    # via the environment/user turns, so we check only the injected user (sim) replies.
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


if __name__ == "__main__":
    import pytest  # noqa: F401
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and fn.__code__.co_argcount == 0:
            fn()
            print(f"PASS {name}")
    print("(run via pytest for the monkeypatch/caplog debug test)")

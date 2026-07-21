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
"""Agent-loop tests for the SPEC path: drive ``ColBenchSpecAgentLoop.run`` end-to-end with a
scripted solver + scripted sim, asserting the same ``terminated_by`` / reward / masking the pinned
``tests/test_env_spec.py::drive`` contract produces.

CONTAINER-ONLY: importing ``colbench.colbench_spec_agent`` pulls in
``verl.experimental.agent_loop`` (needs py>=3.11 + ray), which is NOT importable in the light
conda eval env. The whole module is skipped there via ``importorskip`` and runs inside the
VERL/SGLang training container. It complements the env-level tests (which run everywhere) by
exercising the loop's token/mask bookkeeping and extra_fields, not just the env seams.
"""

import asyncio
import os

import pytest

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

# Skip the entire module unless the verl agent-loop stack is importable (container only).
pytest.importorskip("verl.experimental.agent_loop.agent_loop")

from verl.workers.rollout.replica import TokenOutput  # noqa: E402

from colbench.colbench_spec_agent import ColBenchSpecAgentLoop  # noqa: E402
from colbench.env_spec import ColBenchSpecUserSimEnv  # noqa: E402

# Reuse the env-level fixtures' shape (kept local to avoid importing a test module).
GT = "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n        return x - y\n"
WRONG = "def f(x, y):\n    return x + y\n"  # ignores x<10 -> 0.5 pass-rate
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."
SPEC = {
    "persona": {"who": "an analyst", "domain": "ops", "python_skill": "analyst",
                "communication_style": "brief"},
    "scenario": "Needs a small helper for a report.",
    "requirements": "The user wants f(x,y): if x is at least 10 return x+y, otherwise x-y.",
    "plot": "The user reveals the threshold of 10 only if the assistant asks about the cutoff.",
}


def _code_turn(src):
    return "Here's my function:\n```python\n" + src + "```"


def _scripted_backend(replies):
    seq = list(replies)
    state = {"i": 0}

    def backend(system_content, user_content):
        r = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return r

    return backend


class _FakeTokenizer:
    """UTF-8 byte tokenizer: exact encode/decode roundtrip + realistic token counts."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8", errors="ignore")


class _FakeServerManager:
    """Yields the scripted solver turns as TokenOutput, encoded by the fake tokenizer."""

    def __init__(self, tokenizer, solver_turns):
        self._tok = tokenizer
        self._turns = list(solver_turns)
        self._i = 0

    async def generate(self, **kwargs):
        text = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        ids = self._tok.encode(text)
        return TokenOutput(
            token_ids=ids,
            log_probs=[0.0] * len(ids),
            num_preempted=0,
            extra_fields={"min_global_steps": 0, "max_global_steps": 0},
        )


def _make_loop(solver_turns, sim_replies, *, max_assistant_turns=10, max_code_proposals=2,
               sim_max_tries=8, train_turns="all"):
    """Construct a ColBenchSpecAgentLoop bypassing AgentLoopBase.__init__, wired to fakes."""
    obj = object.__new__(ColBenchSpecAgentLoop)
    tok = _FakeTokenizer()
    obj.tokenizer = tok
    obj.server_manager = _FakeServerManager(tok, solver_turns)
    obj.loop = asyncio.new_event_loop()
    obj.prompt_length = 4096
    obj.response_length = 8192
    obj.max_assistant_turns = max_assistant_turns
    obj.max_new_tokens_per_turn = 1024
    obj.env_step_timeout = 60.0
    obj.reward_time_limit = 6.0
    obj.train_turns = train_turns
    obj.max_code_proposals = max_code_proposals
    obj.sim_max_tries = sim_max_tries

    # apply_chat_template is normally an AgentLoopBase method; override on the instance with a
    # byte-encoding stub (only token COUNTS + mask placement matter to the loop under test).
    async def _fake_act(messages, remove_system_prompt=False):
        text = "".join(m.get("content", "") for m in messages)
        return tok.encode(text)

    obj.apply_chat_template = _fake_act

    # Bind a spec env with the scripted sim backend (the loop builds its own env in run(), but we
    # inject the sim backend via extra_info-independent monkeypatch on ColBenchSpecUserSimEnv is
    # awkward; instead patch the default backend by pre-seeding env creation through kwargs). The
    # loop reads extra_info.spec + ground_truth; we route the scripted backend via a subclass.
    obj._test_sim_backend = _scripted_backend(sim_replies)
    return obj


def _run(obj, spec=SPEC):
    kwargs = {
        "raw_prompt": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": PROBLEM},
        ],
        "extra_info": {
            "spec": spec,
            "ground_truth": {"problem_description": PROBLEM, "ground_truth": GT, "test_cases": CALLS},
        },
        "index": 0,
    }
    # Inject the scripted sim backend into every env the loop builds.
    orig_post_init = ColBenchSpecUserSimEnv.__post_init__

    def patched_post_init(self):
        self.sim_backend = obj._test_sim_backend

    ColBenchSpecUserSimEnv.__post_init__ = patched_post_init
    try:
        return obj.loop.run_until_complete(obj.run({"temperature": 0.7}, **kwargs))
    finally:
        ColBenchSpecUserSimEnv.__post_init__ = orig_post_init


def test_correct_code_then_user_terminate():
    # Solver asks, sim answers, solver shows GT, sim [TERMINATE] -> reward 1.0, terminated_by user.
    obj = _make_loop(
        solver_turns=["What's the cutoff?", _code_turn(GT)],
        sim_replies=["It's 10.", "Perfect, thanks! [TERMINATE]"],
    )
    out = _run(obj)
    ef = out.extra_fields
    assert ef["terminated_by"] == "user"
    assert out.reward_score == 1.0
    assert ef["showed_code"] is True
    assert ef["code_proposals"] == 1
    # Mask: solver turns are 1, the injected sim turn is 0, and at least one of each exists.
    assert any(m == 1 for m in out.response_mask)
    assert any(m == 0 for m in out.response_mask)


def test_code_cap_forces_grade():
    # Two WRONG proposals hit max_code_proposals=2 -> code_cap, grade last (0.5).
    obj = _make_loop(
        solver_turns=[_code_turn(WRONG), _code_turn(WRONG)],
        sim_replies=["Not quite, try again."],
        max_code_proposals=2,
    )
    out = _run(obj)
    assert out.extra_fields["terminated_by"] == "code_cap"
    assert out.extra_fields["code_proposals"] == 2
    assert out.reward_score == 0.5


def test_user_terminate_without_code_is_no_code_zero():
    obj = _make_loop(
        solver_turns=["Tell me more?"],
        sim_replies=["I think you've got it. [TERMINATE]"],
    )
    out = _run(obj)
    assert out.extra_fields["terminated_by"] == "no_code"
    assert out.extra_fields["showed_code"] is False
    assert out.reward_score == 0.0


def test_sim_code_reject_exhaustion_aborts():
    # Sim always writes code -> exhaustion -> terminated_by sim_code_reject; grade last shown code.
    obj = _make_loop(
        solver_turns=[_code_turn(GT), "anything"],
        sim_replies=["```python\ndef f(x, y): return x + y\n```"],
        sim_max_tries=3,
    )
    out = _run(obj)
    assert out.extra_fields["terminated_by"] == "sim_code_reject"
    assert out.extra_fields["sim_code_rejected"] == 3
    assert out.reward_score == 1.0  # GT was shown before the abort


def test_all_turns_mask_keeps_every_solver_turn():
    # train_turns='all' (default) -> every solver span stays 1; only sim turns are 0.
    obj = _make_loop(
        solver_turns=["What's the cutoff?", _code_turn(GT)],
        sim_replies=["It's 10.", "Great, thanks! [TERMINATE]"],
        train_turns="all",
    )
    out = _run(obj)
    # Two solver turns worth of 1s plus one sim turn of 0s.
    assert out.response_mask.count(1) == len(("What's the cutoff?" + _code_turn(GT)).encode("utf-8"))
    assert out.response_mask.count(0) == len("It's 10.".encode("utf-8"))

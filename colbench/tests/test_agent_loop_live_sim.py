"""Agent-loop tests for the GT path's LIVE-weights simulator (``+colbench.sim_live=True``).

The arm under test keeps ONE copy of the weights for the whole run: the user turn is generated
by ``server_manager.generate`` (the training rollout engine) instead of an HTTP call to a frozen
sim server. What must hold:

  * the sim's prompt is BYTE-IDENTICAL to the frozen path's (same env._build_sim_prompt), so the
    two arms differ only in which weights answer -- and it still carries the hidden GT;
  * the sim's generated TOKENS never enter the solver trajectory (only the <=400-char text does,
    at mask=0), so the leak invariant and the mask bookkeeping are unchanged;
  * the sim call uses a FRESH request_id (not the solver's shared KV-prefix session);
  * ``min/max_global_steps`` reflect the SOLVER turns only.

CONTAINER-ONLY in spirit (imports verl.experimental.agent_loop), but the conftest StrEnum
backport + in-process grading let it run in the py3.10 conda env too:
  CODECONTEST_ALLOW_INPROCESS=1 python -m pytest colbench/tests/test_agent_loop_live_sim.py -v
"""

import asyncio
import os

import pytest

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

pytest.importorskip("verl.experimental.agent_loop.agent_loop")

from colbench import templates  # noqa: E402
from colbench.colbench_agent import ColBenchAgentLoop  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

GT = "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n        return x - y\n"
WRONG = "def f(x, y):\n    return x + y\n"  # ignores x<10 -> 0.5 pass-rate
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."


def _answer_turn(src):
    return "I WANT TO ANSWER:\n```python\n" + src + "```"


class _FakeTokenizer:
    """UTF-8 byte tokenizer: exact encode/decode roundtrip + realistic token counts."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8", errors="ignore")

    def apply_chat_template(self, messages, **kwargs):
        # Stand-in for the real template: concatenate contents. The live sim path routes its
        # prompt through here (verl.utils.tokenizer.chat_template.apply_chat_template), so the
        # test can assert on what the sim was actually asked.
        return self.encode("".join(m.get("content", "") for m in messages))


class _RoleAwareServerManager:
    """One engine serving BOTH roles -- exactly the sim_live topology.

    Solver turns and sim turns are told apart by the request_id: the loop reuses ONE id for the
    solver and mints a fresh one per sim call, so any id seen twice is the solver's.
    """

    def __init__(self, tokenizer, solver_turns, sim_replies):
        self._tok = tokenizer
        self._solver = list(solver_turns)
        self._sim = list(sim_replies)
        self._si = 0
        self._ri = 0
        self.solver_ids = []
        self.sim_ids = []
        self.sim_prompts = []  # decoded sim prompts (assert GT presence / prompt identity)
        self.sim_sampling = []  # sampling params each sim call was made with

    async def generate(self, *, request_id, prompt_ids, sampling_params, **kwargs):
        # The solver's request_id is established on call 1 and reused; sim ids are always new.
        if not self.solver_ids or request_id == self.solver_ids[0]:
            self.solver_ids.append(request_id)
            text = self._solver[min(self._si, len(self._solver) - 1)]
            self._si += 1
        else:
            self.sim_ids.append(request_id)
            self.sim_prompts.append(self._tok.decode(prompt_ids))
            self.sim_sampling.append(dict(sampling_params))
            text = self._sim[min(self._ri, len(self._sim) - 1)]
            self._ri += 1
        ids = self._tok.encode(text)
        return TokenOutput(
            token_ids=ids,
            log_probs=[0.0] * len(ids),
            num_preempted=0,
            # Deliberately non-zero: the loop must NOT let sim outputs set these (it reads the
            # first min / last max it sees, which must come from solver turns only).
            extra_fields={"min_global_steps": 7, "max_global_steps": 7},
        )


def _make_loop(
    solver_turns, sim_replies, *, sim_live=True, max_assistant_turns=10, sim_reject_max_tries=0, train_turns="all"
):
    """Construct a ColBenchAgentLoop bypassing AgentLoopBase.__init__, wired to one fake engine."""
    obj = object.__new__(ColBenchAgentLoop)
    tok = _FakeTokenizer()
    obj.tokenizer = tok
    obj.server_manager = _RoleAwareServerManager(tok, solver_turns, sim_replies)
    obj.loop = asyncio.new_event_loop()
    obj.prompt_length = 4096
    obj.response_length = 8192
    obj.max_assistant_turns = max_assistant_turns
    obj.max_new_tokens_per_turn = 1024
    obj.env_step_timeout = 60.0
    obj.reward_time_limit = 6.0
    obj.train_turns = train_turns
    obj.sim_reject_max_tries = sim_reject_max_tries
    obj.sim_live = sim_live
    # Normally set by AgentLoopBase.__init__ from data.apply_chat_template_kwargs.
    obj.apply_chat_template_kwargs = {}

    async def _fake_act(messages, remove_system_prompt=False):
        return tok.encode("".join(m.get("content", "") for m in messages))

    obj.apply_chat_template = _fake_act
    return obj


def _run(obj):
    kwargs = {
        "raw_prompt": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": PROBLEM},
        ],
        "extra_info": {
            "ground_truth": {"problem_description": PROBLEM, "ground_truth": GT, "test_cases": CALLS},
        },
        "index": 0,
    }
    return obj.loop.run_until_complete(obj.run({"temperature": 0.7}, **kwargs))


def test_live_sim_generates_on_the_training_engine():
    # The sim turn must come from server_manager.generate, with a request_id DISTINCT from the
    # solver's shared one, and its reply must be injected at mask=0.
    obj = _make_loop(
        solver_turns=["What's the cutoff?", _answer_turn(GT)],
        sim_replies=["It's 10."],
    )
    out = _run(obj)
    sm = obj.server_manager
    assert len(sm.sim_ids) == 1, "the live sim never hit the rollout engine"
    assert sm.sim_ids[0] not in sm.solver_ids, "sim reused the solver's KV-prefix request_id"
    assert len(set(sm.solver_ids)) == 1, "solver turns must share ONE request_id"
    assert out.reward_score == 1.0
    assert out.response_mask.count(0) == len(b"It's 10.")  # sim text, no gradient


def test_live_sim_prompt_carries_the_hidden_gt_and_never_leaks_into_the_solver():
    obj = _make_loop(
        solver_turns=["What's the cutoff?", _answer_turn(GT)],
        sim_replies=["It's 10."],
    )
    out = _run(obj)
    sim_prompt = obj.server_manager.sim_prompts[0]
    # Same prompt content the frozen path builds: sim system prompt + build_sim_user_message(GT).
    expected_user = templates.build_sim_user_message(
        PROBLEM, GT, [{"role": "user", "content": PROBLEM}, {"role": "assistant", "content": "What's the cutoff?"}]
    )
    assert sim_prompt == templates.SIM_SYSTEM_PROMPT + expected_user
    assert GT in sim_prompt  # the sim (and ONLY the sim) sees the ground truth
    # The solver trajectory contains the GT only because the SOLVER wrote it, never as sim tokens:
    # the sim generated 8 bytes ("It's 10.") and that is exactly what got mask=0.
    assert out.response_mask.count(0) == 8


def test_live_sim_uses_the_shared_sim_sampling_envs():
    # Same SIM_* knobs as the frozen backend -> the arms differ only in which weights answer.
    os.environ["SIM_TEMPERATURE"] = "0.9"
    os.environ["SIM_MAX_TOKENS"] = "256"
    try:
        obj = _make_loop(solver_turns=["Q?", _answer_turn(GT)], sim_replies=["It's 10."])
        _run(obj)
    finally:
        del os.environ["SIM_TEMPERATURE"], os.environ["SIM_MAX_TOKENS"]
    sp = obj.server_manager.sim_sampling[0]
    assert sp["temperature"] == 0.9 and sp["max_new_tokens"] == 256
    assert "min_p" not in sp  # verl rollout has no min_p field; don't smuggle it into the engine


def test_global_steps_come_from_solver_turns_only():
    # The fake engine reports 7 for every call incl. the sim's. Off-policy staleness must still
    # describe the trained turns -- assert the loop never picks a value up from a sim response.
    obj = _make_loop(solver_turns=["Q?", _answer_turn(GT)], sim_replies=["It's 10."])
    out = _run(obj)
    assert out.extra_fields["min_global_steps"] == 7
    assert out.extra_fields["max_global_steps"] == 7
    # 2 solver calls + 1 sim call all report 7, so equality alone is weak; the real guarantee is
    # structural (the sim backend discards extra_fields). Pin it so a refactor that starts
    # threading sim outputs into the bookkeeping fails here.
    assert len(obj.server_manager.sim_ids) == 1


def test_sim_drift_metrics_present_on_every_rollout():
    obj = _make_loop(solver_turns=["Q?", _answer_turn(GT)], sim_replies=["It's 10."])
    rei = _run(obj).extra_fields["reward_extra_info"]
    assert rei["sim_live"] == 1.0
    assert rei["sim_reply_chars"] == float(len("It's 10."))
    assert rei["sim_leak_frac"] == 0.0
    # A trajectory with NO sim turn (solver answers immediately) still carries all three keys --
    # verl reads the reward_extra_info key set from the first sample.
    rei2 = _run(_make_loop(solver_turns=[_answer_turn(WRONG)], sim_replies=[])).extra_fields["reward_extra_info"]
    assert rei2["sim_reply_chars"] == 0.0 and rei2["sim_leak_frac"] == 0.0
    assert rei2["pass_rate"] == 0.5


def test_sim_leak_frac_catches_a_sim_that_hands_over_code():
    # The live sim shares the solver's weights, so "the user starts writing the function" is the
    # failure mode to watch. With rejection sampling OFF the monitor must report it (and the
    # leaked code still only ever reaches the solver as text, never as trainable tokens).
    obj = _make_loop(
        solver_turns=["Just write it for me", _answer_turn(GT)],
        sim_replies=["sure: def f(x, y): return x + y"],
    )
    rei = _run(obj).extra_fields["reward_extra_info"]
    assert rei["sim_leak_frac"] == 1.0


def test_live_sim_rejection_sampling_resamples_until_clean():
    # sim_reject_max_tries>0 on the live path: the async checked loop must resample the leaking
    # reply and inject the clean one (tries counted in sim_reject_tries).
    obj = _make_loop(
        solver_turns=["Just write it for me", _answer_turn(GT)],
        sim_replies=["def f(x, y): return x + y", "I'd rather describe it: use a cutoff of 10."],
        sim_reject_max_tries=4,
    )
    rei = _run(obj).extra_fields["reward_extra_info"]
    assert rei["sim_reject_tries"] == 2.0  # one rejected, one accepted
    assert rei["sim_leak_frac"] == 0.0
    assert len(obj.server_manager.sim_ids) == 2
    assert len(set(obj.server_manager.sim_ids)) == 2  # a fresh request_id per sample


def test_sim_timeout_is_observable_and_kills_the_episode_at_reward_zero():
    # A slow sim is the live arm's main contention risk, and its damage is NOT "the step is slow":
    # wait_for fires -> the loop breaks -> an unanswered episode trains at reward 0. That must be
    # VISIBLE, so the flag rides reward_extra_info (AgentLoopOutput.metrics is a fixed-field
    # pydantic model with extra='ignore' -- keys put there are silently dropped).
    obj = _make_loop(solver_turns=["What's the cutoff?", _answer_turn(GT)], sim_replies=["It's 10."])
    obj.env_step_timeout = 0.05
    real_generate = obj.server_manager.generate

    async def slow_sim(*, request_id, prompt_ids, sampling_params, **kwargs):
        if obj.server_manager.solver_ids and request_id != obj.server_manager.solver_ids[0]:
            await asyncio.sleep(0.5)  # 10x the deadline
        return await real_generate(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, **kwargs
        )

    obj.server_manager.generate = slow_sim
    out = _run(obj)
    rei = out.extra_fields["reward_extra_info"]
    assert rei["sim_turn_timeout"] == 1.0
    assert rei["answered"] == 0.0 and out.reward_score == 0.0  # the artifact GRPO would train on
    assert rei["sim_seconds"] >= 0.05  # latency is recorded, not lost
    assert rei["num_assistant_turns"] == 1.0  # episode truncated after turn 1


def test_timeout_keys_present_on_the_happy_path():
    # verl reads the reward_extra_info key set from the FIRST sample, so these must exist at 0.0
    # on every normal rollout or the whole run loses the columns.
    obj = _make_loop(solver_turns=["Q?", _answer_turn(GT)], sim_replies=["It's 10."])
    rei = _run(obj).extra_fields["reward_extra_info"]
    assert rei["sim_turn_timeout"] == 0.0 and rei["env_score_timeout"] == 0.0
    assert rei["sim_seconds"] > 0.0 and rei["sim_seconds_max"] >= rei["sim_seconds"]


def test_frozen_path_untouched_when_sim_live_false():
    # sim_live=False must NOT call the engine for the sim; it goes through env.sim_backend
    # (patched here to a stub, standing in for the frozen HTTP server).
    from colbench.env import ColBenchUserSimEnv

    obj = _make_loop(solver_turns=["Q?", _answer_turn(GT)], sim_replies=[], sim_live=False)
    orig = ColBenchUserSimEnv.__post_init__

    def patched(self):
        self.sim_backend = lambda s, u: "It's 10."

    ColBenchUserSimEnv.__post_init__ = patched
    try:
        out = _run(obj)
    finally:
        ColBenchUserSimEnv.__post_init__ = orig
    assert obj.server_manager.sim_ids == []  # the engine served the solver only
    assert out.extra_fields["reward_extra_info"]["sim_live"] == 0.0
    assert out.reward_score == 1.0

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
"""CPU unit test for CodeRefineAgentLoop control flow / masking / reward.

The LLM server, tokenizer and chat-template are mocked so the test is pure-Python
and deterministic. The env (GTOracleEnv) is REAL and runs the subprocess sandbox,
so this also exercises the loop<->env<->local_exec integration end to end.
"""

import asyncio

from codecontest.code_refine_agent import CodeRefineAgentLoop
from verl.workers.rollout.replica import TokenOutput

GT_IN = ["2 3\n", "10 20\n", "1 1\n"]
GT_OUT = ["5\n", "30\n", "2\n"]
GROUND_TRUTH = {"test_input": GT_IN, "test_output": GT_OUT, "test_time_limit": 4.0}

PASS_CODE = "```python\nimport sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)\n```"
FAIL_CODE = "```python\nprint(0)\n```"

FEEDBACK_IDS = [101, 102, 103]  # what the mocked apply_chat_template returns


class _FakeTokenizer:
    def __init__(self, id_to_text):
        self._map = id_to_text

    def decode(self, ids, skip_special_tokens=True):
        return self._map[tuple(ids)]


class _FakeServer:
    """Returns canned (token_ids) responses in order."""

    def __init__(self, response_id_lists):
        self._responses = list(response_id_lists)
        self._i = 0

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        ids = self._responses[self._i]
        self._i += 1
        return TokenOutput(token_ids=ids, log_probs=None)


def _build_loop(turns, response_length=4096, max_assistant_turns=3, train_turns="all"):
    """turns: list of (token_ids, decoded_text). Returns (loop, decode_map)."""
    loop = object.__new__(CodeRefineAgentLoop)
    loop.response_length = response_length
    loop.prompt_length = 4096
    loop.max_assistant_turns = max_assistant_turns
    loop.max_new_tokens_per_turn = None
    loop.on_overflow = "end_zero_reward"
    loop.default_exec_timeout = 4.0
    loop.default_max_failures_shown = 3
    loop.default_max_gt_test = 20
    loop.train_turns = train_turns

    id_to_text = {tuple(ids): text for ids, text in turns}
    loop.tokenizer = _FakeTokenizer(id_to_text)
    loop.server_manager = _FakeServer([ids for ids, _ in turns])

    async def fake_apply_chat_template(messages, **kwargs):
        return list(FEEDBACK_IDS)

    loop.apply_chat_template = fake_apply_chat_template
    return loop


def _run(loop):
    async def _go():
        loop.loop = asyncio.get_running_loop()
        return await loop.run(sampling_params={"temperature": 1.0}, raw_prompt=[{"role": "user", "content": "p"}], extra_info={"ground_truth": GROUND_TRUTH, "index": 0})

    return asyncio.run(_go())


def test_fail_fail_pass_gets_reward_1_and_correct_mask():
    out = _run(_build_loop([([1], FAIL_CODE), ([2], FAIL_CODE), ([3], PASS_CODE)]))
    assert out.reward_score == 1.0
    assert out.extra_fields["solved"] is True
    assert out.extra_fields["solved_at_turn"] == 2
    assert out.extra_fields["num_assistant_turns"] == 3
    # assistant token (1) + feedback(0,0,0) + assistant(1) + feedback(0,0,0) + assistant(1)
    assert out.response_mask == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert out.num_turns == 3 + 2 + 1


def test_train_turns_final_only_trains_last_turn():
    # final_only zeroes every solver turn except the last; the rolled-out sequence
    # (response_ids) is unchanged, only the loss mask differs. Feedback turns stay 0.
    out = _run(_build_loop([([1], FAIL_CODE), ([2], FAIL_CODE), ([3], PASS_CODE)], train_turns="final_only"))
    assert out.reward_score == 1.0
    assert out.response_mask == [0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert len(out.response_ids) == len(out.response_mask)


def test_train_turns_final_only_turn0_solve_trains_turn0():
    # Solving at turn 0: the last (== only) solver turn IS turn 0, so it stays trained.
    out = _run(_build_loop([([7], PASS_CODE)], train_turns="final_only"))
    assert out.reward_score == 1.0
    assert out.extra_fields["solved_at_turn"] == 0
    assert out.response_mask == [1]


def test_all_fail_gets_reward_0_no_trailing_feedback():
    out = _run(_build_loop([([1], FAIL_CODE), ([2], FAIL_CODE), ([3], FAIL_CODE)]))
    assert out.reward_score == 0.0
    assert out.extra_fields["solved"] is False
    assert out.extra_fields["num_assistant_turns"] == 3
    # last (3rd) assistant turn is not followed by feedback
    assert out.response_mask == [1, 0, 0, 0, 1, 0, 0, 0, 1]


def test_solved_on_turn_0():
    out = _run(_build_loop([([7], PASS_CODE)]))
    assert out.reward_score == 1.0
    assert out.extra_fields["solved_at_turn"] == 0
    assert out.extra_fields["num_assistant_turns"] == 1
    assert out.response_mask == [1]
    assert out.num_turns == 2


def test_overflow_before_solve_gives_reward_0():
    # response_length=4: turn-0 emits 2 tokens, feedback(3) would push to 5 -> overflow.
    loop = _build_loop([([1, 1], FAIL_CODE), ([2, 2], FAIL_CODE)], response_length=4)
    out = _run(loop)
    assert out.reward_score == 0.0
    assert out.extra_fields["overflow"] is True
    assert out.extra_fields["solved"] is False
    assert out.extra_fields["num_assistant_turns"] == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all agent-loop tests passed")

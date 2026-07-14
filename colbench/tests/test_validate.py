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
"""CPU tests for colbench.validate_colbench.run_eval (no GPU, no SGLang, no sim server).

Drives the offline multi-turn eval loop with a FAKE solver engine (scripted turns) + a STUB
sim backend + the in-process exec grader, and checks: the fractional-reward summary, that a
correct submission scores 1.0, that the saved-conversation dump is bounded by
--max_saved_convos while metrics still cover ALL trajectories, and that the hidden GT never
leaks into a saved (solver-visible) message. Mirrors the stub style of test_env.py.
"""

import json
import os
from argparse import Namespace

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

import pandas as pd  # noqa: E402

from colbench import validate_colbench as vc  # noqa: E402

GT = "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n        return x - y\n"
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."


class FakeTokenizer:
    """Minimal chat tokenizer: join message contents; whitespace 'tokens'."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "\n".join(str(m["content"]) for m in messages)

    def encode(self, text, add_special_tokens=False):
        return (text or "").split()


class FakeLLM:
    """Scripted solver engine. Returns turn `i`'s response for every prompt in the batch,
    advancing the turn counter once per `.generate` call (one call per turn in run_eval)."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.turn = 0

    def generate(self, prompt, sampling_params):
        text = self.scripts[min(self.turn, len(self.scripts) - 1)]
        self.turn += 1
        return [{"text": text, "meta_info": {"completion_tokens": len(text.split())}} for _ in prompt]


def _sim_backend(system_content, user_content):
    """Frozen-sim stub: short reply carrying NO ground truth."""
    return "The cutoff is 10; below it we subtract."


def _val_df(n_problems=2):
    rows = []
    for i in range(n_problems):
        gt = {"problem_description": PROBLEM, "ground_truth": GT, "test_cases": list(CALLS)}
        rows.append({
            "prompt": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": PROBLEM},
            ],
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {"ground_truth": gt, "index": i},
        })
    return pd.DataFrame(rows)


def _args(max_saved_convos=100, sim_reject_max_tries=0):
    return Namespace(
        model="fake-model",
        val_file="fake.parquet",
        max_assistant_turns=3,
        reward_time_limit=6.0,
        max_response_length=4096,
        max_prompt_length=2048,
        top_p=0.95,
        top_k=-1,
        max_new_tokens_per_turn=256,
        grade_concurrency=4,
        seed=0,
        max_saved_convos=max_saved_convos,
        # Rejection sampling: 0 => single-shot sim turns (existing behavior).
        sim_reject_max_tries=sim_reject_max_tries,
        sim_reject_ngram_n=10,
        sim_reject_min_ops=2,
    )


def _run(tmp_path, n_samples=2, max_saved_convos=100, scripts=None,
         sim_reject_max_tries=0, sim_backend=None):
    if scripts is None:
        # turn 0: a clarification question (no answer -> a sim turn is injected);
        # turn 1: submit the exact GT (-> all cases pass, reward 1.0).
        scripts = [
            "What is the cutoff for x?",
            "I WANT TO ANSWER:\n```python\n" + GT + "```",
        ]
    args = _args(max_saved_convos=max_saved_convos, sim_reject_max_tries=sim_reject_max_tries)
    out_path = str(tmp_path / "eval.json")
    llm = FakeLLM(scripts)
    summary = vc.run_eval(
        llm, FakeTokenizer(), _val_df(2), temperature=0.6, n_samples=n_samples,
        args=args, out_path=out_path, max_model_len=args.max_prompt_length + args.max_response_length,
        sim_backend=sim_backend or _sim_backend,
    )
    with open(out_path) as f:
        dump = json.load(f)
    return summary, dump


def test_correct_submission_scores_full_pass_rate(tmp_path):
    summary, dump = _run(tmp_path, n_samples=2)
    assert summary["n_problems"] == 2
    assert summary["n_trajectories"] == 4
    assert summary["mean_pass_rate"] == 1.0
    assert summary["all_pass_rate"] == 1.0
    assert summary["answered_rate"] == 1.0
    # Answered on turn index 1 -> 2 turns taken.
    assert summary["avg_turns_to_answer"] == 2.0


def test_saved_conversations_bounded_but_metrics_over_all(tmp_path):
    # 2 problems x 3 samples = 6 trajectories, but only 2 conversations saved.
    summary, dump = _run(tmp_path, n_samples=3, max_saved_convos=2)
    assert summary["n_trajectories"] == 6           # metrics cover all 6
    assert dump["num_saved_conversations"] == 2     # only 2 dumped
    assert len(dump["trajectories"]) == 2


def _leaking_sim_backend(system_content, user_content):
    """A frozen-sim stub that always hands over code (a `def`) -> always rejected."""
    return "Sure, here it is: def f(x, y): return x + y if x >= 10 else x - y"


def test_rejection_on_clean_backend_records_no_retry(tmp_path):
    # Rejection sampling enabled but the sim is already clean -> accepted on the first try,
    # every turn recorded, no simulation failures, no denominator change.
    summary, dump = _run(tmp_path, n_samples=2, sim_reject_max_tries=8)
    assert summary["n_sim_failures"] == 0
    assert summary["mean_pass_rate"] == 1.0
    rj = summary["rejection_sampling"]
    assert rj["enabled"] is True and rj["max_tries"] == 8
    assert rj["n_user_turns_accepted"] == 4        # 2 problems x 2 samples, one sim turn each
    assert rj["n_user_turns_with_retry"] == 0
    assert rj["reject_reason_counts"] == {}
    # Every saved trajectory logs its (single, first-try) sim turn.
    for traj in dump["trajectories"]:
        assert traj["sim_failed"] is False
        assert [ev["tries"] for ev in traj["sim_reject_events"]] == [1]


def test_rejection_exhaustion_marks_simulation_failure(tmp_path):
    # The sim only ever produces code -> every trajectory becomes a "simulation failure":
    # terminated, excluded from the pass-rate denominator, and its own reported category.
    summary, dump = _run(
        tmp_path, n_samples=2, sim_reject_max_tries=4, sim_backend=_leaking_sim_backend,
    )
    assert summary["n_trajectories"] == 4
    assert summary["n_sim_failures"] == 4
    assert summary["n_scored_trajectories"] == 0
    assert summary["mean_pass_rate"] == 0.0        # not scored as solver failures
    rj = summary["rejection_sampling"]
    assert rj["sim_failure_rate"] == 1.0
    assert rj["reject_reason_counts"].get("def", 0) == 4 * 4   # 4 tries x 4 trajectories
    for traj in dump["trajectories"]:
        assert traj["sim_failed"] is True
        assert traj["sim_reject_events"][-1]["accepted"] is False
        # No GT ever reaches a solver-visible message, even on the failed path.
        for m in traj["messages"]:
            assert GT not in str(m["content"])


def test_no_answer_scores_zero_and_no_gt_leak(tmp_path):
    # Solver never submits; its FINAL turn is short/non-code so the last-turn fallback in
    # templates.final_answer does not accept it as an answer -> reward 0, not answered.
    summary, dump = _run(
        tmp_path, n_samples=1, max_saved_convos=100,
        scripts=["What is the cutoff?", "And the upper behavior?", "ok?"],
    )
    assert summary["mean_pass_rate"] == 0.0
    assert summary["answered_rate"] == 0.0
    # The hidden GT must never appear in any saved (solver-visible) message.
    for traj in dump["trajectories"]:
        for m in traj["messages"]:
            assert GT not in str(m["content"])

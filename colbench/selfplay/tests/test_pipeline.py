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
"""CPU tests for the Phase-0 spec pipeline: dataio, generate, diagnose.

Uses a STUBBED ChatEndpoint (no server / no openai SDK) and the in-process exec fallback for
grading (no sidecar). Run:
    CODECONTEST_ALLOW_INPROCESS=1 pytest colbench/selfplay/tests/test_pipeline.py
"""

import os
import tempfile

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

from colbench.selfplay import diagnose_specs, generate_specs  # noqa: E402
from colbench.selfplay.dataio import (  # noqa: E402
    _resolve_gt, append_jsonl, existing_indices, read_jsonl,
)
from colbench.selfplay.llm_client import ChatEndpoint  # noqa: E402

# Same GT/branch as test_reward so the in-process grader is exercised end-to-end.
GT = "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n        return x - y\n"
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Create def f(x, y). The signature is def f(x, y)."

CORRECT_CODE = "```python\ndef f(x, y):\n    return x + y if x >= 10 else x - y\n```"
WRONG_CODE = "```python\ndef f(x, y):\n    return 0\n```"
SPEC_JSON = ('{"persona": "an analyst", "scenario": "s", '
             '"requirements": "if x>=10 add else subtract"}')


def _endpoint(reply):
    return ChatEndpoint(base_url="x", model="y", backend=lambda msgs: reply)


def test_openai_vendor_drops_vendor_extras_and_uses_completion_tokens():
    ep = ChatEndpoint(base_url="https://api.openai.com/v1", model="gpt-x", vendor="openai")
    assert ep._extra_body() == {}                       # no top_k/min_p to the public API
    psets = ep._param_sets([{"role": "user", "content": "hi"}])
    assert all("max_completion_tokens" in p and "max_tokens" not in p for p in psets)
    assert "extra_body" not in psets[0]
    # progressively-minimal: full sampling -> temperature-only -> bare
    assert "top_p" in psets[0] and "top_p" not in psets[1] and "temperature" not in psets[2]


def test_vllm_vendor_sends_extra_body():
    ep = ChatEndpoint(base_url="http://127.0.0.1:30000/v1", model="specgen")  # default vendor=vllm
    assert ep._extra_body() == {"top_k": 20, "min_p": 0.0}
    p = ep._param_sets([{"role": "user", "content": "hi"}])[0]
    assert p["extra_body"] == {"top_k": 20, "min_p": 0.0} and "max_tokens" in p


# ── dataio._resolve_gt ────────────────────────────────────────────────────────
def test_resolve_gt_preprocessed_schema():
    row = {"extra_info": {"ground_truth": {"problem_description": PROBLEM, "ground_truth": GT,
                                           "test_cases": CALLS}}, "reward_model": {}}
    t = _resolve_gt(row)
    assert t["ground_truth"] == GT and t["test_cases"] == CALLS and t["problem_description"] == PROBLEM


def test_resolve_gt_raw_infopo_schema():
    row = {
        "reward_model": {"problem_description": PROBLEM, "ground_truth": GT},
        "extra_info": {"tools_kwargs": {"interact_with_env": {"create_kwargs": {"task": {
            "test_cases": {"c0": "f(1, 2)", "c1": "f(20, 5)", "pad": None}}}}}},
    }
    t = _resolve_gt(row)
    assert t["ground_truth"] == GT
    assert t["test_cases"] == ["f(1, 2)", "f(20, 5)"]  # None padding dropped


# ── generate_specs ────────────────────────────────────────────────────────────
def test_author_one_builds_record():
    task = {"index": 3, "problem_description": PROBLEM, "ground_truth": GT}
    rec = generate_specs._author_one(_endpoint(SPEC_JSON), task, "selfplay")
    assert rec["index"] == 3 and rec["backend"] == "selfplay" and rec["ok"] is True
    assert rec["requirements"] == "if x>=10 add else subtract"


def test_generate_is_resumable():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "specs.jsonl")
        append_jsonl(p, [{"index": 0, "backend": "selfplay"}, {"index": 2, "backend": "selfplay"}])
        assert existing_indices(p) == {0, 2}
        assert len(read_jsonl(p)) == 2


# ── diagnose_specs ────────────────────────────────────────────────────────────
def _write_specs(d, name="specs.jsonl", backend="selfplay", indices=(0,)):
    p = os.path.join(d, name)
    append_jsonl(p, [{"index": i, "backend": backend, "persona": "an analyst",
                      "scenario": "s", "requirements": "r"} for i in indices])
    return p


def test_diagnose_correct_solver_scores_full():
    tasks_by_index = {0: {"index": 0, "problem_description": PROBLEM, "ground_truth": GT, "test_cases": CALLS}}
    with tempfile.TemporaryDirectory() as d:
        p = _write_specs(d)
        r = diagnose_specs.diagnose_specs_file(p, tasks_by_index, _endpoint(CORRECT_CODE),
                                               n_samples=1, reward_time_limit=6.0, concurrency=2)
    assert r["label"] == "selfplay"
    assert r["metrics"]["solve_rate"] == 1.0
    assert r["metrics"]["mean_pass_rate"] == 1.0


def test_diagnose_wrong_solver_scores_zero():
    tasks_by_index = {0: {"index": 0, "problem_description": PROBLEM, "ground_truth": GT, "test_cases": CALLS}}
    with tempfile.TemporaryDirectory() as d:
        p = _write_specs(d)
        r = diagnose_specs.diagnose_specs_file(p, tasks_by_index, _endpoint(WRONG_CODE),
                                               n_samples=1, reward_time_limit=6.0, concurrency=2)
    assert r["metrics"]["solve_rate"] == 0.0


def test_diagnose_skips_specs_without_matching_task():
    # Spec index 5 has no task -> skipped, no crash, zero rows.
    with tempfile.TemporaryDirectory() as d:
        p = _write_specs(d, indices=(5,))
        r = diagnose_specs.diagnose_specs_file(p, {}, _endpoint(CORRECT_CODE),
                                               n_samples=1, reward_time_limit=6.0, concurrency=2)
    assert r["metrics"]["n_tasks"] == 0 and r["metrics"]["solve_rate"] == 0.0


PLOT_SPEC_JSON = ('{"persona": "an analyst", "scenario": "s", '
                  '"requirements": "if x>=10 add else subtract", '
                  '"plot": "The user forgets the below-10 subtract branch until the assistant asks."}')


def test_author_one_plot_builds_record():
    task = {"index": 4, "problem_description": PROBLEM, "ground_truth": GT}
    rec = generate_specs._author_one(_endpoint(PLOT_SPEC_JSON), task, "selfplay", mode="plot")
    assert rec["index"] == 4 and rec["mode"] == "plot" and rec["ok"] is True
    assert rec["requirements"] == "if x>=10 add else subtract"
    assert "forgets" in rec["plot"]


def _write_plot(d, name="plot.jsonl", indices=(0,)):
    p = os.path.join(d, name)
    append_jsonl(p, [{"index": i, "backend": "selfplay", "mode": "plot", "persona": "an analyst",
                      "scenario": "s", "requirements": "if x>=10 add else subtract",
                      "plot": "forgets the subtract branch at first"} for i in indices])
    return p


def test_diagnose_plot_specs_grade_on_requirements():
    tasks_by_index = {0: {"index": 0, "problem_description": PROBLEM, "ground_truth": GT, "test_cases": CALLS}}
    with tempfile.TemporaryDirectory() as d:
        p = _write_plot(d)
        r = diagnose_specs.diagnose_specs_file(p, tasks_by_index, _endpoint(CORRECT_CODE),
                                               n_samples=1, reward_time_limit=6.0, concurrency=2)
    assert r["label"] == "selfplay"
    assert r["metrics"]["solve_rate"] == 1.0  # graded off requirements, plot ignored


def test_diagnose_skips_specs_missing_requirements():
    tasks_by_index = {0: {"index": 0, "problem_description": PROBLEM, "ground_truth": GT, "test_cases": CALLS}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "noreq.jsonl")
        append_jsonl(p, [{"index": 0, "backend": "selfplay", "mode": "plot", "plot": "x"}])  # no requirements
        r = diagnose_specs.diagnose_specs_file(p, tasks_by_index, _endpoint(CORRECT_CODE),
                                               n_samples=1, reward_time_limit=6.0, concurrency=2)
    assert r["n_missing_field"] == 1 and r["metrics"]["n_tasks"] == 0


def test_aggregate_pass_at_n_counts_task_level():
    rows = [
        {"index": 0, "all_pass": True, "pass_rate": 1.0},
        {"index": 0, "all_pass": False, "pass_rate": 0.5},
        {"index": 1, "all_pass": False, "pass_rate": 0.0},
        {"index": 1, "all_pass": False, "pass_rate": 0.0},
    ]
    m = diagnose_specs._aggregate(rows, n_samples=2)
    assert m["n_tasks"] == 2
    assert m["solve_rate"] == 0.25          # 1 of 4 samples fully correct
    assert m["pass_at_n"] == 0.5            # task 0 has a correct sample, task 1 does not
    assert abs(m["mean_pass_rate"] - 0.375) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all pipeline tests passed")

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
"""CPU test for colbench.preprocess_colbench schema mapping (no GPU).

Applies the row map to a tiny slice of InfoPO's source parquet and asserts the emitted VERL
schema + that the None-padded test_cases are filtered to non-empty call-strings.
"""

import os

import pytest

from colbench import preprocess_colbench as pp

# Source parquet lives in the InfoPO data dir at the repo root (one level above verl/).
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "InfoPO", "data", "colbench_code", "train.parquet")


def _load_rows(n=3):
    pd = pytest.importorskip("pandas")
    if not os.path.exists(_SRC):
        pytest.skip(f"source parquet not found at {_SRC}")
    df = pd.read_parquet(_SRC).head(n)
    return df.to_dict("records")


def test_map_fn_emits_expected_schema():
    rows = _load_rows()
    fn = pp.make_map_fn("train")
    for idx, row in enumerate(rows):
        out = fn(row, idx)

        # prompt: [system(agent prompt), user(problem)]
        assert [m["role"] for m in out["prompt"]] == ["system", "user"]
        assert "clarification" in out["prompt"][0]["content"].lower()
        assert out["prompt"][1]["content"]  # non-empty problem

        # ground_truth mirrored in reward_model + extra_info
        gt = out["reward_model"]["ground_truth"]
        assert set(gt.keys()) == {"problem_description", "ground_truth", "test_cases"}
        assert out["extra_info"]["ground_truth"] == gt
        assert out["extra_info"]["agent_name"] == "colbench_agent"

        # GT function source + non-empty, None-filtered call-strings
        assert "def " in gt["ground_truth"]
        assert isinstance(gt["test_cases"], list)
        assert len(gt["test_cases"]) > 0
        assert all(isinstance(c, str) and c for c in gt["test_cases"])
        assert all(c is not None for c in gt["test_cases"])


def test_extract_test_cases_filters_none():
    extra_info = {
        "tools_kwargs": {
            "interact_with_env": {
                "create_kwargs": {
                    "task": {"test_cases": {"test1": "f(1)", "pad": None, "test2": "f(2)"}}
                }
            }
        }
    }
    assert pp._extract_test_cases(extra_info) == ["f(1)", "f(2)"]


def test_extract_test_cases_handles_missing():
    assert pp._extract_test_cases({}) == []


if __name__ == "__main__":
    test_extract_test_cases_filters_none()
    test_extract_test_cases_handles_missing()
    print("PASS extract test_cases")

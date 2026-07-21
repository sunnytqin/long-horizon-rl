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
"""Shared row reading + JSONL cache helpers for the Phase-0 spec scripts.

``read_tasks`` normalizes a ColBench parquet row (either the raw InfoPO source schema or our
preprocessed schema) into ``{index, problem_description, ground_truth, test_cases}`` -- the
minimal task payload both ``generate_specs`` and ``diagnose_specs`` need. Test-case extraction
for the raw schema mirrors ``colbench.preprocess_colbench._extract_test_cases`` (inlined below
so this Phase-0 subpackage stays free of that module's absl CLI dependency).
"""

import json
import os
from typing import Iterator, Optional

import pandas as pd


def _extract_test_cases(extra_info: dict) -> list:
    """Non-None call-strings from the raw InfoPO nested tools_kwargs payload.

    Kept byte-identical to ``colbench.preprocess_colbench._extract_test_cases``; duplicated
    here only to avoid importing that module (its top-level absl import is CLI-only).
    """
    tools_kwargs = (extra_info or {}).get("tools_kwargs", {}) or {}
    create_kwargs = (tools_kwargs.get("interact_with_env", {}) or {}).get("create_kwargs", {}) or {}
    task = (create_kwargs.get("task", {}) or {})
    test_cases = task.get("test_cases", {}) or {}
    return [str(v) for v in test_cases.values() if v is not None]


def _resolve_gt(row) -> dict:
    """Return the task ground_truth dict from either schema.

    Preprocessed rows carry a ready dict at ``extra_info.ground_truth`` /
    ``reward_model.ground_truth`` (with ``problem_description``, ``ground_truth`` source, and
    ``test_cases``). Raw InfoPO rows carry ``reward_model.{problem_description, ground_truth}``
    and nest test_cases under ``extra_info.tools_kwargs`` -- extracted here.
    """
    extra_info = row.get("extra_info", {}) or {}
    rm = row.get("reward_model", {}) or {}
    gt = extra_info.get("ground_truth")
    if gt is None:
        gt = rm.get("ground_truth")
    # Preprocessed schema: gt is a dict with all three fields.
    if isinstance(gt, dict):
        _tc = gt.get("test_cases")
        return {
            "problem_description": gt.get("problem_description", rm.get("problem_description", "")),
            "ground_truth": gt.get("ground_truth", ""),
            "test_cases": list(_tc) if _tc is not None else [],
        }
    # Raw InfoPO schema: gt is the GT source string; test_cases nested in extra_info.
    return {
        "problem_description": rm.get("problem_description", ""),
        "ground_truth": gt if isinstance(gt, str) else rm.get("ground_truth", ""),
        "test_cases": _extract_test_cases(extra_info),
    }


def read_tasks(data_file: str, max_rows: Optional[int] = None) -> list[dict]:
    """Load tasks from a parquet, normalized to the minimal payload. Index = row position."""
    df = pd.read_parquet(os.path.expanduser(data_file))
    if max_rows is not None:
        df = df.iloc[:max_rows]
    tasks = []
    for i, (_, row) in enumerate(df.iterrows()):
        t = _resolve_gt(row)
        t["index"] = i
        tasks.append(t)
    return tasks


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts (empty list if the file does not exist)."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def existing_indices(path: str) -> set:
    """Indices already present in a JSONL cache (for resumable generation)."""
    return {r["index"] for r in read_jsonl(path) if "index" in r}


def append_jsonl(path: str, records: Iterator[dict]) -> None:
    """Append records to a JSONL file, creating parent dirs as needed."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

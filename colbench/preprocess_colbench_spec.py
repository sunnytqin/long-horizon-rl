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
"""Preprocess the SPEC-path ColBench dataset into our VERL RL schema (Phase 1).

Sibling of ``colbench.preprocess_colbench`` for the spec setting. Joins the Phase-0 authored
specs (``.../colbench_specs/specs/train.selfplay.plot.jsonl``; ``index`` = row position in the
raw InfoPO parquet) back to the raw parquet to recover the GT code + ``test_cases`` for grading,
and attaches the spec to each row's ``extra_info`` so the spec sim can condition on it.

Each output row carries:
  - ``reward_model.ground_truth`` and ``extra_info.ground_truth`` -- {problem, GT source,
    test_cases}, the UNCHANGED grading payload (same as the GT preprocess), and
  - ``extra_info.spec = {persona, scenario, requirements, plot}`` -- read by the spec agent loop /
    spec sim env. The GT code is NEVER placed in ``spec``.
  - ``extra_info.agent_name = "colbench_spec_agent"`` and the spec solver system prompt
    (``COLBENCH_SPEC_AGENT_SYSTEM_PROMPT``, no "I WANT TO ANSWER:" marker).

Only rows that have a usable spec (parsed ``ok`` with non-empty requirements + plot) are emitted;
this is the ~1k spec set that is the eval set for bringing Phase 1 up (train/val split for RL is
deferred). Reuses ``selfplay.dataio`` for the raw-parquet load + GT resolution + jsonl reader.

Usage:
    python colbench/preprocess_colbench_spec.py \
        --raw_parquet InfoPO/data/colbench_code/train.parquet \
        --specs_jsonl /n/netscratch/dam_lab/Lab/sqin/colbench_specs/specs/train.selfplay.plot.jsonl \
        --out ~/data/colbench_spec/train.parquet
    # quick 30-row subset:
    python colbench/preprocess_colbench_spec.py --raw_parquet ... \
        --specs_jsonl .../train.selfplay.plot.cond30.jsonl --out /tmp/colbench_spec/cond30.parquet
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets

from colbench.selfplay import dataio
from colbench.templates import COLBENCH_SPEC_AGENT_SYSTEM_PROMPT, build_initial_user_message

DATA_SOURCE = "colbench_spec_local"  # routes nowhere special; reward comes from the loop

_SPEC_KEYS = ("persona", "scenario", "requirements", "plot")


def _usable(spec_row: dict) -> bool:
    """A spec row is usable iff it parsed ok and has non-empty requirements + plot.

    Guards against the ~3/1000 rows that failed to parse (empty spec) and any row missing the two
    fields the sim actually needs to behave (the substance + the arc).
    """
    if not spec_row.get("ok", True):
        return False
    return bool(str(spec_row.get("requirements", "")).strip()) and bool(str(spec_row.get("plot", "")).strip())


def build_rows(raw_parquet: str, specs_jsonl: str, split: str) -> list[dict]:
    """Join specs (by ``index``) to the raw parquet's resolved GT and build VERL-schema rows."""
    tasks = dataio.read_tasks(raw_parquet)  # tasks[i]["index"] == i (raw row position)
    specs = dataio.read_jsonl(specs_jsonl)
    rows, skipped_unusable, skipped_oob = [], 0, 0
    for sp in specs:
        idx = sp.get("index")
        if idx is None or not (0 <= idx < len(tasks)):
            skipped_oob += 1
            continue
        if not _usable(sp):
            skipped_unusable += 1
            continue
        gt = tasks[idx]  # {problem_description, ground_truth, test_cases}
        problem_description = gt["problem_description"]
        ground_truth = {
            "problem_description": problem_description,
            "ground_truth": gt["ground_truth"],
            "test_cases": gt["test_cases"],
        }
        spec = {k: sp.get(k) for k in _SPEC_KEYS}
        rows.append({
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": COLBENCH_SPEC_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": build_initial_user_message(problem_description)},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": int(idx),
                "agent_name": "colbench_spec_agent",
                "ground_truth": ground_truth,
                "spec": spec,
            },
        })
    print(f"[preprocess_spec] {len(rows)} usable rows "
          f"(skipped {skipped_unusable} unusable, {skipped_oob} out-of-range) from {len(specs)} specs")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_parquet", default="InfoPO/data/colbench_code/train.parquet",
                    help="Raw InfoPO parquet the specs' `index` points into (GT source + test_cases).")
    ap.add_argument("--specs_jsonl", required=True,
                    help="Phase-0 specs JSONL (e.g. train.selfplay.plot.jsonl or the cond30 subset).")
    ap.add_argument("--out", required=True, help="Output parquet path (VERL schema).")
    ap.add_argument("--split", default="train", help="Value stored in extra_info.split (metadata).")
    args = ap.parse_args()

    rows = build_rows(os.path.expanduser(args.raw_parquet), os.path.expanduser(args.specs_jsonl), args.split)
    if not rows:
        raise SystemExit("[preprocess_spec] no usable rows -- check --specs_jsonl / --raw_parquet.")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ds = datasets.Dataset.from_list(rows)
    ds.to_parquet(out)
    print(f"[preprocess_spec] wrote {len(ds)} rows -> {out}")

    ex = ds[0]
    print("Example row:")
    print("  data_source:", ex["data_source"])
    print("  prompt[0].role:", ex["prompt"][0]["role"])
    print("  extra_info.agent_name:", ex["extra_info"]["agent_name"])
    print("  extra_info.spec keys:", list(ex["extra_info"]["spec"].keys()))
    print("  reward_model.ground_truth keys:", list(ex["reward_model"]["ground_truth"].keys()))
    print("  #test cases:", len(ex["extra_info"]["ground_truth"]["test_cases"]))


if __name__ == "__main__":
    main()

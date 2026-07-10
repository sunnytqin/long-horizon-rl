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
"""Preprocess ColBench (Sweet-RL Backend-Programming) parquet into our VERL RL schema.

Source: InfoPO's ``data/colbench_code/{train,test}.parquet`` (10k train rows). Each source
row carries:
  - ``reward_model.{problem_description, ground_truth}`` (GT function source), and
  - ``extra_info.tools_kwargs.interact_with_env.create_kwargs.task.test_cases`` -- a dict of
    ``label -> call-string`` where MANY values are ``None`` (parquet schema-padding); we keep
    only the non-None call-strings.

Each output row carries the task payload in BOTH:
  - ``reward_model.ground_truth`` (standard VERL reward channel), and
  - ``extra_info.ground_truth``   (read by ``colbench.colbench_agent`` at rollout),
so the same {problem, GT source, call-strings} drives the simulator prompt and the final
fractional pass-rate reward.

Usage:
    python colbench/preprocess_colbench.py \
        --src_dir InfoPO/data/colbench_code --local_dir ~/data/colbench
    # quick smoke slice:
    python colbench/preprocess_colbench.py --src_dir InfoPO/data/colbench_code \
        --local_dir /tmp/colbench --max_train 20 --max_val 20
"""

import argparse
import os

import datasets

from colbench.templates import COLBENCH_AGENT_SYSTEM_PROMPT, build_initial_user_message

DATA_SOURCE = "colbench_code_local"  # routes nowhere special; reward comes from the loop


def _extract_test_cases(extra_info: dict) -> list:
    """Pull the non-None call-strings out of the nested tools_kwargs task payload."""
    tools_kwargs = (extra_info or {}).get("tools_kwargs", {}) or {}
    create_kwargs = (tools_kwargs.get("interact_with_env", {}) or {}).get("create_kwargs", {}) or {}
    task = (create_kwargs.get("task", {}) or {})
    test_cases = task.get("test_cases", {}) or {}
    # Keep only non-None values (parquet pads the dict with None keys for schema consistency).
    return [str(v) for v in test_cases.values() if v is not None]


def make_map_fn(split: str):
    def process_fn(example, idx):
        reward_model = example["reward_model"] or {}
        problem_description = reward_model.get("problem_description", "")
        ground_truth_src = reward_model.get("ground_truth", "")
        test_cases = _extract_test_cases(example.get("extra_info", {}) or {})

        ground_truth = {
            "problem_description": problem_description,
            "ground_truth": ground_truth_src,
            "test_cases": test_cases,
        }
        return {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": COLBENCH_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": build_initial_user_message(problem_description)},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "agent_name": "colbench_agent",
                "ground_truth": ground_truth,
            },
        }

    return process_fn


def build_split(src_path: str, out_split: str, max_rows):
    ds = datasets.load_dataset("parquet", data_files=src_path, split="train")
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds.map(make_map_fn(out_split), with_indices=True, remove_columns=ds.column_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default="InfoPO/data/colbench_code",
                   help="dir holding the source train.parquet / test.parquet")
    p.add_argument("--local_dir", default=os.path.expanduser("~/data/colbench"))
    p.add_argument("--max_train", type=int, default=None, help="limit train rows (debug)")
    p.add_argument("--max_val", type=int, default=None, help="limit val rows (debug)")
    args = p.parse_args()

    src_dir = os.path.expanduser(args.src_dir)
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Loading train: {src_dir}/train.parquet")
    train = build_split(os.path.join(src_dir, "train.parquet"), "train", args.max_train)
    print(f"Loading val: {src_dir}/test.parquet")
    val = build_split(os.path.join(src_dir, "test.parquet"), "test", args.max_val)

    train_path = os.path.join(local_dir, "train.parquet")
    val_path = os.path.join(local_dir, "test.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    print(f"Wrote {len(train)} train rows -> {train_path}")
    print(f"Wrote {len(val)} val rows   -> {val_path}")
    print("Example row:")
    ex = train[0]
    print("  prompt[0].role:", ex["prompt"][0]["role"])
    print("  data_source:", ex["data_source"])
    print("  reward_model.ground_truth keys:", list(ex["reward_model"]["ground_truth"].keys()))
    print("  extra_info.agent_name:", ex["extra_info"]["agent_name"])
    print("  #test cases:", len(ex["extra_info"]["ground_truth"]["test_cases"]))


if __name__ == "__main__":
    main()

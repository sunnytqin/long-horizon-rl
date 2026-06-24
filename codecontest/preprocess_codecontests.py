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
"""Preprocess Gen-Verse CodeContests into VERL multi-turn RL parquet.

Train: Gen-Verse/CodeContests_train     Val/Test: Gen-Verse/CodeContests
Both expose: question, task_id, test_input[list], test_output[list],
             test_time_limit, example_input/output, exe_method, solutions.

Each output row carries the ground-truth tests in BOTH:
  - reward_model.ground_truth  (standard VERL reward channel), and
  - extra_info.ground_truth    (read by codecontest.code_refine_agent at rollout)
so the same GT set drives mid-turn oracle feedback and the final binary reward.

Usage:
    python codecontest/preprocess_codecontests.py --local_dir ~/data/codecontests
    # quick smoke slice:
    python codecontest/preprocess_codecontests.py --local_dir /tmp/cc --max_train 20 --max_val 20
"""

import argparse
import os

import datasets

from codecontest.templates import CODE_PROMPT_TEMPLATE, SOLVER_SYSTEM_PROMPT

TRAIN_REPO = "Gen-Verse/CodeContests_train"
VAL_REPO = "Gen-Verse/CodeContests"
DATA_SOURCE = "codecontests_local"  # routes nowhere special; reward comes from the loop


def make_map_fn(split: str):
    def process_fn(example, idx):
        question = example["question"]
        # Ground-truth stdin/stdout tests (lists), plus the per-problem time limit.
        ground_truth = {
            "test_input": list(example["test_input"] or []),
            "test_output": list(example["test_output"] or []),
            "test_time_limit": float(example.get("test_time_limit") or 2.0),
        }
        return {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": CODE_PROMPT_TEMPLATE.format(problem=question)},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "task_id": int(example["task_id"]),
                "agent_name": "code_refine_agent",
                "ground_truth": ground_truth,
            },
        }

    return process_fn


def build_split(repo: str, hf_split: str, out_split: str, max_rows):
    ds = datasets.load_dataset(repo, split=hf_split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    # Drop original columns so the parquet only holds the VERL schema.
    return ds.map(make_map_fn(out_split), with_indices=True, remove_columns=ds.column_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", default=os.path.expanduser("~/data/codecontests"))
    p.add_argument("--max_train", type=int, default=None, help="limit train rows (debug)")
    p.add_argument("--max_val", type=int, default=None, help="limit val rows (debug)")
    args = p.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Loading train: {TRAIN_REPO}")
    train = build_split(TRAIN_REPO, "train", "train", args.max_train)
    print(f"Loading val: {VAL_REPO}")
    val = build_split(VAL_REPO, "test", "test", args.max_val)

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
    print("  #gt tests:", len(ex["extra_info"]["ground_truth"]["test_input"]))


if __name__ == "__main__":
    main()

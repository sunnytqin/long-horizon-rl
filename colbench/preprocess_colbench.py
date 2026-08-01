r"""Preprocess the ColBench parquet into our VERL RL schema.

ColBench here is Sweet-RL Backend-Programming.

Source: InfoPO's ``data/colbench_code/{train,test}.parquet`` (10k train rows).
        Each source row carries:
 - ``reward_model.{problem_description, ground_truth}`` (GT function source),
   and
 - ``extra_info.tools_kwargs.interact_with_env.create_kwargs.task.test_cases``
   -- a dict of ``label -> call-string`` where MANY values are ``None`` (parquet
   schema-padding); we keep only the non-None call-strings.


Each output row carries the task payload in BOTH:
 - ``reward_model.ground_truth`` (standard VERL reward channel), and
 - ``extra_info.ground_truth`` (read by ``colbench.colbench_agent`` at
   rollout),
so the same {problem, GT source, call-strings} drives the simulator prompt and
the final fractional pass-rate reward.


Usage:
   python colbench/preprocess_colbench.py \
       --src_dir InfoPO/data/colbench_code --local_dir ~/data/colbench
   # quick smoke slice:
   python colbench/preprocess_colbench.py --src_dir InfoPO/data/colbench_code \
       --local_dir /tmp/colbench --max_train 20 --max_val 20
"""

import os
import sys
from typing import Any

import datasets
import pandas as pd
from absl import app
from absl import flags

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The module-level setup above (env vars, sys.path) has to run
# before these imports resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench.templates import COLBENCH_AGENT_SYSTEM_PROMPT
from colbench.templates import build_initial_user_message

DATA_SOURCE = (
    "colbench_code_local"  # routes nowhere special; reward comes from the loop
)


def _extract_test_cases(extra_info: dict[str, Any]) -> list[str]:
  """Pull non-None call-strings from the nested tools_kwargs payload."""
  tools_kwargs = (extra_info or {}).get("tools_kwargs", {}) or {}
  create_kwargs = (tools_kwargs.get("interact_with_env", {}) or {}).get(
      "create_kwargs", {}
  ) or {}
  task = create_kwargs.get("task", {}) or {}
  test_cases = task.get("test_cases", {}) or {}
  # Keep only non-None values (parquet pads the dict with None keys for schema
  # consistency).
  return [str(v) for v in test_cases.values() if v is not None]


def make_map_fn(split: str):
  """Build the ``datasets.map`` function for one split.

  Args:
    split: value stored in ``extra_info.split``.

  Returns:
    A ``(example, idx) -> row`` callable in the VERL schema.
  """

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
            {
                "role": "user",
                "content": build_initial_user_message(problem_description),
            },
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
  df = pd.read_parquet(src_path)
  if max_rows is not None:
    df = df.iloc[:max_rows]
  ds = datasets.Dataset.from_pandas(df)
  return ds.map(
      make_map_fn(out_split), with_indices=True, remove_columns=ds.column_names
  )


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "src_dir",
    "InfoPO/data/colbench_code",
    "dir holding the source train.parquet / test.parquet",
)
flags.DEFINE_string(
    "local_dir", os.path.expanduser("~/data/colbench"), "local output dir"
)
flags.DEFINE_integer("max_train", None, "limit train rows (debug)")
flags.DEFINE_integer("max_val", None, "limit val rows (debug)")
flags.DEFINE_integer("val_small", 2000, "limit small val size")


def main(argv):
  del argv
  src_dir = os.path.expanduser(FLAGS.src_dir)
  local_dir = os.path.expanduser(FLAGS.local_dir)
  os.makedirs(local_dir, exist_ok=True)

  print(f"Loading train: {src_dir}/train.parquet")
  train = build_split(
      os.path.join(src_dir, "train.parquet"), "train", FLAGS.max_train
  )
  print(f"Loading val: {src_dir}/test.parquet")
  val = build_split(
      os.path.join(src_dir, "test.parquet"), "test", FLAGS.max_val
  )

  train_path = os.path.join(local_dir, "train.parquet")
  val_path = os.path.join(local_dir, "test.parquet")
  train.to_parquet(train_path)
  val.to_parquet(val_path)
  print(f"Wrote {len(train)} train rows -> {train_path}")
  print(f"Wrote {len(val)} val rows   -> {val_path}")

  # Light in-training validation set: a deterministic first-N slice of the full
  # val set. The full test.parquet is reserved for the offline eval loop
  # (colbench/validate_colbench.py).
  if 0 < FLAGS.val_small < len(val):
    val_small = val.select(range(FLAGS.val_small))
    val_small_path = os.path.join(local_dir, "test_small.parquet")
    val_small.to_parquet(val_small_path)
    print(
        (
            f"Wrote {len(val_small)} val-small rows -> {val_small_path}"
            f" (in-training val)"
        )
    )
  else:
    print(
        (
            f"Skipped test_small.parquet (val_small={FLAGS.val_small}, full "
            f"val={len(val)})"
        )
    )
  print("Example row:")
  ex = train[0]
  print("  prompt[0].role:", ex["prompt"][0]["role"])
  print("  data_source:", ex["data_source"])
  print(
      "  reward_model.ground_truth keys:",
      list(ex["reward_model"]["ground_truth"].keys()),
  )
  print("  extra_info.agent_name:", ex["extra_info"]["agent_name"])
  print("  #test cases:", len(ex["extra_info"]["ground_truth"]["test_cases"]))


if __name__ == "__main__":
  app.run(main)

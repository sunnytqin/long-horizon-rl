# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Pre-scan an SFT parquet through the REAL dataset class before launching.

This is the only check that catches "trained on the wrong tokens". Everything
else -- row counts, the composition report, a manual read of the targets -- is
consistent with a loss mask that covers the prompt, or the system message, or
nothing at all. Those failures do not crash and do not look wrong in the loss
curve; they just train something other than what was intended.

So: build the actual ``MultiTurnSFTDataset`` with the actual tokenizer, take the
tokens the mask selects, decode them, and require that they are the assistant
content. Also surfaces the length distribution IN TOKENS, which is what decides
``max_length`` -- the config default of 1024 raises on row 1 of this dataset.

CPU only, no GPU, ~20 s for a few thousand rows.

Example:
    python -m colbench.simtrain.prescan_sft \\
        --parquet $SIMTRAIN_ROOT/base_partner/sft/sim_sft_train.parquet \\
        --model   $MODEL_ROOT/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<sha>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# pylint: disable=g-import-not-at-top,wrong-import-position
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the pre-scan.

  Returns:
    The parser.
  """
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--parquet", required=True)
  p.add_argument("--model", required=True, help="Tokenizer path or HF id.")
  p.add_argument("--max_length", type=int, default=16384)
  p.add_argument(
      "--limit",
      type=int,
      default=0,
      help="Check only the first N rows (0 = all). The mask contract is a "
      "property of the shape, so a few hundred rows prove it.",
  )
  return p


def main(argv: Optional[list[str]] = None) -> None:
  """Verify the loss mask and report token lengths.

  Args:
    argv: argument vector, defaulting to ``sys.argv[1:]``.

  Raises:
    SystemExit: the decoded masked tokens do not match the assistant content.
  """
  args = build_arg_parser().parse_args(argv)

  import numpy as np
  from omegaconf import OmegaConf
  import pandas as pd
  from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
  from verl.utils import hf_tokenizer

  df = pd.read_parquet(args.parquet, dtype_backend="pyarrow")
  print(f"[prescan] {args.parquet}: {len(df)} rows")
  tokenizer = hf_tokenizer(args.model)
  # `truncation=error` on purpose: a silently truncated row is a row whose
  # target is cut off, which trains the sim to stop mid-sentence.
  cfg = OmegaConf.create({
      "max_length": args.max_length,
      "truncation": "error",
      "messages_key": "messages",
      "pad_mode": "right",
  })
  ds = MultiTurnSFTDataset(args.parquet, tokenizer, cfg)

  n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
  lengths = []
  mismatches = []
  for i in range(n):
    item = ds[i]
    ids = np.asarray(item["input_ids"])
    mask = np.asarray(item["loss_mask"]).astype(bool)
    attn = np.asarray(item["attention_mask"]).astype(bool)
    lengths.append(int(attn.sum()))
    got = tokenizer.decode(ids[mask], skip_special_tokens=True).strip()
    want = df.iloc[i]["messages"][2]["content"].strip()
    # The mask must cover the assistant turn and nothing else. Compared after
    # collapsing whitespace: the chat template legitimately adds newlines
    # around the turn, and that is not a mask error.
    if " ".join(got.split()) != " ".join(want.split()):
      mismatches.append((i, want[:120], got[:120]))

  lengths.sort()
  print(f"[prescan] checked {n} rows")
  print(
      f"[prescan] TOKEN lengths p50={lengths[len(lengths) // 2]} "
      f"p95={lengths[int(len(lengths) * 0.95)]} max={lengths[-1]}  "
      f"(max_length={args.max_length})"
  )
  if lengths[-1] > args.max_length:
    print("[prescan] WARNING: a row exceeds max_length; truncation=error will raise")
  if mismatches:
    print(f"[prescan] FAIL: {len(mismatches)} rows whose masked tokens are not the target")
    for i, want, got in mismatches[:3]:
      print(f"    row {i}\n      want: {want!r}\n      got : {got!r}")
    raise SystemExit(
        "the loss mask does not select the assistant content -- training this "
        "parquet would optimise the wrong tokens"
    )
  print("[prescan] OK: every checked row trains exactly its assistant turn")


if __name__ == "__main__":
  main()

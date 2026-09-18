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
"""Stage 4: turn judged prefixes into the SFT parquet the verl trainer reads.

Pure JSONL -> parquet. No network, no GPU, no tokenizer: this is a selection
pass plus a reshape, and keeping it that way is what makes it unit-testable.

THE ROW IS A THREE-MESSAGE CONVERSATION and nothing more:

    [{"role": "system",    "content": <the sim's system prompt>},
     {"role": "user",      "content": <the MATERIALIZED sim user message>},
     {"role": "assistant", "content": <the judge-selected reply>}]

which is exactly ``MultiTurnSFTDataset``'s native shape (``messages_key`` is
``messages``, and it masks loss to assistant tokens). One row per prefix, and a
prefix is one simulator turn -- the turn-level formulation this whole pipeline
rests on, because ``templates.str_dialogue_history`` renders the entire dialogue
into ONE user message, so a ``(rendered prompt, reply)`` pair is the complete
state.

THE SYSTEM MESSAGE COMES OFF THE PREFIX RECORD (``--system collected``, the
default). That makes the SFT row byte-identical to the prompt that generated the
candidate AND to what the sim is served with, now that ``role_restraint`` is the
default on both the training path and the collector. Passing an explicit variant
instead turns this into a CONTEXT-DISTILLATION run -- draw under the teacher,
train against a different production prompt -- which is a deliberate experiment,
not a default, so it has to be asked for by name.

SPLITTING IS BY TASK, NEVER BY PREFIX. One task contributes several prefixes
(2.14 per episode against the trained partner, 1.29 against the base one) and
they share a ground truth, so a prefix-level split leaks the answer across it.
The real held-out evaluation is a separate FILE (``test_small.fence.parquet``);
the split made here exists only so the trainer's ``val/loss`` means something,
which is also why ``--val_min`` defaults above the SFT batch size: verl reports
``val/loss = nan`` whenever the val set is smaller than one batch.

Example:
    python -m colbench.simtrain.build_sft_parquet \\
        --prefixes $SIMTRAIN_ROOT/base_partner/prefixes.train.fence.c1.jsonl \\
        --judged   $SIMTRAIN_ROOT/base_partner/judged.train.fence.c1.r6.jsonl \\
        --out_dir  $SIMTRAIN_ROOT/base_partner/sft
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from typing import Any, Optional

# pylint: disable=g-import-not-at-top,wrong-import-position
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)

from colbench import prompts
from colbench import templates
from colbench.selfplay.dataio import read_jsonl
from colbench.simtrain import COLLECTOR_VERSION
from colbench.simtrain import JUDGE_HARNESS_VERSION
from colbench.simtrain import judge_rubric


def parse_min_dims(spec: Optional[str]) -> Optional[dict[str, int]]:
  """Parse ``'calibration=3,in_character=3'`` into a floors dict.

  Args:
    spec: the CLI string; ``None`` selects ``DEFAULT_MIN_DIMS``, and an empty
      string means ranking-only (no floors).

  Returns:
    The floors dict, or ``None`` for ranking-only.

  Raises:
    ValueError: a clause is malformed or names a non-integer floor.
  """
  if spec is None:
    return dict(judge_rubric.DEFAULT_MIN_DIMS)
  spec = spec.strip()
  if not spec:
    return None
  out: dict[str, int] = {}
  for clause in spec.split(","):
    if "=" not in clause:
      raise ValueError(f"--min_dims clause {clause!r} is not 'dim=N'")
    dim, _, val = clause.partition("=")
    try:
      out[dim.strip()] = int(val)
    except ValueError as e:
      raise ValueError(f"--min_dims clause {clause!r} has a non-integer floor") from e
  return out


def task_split(task_id: Any, val_frac: float, salt: str = "simtrain-v1") -> str:
  """Assign one TASK to train or val by a stable hash.

  Hash-based rather than index-based so the assignment does not move when the
  collected slice grows: re-running after collecting more tasks must not
  reshuffle which tasks were held out, or every val number becomes
  incomparable with the previous run.

  Args:
    task_id: the task identifier.
    val_frac: fraction of tasks to hold out, 0..1.
    salt: namespace for the hash.

  Returns:
    ``"train"`` or ``"val"``.
  """
  if val_frac <= 0:
    return "train"
  h = hashlib.sha256(f"{salt}:{task_id}".encode()).hexdigest()
  # 4 hex digits = 16 bits of resolution, plenty for a percentage split.
  return "val" if (int(h[:4], 16) / 0xFFFF) < val_frac else "train"


def build_rows(
    prefixes: dict[str, dict[str, Any]],
    judged: list[dict[str, Any]],
    min_dims: Optional[dict[str, int]],
    system_variant: str = "collected",
    min_total: int = 0,
    min_margin: int = 0,
    all_tied: bool = True,
    near_dup_ratio: float = 0.9,
    max_tokens: int = 0,
    tokenizer: Any = None,
) -> tuple[list[dict[str, Any]], collections.Counter]:
  """Select the target(s) per judged prefix and shape them into SFT rows.

  Args:
    prefixes: ``prefix_id`` -> collector record.
    judged: ``judge_candidates`` records.
    min_dims: per-dimension floors handed to ``judge_rubric.select``.
    system_variant: ``"collected"`` uses the prefix's recorded ``sim_system``;
      any other value is resolved through ``templates.resolve_sim_system`` and
      makes this a context-distillation build.
    min_total: total floor handed to ``select``.
    min_margin: margin floor handed to ``select``.
    all_tied: emit one row per candidate TIED at the group max instead of one
      row per prefix. See ``judge_rubric.select_all``: 2.2x the rows on the
      grounded pilot at no cost to prefix, task or turn coverage.
    near_dup_ratio: passed to ``select_all``; ignored when ``all_tied`` is off.
    max_tokens: drop a row whose chat-templated length exceeds this; 0 disables.
    tokenizer: required when ``max_tokens`` > 0.

  Returns:
    ``(rows, stats)``. ``stats`` counts keeps and every drop reason, plus
    ``missing_prefix`` for a judged row whose prefix is not in the prefix file.

  Raises:
    ValueError: two judged records share a ``prefix_id``, or a selected target
      is empty (both mean the upstream file is corrupt, and a silent skip here
      would be a silently smaller dataset).
  """
  stats: collections.Counter = collections.Counter()
  rows: list[dict[str, Any]] = []
  seen: set[str] = set()

  override = None
  if system_variant != "collected":
    override = templates.resolve_sim_system(system_variant)

  for rec in judged:
    pid = rec.get("prefix_id")
    if pid in seen:
      raise ValueError(f"duplicate prefix_id {pid!r} in the judged input")
    seen.add(pid)
    pre = prefixes.get(pid)
    if pre is None:
      stats["missing_prefix"] += 1
      continue

    if all_tied:
      idxs, reason = judge_rubric.select_all(
          rec,
          min_total=min_total,
          min_dim=0,
          min_margin=min_margin,
          min_dims=min_dims,
          near_dup_ratio=near_dup_ratio,
      )
    else:
      idx, reason = judge_rubric.select(
          rec,
          min_total=min_total,
          min_dim=0,
          min_margin=min_margin,
          min_dims=min_dims,
      )
      idxs = [] if idx is None else [idx]
    if not idxs:
      stats[reason] += 1
      continue

    # One drop reason per PREFIX but one KEPT per ROW, so the two counters stay
    # comparable with the single-target build: `prefixes_kept` is the figure to
    # compare against a `--no-all_tied` run.
    stats["prefixes_kept"] += 1
    system = override if override is not None else pre["sim_system"]
    for idx in idxs:
      target = rec["candidates"][idx]
      if not (target or "").strip():
        raise ValueError(f"{pid}: selected an empty target at index {idx}")
      verdict = rec["scores"][idx]
      msgs_for_row = [
          {"role": "system", "content": system},
          {"role": "user", "content": pre["sim_user"]},
          {"role": "assistant", "content": target},
      ]

      # verl's MultiTurnSFTDataset PADS every sequence to `max_length` and is
      # configured to RAISE (not truncate) above it, so one over-long row kills
      # the run while raising max_length taxes every batch. Dropping the tail is
      # cheaper than either: measured on sft_g3_third, 2 of 12,929 rows exceed
      # 4096 tokens and none exceed 6144, against a p99 of 2,053.
      if max_tokens:
        n_tok = len(
            tokenizer.apply_chat_template(msgs_for_row, tokenize=True)
        )
        if n_tok > max_tokens:
          stats["over_max_tokens"] += 1
          continue

      stats["KEPT"] += 1
      rows.append({
          "messages": msgs_for_row,
          # Provenance. Every one of these has been needed at least once to
          # answer "where did this row come from?" without re-deriving it.
          "prefix_id": pid,
          "task_id": str(pre["task_id"]),
          "task_index": int(pre["task_index"]),
          "turn_idx": int(pre["turn_idx"]),
          "episode_idx": int(pre.get("episode_idx", -1)),
          "cand_idx": int(idx),
          "total": int(verdict.get("total", 0)),
          "margin": int(rec.get("margin", 0)),
          "n_candidates": len(rec["candidates"]),
          "sim_system_variant": (
              system_variant
              if override is not None
              else pre.get("sim_system_variant", "default")
          ),
          "sim_user_sha16": pre.get("sim_user_sha16", ""),
          "rubric_version": rec.get("rubric_version", ""),
          "harness_version": rec.get("harness_version", ""),
          "collector_version": pre.get("collector_version", ""),
      })
  return rows, stats


def compose_report(rows: list[dict[str, Any]], prefixes) -> str:
  """Describe the built dataset in the terms that decide whether to train on it.

  Args:
    rows: the built SFT rows.
    prefixes: ``prefix_id`` -> collector record, for the ground truth.

  Returns:
    The report text.
  """
  if not rows:
    return "  (no rows)"
  by_turn = collections.Counter(r["turn_idx"] for r in rows)
  tasks = {r["task_id"] for r in rows}
  lens = sorted(len(r["messages"][2]["content"]) for r in rows)
  leaks = 0
  for r in rows:
    gt = prefixes[r["prefix_id"]]["ground_truth"]
    if templates.detect_code_leak(
        r["messages"][2]["content"], gt, ngram_n=0, expr_over_gt_names=True
    ):
      leaks += 1
  out = [
      f"  rows                    {len(rows)}",
      f"  distinct tasks          {len(tasks)}  ({len(rows) / len(tasks):.2f} rows/task)",
      f"  target chars            p50={lens[len(lens) // 2]} p95={lens[int(len(lens) * 0.95)]} max={lens[-1]}",
      f"  targets that leak code  {leaks} ({leaks / len(rows):.4f})   <-- MUST be 0",
      "  rows by turn_idx        "
      + " ".join(f"{t}:{n}" for t, n in sorted(by_turn.items())),
  ]
  return "\n".join(out)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the SFT-parquet build.

  Returns:
    The parser.
  """
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--prefixes", required=True)
  p.add_argument(
      "--judged",
      required=True,
      action="append",
      help="Judged JSONL. Repeatable, but every file must carry the SAME "
      "rubric_version unless --allow_mixed_rubrics is given.",
  )
  p.add_argument("--out_dir", required=True)
  p.add_argument("--train_name", default="sim_sft_train.parquet")
  p.add_argument("--val_name", default="sim_sft_val.parquet")
  p.add_argument(
      "--min_dims",
      default=None,
      help="Per-dimension floors as 'dim=N,dim=N'. Default: the pipeline "
      "policy (judge_rubric.DEFAULT_MIN_DIMS); '' for ranking-only.",
  )
  p.add_argument(
      "--all_tied",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Emit one row per candidate TIED at the group max instead of one "
      "row per prefix (default on). Measured on the grounded pilot: 168 rows "
      "instead of 76, same 76/85 prefixes, same 39/40 tasks, same 39/32/5 turn "
      "split, same 22 [TERMINATE] targets, mean score 10.25 -> 10.29/12. Pass "
      "--no-all_tied for the old one-target-per-prefix behaviour.",
  )
  p.add_argument(
      "--near_dup_ratio",
      type=float,
      default=0.9,
      help="With --all_tied, drop a tied candidate at least this similar to one "
      "already kept, so a prefix that produced many paraphrases does not "
      "silently get extra loss weight. 1.0 keeps exact duplicates.",
  )
  p.add_argument(
      "--max_tokens",
      type=int,
      default=0,
      help="Drop rows whose chat-templated length exceeds this. 0 (default) "
      "keeps the builder tokenizer-free. Set it to the SFT run's MAX_LENGTH: "
      "verl's MultiTurnSFTDataset RAISES above max_length rather than "
      "truncating, so one over-long row kills the run, and raising max_length "
      "instead taxes every batch because it pads to that length.",
  )
  p.add_argument(
      "--tokenizer",
      default="",
      help="Model path for --max_tokens. Required when --max_tokens > 0.",
  )
  p.add_argument("--min_total", type=int, default=0)
  p.add_argument("--min_margin", type=int, default=0)
  p.add_argument(
      "--system",
      default="collected",
      help="'collected' (default) reuses the prefix's recorded sim_system, so "
      "the row matches both the draw and what the sim is served with. Naming a "
      "variant instead (default/role/role_restraint) makes this a "
      "CONTEXT-DISTILLATION build and is checked against "
      "templates.GT_SIM_SYSTEM_PROMPTS.",
  )
  p.add_argument(
      "--val_frac",
      type=float,
      default=0.06,
      help="Fraction of TASKS held out for the trainer's val/loss. The real "
      "evaluation is a separate file (test_small.fence.parquet); this split "
      "only makes val/loss meaningful.",
  )
  p.add_argument(
      "--val_min",
      type=int,
      default=64,
      help="Fail if the val split is smaller than this. verl reports "
      "val/loss = nan whenever N_val < train_batch_size, which reads as a "
      "broken run rather than as a too-small split.",
  )
  p.add_argument("--allow_mixed_rubrics", action="store_true")
  return p


def main(argv: Optional[list[str]] = None) -> None:
  """Build the train/val SFT parquets and print the composition report.

  Args:
    argv: argument vector, defaulting to ``sys.argv[1:]``.

  Raises:
    SystemExit: on any refusal (mixed rubrics, an undersized val split, a
      target that still contains code).
  """
  args = build_arg_parser().parse_args(argv)
  import pandas as pd  # local: keeps the module importable without pandas

  prefixes = {r["prefix_id"]: r for r in read_jsonl(args.prefixes)}
  judged: list[dict[str, Any]] = []
  for path in args.judged:
    judged.extend(read_jsonl(path))
  if not judged:
    raise SystemExit(f"no judged rows in {args.judged}")

  versions = {r.get("rubric_version", "") for r in judged}
  if len(versions) > 1 and not args.allow_mixed_rubrics:
    raise SystemExit(
        f"judged inputs mix rubric versions {sorted(versions)}. Two rubrics "
        "score 'good' differently, so mixing them makes the dataset mean two "
        "things at once. Re-judge under one, or pass --allow_mixed_rubrics."
    )

  min_dims = parse_min_dims(args.min_dims)
  tok = None
  if args.max_tokens:
    if not args.tokenizer:
      raise SystemExit("--max_tokens needs --tokenizer <model path>")
    from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
  rows, stats = build_rows(
      prefixes,
      judged,
      min_dims,
      system_variant=args.system,
      min_total=args.min_total,
      min_margin=args.min_margin,
      all_tied=args.all_tied,
      near_dup_ratio=args.near_dup_ratio,
      max_tokens=args.max_tokens,
      tokenizer=tok,
  )
  n_judged = len(judged)
  print("=" * 78)
  print("SELECTION")
  print("=" * 78)
  print(f"  judged prefixes         {n_judged}")
  print(f"  rubric/harness          {sorted(versions)} / {JUDGE_HARNESS_VERSION}")
  print(f"  collector               {COLLECTOR_VERSION}")
  print(f"  min_dims                {min_dims}")
  print(f"  system                  {args.system}")
  kept = stats["KEPT"]
  n_pref = stats.get("prefixes_kept", kept)
  # ROWS and PREFIXES are different numbers under --all_tied, and conflating
  # them would read as a keep-rate change when it is only a rows-per-prefix
  # change. The keep RATE is per prefix; the dataset SIZE is rows.
  if args.max_tokens:
    print(f"  max_tokens              {args.max_tokens} (tokenizer {args.tokenizer})")
  print(f"  all_tied                {args.all_tied}"
        f"{f' (near_dup_ratio={args.near_dup_ratio})' if args.all_tied else ''}")
  print(f"  prefixes KEPT           {n_pref} ({n_pref / max(n_judged, 1):.3f})")
  print(f"  ROWS                    {kept} ({kept / max(n_pref, 1):.2f} per kept prefix)")
  for reason, n in sorted(stats.items()):
    if reason not in ("KEPT", "prefixes_kept"):
      print(f"    {reason:<22} {n}")
  if not rows:
    raise SystemExit("selection kept nothing; nothing to write")

  print()
  print("=" * 78)
  print("DATASET")
  print("=" * 78)
  print(compose_report(rows, prefixes))

  # A target containing code would train the simulator to emit code -- the one
  # thing this pipeline exists to remove. Refuse rather than warn.
  bad = [
      r["prefix_id"]
      for r in rows
      if templates.detect_code_leak(
          r["messages"][2]["content"],
          prefixes[r["prefix_id"]]["ground_truth"],
          ngram_n=0,
          expr_over_gt_names=True,
      )
  ]
  if bad:
    raise SystemExit(
        f"{len(bad)} selected targets still contain code (e.g. {bad[:5]}). "
        "Stage 1 and the judge disagree -- fix that before training."
    )

  split = {r["prefix_id"]: task_split(r["task_id"], args.val_frac) for r in rows}
  train = [r for r in rows if split[r["prefix_id"]] == "train"]
  val = [r for r in rows if split[r["prefix_id"]] == "val"]
  # No task may appear on both sides: one task's prefixes share a ground truth.
  overlap = {r["task_id"] for r in train} & {r["task_id"] for r in val}
  assert not overlap, f"task split leaked: {sorted(overlap)[:5]}"
  if len(val) < args.val_min:
    raise SystemExit(
        f"val split is {len(val)} rows, below --val_min {args.val_min}. verl "
        "reports val/loss = nan when N_val < train_batch_size; raise "
        "--val_frac or collect more tasks."
    )

  os.makedirs(args.out_dir, exist_ok=True)
  train_path = os.path.join(args.out_dir, args.train_name)
  val_path = os.path.join(args.out_dir, args.val_name)
  pd.DataFrame(train).to_parquet(train_path, index=False)
  pd.DataFrame(val).to_parquet(val_path, index=False)
  print()
  print(f"  train -> {train_path}  ({len(train)} rows, "
        f"{len({r['task_id'] for r in train})} tasks)")
  print(f"  val   -> {val_path}  ({len(val)} rows, "
        f"{len({r['task_id'] for r in val})} tasks)")
  print()
  print("  NEXT: run the dataset pre-scan before launching. It is the only")
  print("  check that catches 'trained on the wrong tokens':")
  print("    python -m colbench.simtrain.prescan_sft --parquet " + train_path)


if __name__ == "__main__":
  main()

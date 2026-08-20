r"""Merge K code-anchored spec draws into ONE augmented training JSONL.

Augmentation is applied SELECTIVELY: a seed gets all K draws only if its ground
truth has enough distinct behavior to support K different plots. Seeds whose GT
is a one-liner (single statement, no branching) keep exactly ONE draw -- they
collapse to the same withheld detail across draws, and they are where the author
confabulates edge-case handling the reference never implements (verified on the
300-seed pilot: `return population / area` -> "if the area is zero return zero").

Classification is structural, straight off the GT AST, so it never depends on
comparing plots to each other (plot divergence is the CONFABULATION signal, so
selecting on it would preferentially retain hallucinated draws).

Usage:
    python -m colbench.selfplay.merge_augmented_specs \
        --draws $S/train.selfplay.plot.ca1.jsonl $S/train.selfplay.plot.ca2.jsonl \
                $S/train.selfplay.plot.ca3.jsonl \
        --raw_parquet InfoPO/data/colbench_code/train.parquet \
        --out $S/train.selfplay_aug.plot.jsonl
"""

import argparse
import ast
import json
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench.selfplay.dataio import read_jsonl
from colbench.selfplay.dataio import read_tasks


def gt_is_one_liner(source: str) -> bool:
  """True if the GT is too thin to support more than one distinct plot.

  One statement (imports excluded) with no branching and at most one literal, or
  at most two statements with at most one branch point. Both bands were checked
  against the pilot: every seed whose draws invented edge-case behavior fell
  inside them, and the strict band alone missed two of the three.

  Args:
    source: the ground-truth function source.

  Returns:
    True when the seed should keep a single draw.
  """
  try:
    tree = ast.parse(source)
  except SyntaxError:
    return True  # unparseable -> do not augment
  fn = next(
      (
          n
          for n in ast.walk(tree)
          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
      ),
      None,
  )
  if fn is None:
    return True
  body = [
      n for n in fn.body if not isinstance(n, (ast.Import, ast.ImportFrom))
  ]
  branches = sum(
      1
      for n in ast.walk(fn)
      if isinstance(n, (ast.If, ast.For, ast.While, ast.IfExp, ast.Compare))
  )
  consts = len({
      ast.dump(n)
      for n in ast.walk(fn)
      if isinstance(n, ast.Constant) and not isinstance(n.value, bool)
  })
  if len(body) <= 1 and branches == 0 and consts <= 1:
    return True
  return len(body) <= 2 and branches <= 1


def main():
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--draws", nargs="+", required=True, help="Draw JSONLs, in order.")
  ap.add_argument(
      "--raw_parquet",
      default="InfoPO/data/colbench_code/train.parquet",
      help="Raw parquet the draws' `index` points into (for GT classification).",
  )
  ap.add_argument("--out", required=True, help="Merged JSONL path.")
  args = ap.parse_args()

  tasks = read_tasks(os.path.expanduser(args.raw_parquet))
  one_liner = {t["index"]: gt_is_one_liner(t["ground_truth"]) for t in tasks}

  kept, n_aug, n_single = [], 0, 0
  seen_single = set()
  for draw_i, path in enumerate(args.draws):
    for rec in read_jsonl(os.path.expanduser(path)):
      idx = rec.get("index")
      if idx is None or idx not in one_liner:
        continue
      if one_liner[idx]:
        if idx in seen_single:
          continue  # one-liner seeds keep the FIRST draw only
        seen_single.add(idx)
        n_single += 1
      elif draw_i == 0:
        n_aug += 1
      rec["draw"] = draw_i  # provenance; ignored by preprocess
      kept.append(rec)

  out = os.path.expanduser(args.out)
  os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
  with open(out, "w", encoding="utf-8") as fh:
    for rec in kept:
      fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
  print(
      f"[merge] {len(kept)} rows -> {out}\n"
      f"[merge]   augmented seeds (K={len(args.draws)}): {n_aug}\n"
      f"[merge]   one-liner seeds (K=1):                {n_single}"
  )


if __name__ == "__main__":
  main()

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
"""Stage 6: did the SFT improve the simulator? A PAIRED, held-out comparison.

Deliberately ONLY the comparison. Drawing the replies and judging them already
have tools -- ``collect_candidates.py`` draws K replies per prefix from
whichever sim endpoint it is pointed at, and ``judge_candidates.py`` scores them
-- so an evaluation is those two run once per arm and this file diffing the
results. Re-implementing either here would mean the two arms no longer went
through identical code, which is the one property a comparison needs.

    # 1. held-out prefixes, ONCE. Both arms are graded on the SAME prefixes,
    #    so the transcripts must not be regenerated per arm.
    DATA_FILE=$DATA_ROOT/colbench/test_small.fence.parquet \\
    END=500 RUN_TAG=eval STAGE=collect \\
      sbatch colbench/simtrain/run_collect_slurm.sh

    # 2. draw from each arm over those prefixes, same sampling
    STAGE=candidates RUN_TAG=eval SIM_MODEL=<S_0 path> \\
      CAND_FILE=.../candidates.eval.base.jsonl sbatch ...
    STAGE=candidates RUN_TAG=eval SIM_MODEL=<S_1 path> \\
      CAND_FILE=.../candidates.eval.sft.jsonl  sbatch ...

    # 3. judge both (same rubric, same seed), then
    python -m colbench.simtrain.eval_sim_sft \\
        --prefixes .../prefixes.test_small.fence.c1.jsonl \\
        --base .../judged.eval.base.jsonl --sft .../judged.eval.sft.jsonl

WHAT IT REPORTS, and why each line is here rather than one headline number:

  * ``select`` KEEP RATE -- "how often is this simulator's best draw usable".
    The closest single quantity to what the SFT optimised.
  * per-dimension means, PAIRED over prefixes kept in BOTH arms. Averaging over
    different prefix subsets would compare different tasks and call it progress.
  * the JUDGE-FREE over-release rate -- a GT token the reply introduces that
    was not already in the transcript. The judge selected the training data, so
    a judge-only improvement is partly circular; this number the rubric cannot
    reach.
  * stage-1 code leak and stage-2 wrong/unsure over ALL candidates. These are
    the failure modes; the score is a summary of them.
  * WIN/LOSS/TIE per prefix plus an exact sign test. Most prefixes tie by
    construction -- after two vetoes the survivors score alike -- so a small
    mean difference over a few hundred prefixes needs the split, not the mean.

THE LIMIT, stated here because it is easy to forget once a table exists:
held-out TASKS control for memorising prefixes, NOT for inheriting this judge's
blind spots. Only re-judging both arms with a different ``--judge_model`` does
that. The judge-free line is the one that is never circular.
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import re
import sys
from typing import Any, Optional

# pylint: disable=g-import-not-at-top,wrong-import-position
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)

from colbench import templates
from colbench.selfplay.dataio import read_jsonl
from colbench.simtrain import judge_rubric

# What counts as "content that could carry hidden information": decimals,
# multi-digit numbers, quoted strings, and identifiers of 3+ characters.
# SINGLE DIGITS ARE EXCLUDED -- a bare 0, 1 or 2 occurs in almost any sentence
# about code, and with them in the metric read 0.58 against 0.52 for the
# tightened version, with every extra hit noise ('elif', '1', '2').
_NUM = re.compile(r"(?<![\w.])(?:\d+\.\d+|\d{2,})(?![\w.])")
_STR = re.compile(r"['\"]([A-Za-z_][\w \-/]{1,30})['\"]")
_IDENT = re.compile(r"\b[A-Za-z_]\w{2,}\b")
_STOP = frozenset("""the a an and or of to in for is are be with that this it if then else
return def class import from as not none true false int str float bool list dict set tuple
print len range you your function python code value values input output param parameter
arguments elif for while try except with pass break continue get add sum min max abs round
sort sorted append extend items keys lower upper split join strip format number numbers
result total count index length string text data name type each other should would could
""".split())


def released_tokens(text: str) -> set[str]:
  """Tokens in ``text`` that could carry hidden information.

  Args:
    text: a reply, a problem statement, or a transcript fragment.

  Returns:
    The comparable token set.
  """
  out = set(_NUM.findall(text or ""))
  out |= {m.lower() for m in _STR.findall(text or "")}
  out |= {m.lower() for m in _IDENT.findall(text or "") if m.lower() not in _STOP}
  return out


def new_gt_tokens(prefix: dict[str, Any], reply: str) -> set[str]:
  """GT tokens the reply introduces that were NOT already in the transcript.

  The judge-free over-release measure. Everything the agent could already see
  -- the problem statement, the dialogue so far, its own last turn -- is
  subtracted, so confirming what the agent itself proposed scores zero instead
  of counting as a release. That subtraction is the whole reason this is a
  usable metric rather than a length proxy.

  Args:
    prefix: one ``collect_prefixes`` record.
    reply: the simulator turn being measured.

  Returns:
    The newly released GT tokens.
  """
  gt = released_tokens(prefix["ground_truth"])
  seen = released_tokens(prefix["problem_description"])
  for msg in prefix.get("sim_dialogue") or []:
    seen |= released_tokens(
        msg.get("content", "") if isinstance(msg, dict) else str(msg)
    )
  seen |= released_tokens(prefix.get("partner_reply") or "")
  return (released_tokens(reply) & gt) - seen


def sign_test_p(wins: int, losses: int) -> float:
  """Exact two-sided sign-test p-value for ``wins`` vs ``losses``.

  Ties are excluded, which is both the standard treatment and necessary here:
  after the code and truth vetoes the surviving candidates score alike, so most
  prefixes tie and an unconditional test would never move.

  Args:
    wins: prefixes where S_1 scored higher.
    losses: prefixes where S_1 scored lower.

  Returns:
    The p-value, or 1.0 when there is nothing to test.
  """
  n = wins + losses
  if n == 0:
    return 1.0
  k = min(wins, losses)
  tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
  return min(1.0, 2.0 * tail)


def arm_stats(
    prefixes: dict[str, dict[str, Any]],
    judged: list[dict[str, Any]],
    min_dims: Optional[dict[str, int]],
) -> dict[str, Any]:
  """Summarise one arm, keyed by ``prefix_id`` so the arms can be paired.

  Args:
    prefixes: ``prefix_id`` -> collector record.
    judged: that arm's ``judge_candidates`` records.
    min_dims: per-dimension floors handed to ``judge_rubric.select``.

  Returns:
    ``{"per_prefix": {pid: {...}}, "counts": Counter}``. ``per_prefix`` carries
    ``kept`` for every judged prefix, and the target-level fields only where
    something was selected.
  """
  per: dict[str, dict[str, Any]] = {}
  counts: collections.Counter = collections.Counter()
  for rec in judged:
    pid = rec.get("prefix_id")
    pre = prefixes.get(pid)
    if pre is None:
      counts["missing_prefix"] += 1
      continue
    idx, reason = judge_rubric.select(rec, min_dims=min_dims)
    counts["kept" if idx is not None else reason] += 1
    for flag in rec.get("truth_flags") or []:
      if flag is not None:
        counts[f"truth_{flag}"] += 1
        counts["truth_checked"] += 1
    counts["code_vetoed"] += len(rec.get("code_vetoed") or [])
    counts["candidates"] += len(rec.get("candidates") or [])

    # THE SERVING METRIC. Everything else in this record describes the SELECTED
    # target, i.e. best-of-K. That is the right readout for the selection
    # pipeline but the WRONG one for deployment: in training and in eval the
    # simulator is served plainly and emits ONE sample, so what matters is the
    # expected quality of an arbitrary draw, not of the best of eight.
    # dup_counts weighting is not optional -- the candidate list is DEDUPED, so
    # an unweighted mean over it reweights toward the sim's RARE outputs and has
    # produced two wrong conclusions on this pipeline before.
    dups = rec.get("dup_counts") or [1] * len(rec.get("candidates") or [])
    vetoed_idx = (
        set(rec.get("code_vetoed") or [])
        | set(rec.get("term_vetoed") or [])
        | set(rec.get("form_vetoed") or [])
    )
    draws = sum(dups)
    scored_w = 0
    dim_w: dict[str, float] = {}
    for i, v in enumerate(rec.get("scores") or []):
      if v is None:
        continue
      scored_w += dups[i]
      for d in judge_rubric.graded_dims_of(v):
        dim_w[d] = dim_w.get(d, 0.0) + dups[i] * int(v[d])
    counts["draws"] += draws
    for kind in ("code_vetoed", "term_vetoed", "form_vetoed"):
      counts[f"w_{kind}"] += sum(dups[i] for i in (rec.get(kind) or []))
    counts["w_any_vetoed"] += sum(dups[i] for i in vetoed_idx)

    entry: dict[str, Any] = {"kept": idx is not None, "reason": reason}
    entry["draws"] = draws
    entry["scored_w"] = scored_w
    entry["dim_w"] = dim_w
    entry["vetoed_w"] = sum(dups[i] for i in vetoed_idx)
    if idx is not None:
      target = rec["candidates"][idx]
      verdict = rec["scores"][idx]
      entry.update({
          "total": int(verdict.get("total", 0)),
          "dims": {
              d: int(verdict[d]) for d in judge_rubric.graded_dims_of(verdict)
          },
          "chars": len(target),
          "released": bool(new_gt_tokens(pre, target)),
          "leak": bool(
              templates.detect_code_leak(
                  target,
                  pre["ground_truth"],
                  ngram_n=0,
                  expr_over_gt_names=True,
              )
          ),
      })
    per[pid] = entry
  return {"per_prefix": per, "counts": counts}


def compare(base: dict[str, Any], sft: dict[str, Any]) -> str:
  """Render the paired comparison.

  Args:
    base: ``arm_stats`` output for S_0 (the base simulator).
    sft: ``arm_stats`` output for S_1 (the SFT'd simulator).

  Returns:
    The report text.
  """
  bp, sp = base["per_prefix"], sft["per_prefix"]
  shared = sorted(set(bp) & set(sp))
  out = [
      "=" * 78,
      "PAIRED EVALUATION   S_0 = base sim, S_1 = SFT sim, held-out tasks",
      "=" * 78,
      f"  prefixes judged in BOTH arms   {len(shared)}"
      f"   (base-only {len(set(bp) - set(sp))}, sft-only {len(set(sp) - set(bp))})",
  ]
  if not shared:
    return "\n".join(
        out + ["  (no shared prefixes -- the arms used different prefix files)"]
    )

  def rate(arm: dict[str, Any], key: str) -> float:
    return sum(1 for p in shared if arm[p].get(key)) / len(shared)

  b_keep, s_keep = rate(bp, "kept"), rate(sp, "kept")
  out += [
      "",
      f"  {'metric':<34} {'S_0':>8} {'S_1':>8} {'delta':>9}",
      f"  {'keep rate (select)':<34} {b_keep:>8.3f} {s_keep:>8.3f} {s_keep - b_keep:>+9.3f}",
  ]

  # Everything below is over prefixes KEPT IN BOTH arms: a metric on the
  # selected target is undefined where nothing was selected, and averaging over
  # different subsets would silently compare different tasks.
  both = [p for p in shared if bp[p]["kept"] and sp[p]["kept"]]
  out.append(f"  {'(both arms kept -- rows below)':<34} {len(both):>8}")
  if both:
    for label, key in (("judge total", "total"), ("target chars", "chars")):
      b = sum(bp[p][key] for p in both) / len(both)
      s = sum(sp[p][key] for p in both) / len(both)
      out.append(f"  {label:<34} {b:>8.2f} {s:>8.2f} {s - b:>+9.2f}")
    for dim in sorted(set().union(*(set(bp[p]["dims"]) for p in both))):
      b = sum(bp[p]["dims"].get(dim, 0) for p in both) / len(both)
      s = sum(sp[p]["dims"].get(dim, 0) for p in both) / len(both)
      out.append(f"  {'  dim ' + dim:<34} {b:>8.2f} {s:>8.2f} {s - b:>+9.2f}")
    for label, key in (
        ("JUDGE-FREE over-release", "released"),
        ("code leak in the target", "leak"),
    ):
      b = sum(1 for p in both if bp[p][key]) / len(both)
      s = sum(1 for p in both if sp[p][key]) / len(both)
      out.append(f"  {label:<34} {b:>8.3f} {s:>8.3f} {s - b:>+9.3f}")

    wins = sum(1 for p in both if sp[p]["total"] > bp[p]["total"])
    losses = sum(1 for p in both if sp[p]["total"] < bp[p]["total"])
    out += [
        "",
        f"  per-prefix judge total: S_1 wins {wins}, loses {losses}, "
        f"ties {len(both) - wins - losses}",
        f"  exact two-sided sign test (ties excluded)   p = {sign_test_p(wins, losses):.4f}",
        "     ^ most prefixes tie BY CONSTRUCTION: after the code and truth",
        "       vetoes the survivors score alike. Read the win/loss split.",
    ]

  # ── EXPECTED DRAW: what plain serving gives, not what BoN could give ──────
  out += [
      "",
      f"  EXPECTED DRAW (dup-weighted, ALL candidates) {'S_0':>8} {'S_1':>8} {'delta':>9}",
      "     ^ THIS is the serving metric: the sim emits ONE sample in training",
      "       and in eval, so read these rows, not the best-of-K rows above.",
  ]

  def _wrate(arm: dict[str, Any], key: str) -> float:
    num = sum(arm[p][key] for p in shared)
    den = sum(arm[p]["draws"] for p in shared)
    return num / max(den, 1)

  for label, key in (
      ("P(draw is vetoed, any kind)", "w_any_vetoed"),
      ("  writes code", "w_code_vetoed"),
      ("  ends before code shown", "w_term_vetoed"),
      ("  speaks AND terminates", "w_form_vetoed"),
  ):
    b = base["counts"][key] / max(base["counts"]["draws"], 1)
    s_ = sft["counts"][key] / max(sft["counts"]["draws"], 1)
    out.append(f"  {label:<34} {b:>8.3f} {s_:>8.3f} {s_ - b:>+9.3f}")

  # Dimension means are conditional on the draw surviving the vetoes: a vetoed
  # draw has no verdict, and scoring it 0 would mix two different quantities
  # into one column. The unconditional part is the veto rows immediately above.
  dims_seen = sorted(
      set().union(*(set(bp[p]["dim_w"]) for p in shared)) if shared else set()
  )
  for dim in dims_seen:
    b_n = sum(bp[p]["dim_w"].get(dim, 0.0) for p in shared)
    b_d = sum(bp[p]["scored_w"] for p in shared)
    s_n = sum(sp[p]["dim_w"].get(dim, 0.0) for p in shared)
    s_d = sum(sp[p]["scored_w"] for p in shared)
    b, s_ = b_n / max(b_d, 1), s_n / max(s_d, 1)
    out.append(
        f"  {'  E[' + dim + ' | not vetoed]':<34} {b:>8.2f} {s_:>8.2f} {s_ - b:>+9.2f}"
    )

  out += [
      "",
      f"  FAILURE MODES over ALL candidates   {'S_0':>8} {'S_1':>8} {'delta':>9}",
      "     ^ DEDUPED, unweighted -- kept for continuity with the r6 report.",
      "       Prefer the dup-weighted rows above.",
  ]
  for label, key, denom in (
      ("stage-1 code veto", "code_vetoed", "candidates"),
      ("stage-2 wrong", "truth_wrong", "truth_checked"),
      ("stage-2 unsure", "truth_unsure", "truth_checked"),
  ):
    b = base["counts"][key] / max(base["counts"][denom], 1)
    s = sft["counts"][key] / max(sft["counts"][denom], 1)
    out.append(f"  {label:<34} {b:>8.3f} {s:>8.3f} {s - b:>+9.3f}")

  out += [
      "",
      "  DROP REASONS (share of judged prefixes)",
      f"  {'reason':<34} {'S_0':>8} {'S_1':>8}",
  ]
  # Each arm is divided by ITS OWN judged count, not by the shared set: a
  # prefix the other arm failed to judge is not a drop for this one.
  reasons = sorted(
      r
      for r in set(base["counts"]) | set(sft["counts"])
      if r.startswith("all_")
      or r in (judge_rubric.DROP_BELOW_FLOOR, judge_rubric.DROP_JUDGE_FAILED)
  )
  for r in reasons:
    db = base["counts"][r] / max(len(bp), 1)
    ds = sft["counts"][r] / max(len(sp), 1)
    out.append(f"  {r:<34} {db:>8.3f} {ds:>8.3f}")

  out += [
      "",
      "  Held-out TASKS control for memorising prefixes, NOT for inheriting",
      "  this judge's blind spots -- the SFT data was selected by this same",
      "  rubric. Re-judge both arms with a different --judge_model before",
      "  believing the judge deltas; JUDGE-FREE over-release is the line that",
      "  cannot be circular.",
  ]
  return "\n".join(out)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the paired evaluation.

  Returns:
    The parser.
  """
  p = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  p.add_argument("--prefixes", required=True, help="Held-out prefix JSONL.")
  p.add_argument("--base", required=True, help="Judged JSONL for S_0.")
  p.add_argument("--sft", required=True, help="Judged JSONL for S_1.")
  p.add_argument(
      "--min_dims",
      default=None,
      help="Per-dimension floors as 'dim=N,...'. Default: the pipeline policy; "
      "'' for ranking-only. MUST match what built the SFT set, or the keep "
      "rates are not comparable to training.",
  )
  p.add_argument("--out", default="")
  return p


def main(argv: Optional[list[str]] = None) -> None:
  """Print (and optionally save) the paired comparison.

  Args:
    argv: argument vector, defaulting to ``sys.argv[1:]``.

  Raises:
    SystemExit: the two arms were judged under different rubric versions, which
      would make this measure the rubric change rather than the SFT.
  """
  args = build_arg_parser().parse_args(argv)
  from colbench.simtrain.build_sft_parquet import parse_min_dims

  prefixes = {r["prefix_id"]: r for r in read_jsonl(args.prefixes)}
  base_rows = read_jsonl(args.base)
  sft_rows = read_jsonl(args.sft)
  if not base_rows or not sft_rows:
    raise SystemExit("one of the arms has no judged rows")

  vb = {r.get("rubric_version") for r in base_rows}
  vs = {r.get("rubric_version") for r in sft_rows}
  if vb != vs:
    raise SystemExit(
        f"arms judged under different rubrics: base={sorted(vb)} "
        f"sft={sorted(vs)}. The comparison would measure the rubric change, "
        "not the SFT. Re-judge one arm."
    )

  min_dims = parse_min_dims(args.min_dims)
  report = compare(
      arm_stats(prefixes, base_rows, min_dims),
      arm_stats(prefixes, sft_rows, min_dims),
  )
  print(report)
  if args.out:
    with open(args.out, "w") as f:
      f.write(report + "\n")
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
  main()

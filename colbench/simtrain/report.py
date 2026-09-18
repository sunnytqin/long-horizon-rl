r"""Aggregate report over collected / judged data. Pure CPU, no network.

Reads whichever of the three JSONL files it is given and prints the numbers that
decide whether the dataset is usable:

  * COLLECTION -- episodes per termination reason, prefixes per episode, and the
    per-``turn_idx`` leak curve of the base sim's own replies. The turn curve is
    the honest version of the poisoned-prefix effect: prefixes are NOT filtered,
    so if the simulator degrades with dialogue depth, it shows up here as a
    rising leak rate rather than as an invisible bias.
  * JUDGING    -- parse-fail rate, score distributions per dimension, and the
    within-group spread (a spread near zero means best-of-N has nothing to
    choose between, which is the negative answer to the PoC's first question).
  * SELECTION  -- keep rate and drop reasons, broken out BY ``turn_idx``. If
    ``all_vetoed`` climbs with depth, later turns are silently under-represented
    in the SFT set and the trained sim will be best at turn 0.

Example:
    python -m colbench.simtrain.report \
        --prefixes $SIMTRAIN_ROOT/gs200/prefixes.train.fence.c1.jsonl \
        --episodes $SIMTRAIN_ROOT/gs200/episodes.train.fence.c1.jsonl \
        --judged   $SIMTRAIN_ROOT/gs200/judged.train.fence.c1.r1.jsonl
"""

# pylint: disable=g-importing-member
import argparse
import collections
import os
import sys
from typing import Any
from typing import Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench import templates
from colbench.selfplay.dataio import read_jsonl
from colbench.simtrain import judge_rubric
from colbench.simtrain.collect_prefixes import read_prefixes

RULE = "=" * 78


def _bar(frac: float, width: int = 24) -> str:
  """A fixed-width text bar for a 0..1 fraction.

  Args:
    frac: the fraction to draw.
    width: bar width in characters.

  Returns:
    The bar.
  """
  n = max(0, min(width, int(round(frac * width))))
  return "#" * n + "." * (width - n)


def report_collection(
    prefixes: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> str:
  """Episode outcomes, prefix yield, and the per-turn leak curve.

  Args:
    prefixes: ``collect_prefixes`` records.
    episodes: ``episodes.*.jsonl`` records.

  Returns:
    The section, as plain text.
  """
  out = [RULE, "COLLECTION", RULE]
  if episodes:
    term = collections.Counter(
        e.get("episode_terminated_by", "?") for e in episodes
    )
    n = len(episodes)
    out.append(f"  episodes                {n}")
    for reason, count in term.most_common():
      out.append(f"    {reason:<20} {count:>6}  ({count / n:.3f})")
    turns = [e.get("n_assistant_turns", 0) for e in episodes]
    out.append(
        f"  assistant turns/episode mean={sum(turns) / n:.2f} "
        f"min={min(turns)} max={max(turns)}"
    )
    out.append(f"  prefixes/episode        {len(prefixes) / n:.2f}")
  if not prefixes:
    return "\n".join(out)

  out += ["", f"  prefixes                {len(prefixes)}"]
  by_turn = collections.defaultdict(list)
  for rec in prefixes:
    by_turn[rec["turn_idx"]].append(rec)
  out += [
      "",
      "  BASE SIM leak rate by turn_idx, on its own single-shot reply during",
      "  collection. `def/fence` = detect_code_leak(ngram_n=0), the exact call",
      "  behind sim_leak_frac. `+ngram` ADDS the operator-gated n-gram detector",
      "  (10, 2), so it is always >= the first: the gap is prose that copies a",
      "  code expression without a def or a fence.",
      f"    {'turn':>4} {'n':>6} {'def/fence':>10} {'+ngram':>8}",
  ]
  for turn in sorted(by_turn):
    rows = by_turn[turn]
    loose = sum(1 for r in rows if r.get("sim_reply_leak_reason")) / len(rows)
    strict = sum(
        1 for r in rows if r.get("sim_reply_leak_reason_strict")
    ) / len(rows)
    out.append(
        f"    {turn:>4} {len(rows):>6} {loose:>10.3f} {strict:>8.3f}  "
        f"{_bar(loose)}"
    )
  reasons = collections.Counter(
      r.get("sim_reply_leak_reason") for r in prefixes
  )
  out.append(f"  leak reasons            {dict(reasons)}")
  chars = [len(r.get("sim_reply") or "") for r in prefixes]
  chars.sort()
  out.append(
      f"  sim reply chars         p50={chars[len(chars) // 2]} "
      f"p95={chars[int(len(chars) * 0.95)]} max={chars[-1]}"
  )
  return "\n".join(out)


def report_judging(
    judged: list[dict[str, Any]], prefixes: dict[str, dict[str, Any]]
) -> str:
  """Parse health, per-dimension distributions, and within-group spread.

  Args:
    judged: ``judge_candidates`` records.
    prefixes: ``prefix_id`` -> prefix record, for the judge-vs-regex table.

  Returns:
    The section, as plain text.
  """
  out = [RULE, "JUDGING", RULE]
  if not judged:
    return "\n".join(out + ["  (no judged rows)"])
  n = len(judged)
  n_ok = sum(1 for r in judged if r.get("ok"))
  n_retried = sum(1 for r in judged if r.get("retried"))
  n_clamped = sum(1 for r in judged if r.get("clamped"))
  n_incoh = sum(1 for r in judged if r.get("judge_incoherent"))
  fail = 1 - n_ok / n
  out += [
      f"  rubric/harness          {judged[0].get('rubric_version')} / "
      f"{judged[0].get('harness_version')}  judge="
      f"{judged[0].get('judge_model')}",
      f"  judged prefixes         {n}",
      f"  judge_parse_fail_rate   {fail:.4f}"
      + ("   <-- ABOVE 0.02: the OUTPUT CONTRACT is ambiguous" if fail > 0.02 else ""),
      f"  needed temp-0 retry     {n_retried} ({n_retried / n:.3f})",
      f"  clamped a score         {n_clamped} ({n_clamped / n:.3f})",
      f"  judge_incoherent        {n_incoh} ({n_incoh / n:.3f})   "
      "(its best_label disagrees with its own arithmetic)",
  ]

  # Keyed off the RECORD's dimensions, not the imported rubric's, so a report
  # over an older judged file shows that rubric's dimensions instead of raising.
  hist = collections.defaultdict(collections.Counter)
  leak_hist = collections.Counter()
  totals: list[int] = []
  spreads: list[int] = []
  cells = collections.Counter()
  for rec in judged:
    if not rec.get("ok"):
      continue
    gt = (prefixes.get(rec["prefix_id"]) or {}).get("ground_truth", "")
    group = []
    for cand, v in zip(rec["candidates"], rec["scores"], strict=True):
      if v is None:
        continue
      leak_hist[v["code_leak"]] += 1
      for d in judge_rubric.graded_dims_of(v):
        hist[d][v[d]] += 1
      totals.append(v["total"])
      group.append(v["total"])
      if gt:
        cells[
            (
                int(v["code_leak"]),
                int(
                    templates.detect_code_leak(
                        cand, gt, ngram_n=0, expr_over_gt_names=True
                    )
                    is None
                ),
            )
        ] += 1
    if len(group) > 1:
      spreads.append(max(group) - min(group))

  if totals:
    n_c = len(totals)
    out += [
        "",
        f"  candidates scored       {n_c}",
        f"  code_leak veto          clean={leak_hist[1]} "
        f"({leak_hist[1] / n_c:.3f})  vetoed={leak_hist[0]} "
        f"({leak_hist[0] / n_c:.3f})",
        f"  total 0..16             mean={sum(totals) / n_c:.2f} "
        f"min={min(totals)} max={max(totals)}",
        "",
        "  per-dimension histogram (0..4). A dimension that is a single spike",
        "  is DECORATION -- it contributes no variance to the ranking:",
    ]
    for d in sorted(hist):
      row = "  ".join(
          f"{k}:{hist[d][k] / n_c:.2f}" for k in range(judge_rubric.MAX_DIM + 1)
      )
      out.append(f"    {d:<22} {row}")
  if spreads:
    flat = sum(1 for s in spreads if s == 0)
    out += [
        "",
        f"  WITHIN-GROUP SPREAD     mean={sum(spreads) / len(spreads):.2f} "
        f"max={max(spreads)}  flat groups {flat}/{len(spreads)} "
        f"({flat / len(spreads):.3f})",
        "     ^ this is the PoC's question 1. Near-zero spread across most",
        "       groups means best-of-N selection has nothing to select.",
    ]
  if cells:
    out += [
        "",
        "  judge veto vs detect_code_leak (candidates; the STAGE-1 screen, i.e.",
      "  expr_over_gt_names=True -- so a backticked function body counts):",
        f"    both clean            {cells[(1, 1)]}",
        f"    both flag             {cells[(0, 0)]}",
        f"    JUDGE only            {cells[(0, 1)]}   <-- prose leaks the "
        "regex cannot see",
        f"    REGEX only            {cells[(1, 0)]}   <-- syntactic leaks the "
        "judge missed",
    ]
  return "\n".join(out)


def report_selection(
    judged: list[dict[str, Any]],
    min_total: int,
    min_dim: int,
    min_margin: int,
    min_dims: Optional[dict[str, int]] = None,
) -> str:
  """Keep rate and drop reasons, overall and by ``turn_idx``.

  Args:
    judged: ``judge_candidates`` records.
    min_total: total floor passed to ``select``.
    min_dim: per-dimension floor passed to ``select``.
    min_margin: margin floor passed to ``select``.
    min_dims: per-dimension floors passed to ``select``.

  Returns:
    The section, as plain text.
  """
  out = [
      RULE,
      f"SELECTION  (min_total={min_total} min_dim={min_dim} "
      f"min_margin={min_margin} min_dims={min_dims or {}})",
      RULE,
  ]
  if not judged:
    return "\n".join(out + ["  (no judged rows)"])
  reasons = collections.Counter()
  by_turn = collections.defaultdict(collections.Counter)
  for rec in judged:
    chosen, reason = judge_rubric.select(
        rec, min_total, min_dim, min_margin, min_dims
    )
    key = "KEPT" if chosen is not None else reason
    reasons[key] += 1
    by_turn[rec.get("turn_idx", -1)][key] += 1
  n = len(judged)
  keep = reasons["KEPT"]
  out.append(f"  group_keep_rate         {keep / n:.3f}  ({keep}/{n})")
  if keep / n < 0.3:
    out.append(
        "     ^ BELOW 0.3. Lower min_total to 11 and re-run this report; "
        "selection is a separate pass, so re-tuning costs no judge calls."
    )
  for reason, count in reasons.most_common():
    if reason != "KEPT":
      out.append(f"    {reason:<20} {count:>6}  ({count / n:.3f})")
  # Stage 2 is a VETO, so its rate is the number that decides whether r4's
  # split earned its extra call. Reported per verdict because "wrong" and
  # "unsure" are different failures: one misinforms the agent, the other
  # starves it.
  n_truth = sum(1 for r in judged if r.get("truth_flags"))
  if n_truth:
    tc = collections.Counter()
    checked = 0
    for rec in judged:
      for f in rec.get("truth_flags") or []:
        if f is not None:
          tc[f] += 1
          checked += 1
    bad = tc[judge_rubric.TRUTH_WRONG] + tc[judge_rubric.TRUTH_UNSURE]
    out += [
        "",
        f"  STAGE 2 truth veto      {checked} candidates checked "
        f"(after the programmatic code screen)",
        f"    ok                    {tc[judge_rubric.TRUTH_OK]:>5} "
        f"({tc[judge_rubric.TRUTH_OK] / max(checked, 1):.3f})",
        f"    wrong                 {tc[judge_rubric.TRUTH_WRONG]:>5} "
        f"({tc[judge_rubric.TRUTH_WRONG] / max(checked, 1):.3f})"
        "   <-- misstates the hidden spec",
        f"    unsure                {tc[judge_rubric.TRUTH_UNSURE]:>5} "
        f"({tc[judge_rubric.TRUTH_UNSURE] / max(checked, 1):.3f})"
        "   <-- withholds what the spec settles",
        f"    vetoed                {bad:>5} ({bad / max(checked, 1):.3f})",
    ]
    nfail = sum(1 for r in judged if r.get("truth_ok") is False)
    if nfail:
      out.append(
          f"    stage-2 PARSE FAILS   {nfail} rows failed CLOSED (nothing "
          "selected) rather than falling through to ranking"
      )
  out += [
      "",
      "  by turn_idx (a keep rate that FALLS with depth means later turns are",
      "  under-represented in the SFT set -- the honest form of the",
      "  poisoned-prefix effect, since prefixes are deliberately not filtered):",
      f"    {'turn':>4} {'n':>6} {'keep':>6} {'all_vetoed':>11} "
      f"{'all_untrue':>11} {'below_floor':>12}",
  ]
  for turn in sorted(by_turn):
    c = by_turn[turn]
    tot = sum(c.values())
    out.append(
        f"    {turn:>4} {tot:>6} {c['KEPT'] / tot:>6.3f} "
        f"{c[judge_rubric.DROP_ALL_VETOED] / tot:>11.3f} "
        f"{c[judge_rubric.DROP_ALL_UNTRUE] / tot:>11.3f} "
        f"{c[judge_rubric.DROP_BELOW_FLOOR] / tot:>12.3f}"
    )
  return "\n".join(out)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the report.

  Returns:
    The configured parser.
  """
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--prefixes", default="")
  ap.add_argument("--episodes", default="")
  ap.add_argument("--judged", default="")
  # Mirror judge_rubric.select's r2 defaults: ranking-only, veto retained.
  # Pass nonzero values to MEASURE what a floor would buy -- selection re-reads
  # the judged JSONL, so it costs no judge calls.
  ap.add_argument("--min_total", type=int, default=0)
  ap.add_argument("--min_dim", type=int, default=0)
  ap.add_argument("--min_margin", type=int, default=0)
  ap.add_argument(
      "--min_dims",
      default=",".join(f"{k}={v}" for k, v in judge_rubric.DEFAULT_MIN_DIMS.items()),
      help="Per-dimension floors as 'dim=N,dim=N'. Defaults to the pipeline "
      "policy (judge_rubric.DEFAULT_MIN_DIMS); pass '' for ranking-only.",
  )
  ap.add_argument("--out", default="")
  return ap


def main(argv: Optional[list[str]] = None) -> None:
  """Print (or write) the report for whichever files were given."""
  args = build_arg_parser().parse_args(argv)
  prefixes = read_prefixes(args.prefixes) if args.prefixes else []
  episodes = read_jsonl(args.episodes) if args.episodes else []
  judged = read_jsonl(args.judged) if args.judged else []
  by_id = {p["prefix_id"]: p for p in prefixes}

  sections = []
  if prefixes or episodes:
    sections.append(report_collection(prefixes, episodes))
  if judged:
    sections.append(report_judging(judged, by_id))
    min_dims = {}
    for part in (args.min_dims or "").split(","):
      if part.strip():
        k, _, v = part.partition("=")
        min_dims[k.strip()] = int(v)
    sections.append(
        report_selection(
            judged, args.min_total, args.min_dim, args.min_margin, min_dims
        )
    )
  text = "\n\n".join(sections) + "\n"
  if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(text)
    print(f"[report] -> {args.out}")
  else:
    sys.stdout.write(text)


if __name__ == "__main__":
  main()

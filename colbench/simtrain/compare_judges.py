r"""Compare two judged files over the SAME candidates: is judge B as good as A?

Built to answer one question -- can a locally served open model replace the
metered API judge -- and the honest answer is not a single agreement number.
Agreement with A is only evidence if A is right, and this project has already
retracted two conclusions drawn from judge output (see ``RESULTS.md``). So this
script does two things, and the SECOND one is the deliverable:

  1. quantify WHERE the two judges diverge, stage by stage, and
  2. dump the disagreeing prefixes as readable pages so they can be ADJUDICATED
     BY HAND against the hidden ground truth, which is the only arbiter here.

Read the dump before quoting the table.

WHAT IS COMPARABLE, AND WHAT IS NOT:
  * Stage 1 (``code_vetoed``) is PROGRAMMATIC -- the same regex on the same
    strings. It MUST be identical, so a mismatch means the two files do not
    describe the same candidates and every other number is meaningless. Checked
    first and treated as fatal.
  * Stage 2 (``truth_flags``) and stage 3 (``scores``) are the model's own
    judgement and are the real subject.
  * ``select()`` is the only thing that changes the SFT set, so keep/drop and
    the chosen candidate are the HEADLINE, not the per-dimension scores.

ON ``dup_counts``: candidate lists are DEDUPLICATED. Agreement is reported
UNWEIGHTED because the unit being measured is one judgement on one distinct
reply. Every rate describing the SIMULATOR (untrue, kept) is additionally
reported dup-weighted, because the deduped average reweights toward the sim's
RARE outputs and has produced wrong conclusions here twice.

Example:
    python -m colbench.simtrain.compare_judges \
        --prefixes $SIMTRAIN_ROOT/eval_heldout/prefixes.test_small.fence.c1.jsonl \
        --a $SIMTRAIN_ROOT/eval_heldout/judged.test_small.fence.c1.r6.jsonl \
        --b $SIMTRAIN_ROOT/eval_heldout/judged.test_small.fence.c1.r6.qwen...jsonl \
        --out_dump disagreements.txt
"""

# pylint: disable=g-importing-member
import argparse
import collections
import json
import os
import sys
from typing import Any
from typing import Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench.selfplay.dataio import read_jsonl
from colbench.simtrain import judge_rubric
from colbench.simtrain.collect_prefixes import read_prefixes
from colbench.simtrain.dump_judged import render_prefix

RULE = "=" * 88
THIN = "-" * 88


def load_judged(path: str) -> dict[str, dict[str, Any]]:
  """Judged rows by ``prefix_id``.

  Args:
    path: a ``judge_candidates`` JSONL.

  Returns:
    Mapping from prefix id to the row.

  Raises:
    SystemExit: the file mixes judges, rubrics or harness versions, which would
      make every aggregate below an average over two different standards.
  """
  rows = {r["prefix_id"]: r for r in read_jsonl(path) if "prefix_id" in r}
  stamps = {
      (r.get("judge_model", "?"), r.get("rubric_version", "?"),
       r.get("harness_version", "?"))
      for r in rows.values()
  }
  if len(stamps) > 1:
    raise SystemExit(
        f"[compare_judges] {path} mixes scoring standards: {sorted(stamps)}"
    )
  return rows


def stamp_of(rows: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
  """The single (judge, rubric, harness) triple describing a judged file.

  Args:
    rows: judged rows by prefix id.

  Returns:
    The triple, or a placeholder when there are no rows.
  """
  for r in rows.values():
    return (r.get("judge_model", "?"), r.get("rubric_version", "?"),
            r.get("harness_version", "?"))
  return ("?", "?", "?")


def kappa(pairs: list[tuple[Any, Any]]) -> Optional[float]:
  """Cohen's kappa: agreement ABOVE what the two marginals give by chance.

  Reported because raw agreement is inflated whenever one label dominates, and
  here one does -- most candidates are ``ok``. Two judges that both say ``ok``
  almost always will agree ~85% while sharing no discriminating signal at all.

  Args:
    pairs: (label_a, label_b) per item.

  Returns:
    Kappa, or None when it is undefined (no items, or chance agreement of 1).
  """
  if not pairs:
    return None
  n = len(pairs)
  observed = sum(1 for a, b in pairs if a == b) / n
  count_a = collections.Counter(a for a, _ in pairs)
  count_b = collections.Counter(b for _, b in pairs)
  expected = sum(
      (count_a[k] / n) * (count_b[k] / n) for k in set(count_a) | set(count_b)
  )
  if expected >= 1.0:
    return None
  return (observed - expected) / (1.0 - expected)


def _fmt(value: Optional[float], width: int = 6) -> str:
  """Format a possibly-undefined rate.

  Args:
    value: the number, or None.
    width: field width.

  Returns:
    A fixed-width string.
  """
  return f"{'n/a':>{width}}" if value is None else f"{value:{width}.3f}"


def compare(
    prefixes: dict[str, dict[str, Any]],
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    min_dims: Optional[dict[str, int]] = None,
) -> tuple[str, list[dict[str, Any]]]:
  """Build the report and the list of prefixes worth reading by hand.

  Args:
    prefixes: prefix records by id (for turn/task context).
    a_rows: judged rows from judge A (the incumbent).
    b_rows: judged rows from judge B (the candidate).
    min_dims: per-dimension floors handed to ``select``; None uses the
      pipeline default, so keep/drop here matches what the SFT build would do.

  Returns:
    ``(report_text, disagreements)``.

  Raises:
    SystemExit: the two files disagree on the PROGRAMMATIC stage-1 veto, i.e.
      they were not computed over the same candidates.
  """
  if min_dims is None:
    min_dims = judge_rubric.DEFAULT_MIN_DIMS
  shared = sorted(set(a_rows) & set(b_rows))
  if not shared:
    raise SystemExit("[compare_judges] the two files share no prefix ids")

  # Integrity first. Stage 1 is a regex over the candidate strings, so any
  # difference means the rows are not about the same replies and nothing below
  # would mean anything.
  for pid in shared:
    a, b = a_rows[pid], b_rows[pid]
    if a.get("candidates") != b.get("candidates"):
      raise SystemExit(
          f"[compare_judges] prefix {pid}: the two files hold DIFFERENT "
          "candidate texts -- they were judged over different draws."
      )
    if sorted(a.get("code_vetoed") or []) != sorted(b.get("code_vetoed") or []):
      raise SystemExit(
          f"[compare_judges] prefix {pid}: stage-1 code veto differs "
          f"({a.get('code_vetoed')} vs {b.get('code_vetoed')}). That stage is "
          "programmatic, so this is a harness mismatch, not a judgement one."
      )

  truth_pairs: list[tuple[str, str]] = []
  truth_pairs_w: list[tuple[str, str, int]] = []
  dim_stats: dict[str, dict[str, float]] = collections.defaultdict(
      lambda: {"n": 0, "same": 0, "absdiff": 0.0, "sum_a": 0.0, "sum_b": 0.0}
  )
  sel_same = sel_diff = 0
  keep_a = keep_b = keep_both = 0
  keepw_a = keepw_b = 0.0
  dupw_total = 0.0
  parse_fail_a = parse_fail_b = 0
  truth_fail_a = truth_fail_b = 0
  drop_a: collections.Counter = collections.Counter()
  drop_b: collections.Counter = collections.Counter()
  disagreements: list[dict[str, Any]] = []

  for pid in shared:
    a, b = a_rows[pid], b_rows[pid]
    dups = a.get("dup_counts") or [1] * len(a.get("candidates") or [])
    reasons: list[str] = []

    if not a.get("ok"):
      parse_fail_a += 1
    if not b.get("ok"):
      parse_fail_b += 1
    if not a.get("truth_ok", True):
      truth_fail_a += 1
    if not b.get("truth_ok", True):
      truth_fail_b += 1
    if bool(a.get("ok")) != bool(b.get("ok")) or bool(
        a.get("truth_ok", True)
    ) != bool(b.get("truth_ok", True)):
      reasons.append("parse")

    # Stage 2, per candidate. Only where BOTH judges recorded a flag.
    fa = a.get("truth_flags") or []
    fb = b.get("truth_flags") or []
    for i in range(min(len(fa), len(fb))):
      if fa[i] is None or fb[i] is None:
        continue
      truth_pairs.append((fa[i], fb[i]))
      truth_pairs_w.append((fa[i], fb[i], dups[i] if i < len(dups) else 1))
      if fa[i] != fb[i]:
        reasons.append(f"truth[{i}] {fa[i]}->{fb[i]}")

    # Stage 3, per candidate per dimension.
    sa = a.get("scores") or []
    sb = b.get("scores") or []
    for i in range(min(len(sa), len(sb))):
      if not sa[i] or not sb[i]:
        continue
      for dim in judge_rubric.graded_dims_of(sa[i]):
        if dim not in sb[i]:
          continue
        va, vb = int(sa[i][dim]), int(sb[i][dim])
        st = dim_stats[dim]
        st["n"] += 1
        st["same"] += int(va == vb)
        st["absdiff"] += abs(va - vb)
        st["sum_a"] += va
        st["sum_b"] += vb

    # Selection -- the only thing that reaches the SFT set.
    pick_a, why_a = judge_rubric.select(a, min_dims=min_dims)
    pick_b, why_b = judge_rubric.select(b, min_dims=min_dims)
    dup_total = sum(dups) or 1
    dupw_total += dup_total
    if pick_a is not None:
      keep_a += 1
      keepw_a += dups[pick_a] if pick_a < len(dups) else 1
    else:
      drop_a[why_a] += 1
    if pick_b is not None:
      keep_b += 1
      keepw_b += dups[pick_b] if pick_b < len(dups) else 1
    else:
      drop_b[why_b] += 1
    if pick_a is not None and pick_b is not None:
      keep_both += 1
      if pick_a == pick_b:
        sel_same += 1
      else:
        sel_diff += 1
        reasons.append(f"pick {pick_a}->{pick_b}")
    elif pick_a != pick_b:
      reasons.append(f"keep {why_a or 'KEEP'}->{why_b or 'KEEP'}")

    if reasons:
      disagreements.append({"prefix_id": pid, "reasons": reasons,
                            "pick_a": pick_a, "pick_b": pick_b,
                            "why_a": why_a, "why_b": why_b})

  n = len(shared)
  name_a = stamp_of(a_rows)
  name_b = stamp_of(b_rows)
  out = [
      RULE,
      "JUDGE COMPARISON",
      f"  A (incumbent): judge={name_a[0]} rubric={name_a[1]} harness={name_a[2]}",
      f"  B (candidate): judge={name_b[0]} rubric={name_b[1]} harness={name_b[2]}",
      f"  {n} shared prefixes "
      f"(A holds {len(a_rows)}, B holds {len(b_rows)})",
      RULE,
      "",
      "STAGE 1 (code veto, programmatic): identical by construction -- verified.",
      "",
      "PARSE FAILURES (a content failure is recorded, not retried forever):",
      f"  rank stage   A {parse_fail_a:4d}/{n}  ({parse_fail_a / n:.3f})"
      f"   B {parse_fail_b:4d}/{n}  ({parse_fail_b / n:.3f})",
      f"  truth stage  A {truth_fail_a:4d}/{n}  ({truth_fail_a / n:.3f})"
      f"   B {truth_fail_b:4d}/{n}  ({truth_fail_b / n:.3f})",
      "  >0.02 means the OUTPUT CONTRACT is ambiguous for that model, not that",
      "  it was unlucky. A judge that cannot hold the format is not a judge.",
      "",
  ]

  # Stage 2.
  out += ["STAGE 2 (truth veto) -- per candidate:"]
  if truth_pairs:
    agree = sum(1 for x, y in truth_pairs if x == y) / len(truth_pairs)
    k = kappa(truth_pairs)
    out += [
        f"  n={len(truth_pairs)}  agreement={agree:.3f}  kappa={_fmt(k)}",
        "  kappa, not agreement, is the number to read: 'ok' dominates, so two",
        "  judges that both say ok almost always look ~0.85 agreed on nothing.",
        "",
        "  confusion (rows = A, cols = B):",
    ]
    labels = sorted({x for x, _ in truth_pairs} | {y for _, y in truth_pairs})
    counts = collections.Counter(truth_pairs)
    out.append("            " + "".join(f"{l:>10}" for l in labels))
    for la in labels:
      out.append(
          f"  {la:>9} " + "".join(f"{counts[(la, lb)]:>10d}" for lb in labels)
      )
    out.append("")
    # Marginals, weighted and not: these describe the SIMULATOR.
    out += ["  marginal rates (what each judge SAYS about the sim):"]
    for label in labels:
      ra = sum(1 for x, _ in truth_pairs if x == label) / len(truth_pairs)
      rb = sum(1 for _, y in truth_pairs if y == label) / len(truth_pairs)
      wt = sum(w for _, _, w in truth_pairs_w) or 1
      wa = sum(w for x, _, w in truth_pairs_w if x == label) / wt
      wb = sum(w for _, y, w in truth_pairs_w if y == label) / wt
      out.append(
          f"    {label:>8}  A {ra:.3f} (dup-weighted {wa:.3f})"
          f"   B {rb:.3f} (dup-weighted {wb:.3f})"
      )
  else:
    out.append("  no comparable truth flags")
  out.append("")

  # Stage 3.
  out += [
      "STAGE 3 (ranking dimensions) -- per candidate, where both scored it:",
      f"  {'dim':<16}{'n':>7}{'exact':>8}{'mean|d|':>9}{'meanA':>8}{'meanB':>8}",
  ]
  for dim in sorted(dim_stats):
    st = dim_stats[dim]
    cnt = st["n"] or 1
    out.append(
        f"  {dim:<16}{int(st['n']):>7}{st['same'] / cnt:>8.3f}"
        f"{st['absdiff'] / cnt:>9.3f}{st['sum_a'] / cnt:>8.2f}"
        f"{st['sum_b'] / cnt:>8.2f}"
    )
  out += [
      "  A dim where meanA and meanB are close but 'exact' is low is NOISE, not",
      "  agreement -- the two judges are disagreeing in both directions equally.",
      "",
  ]

  # Selection -- the headline.
  out += [
      "SELECTION (the only stage that changes the SFT set):",
      f"  keep rate      A {keep_a / n:.3f} ({keep_a}/{n})"
      f"   B {keep_b / n:.3f} ({keep_b}/{n})",
      f"  dup-weighted   A {keepw_a / (dupw_total or 1):.3f}"
      f"   B {keepw_b / (dupw_total or 1):.3f}",
      f"  both kept      {keep_both}/{n}"
      f"  -- of those, SAME candidate {sel_same}, different {sel_diff}"
      + (f" ({sel_same / keep_both:.3f} same)" if keep_both else ""),
      "",
      "  drop reasons:",
      f"  {'reason':<20}{'A':>8}{'B':>8}",
  ]
  for reason in sorted(set(drop_a) | set(drop_b)):
    out.append(f"  {reason:<20}{drop_a[reason]:>8}{drop_b[reason]:>8}")
  out += [
      "",
      f"DISAGREEING PREFIXES: {len(disagreements)}/{n} "
      f"({len(disagreements) / n:.3f}) -- these are what to read.",
      RULE,
  ]
  return "\n".join(out), disagreements


def render_disagreements(
    prefixes: dict[str, dict[str, Any]],
    a_rows: dict[str, dict[str, Any]],
    b_rows: dict[str, dict[str, Any]],
    disagreements: list[dict[str, Any]],
    limit: Optional[int] = None,
    gt_lines: int = 24,
) -> str:
  """Readable pages for hand-adjudication, A and B side by side.

  Args:
    prefixes: prefix records by id.
    a_rows: judged rows from judge A.
    b_rows: judged rows from judge B.
    disagreements: the rows produced by ``compare``.
    limit: cap the number of pages.
    gt_lines: lines of hidden ground truth to show.

  Returns:
    The dump text.
  """
  pages = []
  for item in disagreements[:limit]:
    pid = item["prefix_id"]
    a, b = a_rows[pid], b_rows[pid]
    prefix = prefixes.get(pid)
    if prefix is not None:
      # The shared context (problem, hidden GT, agent turn) exactly as the
      # single-judge dump renders it, so pages read the same way.
      pages.append(render_prefix(prefix, a, gt_lines=gt_lines))
    pages.append(f"\n{THIN}\nWHY THIS PREFIX IS HERE: "
                 f"{'; '.join(item['reasons'])}\n{THIN}")
    fa = a.get("truth_flags") or []
    fb = b.get("truth_flags") or []
    sa = a.get("scores") or []
    sb = b.get("scores") or []
    for i, text in enumerate(a.get("candidates") or []):
      mark = []
      if item["pick_a"] == i:
        mark.append("A-PICK")
      if item["pick_b"] == i:
        mark.append("B-PICK")
      pages.append(f"\ncandidate [{i}] {' '.join(mark)}")
      pages.append(f"  {text}")
      va = fa[i] if i < len(fa) else None
      vb = fb[i] if i < len(fb) else None
      flag = "  <-- DISAGREE" if va != vb else ""
      pages.append(f"  truth:  A={va}  B={vb}{flag}")
      ca = a.get("truth_checks", {}).get(str(i), {}) or {}
      cb = b.get("truth_checks", {}).get(str(i), {}) or {}
      for who, chk in (("A", ca), ("B", cb)):
        if chk.get("quote") or chk.get("code"):
          pages.append(
              f"    {who} evidence: reply={chk.get('quote', '')!r} "
              f"code={chk.get('code', '')!r}"
          )
      da = sa[i] if i < len(sa) and sa[i] else None
      db = sb[i] if i < len(sb) and sb[i] else None
      if da or db:
        dims = sorted(
            set(judge_rubric.graded_dims_of(da or {}))
            | set(judge_rubric.graded_dims_of(db or {}))
        )
        pages.append(
            "  scores: "
            + "  ".join(
                f"{d}={(da or {}).get(d, '-')}/{(db or {}).get(d, '-')}"
                for d in dims
            )
            + f"   total={(da or {}).get('total', '-')}"
            f"/{(db or {}).get('total', '-')}"
        )
        if (da or {}).get("note"):
          pages.append(f"    A note: {(da or {})['note']}")
        if (db or {}).get("note"):
          pages.append(f"    B note: {(db or {})['note']}")
    pages.append(
        f"\nOUTCOME: A {'keep #%d' % item['pick_a'] if item['pick_a'] is not None else 'DROP ' + item['why_a']}"
        f"   |   B {'keep #%d' % item['pick_b'] if item['pick_b'] is not None else 'DROP ' + item['why_b']}"
    )
    pages.append("\n" + RULE + "\n")
  return "\n".join(pages)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI.

  Returns:
    The configured parser.
  """
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--prefixes", required=True)
  ap.add_argument("--a", required=True, help="judged JSONL, the INCUMBENT judge")
  ap.add_argument("--b", required=True, help="judged JSONL, the CANDIDATE judge")
  ap.add_argument("--out", default="", help="report text (default: stdout)")
  ap.add_argument(
      "--out_dump",
      default="",
      help="readable side-by-side pages for the disagreeing prefixes. THE "
      "DELIVERABLE: agreement with A is only evidence if A is right.",
  )
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--gt_lines", type=int, default=24)
  ap.add_argument(
      "--json_out", default="", help="machine-readable disagreement list"
  )
  return ap


def main(argv: Optional[list[str]] = None) -> None:
  """Entry point."""
  args = build_arg_parser().parse_args(argv)
  prefixes = {p["prefix_id"]: p for p in read_prefixes(args.prefixes)}
  a_rows = load_judged(args.a)
  b_rows = load_judged(args.b)
  report, disagreements = compare(prefixes, a_rows, b_rows)
  if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(report + "\n")
    print(f"[compare_judges] report -> {args.out}")
  print(report)
  if args.out_dump:
    with open(args.out_dump, "w", encoding="utf-8") as f:
      f.write(
          render_disagreements(
              prefixes, a_rows, b_rows, disagreements,
              limit=args.limit, gt_lines=args.gt_lines,
          )
      )
    print(f"[compare_judges] {len(disagreements)} disagreements -> {args.out_dump}")
    print("[compare_judges] READ IT. Adjudicate against the hidden GT; do not")
    print("[compare_judges] assume the incumbent is right where they differ.")
  if args.json_out:
    with open(args.json_out, "w", encoding="utf-8") as f:
      for item in disagreements:
        f.write(json.dumps(item) + "\n")
    print(f"[compare_judges] json -> {args.json_out}")


if __name__ == "__main__":
  main()

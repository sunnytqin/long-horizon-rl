r"""Render judged prefixes as plain text, one prefix per page, for reading.

THIS IS THE DECISION-POINT TOOL. The PoC's first question -- "does the rubric
give meaningful, separable scores?" -- is not settled by a threshold table; it
is settled by reading fifty judged prefixes and forming an opinion. That is only
practical if the artifact is readable, so this script exists to make it
readable: the agent turn that has to be answered, the hidden information the
simulator holds, then every candidate with its five dimension scores and the
judge's one-line note, sorted best-first.

``--slice i/n`` splits the dump into n contiguous parts so the reading can be
fanned out across several agents (or several sittings) without anyone reading
the same page twice.

What to look for, and what is worth discussing before the bulk spend:
  * Do the top- and bottom-scored candidates in a group actually differ in
    quality, or is the spread noise?
  * Does anything scored HIGH leak the hidden information in PROSE? That case is
    the entire reason for using a judge instead of ``detect_code_leak``.
  * Is anything scored high a stonewall -- i.e. is ``fidelity`` doing its
    counterweight job against ``calibration`` and ``volunteering``?
  * Is one dimension carrying all the variance, leaving the other four as
    decoration?
  * Do the notes show reasoning about the rubric, or pattern-matching on length?

The footer prints two mechanical numbers that ride along free and are NOT gates:
``judge_parse_fail_rate``, and the agreement table between the judge's
``code_leak`` veto and ``templates.detect_code_leak``. The disagreement cell
where the JUDGE flags a leak the regex misses is the one to eyeball.

Example:
    python -m colbench.simtrain.dump_judged \
        --prefixes $SIMTRAIN_ROOT/gs200/prefixes.train.fence.c1.jsonl \
        --judged   $SIMTRAIN_ROOT/gs200/judged.train.fence.c1.r1.jsonl \
        --slice 1/4 --out /tmp/pilot_r1_part1.txt
"""

# pylint: disable=g-importing-member
import argparse
import collections
import os
import sys
import textwrap
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

RULE = "=" * 88
THIN = "-" * 88


def _wrap(text: str, indent: str = "  ", width: int = 86) -> str:
  """Wrap ``text`` for terminal reading, preserving its blank-line structure.

  Args:
    text: the text to wrap.
    indent: prefix for every emitted line.
    width: total line width including the indent.

  Returns:
    The wrapped text.
  """
  out = []
  for para in (text or "").split("\n"):
    if not para.strip():
      out.append(indent.rstrip())
      continue
    out.extend(
        textwrap.wrap(
            para,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        or [indent.rstrip()]
    )
  return "\n".join(out)


def _abbrev(dim: str, dims: tuple[str, ...]) -> str:
  """Shortest UNAMBIGUOUS abbreviation of ``dim`` within ``dims``.

  A fixed 4-character truncation collided under g1 (``gt_adherence`` and
  ``plot_adherence`` are distinct at 4, but any future pair need not be), and
  two score columns sharing a header is worse than one long header.

  Args:
    dim: the dimension name to abbreviate.
    dims: every graded dimension in the same verdict.

  Returns:
    A prefix of ``dim`` that no other name in ``dims`` shares.
  """
  # Starts at 4 so r* dumps keep the width they have always had (cali / in_c /
  # volu); widens only when a rubric makes 4 ambiguous.
  for width in range(4, max(len(d) for d in dims) + 1):
    if sum(1 for d in dims if d[:width] == dim[:width]) == 1:
      # Never end on the separator: "gt_" reads as a truncation artefact
      # where "gt" reads as a name.
      return dim[:width].rstrip("_")
  return dim


def render_prefix(
    prefix: dict[str, Any], judged: dict[str, Any], gt_lines: int = 24
) -> str:
  """Render one judged prefix as a readable page.

  Args:
    prefix: the ``collect_prefixes`` record.
    judged: the matching ``judge_candidates`` record.
    gt_lines: how many lines of the hidden GT to show.

  Returns:
    The page, as plain text.
  """
  lines = [
      RULE,
      f"prefix {judged['prefix_id']}  (task {judged.get('task_id')}, turn "
      f"{judged.get('turn_idx')}, terminated_by "
      f"{prefix.get('episode_terminated_by')})",
      RULE,
      "",
      "PROBLEM (public):",
      _wrap(prefix["problem_description"]),
      "",
      f"HIDDEN INFORMATION (the simulator sees this, the agent never does):",
  ]
  gt = (prefix["ground_truth"] or "").split("\n")
  lines += ["  " + ln for ln in gt[:gt_lines]]
  if len(gt) > gt_lines:
    lines.append(f"  ... ({len(gt) - gt_lines} more lines)")
  lines += [
      "",
      "AGENT'S LAST TURN (this is what must be replied to):",
      _wrap(prefix.get("partner_reply", "")),
      "",
      "BASE SIM'S OWN REPLY during collection (the un-judged reference draw):",
      _wrap(prefix.get("sim_reply", "")),
      f"    [detect_code_leak: {prefix.get('sim_reply_leak_reason')} / strict "
      f"{prefix.get('sim_reply_leak_reason_strict')}]",
      "",
      THIN,
  ]

  if not judged.get("ok"):
    lines += [
        f"JUDGE FAILED: {judged.get('parse_error')!r}"
        f"{'  (after a temp-0 retry)' if judged.get('retried') else ''}",
        "",
    ]
    raw = judged.get("raw")
    if raw:
      lines += ["RAW JUDGE REPLY (first 1500 chars):", _wrap(raw[:1500]), ""]
    return "\n".join(lines)

  candidates = judged["candidates"]
  scores = judged["scores"]
  gt_source = prefix["ground_truth"]
  order = sorted(
      range(len(candidates)),
      key=lambda i: -(scores[i]["total"] if scores[i] else -1),
  )
  chosen, drop_reason = judge_rubric.select(judged)
  header = (
      f"CANDIDATES, best first    margin={judged.get('margin')}  "
      f"incoherent={judged.get('judge_incoherent')}  "
      f"selected={'cand ' + str(chosen) if chosen is not None else 'DROPPED (' + drop_reason + ')'}"
  )
  lines += [header]
  # g1 only: whether ending the conversation was even permitted here. Without
  # it a premature-termination veto in the list below reads as unexplained.
  if "terminate_allowed" in judged:
    lines.append(f"  terminate_allowed={judged['terminate_allowed']}")
  lines += [""]
  for i in order:
    v = scores[i]
    dup = (judged.get("dup_counts") or [1] * len(candidates))[i]
    if v is None:
      # r4: never ranked. Say WHICH stage removed it, and for a stage-2 veto
      # print the evidence it was required to cite -- the whole point of the
      # separate call is that its verdicts are auditable.
      tf = (judged.get("truth_flags") or [None] * len(candidates))[i]
      if i in set(judged.get("code_vetoed") or []):
        why = "code veto"
      elif i in set(judged.get("term_vetoed") or []):
        # g1's second programmatic veto: ended the conversation before the
        # agent had shown a complete function.
        why = "PREMATURE-TERMINATION veto (no code on the table yet)"
      elif i in set(judged.get("form_vetoed") or []):
        # g3's third: spoke and ended in the same reply, which the prompt
        # forbids outright and whose spoken half a rollout discards.
        why = "TERMINATION-FORM veto (spoke AND emitted the sentinel)"
      elif tf in (judge_rubric.TRUTH_WRONG, judge_rubric.TRUTH_UNSURE):
        why = f"STAGE 2 truth veto: {tf}"
      else:
        why = "NO VERDICT"
      lines += [f"  [cand {i}] {why}"]
      chk = (judged.get("truth_checks") or {}).get(str(i))
      if chk:
        if chk.get("quote"):
          lines.append(_wrap(f'quote: "{chk["quote"]}"', "      "))
        if chk.get("code"):
          lines.append(_wrap(f'code:  {chk["code"]}', "      "))
      lines += [_wrap(candidates[i], "      "), ""]
      continue
    regex_leak = templates.detect_code_leak(candidates[i], gt_source, ngram_n=0)
    flags = []
    # `code_leak` is an r* ranker dimension (a second opinion alongside the
    # regex). g1 has no such dimension -- it vetoes code programmatically -- so
    # the judge-vs-regex comparison simply does not apply there.
    judge_leak = v.get("code_leak")
    if judge_leak == 0:
      flags.append("JUDGE:LEAK")
    if regex_leak:
      flags.append(f"REGEX:{regex_leak}")
    if judge_leak == 0 and not regex_leak:
      flags.append("<-- judge-only leak (the reason a judge exists)")
    if judge_leak == 1 and regex_leak:
      flags.append("<-- regex-only leak (judge missed it)")
    # Built from GRADED_DIMS rather than hardcoded names, so a rubric that
    # renames or replaces a dimension does not silently KeyError here -- or,
    # worse, keep printing a stale one.
    # Abbreviations must stay DISTINCT: g1's gt_adherence / plot_adherence
    # both start "p"/"g" at 4 chars but collide at fewer, and two columns with
    # the same header is worse than a long one.
    graded = judge_rubric.graded_dims_of(v)
    dims = " ".join(f"{_abbrev(d, graded)}={v[d]}" for d in graded)
    # The max depends on how many dimensions THIS rubric graded, not on the
    # module-level constant of whichever rubric happens to be imported.
    max_total = judge_rubric.MAX_DIM * len(graded)
    leak = "" if judge_leak is None else f"leak={judge_leak} "
    lines.append(
        f"  [cand {i}] total {v['total']:>2}/{max_total}   "
        f"{leak}{dims}   x{dup} draws"
        + (("   " + "  ".join(flags)) if flags else "")
    )
    lines.append(_wrap(candidates[i], "      "))
    lines.append(_wrap(f"note: {v.get('note', '')}", "      "))
    lines.append("")
  return "\n".join(lines)


def summarize(judged_rows: list[dict[str, Any]], prefixes) -> str:
  """The free, non-gating footer numbers.

  Args:
    judged_rows: every judged record in this dump.
    prefixes: ``prefix_id`` -> prefix record.

  Returns:
    The footer, as plain text.
  """
  n = len(judged_rows)
  n_ok = sum(1 for r in judged_rows if r.get("ok"))
  n_retried = sum(1 for r in judged_rows if r.get("retried"))
  # judge veto x regex detector, over every scored candidate.
  cells = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  dim_spread_sums: dict[str, int] = collections.defaultdict(int)
  dim_spread_n: dict[str, int] = collections.defaultdict(int)
  totals: list[int] = []
  spreads: list[int] = []
  # Accumulated per dimension NAME seen in the records, not from the module's
  # tuple: r* and g* grade different dimensions, and a g1 file iterated over
  # r6's names KeyErrors on every row.
  dim_sums: dict[str, int] = collections.defaultdict(int)
  dim_counts: dict[str, int] = collections.defaultdict(int)
  dim_n = 0
  keeps = 0
  drops: dict[str, int] = {}
  for rec in judged_rows:
    chosen, reason = judge_rubric.select(rec)
    if chosen is None:
      drops[reason] = drops.get(reason, 0) + 1
    else:
      keeps += 1
    if not rec.get("ok"):
      continue
    gt = prefixes[rec["prefix_id"]]["ground_truth"]
    group = []
    for cand, v in zip(rec["candidates"], rec["scores"], strict=True):
      if v is None:
        continue
      # `code_leak` is an r* ranker dimension, so the judge-vs-regex table only
      # exists for r*. g1 vetoes code programmatically before the judge is
      # called, which is why there is nothing to compare there.
      judge_leak = v.get("code_leak")
      if judge_leak is not None:
        regex_clean = int(
            templates.detect_code_leak(cand, gt, ngram_n=0) is None
        )
        cells[(int(judge_leak), regex_clean)] += 1
      totals.append(v["total"])
      group.append(v["total"])
      for d in judge_rubric.graded_dims_of(v):
        dim_sums[d] += int(v[d])
        dim_counts[d] += 1
      dim_n += 1
    if len(group) > 1:
      spreads.append(max(group) - min(group))
    # Per-dimension within-group spread. A dimension that scores identically
    # across every candidate in a group taxes uniformly and can NEVER influence
    # the pick -- that is how r6's `in_character` was inert in 80% of groups.
    # Mean total is not enough to see it.
    scored = [v for v in rec["scores"] if v]
    if len(scored) > 1:
      for d in judge_rubric.graded_dims_of(scored[0]):
        vals = [int(v[d]) for v in scored if d in v]
        if len(vals) > 1:
          dim_spread_sums[d] += max(vals) - min(vals)
          dim_spread_n[d] += 1

  dim_spreads = {
      d: dim_spread_sums[d] / dim_spread_n[d]
      for d in dim_spread_sums
      if dim_spread_n[d]
  }
  out = [
      RULE,
      f"SUMMARY over {n} judged prefixes",
      RULE,
      f"  judge_parse_fail_rate  {1 - n_ok / n if n else 0:.4f}  "
      f"({n - n_ok} of {n}; {n_retried} needed the temp-0 retry)",
      f"  selection keep rate    {keeps / n if n else 0:.3f}  "
      f"drops: {drops or '{}'}",
  ]
  if totals:
    mean = sum(totals) / len(totals)
    out += [
        f"  candidate totals       n={len(totals)} mean={mean:.2f} "
        f"min={min(totals)} max={max(totals)}",
        "  mean per dimension     "
        + "  ".join(
            f"{d}={dim_sums[d] / dim_counts[d]:.2f}"
            for d in sorted(dim_sums)
            if dim_counts[d]
        ),
        "  per-dimension SPREAD   "
        + "  ".join(
            f"{d}={dim_spreads[d]:.2f}" for d in sorted(dim_spreads)
        )
        + "\n     ^ a dimension whose spread is ~0 is a UNIFORM TAX: it cannot"
        " influence ranking, however well worded it is.",
    ]
  if spreads:
    zero = sum(1 for s in spreads if s == 0)
    out += [
        f"  WITHIN-GROUP SPREAD    mean={sum(spreads) / len(spreads):.2f} "
        f"max={max(spreads)}  ({zero}/{len(spreads)} groups completely flat)",
        "     ^ a spread near zero across most groups means best-of-N has "
        "nothing to choose between.",
    ]
  if not any(cells.values()):
    out.append(
        "  (no judge-vs-regex table: this rubric vetoes code programmatically)"
    )
    out.append(RULE)
    return "\n".join(out)
  out += [
      "  judge veto vs detect_code_leak (counts of CANDIDATES):",
      f"     both clean            {cells[(1, 1)]}",
      f"     both flag a leak      {cells[(0, 0)]}",
      f"     JUDGE only            {cells[(0, 1)]}   <-- prose leaks the regex "
      "cannot see; the justification for a judge",
      f"     REGEX only            {cells[(1, 0)]}   <-- the judge missed a "
      "syntactic leak; a rubric problem",
      RULE,
  ]
  return "\n".join(out)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the dump.

  Returns:
    The configured parser.
  """
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--prefixes", required=True)
  ap.add_argument("--judged", required=True)
  ap.add_argument(
      "--out", default="", help="Write here instead of stdout."
  )
  ap.add_argument(
      "--slice",
      default="",
      dest="slice_spec",
      help="'i/n' -- render contiguous part i of n, so the reading can be "
      "fanned out across agents.",
  )
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument(
      "--only",
      default="",
      choices=["", "kept", "dropped", "failed", "leak_disagree"],
      help="Filter the pages: 'leak_disagree' shows only prefixes where the "
      "judge and detect_code_leak disagree about some candidate.",
  )
  ap.add_argument("--gt_lines", type=int, default=24)
  return ap


def main(argv: Optional[list[str]] = None) -> None:
  """Render the dump to stdout or a file."""
  args = build_arg_parser().parse_args(argv)
  prefixes = {p["prefix_id"]: p for p in read_prefixes(args.prefixes)}
  rows = [
      r
      for r in read_jsonl(args.judged)
      if r.get("prefix_id") in prefixes
  ]
  rows.sort(key=lambda r: r["prefix_id"])

  def _keep(rec):
    chosen, _ = judge_rubric.select(rec)
    if args.only == "kept":
      return chosen is not None
    if args.only == "dropped":
      return chosen is None
    if args.only == "failed":
      return not rec.get("ok")
    if args.only == "leak_disagree":
      if not rec.get("ok"):
        return False
      gt = prefixes[rec["prefix_id"]]["ground_truth"]
      return any(
          v is not None
          and (int(v["code_leak"]) == 1)
          != (templates.detect_code_leak(c, gt, ngram_n=0) is None)
          for c, v in zip(rec["candidates"], rec["scores"], strict=True)
      )
    return True

  rows = [r for r in rows if _keep(r)]
  if args.slice_spec:
    i, n = (int(x) for x in args.slice_spec.split("/"))
    if not 1 <= i <= n:
      raise SystemExit(f"--slice {args.slice_spec}: need 1 <= i <= n")
    size = (len(rows) + n - 1) // n
    rows = rows[(i - 1) * size : i * size]
  if args.limit is not None:
    rows = rows[: args.limit]

  pages = [
      render_prefix(prefixes[r["prefix_id"]], r, args.gt_lines) for r in rows
  ]
  text = "\n".join(pages) + "\n" + summarize(rows, prefixes) + "\n"
  if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(text)
    print(f"[dump_judged] {len(rows)} pages -> {args.out}")
  else:
    sys.stdout.write(text)


if __name__ == "__main__":
  main()

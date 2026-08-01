#!/usr/bin/env python3
"""Two focused reports on spec-eval conversation dumps.

Offline and JSON-only -- nothing is re-graded.

Report 1 -- WITHIN-conversation degeneration, by code-submission segment:
    A = start .. 1st code turn (inclusive)          [always]
    B = after 1st code .. 2nd code turn (inclusive) [sim pressed on only]
  per segment: avg agent response length/turn (raw + prose-only), capitulation
  rate. Claim "agent degrades after showing code" == B > A.

Report 2 -- the sim's terminate/press decision vs first-code correctness, and
the value of round 2:
    Branch 1: first code CORRECT   -> accepted? pressed on?
                                      if pressed, did round 2 DEGRADE?
    Branch 2: first code INCORRECT -> accepted (falsely)? pressed?
                                      if pressed, did round 2 FIX?

Detection from saved fields only (max_code_proposals=2 assumed):
    fc/sc     = 1st / 2nd code turn index, from turn_records[].showed_code
    accepted  = code_proposals==1 and terminated_by=="user"
                  (sim responded + ended)
    no_decision = code_proposals==1 and fc==last turn
                  and terminated_by=="turn_cap" (loop hit the turn cap AT the
                  first code; the sim never got to respond -> excluded)
    pressed   = showed_code and not accepted and not no_decision
    2nd code  = code_proposals>=2 ; round-2 correctness from all_pass(final)
                  vs first_code_all_pass.

Usage:
    python colbench/spec_eval_reports.py <eval1.json> [<eval2.json> ...]
"""

import glob
import json
import re
import statistics as st
import sys

CAP = re.compile(
    r"\b(you'?re absolutely right|you'?re right|you'?re correct|great "
    r"catch|good catch|"
    r"my apologies|i apologize|apologies)\b",
    re.I,
)
FENCE = re.compile(r"```.*?```", re.S)


def prose(t):
  return FENCE.sub("", t or "")


def is_capit(t):
  return bool(CAP.search(t or ""))


def code_turn_indices(tr):
  idxs = [i for i, r in enumerate(tr) if r.get("showed_code")]
  fc = idxs[0] if idxs else None
  sc = idxs[1] if len(idxs) > 1 else None
  return fc, sc


def report1(trajs):
  """Print report 1: within-conversation degeneration by segment.

  Args:
    trajs: the loaded trajectory dumps.
  """
  seg = {
      "A": {"raw": [], "prose": [], "cap": []},
      "B": {"raw": [], "prose": [], "cap": []},
  }
  n_with_b = 0
  for t in trajs:
    atext = [
        m["content"]
        for m in t.get("messages", [])
        if m.get("role") == "assistant"
    ]
    tr = t.get("turn_records", [])
    fc, _ = code_turn_indices(tr)
    if fc is None:
      continue
    has_b = False
    for i, txt in enumerate(atext):
      s = "A" if i <= fc else "B"
      if s == "B":
        has_b = True
      seg[s]["raw"].append(len(txt))
      seg[s]["prose"].append(len(prose(txt)))
      seg[s]["cap"].append(1 if is_capit(txt) else 0)
    n_with_b += int(has_b)

  def row(name, d):
    n = len(d["raw"])
    if not n:
      return f"  {name}:  (no turns)"
    return (
        f"  {name}:  turns={n:<6d} avg_len_raw={st.mean(d['raw']):7.0f} "
        f" avg_len_prose={st.mean(d['prose']):7.0f} "
        f" capitulation={st.mean(d['cap']):.3f}"
    )

  print("REPORT 1 -- behavior by segment (per agent turn)")
  print(row("A  (start->1st code)", seg["A"]))
  print(row("B  (1st->2nd code, sim pressed)", seg["B"]))
  print(
      (
          f"  # conversations with a segment B (sim pressed after 1st code): "
          f"{n_with_b}"
      )
  )


def _seg_b_stats(t, fc):
  """Prose-length and capitulation-flag lists over segment B.

  Segment B is the agent's post-1st-code turns.

  Args:
    t: one trajectory dump.
    fc: index of the first code turn; segment B is everything after it.

  Returns:
    ``(prose_lengths, capitulation_flags)`` over the agent's segment-B turns.
  """
  atext = [
      m["content"]
      for m in t.get("messages", [])
      if m.get("role") == "assistant"
  ]
  lens, caps = [], []
  for i, txt in enumerate(atext):
    if i > fc:
      lens.append(len(prose(txt)))
      caps.append(1 if is_capit(txt) else 0)
  return lens, caps


class Bucket:
  """A group of trajectories.

  Count + pooled segment-B prose lengths / capitulation flags.
  """

  def __init__(self):
    self.n = 0
    self.lens = []
    self.caps = []

  def add(self, t, fc):
    self.n += 1
    seg_lens, seg_caps = _seg_b_stats(t, fc)
    self.lens += seg_lens
    self.caps += seg_caps

  def behav(self):
    if not self.lens:
      return "segB: -"
    return (
        f"segB prose_len/turn={st.mean(self.lens):5.0f} "
        f" capitulation={st.mean(self.caps):.3f}"
    )


def report2(trajs):
  """Print report 2: sim decision, correctness, and round-2 outcome.

  Args:
    trajs: the loaded trajectory dumps.
  """
  acc = {"correct": 0, "incorrect": 0}  # Branch 1: sim accepted
  no_decision = 0
  # Branch 2: sim pressed -> [first correct?] -> outcome bucket
  pressed = {
      True: {
          "fixed_or_stillcorrect": Bucket(),
          "worse": Bucket(),
          "no2nd": Bucket(),
      },
      False: {
          "fixed_or_stillcorrect": Bucket(),
          "worse": Bucket(),
          "no2nd": Bucket(),
      },
  }
  pressed_n = {True: 0, False: 0}
  for t in trajs:
    fc, _ = code_turn_indices(t.get("turn_records", []))
    if fc is None:
      continue
    n_turns = len(
        [m for m in t.get("messages", []) if m.get("role") == "assistant"]
    )
    cp = t.get("code_proposals", 0)
    term = t.get("terminated_by")
    correct1 = bool(t.get("first_code_all_pass"))
    if cp == 1 and term == "user":  # Branch 1: accepted
      acc["correct" if correct1 else "incorrect"] += 1
    elif (
        cp == 1 and fc == n_turns - 1 and term == "turn_cap"
    ):  # sim never got to decide
      no_decision += 1
    else:  # Branch 2: pressed
      pressed_n[correct1] += 1
      if cp >= 2:
        final_ok = bool(t.get("all_pass"))
        # correct-first: ok=still-correct, not-ok=made-worse ; incorrect-first:
        # ok=fixed, not-ok=still-wrong
        key = "fixed_or_stillcorrect" if final_ok else "worse"
        if not correct1 and not final_ok:
          key = "worse"  # "still wrong" shares the "worse" slot label below
        pressed[correct1][key].add(t, fc)
      else:
        pressed[correct1]["no2nd"].add(t, fc)

  def pct(x, n):
    return f"{x}/{n} = {x / n:.3f}" if n else f"{x}/0"

  print(
      (
          "\nREPORT 2 -- sim decision -> correctness -> round-2 outcome (+ "
          "segment-B behavior)"
      )
  )
  nacc = acc["correct"] + acc["incorrect"]
  print(f"  BRANCH 1  sim ACCEPTED first code:  N={nacc}")
  print(
      f"     first code CORRECT   (good accept) : {pct(acc['correct'], nacc)}"
  )
  print(
      f"     first code INCORRECT (FALSE accept): {pct(acc['incorrect'], nacc)}"
  )
  print(f"  (sim never got to decide - turn cap at 1st code: {no_decision})")
  print(f"  BRANCH 2  sim PRESSED ON:  N={pressed_n[True] + pressed_n[False]}")
  for correct1, good_lbl in [
      (True, "still correct"),
      (False, "FIXED to correct"),
  ]:
    grp = pressed[correct1]
    n = pressed_n[correct1]
    title = "first code CORRECT" if correct1 else "first code INCORRECT"
    n2 = grp["fixed_or_stillcorrect"].n + grp["worse"].n
    print(
        (
            f"     [{title}]  n={n}   (2nd code produced: {n2}, no 2nd code: "
            f"{grp['no2nd'].n})"
        )
    )
    gb = grp["fixed_or_stillcorrect"]
    wb = grp["worse"]
    nb = grp["no2nd"]
    worse_lbl = "made WORSE" if correct1 else "still wrong"
    print(f"        {good_lbl:16s}: {pct(gb.n, n2)}   {gb.behav()}")
    print(f"        {worse_lbl:16s}: {pct(wb.n, n2)}   {wb.behav()}")
    if nb.n:
      print(f"        {'no 2nd code':16s}: {nb.n:>13}   {nb.behav()}")


def main():
  files = []
  for pattern in sys.argv[1:]:
    if any(c in pattern for c in "*?["):
      files.extend(sorted(glob.glob(pattern)))
    else:
      files.append(pattern)
  if not files:
    files = sorted(glob.glob("runs/spec/*.json"))
  for f in files:
    d = json.load(open(f, encoding="utf-8"))
    trajs = d.get("trajectories", [])
    print("=" * 90)
    print(f.split("/")[-1], f"  (n_traj={len(trajs)})")
    report1(trajs)
    report2(trajs)


if __name__ == "__main__":
  main()

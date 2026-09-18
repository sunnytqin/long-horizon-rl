r"""The judge's PURE core: prompt builder, parser, aggregation, selection.

No file I/O and no network -- every function here is a value-to-value transform,
so the part of the pipeline that is easiest to get subtly wrong is also the part
that is fully covered by CPU tests.

Split of responsibilities:
  * ``colbench.prompts``  -- the rubric TEXT (``SIM_JUDGE_*``,
    ``JUDGE_RUBRIC_VERSION``). Bytes are the experiment; see the EDITING RULE.
  * this module           -- how the rubric is administered
    (``simtrain.JUDGE_HARNESS_VERSION``).
  * ``judge_candidates``  -- the CLI, the metered API loop, the JSONL.

K-IN-ONE-CALL. All K candidates for a prefix are scored in a single call under
letter labels. One call amortizes the ~4 KB of shared context (problem + hidden
GT + dialogue) K-fold, and -- more importantly -- removes cross-call judge
variance from the WITHIN-GROUP ranking, which is the only comparison
best-of-N selection actually uses. Label order is permuted per prefix; the
rubric's "score each candidate on its own" instruction, not the shuffle, is what
suppresses position bias.

SELECTION IS A SEPARATE PASS from judging on purpose: re-tuning the floor costs
nothing because it re-reads the same judged JSONL.
"""

# This tree imports names directly rather than the enclosing module, matching
# how the rest of verl is written.
# pylint: disable=g-importing-member
import difflib
import hashlib
import json
import random
import re
from typing import Any
from typing import Optional

from colbench import prompts
from colbench import templates

# The four GRADED dimensions, 0..4 each. ``code_leak`` is deliberately not in
# this tuple: it is a 0/1 veto, and treating it as a fifth graded dimension is
# exactly the mistake that would let a strong reply buy its way past a leak.
#
# r2 order matters only for display. `volunteering` and `calibration` are the
# two halves of r1's `information_release`, split because a shotgun question
# makes them disagree: nothing was unasked-for (volunteering 4) while everything
# came out (calibration 0), and one dimension cannot say both.
# r1's `responsiveness` is gone -- it scored 4 on 100% of 225 pilot candidates,
# so it was pure decoration.
#
# `fidelity` is gone as of r4, for the opposite reason: it was not decoration,
# it was LOAD-BEARING AND UNEXECUTED. Four of the twelve r3-selected targets
# contradicted the hidden code outright and all four were scored 4, because
# grading truth needs the code read and a model doing that as one of five
# simultaneous judgements skips it. It is now stage 2, its own focused call with
# a citation requirement (`build_truth_messages`). It must NOT also exist here:
# a fidelity 4 in the ranker would outvote a veto in stage 2, which is the r3
# failure exactly.
#
# ``graded_dims_of`` reads the dimensions off each RECORD, so r1/r2/r3 files
# still report and select correctly against this shorter tuple.
GRADED_DIMS = (
    "volunteering",
    "calibration",
    "in_character",
)
MAX_DIM = 4
MAX_TOTAL = MAX_DIM * len(GRADED_DIMS)  # 12

# Candidate labels. Letters, not indices, so the judge cannot read an ordering
# into them the way it would from 1..K.
LABELS = "ABCDEFGHIJKLMNOP"

# Tolerant JSON extraction, mirroring selfplay.spec_templates.parse_spec: models
# wrap JSON in prose or fences often enough that a strict json.loads on the raw
# reply throws away otherwise-perfect verdicts.
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def permute(prefix_id: str, k: int, perm_seed: int = 0) -> list[int]:
  """Deterministic label order for one prefix's candidates.

  ``sha256(prefix_id) ^ perm_seed`` seeds the shuffle, so the permutation is
  reproducible from the record alone -- no stored state, and a re-judge of the
  same prefix under the same seed presents the candidates identically.
  (``label_to_candidate`` is recorded anyway, because reconstructing an
  ordering from a hash at debug time is miserable.)

  Args:
    prefix_id: the prefix's primary key.
    k: number of candidates.
    perm_seed: run-level seed; change it to re-judge under a different order.

  Returns:
    A permutation of ``range(k)``: position *i* holds the candidate index that
    gets label ``LABELS[i]``.
  """
  h = int(hashlib.sha256(prefix_id.encode("utf-8")).hexdigest()[:16], 16)
  order = list(range(k))
  random.Random(h ^ int(perm_seed)).shuffle(order)
  return order


def build_candidate_block(
    candidates: list[str], order: list[int]
) -> tuple[str, dict[str, int]]:
  """Render the labelled candidate list and the label -> candidate index map.

  Args:
    candidates: the candidate reply strings, in their own (stable) order.
    order: a permutation of ``range(len(candidates))`` from ``permute``.

  Returns:
    ``(block, label_to_candidate)``.
  """
  lines, mapping = [], {}
  for pos, cand_idx in enumerate(order):
    label = LABELS[pos]
    mapping[label] = cand_idx
    lines.append(f"--- Candidate {label} ---\n{candidates[cand_idx]}")
  return "\n\n".join(lines), mapping


def build_judge_messages(
    prefix: dict[str, Any],
    candidates: list[str],
    perm_seed: int = 0,
) -> tuple[list[dict[str, str]], dict[str, int]]:
  """Build the one chat call that scores every candidate for one prefix.

  The dialogue is rendered with ``templates.str_dialogue_history`` -- the SAME
  function that built the simulator's own prompt -- so the judge reads the
  transcript in exactly the form the simulator was asked to continue, down to
  the trailing ``agent:`` cue.

  Args:
    prefix: one ``collect_prefixes`` record; reads ``prefix_id``,
      ``problem_description``, ``ground_truth`` and ``sim_dialogue``.
    candidates: the candidate replies to score.
    perm_seed: run-level permutation seed.

  Returns:
    ``(messages, label_to_candidate)``.

  Raises:
    ValueError: ``candidates`` is empty or longer than ``LABELS``.
  """
  if not candidates:
    raise ValueError("build_judge_messages: no candidates")
  if len(candidates) > len(LABELS):
    raise ValueError(
        f"build_judge_messages: {len(candidates)} candidates exceeds the "
        f"{len(LABELS)} available labels"
    )
  order = permute(prefix["prefix_id"], len(candidates), perm_seed)
  block, mapping = build_candidate_block(candidates, order)
  user = prompts.SIM_JUDGE_USER_TEMPLATE.format(
      problem_description=prefix["problem_description"],
      hidden_information=prefix["ground_truth"],
      dialogue_history=templates.str_dialogue_history(prefix["sim_dialogue"]),
      n_candidates=len(candidates),
      candidate_block=block,
  )
  return (
      [
          {"role": "system", "content": prompts.SIM_JUDGE_SYSTEM_PROMPT},
          {"role": "user", "content": user},
      ],
      mapping,
  )


# ── The GROUNDED arm (rubric g1) ──────────────────────────────────────────────
# Three RANKING dimensions and no LLM veto: both vetoes are programmatic (see
# `grounded_vetoes`). `graded_dims_of` reads dimensions off each record, so
# `select`, the report and the dump all work across r* and g* files without
# knowing which rubric wrote them.
GROUNDED_GRADED_DIMS = (
    "gt_adherence",
    "plot_adherence",
    "not_overhelpful",
)
GROUNDED_MAX_TOTAL = MAX_DIM * len(GROUNDED_GRADED_DIMS)  # 12


def grounded_vetoes(
    prefix: dict[str, Any],
    candidates: list[str],
    expr_over_gt_names: bool = True,
) -> tuple[list[int], list[int], list[int]]:
  """The three PROGRAMMATIC vetoes for the grounded arm. No API call.

  Both are things the grounded simulator's own system prompt forbids outright,
  and both are already enforced in ``env_spec`` during a real rollout -- so
  paying a judge to re-derive them would buy nothing and would let a model's
  opinion override a mechanical fact.

  1. WRITING CODE. The grounded sim is handed the GT source, so unlike the spec
     arm a leak is not structurally impossible -- it is the main hazard.
     ``expr_over_gt_names`` defaults True here (it is opt-in elsewhere) because
     the grounded prompt forbids quoting "a variable name, or a literal value as
     code", which is exactly what detector (C) catches.
  2. PREMATURE TERMINATION. Ending before the agent has shown a complete
     function. The grounded prompt states this as a MINIMUM bar ("until you have
     seen one you MUST NOT end the conversation") and ``env_spec`` enforces it by
     rejection sampling; ``terminate_allowed`` on the prefix is the same fact,
     recorded at collection time.
  3. SPEAKING AND THEN ENDING IN ONE REPLY. The prompt is explicit that the
     sentinel is a signal and not a message -- "your ENTIRE reply must be
     exactly [TERMINATE] ... No goodbye, no thanks, no explanation, nothing
     before or after it", and "any reply that is still part of the conversation
     must not contain that sentinel anywhere at all". A trailing "Looks good,
     thanks! [TERMINATE]" is therefore malformed, whatever it says: at rollout
     time ``sim_terminated``'s unanchored match ends the episode and the spoken
     half is never delivered, so training on it teaches a reply shape whose
     content is discarded. Measured on the g1 pilot: 42 of 491 candidates and 9
     of 77 selected targets were this shape.
     This is a veto on FORM, not on the decision to end -- a bare sentinel is
     always fine, and nothing here judges whether ending was deserved.

  Args:
    prefix: one ``collect_prefixes`` record from the grounded arm. Reads
      ``ground_truth`` and ``terminate_allowed``.
    candidates: the candidate reply strings.
    expr_over_gt_names: enable detector (C), expressions over the GT's own
      identifiers.

  Returns:
    ``(code_vetoed, term_vetoed, form_vetoed)``, each a list of candidate
    indices. Kept as three lists rather than one so the dump can say which rule
    a candidate broke; a candidate can appear in more than one.
  """
  ground_truth = prefix.get("ground_truth", "") or ""
  # `terminate_allowed` missing is treated as NOT allowed: a prefix file that
  # predates the field must not silently let every premature termination pass.
  allowed = bool(prefix.get("terminate_allowed", False))
  code_vetoed, term_vetoed, form_vetoed = [], [], []
  for i, text in enumerate(candidates):
    if templates.detect_code_leak(
        text, ground_truth, ngram_n=0, expr_over_gt_names=expr_over_gt_names
    ):
      code_vetoed.append(i)
    if not allowed and templates.sim_terminated(text):
      term_vetoed.append(i)
    if templates.sim_terminated(text) and not templates.sim_terminate_standalone(
        text
    ):
      form_vetoed.append(i)
  return code_vetoed, term_vetoed, form_vetoed


def select_all(
    judged: dict[str, Any],
    min_total: int = 0,
    min_dim: int = 0,
    min_margin: int = 0,
    min_dims: Optional[dict[str, int]] = None,
    near_dup_ratio: float = 0.9,
) -> tuple[list[int], str]:
  """Every candidate TIED at the group max, not just one of them.

  ``select`` breaks a tie and returns a single index, which throws away real
  training data: on the 85-prefix grounded pilot, taking the whole tied set is
  168 rows instead of 76 -- 2.2x -- at NO cost to prefix coverage (76/85 either
  way), task coverage (39/40), turn spread (39/32/5) or the 22 `[TERMINATE]`
  targets, and mean target score is unchanged (10.25 -> 10.29/12). The tie is
  the judge saying it cannot separate these; imitating all of them is a truer
  reading of that than imitating whichever one happened to be shortest.

  WHY NEAR-DUPLICATES ARE DROPPED. The draws in a tie are frequently the same
  sentence reworded, and duplicate rows would silently reweight the SFT loss
  toward whichever prefix happened to produce the most paraphrases. Deduping at
  a similarity ratio removes that without discarding genuinely distinct replies:
  on the pilot, within-tie median similarity is 0.37, so the cut is not doing
  much work -- but where it does, it is removing a reweighting artefact.

  This does NOT subtract a group baseline. A GRPO-style "beat the group mean"
  filter was measured on the same pilot and rejected: it cuts prefix coverage
  76 -> 39, task coverage 40 -> 32, removes turn 2 entirely, and deletes ALL 22
  `[TERMINATE]` targets including the 14 scored 12/12. Zero group variance means
  no advantage under a subtracted baseline, but SFT has no baseline -- it is
  positive-only imitation, so a unanimously good draw still teaches something.

  Args:
    judged: one ``judge_candidates`` record.
    min_total: floor on the total; 0 disables.
    min_dim: floor on EVERY graded dimension; 0 disables.
    min_margin: floor on the best-vs-second gap; 0 disables.
    min_dims: per-dimension floors, as in ``select``.
    near_dup_ratio: drop a tied candidate whose text is at least this similar
      to one already kept. 1.0 keeps exact duplicates too.

  Returns:
    ``(candidate_indices, "")`` on a keep -- ordered as ``select`` would rank
    them, so element 0 is exactly what ``select`` returns -- or
    ``([], reason)`` on a drop, with the same reasons ``select`` uses.
  """
  first, reason = select(
      judged,
      min_total=min_total,
      min_dim=min_dim,
      min_margin=min_margin,
      min_dims=min_dims,
  )
  if first is None:
    return [], reason
  candidates = judged["candidates"]
  scores = judged["scores"]
  # The tie is taken ONLY among candidates that cleared every veto and floor:
  # `passing_candidates` is the single definition of that, and tying on `total`
  # alone would promote a candidate that fails a per-dimension floor.
  passing, _ = passing_candidates(
      judged,
      min_total=min_total,
      min_dim=min_dim,
      min_margin=min_margin,
      min_dims=min_dims,
  )
  best = scores[first]["total"]
  rest = sorted(
      (i for i in passing if i != first and scores[i]["total"] == best),
      key=lambda i: (len(candidates[i]), i),
  )
  kept = [first]
  for i in rest:
    if near_dup_ratio < 1.0 and any(
        difflib.SequenceMatcher(None, candidates[i], candidates[j]).ratio()
        >= near_dup_ratio
        for j in kept
    ):
      continue
    kept.append(i)
  return kept, ""


def build_grounded_judge_messages(
    prefix: dict[str, Any],
    candidates: list[str],
    perm_seed: int = 0,
) -> tuple[list[dict[str, str]], dict[str, int]]:
  """Build the single ranking call for one grounded-arm prefix.

  The dialogue is rendered with ``templates.str_dialogue_history`` -- the SAME
  function that built the simulator's own prompt -- so the judge reads the
  transcript in exactly the form the simulator was asked to continue.

  Args:
    prefix: one grounded ``collect_prefixes`` record; reads ``prefix_id``,
      ``problem_description``, ``ground_truth``, ``plot`` and ``sim_dialogue``.
    candidates: the candidate replies to score (vetoed ones already removed).
    perm_seed: run-level permutation seed.

  Returns:
    ``(messages, label_to_candidate)``.

  Raises:
    ValueError: ``candidates`` is empty or longer than ``LABELS``.
  """
  if not candidates:
    raise ValueError("build_grounded_judge_messages: no candidates")
  if len(candidates) > len(LABELS):
    raise ValueError(
        f"build_grounded_judge_messages: {len(candidates)} candidates exceeds "
        f"the {len(LABELS)} available labels"
    )
  order = permute(prefix["prefix_id"], len(candidates), perm_seed)
  block, mapping = build_candidate_block(candidates, order)
  user = prompts.GROUNDED_JUDGE_USER_TEMPLATE.format(
      problem_description=prefix["problem_description"],
      ground_truth=prefix["ground_truth"],
      # An empty plot renders as a visible placeholder rather than a blank gap,
      # so the judge is told there is nothing to adhere to instead of being left
      # to guess from whitespace.
      plot=(prefix.get("plot") or "").strip() or "(no plot for this task)",
      dialogue_history=templates.str_dialogue_history(prefix["sim_dialogue"]),
      n_candidates=len(candidates),
      candidate_block=block,
  )
  return (
      [
          {"role": "system", "content": prompts.GROUNDED_JUDGE_SYSTEM_PROMPT},
          {"role": "user", "content": user},
      ],
      mapping,
  )


def _clamp_dim(value: Any) -> tuple[int, bool]:
  """Coerce one graded score into 0..4.

  A model that returns ``7``, ``"3"`` or ``3.0`` has understood the rubric and
  fumbled the format. Clamping and RECORDING that it was clamped keeps the
  verdict usable while leaving the fumble visible; failing the whole call over
  it would throw away K-1 good verdicts.

  Args:
    value: whatever the judge put in the field.

  Returns:
    ``(score, clamped)``.

  Raises:
    ValueError: the value is not numeric at all.
  """
  n = int(round(float(value)))
  c = max(0, min(MAX_DIM, n))
  return c, c != n


# Keys a verdict carries that are NOT graded dimensions.
_NON_DIM_KEYS = frozenset({"code_leak", "note", "total", "label"})


def graded_dims_of(verdict: dict[str, Any]) -> tuple[str, ...]:
  """The graded dimensions actually present in ``verdict``.

  ``select`` reads a judged JSONL back off disk, and that file may have been
  written under a DIFFERENT rubric version than the one currently imported --
  comparing r1 and r2 over the same pilot batch is the normal case, not an edge
  case. Iterating the module-level ``GRADED_DIMS`` there raises ``KeyError`` on
  a name the old rubric never had, which reads as a crash rather than as the
  version mismatch it is. Reading the dimensions off the record instead makes
  the floor mean "every graded dimension of WHATEVER rubric scored this", which
  is the intended semantics for any rubric.

  Args:
    verdict: one parsed verdict dict.

  Returns:
    Its graded dimension names, sorted for determinism.
  """
  return tuple(sorted(k for k in verdict if k not in _NON_DIM_KEYS))


def total_score(verdict: dict[str, Any]) -> int:
  """0 when the leak veto fires, else the sum of the four graded dimensions.

  Args:
    verdict: a parsed verdict dict.

  Returns:
    The verdict's total, 0..16.
  """
  # `code_leak` ABSENT means this rubric has no such dimension (g1 vetoes code
  # programmatically), not that the veto fired -- so it defaults to 1. Every
  # r1-r6 verdict carries the field explicitly, so their behaviour is unchanged.
  if int(verdict.get("code_leak", 1)) == 0:
    return 0
  # Dimensions come off the RECORD, not the module-level tuple, for the same
  # reason `graded_dims_of` exists: one judged file may have been written under
  # a different rubric than the one currently imported.
  return sum(int(verdict.get(d, 0)) for d in graded_dims_of(verdict))


def parse_verdicts(
    raw: str,
    labels: list[str],
    dims: tuple[str, ...] = GRADED_DIMS,
    require_code_leak: bool = True,
) -> dict[str, Any]:
  """Parse one judge reply into per-label verdicts.

  Tolerant of prose/fence wrapping and of out-of-range scores; STRICT about the
  label set, because a missing or duplicated label means we cannot say which
  candidate a score belongs to, and a mis-attributed score is worse than no
  score.

  Args:
    raw: the judge's reply text.
    labels: the labels that must each appear exactly once.
    dims: the graded dimensions this rubric requires on every verdict.
      Defaults to r*'s; the grounded arm passes ``GROUNDED_GRADED_DIMS``.
    require_code_leak: whether a ``code_leak`` field is mandatory. True for r*,
      where it is the ranker's second-opinion veto; False for g1, which vetoes
      code programmatically before the call and has no such dimension.

  Returns:
    ``{"ok", "parse_error", "verdicts": {label: verdict}, "best_label",
    "clamped", "raw"}``. On failure ``verdicts`` is empty and ``parse_error``
    says why; the caller records the row with ``ok=False`` rather than
    retrying forever (a content failure is deterministic for that prompt).
  """
  out: dict[str, Any] = {
      "ok": False,
      "parse_error": "",
      "verdicts": {},
      "best_label": None,
      "clamped": False,
      "raw": raw,
  }
  text = (raw or "").strip()
  m = _JSON_OBJ_RE.search(text)
  if not m:
    out["parse_error"] = "no JSON object found"
    return out
  try:
    obj = json.loads(m.group(0))
  except (json.JSONDecodeError, ValueError) as e:
    out["parse_error"] = f"json.loads failed: {e}"
    return out
  if not isinstance(obj, dict):
    out["parse_error"] = f"top level is {type(obj).__name__}, not an object"
    return out
  raw_verdicts = obj.get("verdicts")
  if not isinstance(raw_verdicts, list):
    out["parse_error"] = "no 'verdicts' array"
    return out

  parsed: dict[str, Any] = {}
  clamped_any = False
  for item in raw_verdicts:
    if not isinstance(item, dict):
      out["parse_error"] = "a verdict entry is not an object"
      return out
    label = str(item.get("label", "")).strip().upper()[:1]
    if label not in labels:
      out["parse_error"] = f"unknown label {item.get('label')!r}"
      return out
    if label in parsed:
      out["parse_error"] = f"duplicate label {label}"
      return out
    # `code_leak` is an r* dimension. g1 vetoes code PROGRAMMATICALLY before
    # this call, so demanding it there would reject every well-formed reply.
    verdict: dict[str, Any] = {}
    if require_code_leak:
      try:
        leak_raw = int(round(float(item["code_leak"])))
      except (KeyError, TypeError, ValueError):
        out["parse_error"] = f"label {label}: missing/non-numeric code_leak"
        return out
      if leak_raw not in (0, 1):
        out["parse_error"] = f"label {label}: code_leak={leak_raw} not in (0,1)"
        return out
      verdict["code_leak"] = leak_raw
    for dim in dims:
      try:
        value, was_clamped = _clamp_dim(item[dim])
      except (KeyError, TypeError, ValueError):
        out["parse_error"] = f"label {label}: missing/non-numeric {dim}"
        return out
      verdict[dim] = value
      clamped_any = clamped_any or was_clamped
    verdict["note"] = str(item.get("note", ""))[:400]
    verdict["total"] = total_score(verdict)
    parsed[label] = verdict

  missing = [x for x in labels if x not in parsed]
  if missing:
    out["parse_error"] = f"missing labels {''.join(missing)}"
    return out

  best = obj.get("best_label")
  best = str(best).strip().upper()[:1] if best is not None else None
  out.update({
      "ok": True,
      "verdicts": parsed,
      "best_label": best if best in labels else None,
      "clamped": clamped_any,
  })
  return out


def score_candidates(
    parsed: dict[str, Any],
    label_to_candidate: dict[str, int],
    n_candidates: int,
) -> dict[str, Any]:
  """Map per-LABEL verdicts back onto candidate indices and aggregate.

  ``judge_incoherent`` compares the judge's own ``best_label`` against
  ``argmax(total)``. It is redundant by construction, which is the point: it
  costs nothing and it is the one free signal that the judge's holistic
  preference and its own rubric arithmetic disagree.

  Args:
    parsed: ``parse_verdicts`` output with ``ok=True``.
    label_to_candidate: from ``build_judge_messages``.
    n_candidates: how many candidates were sent.

  Returns:
    ``{"scores": [verdict|None, ...] indexed by candidate, "best_cand_idx",
    "margin", "judge_incoherent", "judge_best_cand_idx"}``. ``margin`` is the
    gap between the best and second-best totals (0 when K == 1).
  """
  scores: list[Optional[dict[str, Any]]] = [None] * n_candidates
  for label, verdict in parsed["verdicts"].items():
    scores[label_to_candidate[label]] = dict(verdict)

  totals = [
      (v["total"] if v is not None else -1) for v in scores
  ]
  # Tie-break on the LOWEST candidate index so argmax is deterministic; the
  # richer tie-break (judge's own pick, then shortest reply) lives in select().
  best_idx = max(range(n_candidates), key=lambda i: (totals[i], -i))
  ordered = sorted(totals, reverse=True)
  margin = (ordered[0] - ordered[1]) if n_candidates > 1 else 0

  judge_best = parsed.get("best_label")
  judge_best_idx = (
      label_to_candidate.get(judge_best) if judge_best is not None else None
  )
  incoherent = (
      judge_best_idx is not None
      and totals[judge_best_idx] < totals[best_idx]
  )
  return {
      "scores": scores,
      "best_cand_idx": best_idx,
      "margin": margin,
      "judge_incoherent": bool(incoherent),
      "judge_best_cand_idx": judge_best_idx,
  }


# The selection policy the pipeline uses, kept next to the evidence for it.
#
# `select` itself defaults to ranking-only (veto + argmax) so its behaviour is
# never surprising; the POLICY lives here, and callers pass it explicitly, so a
# floor is always a visible choice rather than a buried default.
#
# Measured over the 50-prefix r2 pilot, sweeping the calibration floor:
#     floor   kept   new-leak targets   mean chars   "don't know"
#       0      58%        5 (0.17)          316           0
#       2      36%        1 (0.06)          318           0
#       3      34%        0 (0.00)          312           0
#       4      24%        0 (0.00)          318           0
# where a "new-leak target" introduces a ground-truth constant that was not
# already in the transcript. A floor of 3 takes the new-leak rate to ZERO
# without collapsing reply length (312 vs a 353-char pool mean) and without
# admitting a single "I don't know" -- i.e. it buys safety without buying the
# mute. 4 costs another 10 points of keep rate for nothing measurable.
#
# fidelity>=1 vetoes a target the judge says CONTRADICTS the hidden information.
# It is redundant at calibration>=3 on this pilot (both fidelity-0 cases were
# also calibration-0) and is kept as a cheap standing guard: training the
# simulator to say false things is never right, whatever else it scores.
#
# in_character>=3 closes a hole the calibration floor cannot. A reply that says
# NOTHING is the global optimum of a rubric built around "do not over-release":
# at pilot prefix 12-0-0 the reply "I have written a Python function that
# estimates the peak signal power level..." -- a total role inversion, the USER
# claiming to have written the code -- scored calibration=4 (correctly; it
# reveals nothing) and was SELECTED at 13/16. Nothing in the leak-facing
# dimensions can reject that, because on those dimensions it is genuinely good.
# Measured on the role_restraint arm (87 prefixes, calibration>=3 already on):
#     in_character >=   kept   keep%   analyst-voiced targets
#            0 / 2       53     0.61            0.28
#            3           42     0.48            0.17
#            4           41     0.47            0.15
# 3 is the knee: it removes 40% of the analyst-voiced targets for 13 points of
# keep rate, and 4 buys almost nothing more.
# ── Stage 2: the truth veto ──────────────────────────────────────────────────
# Verdict vocabulary. "unsure" is a veto and not a low score on purpose: a reply
# that withholds what the code settles makes the episode unwinnable for the
# agent, which is not a milder version of over-releasing -- it is a different
# way of destroying the task.
TRUTH_OK = "ok"
TRUTH_WRONG = "wrong"
TRUTH_UNSURE = "unsure"
TRUTH_VERDICTS = (TRUTH_OK, TRUTH_WRONG, TRUTH_UNSURE)


def build_truth_messages(
    prefix: dict[str, Any],
    candidates: list[str],
    perm_seed: int = 0,
) -> tuple[list[dict[str, str]], dict[str, int]]:
  """Build the stage-2 truth-check call for one prefix.

  Permuted with the SAME seed and prefix_id as the ranking call, so a candidate
  wears the same label in both stages and the two records can be read together.

  Args:
    prefix: one ``collect_prefixes`` record.
    candidates: the code-screened survivors to check.
    perm_seed: run-level permutation seed.

  Returns:
    ``(messages, label_to_candidate)``.

  Raises:
    ValueError: ``candidates`` is empty or longer than ``LABELS``.
  """
  if not candidates:
    raise ValueError("build_truth_messages: no candidates")
  if len(candidates) > len(LABELS):
    raise ValueError(
        f"build_truth_messages: {len(candidates)} candidates exceeds the "
        f"{len(LABELS)} available labels"
    )
  order = permute(prefix["prefix_id"], len(candidates), perm_seed)
  block, mapping = build_candidate_block(candidates, order)
  user = prompts.SIM_TRUTH_USER_TEMPLATE.format(
      ground_truth=prefix["ground_truth"],
      problem_description=prefix["problem_description"],
      dialogue_history=templates.str_dialogue_history(prefix["sim_dialogue"]),
      n_candidates=len(candidates),
      candidate_block=block,
  )
  return (
      [
          {"role": "system", "content": prompts.SIM_TRUTH_SYSTEM_PROMPT},
          {"role": "user", "content": user},
      ],
      mapping,
  )


def parse_truth(raw: str, labels: list[str]) -> dict[str, Any]:
  """Parse one stage-2 reply into per-label truth verdicts.

  Same contract as ``parse_verdicts``: tolerant of fence/prose wrapping, STRICT
  about the label set. An unrecognised verdict string is a parse failure rather
  than a silent "ok" -- defaulting a veto stage to "pass" is how a broken stage
  becomes invisible.

  Args:
    raw: the model's reply text.
    labels: the labels that must each appear exactly once.

  Returns:
    ``{"ok", "parse_error", "checks": {label: {"verdict", "quote", "code"}}}``.
  """
  m = _JSON_OBJ_RE.search((raw or "").strip())
  if not m:
    return {"ok": False, "parse_error": "no JSON object found", "checks": {}}
  try:
    obj = json.loads(m.group(0))
  except (json.JSONDecodeError, ValueError) as e:
    return {"ok": False, "parse_error": f"json.loads failed: {e}", "checks": {}}
  if not isinstance(obj, dict):
    return {
        "ok": False,
        "parse_error": f"top level is {type(obj).__name__}, not an object",
        "checks": {},
    }
  rows = obj.get("checks")
  if not isinstance(rows, list):
    return {"ok": False, "parse_error": "no 'checks' array", "checks": {}}
  checks: dict[str, Any] = {}
  for row in rows:
    if not isinstance(row, dict):
      return {"ok": False, "parse_error": "check is not an object", "checks": {}}
    label = str(row.get("label", "")).strip().upper()[:1]
    if label not in labels:
      return {
          "ok": False,
          "parse_error": f"unknown label {label!r}",
          "checks": {},
      }
    if label in checks:
      return {
          "ok": False,
          "parse_error": f"duplicate label {label!r}",
          "checks": {},
      }
    verdict = str(row.get("verdict", "")).strip().lower()
    if verdict not in TRUTH_VERDICTS:
      return {
          "ok": False,
          "parse_error": f"bad verdict {verdict!r} for {label!r}",
          "checks": {},
      }
    checks[label] = {
        "verdict": verdict,
        "quote": str(row.get("quote", ""))[:400],
        "code": str(row.get("code", ""))[:400],
    }
  missing = [l for l in labels if l not in checks]
  if missing:
    return {
        "ok": False,
        "parse_error": f"missing labels {''.join(missing)}",
        "checks": {},
    }
  return {"ok": True, "parse_error": "", "checks": checks}


def truth_flags(
    checks: dict[str, Any],
    label_to_candidate: dict[str, int],
    n_candidates: int,
) -> list[Optional[str]]:
  """Map per-label truth verdicts onto candidate indices.

  Args:
    checks: ``parse_truth`` output's ``checks``.
    label_to_candidate: from ``build_truth_messages``.
    n_candidates: how many candidates were sent.

  Returns:
    A list of verdict strings indexed by candidate, ``None`` where the label
    was not checked (a candidate screened out before stage 2).
  """
  out: list[Optional[str]] = [None] * n_candidates
  for label, row in checks.items():
    out[label_to_candidate[label]] = row["verdict"]
  return out


# r4 drops the `fidelity` floor because the dimension is gone -- it is the
# stage-2 veto now, which is strictly stronger than a floor at 1 was.
DEFAULT_MIN_DIMS = {"calibration": 3, "in_character": 3}

# Drop reasons, in the order they are tested. Recorded per dropped prefix so the
# keep-rate can be attributed instead of guessed at.
DROP_ALL_EMPTY = "all_empty"
DROP_JUDGE_FAILED = "judge_failed"
DROP_ALL_VETOED = "all_vetoed"
# r4 splits the veto in two so the keep-rate can be attributed to the STAGE
# that spent it. "all_vetoed" stays the code veto, so older reports and the
# r1-r3 files keep meaning what they meant.
DROP_ALL_UNTRUE = "all_untrue"
DROP_BELOW_FLOOR = "below_floor"
DROP_BELOW_MARGIN = "below_margin"


def passing_candidates(
    judged: dict[str, Any],
    min_total: int = 0,
    min_dim: int = 0,
    min_margin: int = 0,
    min_dims: Optional[dict[str, int]] = None,
) -> tuple[list[int], str]:
  """Candidates eligible to be an SFT target, or why the group has none.

  ONE definition of eligibility, shared by ``select`` and ``select_all``. It is
  factored out because it was duplicated once and the copy drifted: the first
  ``select_all`` matched tied candidates on ``total`` alone, which promoted a
  candidate that tied on total while FAILING a per-dimension floor. Any future
  caller that needs "which candidates could be trained on" must come here
  rather than re-deriving it.

  Args:
    judged: one ``judge_candidates`` record.
    min_total: floor on the total; 0 disables.
    min_dim: floor on EVERY graded dimension; 0 disables.
    min_margin: floor on the best-vs-second gap; 0 disables. A group under it
      is rejected WHOLE, so this returns no candidates rather than fewer.
    min_dims: per-dimension floors, e.g. ``DEFAULT_MIN_DIMS``.

  Returns:
    ``(indices, "")`` in candidate order, or ``([], reason)`` with the drop
    reason attributed to the stage that actually emptied the group.
  """
  candidates = judged.get("candidates") or []
  if not candidates:
    return [], DROP_ALL_EMPTY
  if not judged.get("ok"):
    return [], DROP_JUDGE_FAILED
  scores = judged.get("scores") or []

  # r4: a candidate reaches `scores` only if it already passed the programmatic
  # code screen AND the stage-2 truth veto, so `scores[i] is None` covers both.
  # `code_leak` is still checked because the ranker scores it as a second
  # opinion and does catch prose leaks the regex cannot see.
  truth = judged.get("truth_flags") or []
  eligible = [
      i
      for i, v in enumerate(scores)
      if v is not None
      and int(v.get("code_leak", 1)) == 1
      and (i >= len(truth) or truth[i] in (None, TRUTH_OK))
  ]
  if not eligible:
    # Attribute the drop to whichever stage actually emptied the group.
    n_code = len(judged.get("code_vetoed") or [])
    if n_code >= len(candidates):
      return [], DROP_ALL_VETOED
    return [], (
        DROP_ALL_UNTRUE
        if any(f in (TRUTH_WRONG, TRUTH_UNSURE) for f in truth)
        else DROP_ALL_VETOED
    )

  floors = dict(min_dims or {})

  def _clears(v: dict[str, Any]) -> bool:
    """Does one verdict clear the total, uniform-dim and per-dim floors?

    Args:
      v: a parsed verdict.

    Returns:
      True when every configured floor is met.
    """
    if v["total"] < min_total:
      return False
    if any(v[d] < min_dim for d in graded_dims_of(v)):
      return False
    # `d in v` and not `.get(d, 0)`: a rubric that never had this dimension
    # must not be silently failed by a floor written for a later one.
    return all(v[d] >= f for d, f in floors.items() if d in v)

  passing = [i for i in eligible if _clears(scores[i])]
  if not passing:
    return [], DROP_BELOW_FLOOR
  if min_margin > 0 and int(judged.get("margin", 0)) < min_margin:
    return [], DROP_BELOW_MARGIN
  return passing, ""


def select(
    judged: dict[str, Any],
    min_total: int = 0,
    min_dim: int = 0,
    min_margin: int = 0,
    min_dims: Optional[dict[str, int]] = None,
) -> tuple[Optional[int], str]:
  """Pick the SFT target for one judged prefix, or drop it with a reason.

  RANKING ONLY, with the leak veto retained: among the candidates that clear
  the veto, take the highest total. Both floors default to 0 (off).

  They defaulted to ``min_total=12, min_dim=2`` under rubric r1, and the pilot
  showed why that was the wrong shape. The floor fired ONCE in 50 prefixes,
  while admitting a turn the judge itself had labelled a partial dump
  (``information_release=2``, total 14) -- so it was simultaneously inert and
  too permissive. A threshold cannot substitute for a rubric that discriminates;
  r2 puts the discrimination in the anchors and lets selection just take the
  best available. The knobs remain because selection is a separate pass over the
  judged JSONL, so re-tuning them costs no judge calls -- use them to MEASURE
  what a floor would buy, not as the primary defence.

  ``min_margin`` defaults to 0 on purpose. A margin gate is a pairwise-
  preference device (it protects a DPO pair from being trained on noise); SFT
  trains on the chosen TEXT alone, so a tie is harmless, and gating on margin
  would discard exactly the prefixes where the base simulator is RELIABLY good.
  ``margin`` is recorded regardless, for a later DPO stage (where >=3 is the
  sane floor).

  THERE IS NO BEST-AVAILABLE FALLBACK. A prefix whose every candidate leaks is
  dropped, not softened -- training on the least-bad leak would reinstate the
  behaviour this pipeline exists to remove.

  Args:
    judged: one ``judge_candidates`` record.
    min_total: floor on the 0..16 total; 0 disables the check.
    min_dim: floor on EVERY graded dimension; 0 disables the check.
    min_dims: PER-DIMENSION floors, e.g. ``DEFAULT_MIN_DIMS``. This is the one
      that matters: the pilot showed the useful floor is on ``calibration``
      alone, because a prefix where every clean draw dumps the specification
      should be DROPPED rather than contribute its least-bad dump. A dimension
      absent from the dict is unconstrained; a dimension absent from the
      RECORD (an older rubric) is ignored rather than treated as failing.
    min_margin: floor on the best-vs-second gap; 0 disables the check.

  Returns:
    ``(candidate_index, "")`` on a keep, or ``(None, reason)`` on a drop.
  """
  passing, reason = passing_candidates(
      judged,
      min_total=min_total,
      min_dim=min_dim,
      min_margin=min_margin,
      min_dims=min_dims,
  )
  if not passing:
    return None, reason
  candidates = judged["candidates"]
  scores = judged["scores"]

  judge_best = judged.get("judge_best_cand_idx")
  best_total = max(scores[i]["total"] for i in passing)
  top = [i for i in passing if scores[i]["total"] == best_total]
  if len(top) == 1:
    return top[0], ""
  # Tie-break: the judge's own holistic pick, then the SHORTEST reply (the sim
  # prompt says "IN TWO SENTENCES", so brevity is the tie-breaking virtue),
  # then the lowest candidate index for determinism.
  if judge_best in top:
    return judge_best, ""
  return min(top, key=lambda i: (len(candidates[i]), i)), ""

r"""Stage 3b: score every candidate reply against the rubric, K in one call.

The metered-API script of this pipeline. It follows
``selfplay/generate_specs.py`` in every respect that costs money -- resume by
written row, ``raise_on_exhausted=True`` so a transient failure is DEFERRED
rather than persisted, a hard ``--max_cost`` cap (exit 4), fatal-abort on a
non-retryable error (exit 3), server-reported usage accounting, buffered flush
and a ``finally``-persist -- because every one of those behaviours was learned
the expensive way on that script.

PARSE-FAILURE POLICY, and why it differs from an HTTP failure: an HTTP failure
is transient, so the row must stay UNWRITTEN and be re-paid on resume. A content
failure (the judge returned something the parser cannot use) is DETERMINISTIC
for that prompt -- deferring it means every resume retries it forever and the
run never converges. So: one content retry at temperature 0, then write the row
with ``ok=false``. Selection treats ``ok=false`` as "no acceptable candidate".

``judge_parse_fail_rate > 0.02`` means the output contract is ambiguous rather
than the model being unlucky: fix the rubric text and bump
``prompts.JUDGE_RUBRIC_VERSION``.

Example (~$16 for 1,800 prefixes; halve it with --service_tier flex):
    python -m colbench.simtrain.judge_candidates \
        --prefixes   $SIMTRAIN_ROOT/gs200/prefixes.train.fence.c1.jsonl \
        --candidates $SIMTRAIN_ROOT/gs200/candidates.train.fence.c1.jsonl \
        --out        $SIMTRAIN_ROOT/gs200/judged.train.fence.c1.r1.jsonl \
        --judge_model gpt-5.4-mini --judge_vendor openai \
        --judge_api_key_file ~/.openai_key \
        --price_in 0.25 --price_out 2.0 --max_cost 40
"""

# pylint: disable=g-importing-member
import argparse
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
from typing import Any
from typing import Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench import prompts
from colbench import templates
from colbench.selfplay.dataio import append_jsonl
from colbench.selfplay.dataio import read_jsonl
from colbench.selfplay.llm_client import ChatCallFailedError
from colbench.selfplay.llm_client import ChatCallFatalError
from colbench.selfplay.llm_client import ChatCallRefusedError
from colbench.selfplay.llm_client import ChatEndpoint
from colbench.simtrain import JUDGE_HARNESS_VERSION
from colbench.simtrain import judge_rubric
from colbench.simtrain.collect_prefixes import read_prefixes


def rubric_version_for(arm: str) -> str:
  """The rubric version stamped on every row and used by the resume guard.

  Resolved in ONE place because it keys the output FILENAME, the mixed-standard
  refusal and the per-row stamp; retyping it is how two rubrics end up in one
  file.

  Args:
    arm: "gt" (rubric r*) or "grounded" (rubric g*).

  Returns:
    The rubric version string.
  """
  return (
      prompts.GROUNDED_JUDGE_RUBRIC_VERSION
      if arm == "grounded"
      else prompts.JUDGE_RUBRIC_VERSION
  )


def judge_one(
    prefix: dict[str, Any],
    cand_rec: dict[str, Any],
    endpoint: ChatEndpoint,
    retry_endpoint: ChatEndpoint,
    perm_seed: int = 0,
    truth_endpoint: Optional[ChatEndpoint] = None,
    truth_retry_endpoint: Optional[ChatEndpoint] = None,
) -> dict[str, Any]:
  """Run the three r4 stages over one prefix's candidates.

  STAGE 1, code veto, PROGRAMMATIC -- ``templates.detect_code_leak``, no API
  call. The judge and the regex agreed on 217/217 candidates in the r2 pilot, so
  paying a model to re-derive it bought nothing; screening first also removes
  ~24% of candidates from both prompts below.

  ``expr_over_gt_names=True`` here and nowhere else. It adds detector (C),
  which catches a function body handed over as a BACKTICKED EXPRESSION rather
  than as a def or a fence -- "the function should return
  ``(num_teams * (new_plan_cost_per_team - pension_cost_per_team)) *
  years_savings``". Measured: 9.0% of base-partner candidates, and the judge's
  own code_leak scored them clean too, so both screens were missing them. It
  stays OFF in ``env.py`` so ``sim_leak_frac`` remains comparable with every
  training run to date (which means those numbers UNDER-count the leak).

  STAGE 2, truth veto, ITS OWN CALL -- one job, evidence required. This is the
  check r3 proved cannot ride along with the ranking dimensions: four of twelve
  r3-selected targets contradicted the hidden code and all four scored
  fidelity=4, because grading truth means reading the code and a model doing
  that as one of five simultaneous judgements skips it.

  STAGE 3, rank, over the SURVIVORS ONLY -- the remaining dimensions. Fewer
  candidates per prompt and no truth dimension to outvote stage 2.

  A stage-2 parse failure fails the row CLOSED (``ok=False``, nothing selected)
  rather than falling through to ranking: defaulting a veto to "pass" is how a
  broken veto becomes invisible.

  Args:
    prefix: the ``collect_prefixes`` record (problem, hidden GT, dialogue).
    cand_rec: the matching ``collect_candidates`` record.
    endpoint: the judge at its sampling temperature.
    retry_endpoint: the SAME judge at temperature 0, used for the one content
      retry. A deterministic re-ask is the only retry that makes sense for a
      parse failure -- re-sampling the same temperature is just a coin flip.
    perm_seed: run-level permutation seed.
    truth_endpoint: the stage-2 checker; defaults to ``endpoint``. Split out so
      the truth stage can run on a STRONGER model than the ranker -- it is the
      stage that needs code comprehension, and it is the cheaper of the two.
    truth_retry_endpoint: stage-2's temperature-0 retry; defaults to
      ``retry_endpoint``.

  Returns:
    The judged JSONL record, written whether or not parsing succeeded.

  Raises:
    ChatCallFailedError: transient API exhaustion; the caller must leave the
      row unwritten so a resume re-pays for it.
    ChatCallFatalError: non-retryable, aborts the batch.
    ChatCallRefusedError: this prompt is permanently refused.
  """
  candidates = cand_rec["candidates"]
  rec: dict[str, Any] = {
      "prefix_id": prefix["prefix_id"],
      "task_index": prefix["task_index"],
      "task_id": prefix["task_id"],
      "episode_idx": prefix["episode_idx"],
      "turn_idx": prefix["turn_idx"],
      "sim_user_sha16": prefix["sim_user_sha16"],
      "candidates": candidates,
      "dup_counts": cand_rec.get("dup_counts", []),
      "n_empty": cand_rec.get("n_empty", 0),
      "rubric_version": prompts.JUDGE_RUBRIC_VERSION,
      "harness_version": JUDGE_HARNESS_VERSION,
      "judge_model": endpoint.model,
      "perm_seed": perm_seed,
  }
  if not candidates:
    # Every draw was empty. Nothing to score and nothing to pay for.
    rec.update({
        "ok": False,
        "parse_error": "no candidates",
        "scores": [],
        "label_to_candidate": {},
        "best_cand_idx": None,
        "margin": 0,
        "judge_incoherent": False,
        "judge_best_cand_idx": None,
        "retried": False,
        "code_vetoed": [],
        "truth_flags": [],
        "truth_checks": {},
        "truth_ok": True,
        "truth_parse_error": "",
    })
    return rec

  truth_endpoint = truth_endpoint or endpoint
  truth_retry_endpoint = truth_retry_endpoint or retry_endpoint
  n = len(candidates)

  # ── Stage 1: code veto, programmatic ──────────────────────────────────────
  gt = prefix["ground_truth"]
  code_vetoed = [
      i for i, c in enumerate(candidates)
      if templates.detect_code_leak(c, gt, expr_over_gt_names=True) is not None
  ]
  rec["code_vetoed"] = code_vetoed
  survivors = [i for i in range(n) if i not in set(code_vetoed)]
  if not survivors:
    # Nothing to check or rank, and nothing to pay for. `ok` is True because
    # the pipeline ran correctly; `select` drops this as all_vetoed.
    rec.update({
        "ok": True,
        "parse_error": "",
        "clamped": False,
        "retried": False,
        "label_to_candidate": {},
        "scores": [None] * n,
        "best_cand_idx": None,
        "margin": 0,
        "judge_incoherent": False,
        "judge_best_cand_idx": None,
        "truth_flags": [None] * n,
        "truth_checks": {},
        "truth_ok": True,
        "truth_parse_error": "",
    })
    return rec

  # ── Stage 2: truth veto, its own focused call ─────────────────────────────
  t_msgs, t_map = judge_rubric.build_truth_messages(
      prefix, [candidates[i] for i in survivors], perm_seed
  )
  t_labels = sorted(t_map)
  t_raw = truth_endpoint.chat(t_msgs)
  t_parsed = judge_rubric.parse_truth(t_raw, t_labels)
  t_retried = False
  if not t_parsed["ok"]:
    t_retried = True
    t_raw = truth_retry_endpoint.chat(t_msgs)
    t_parsed = judge_rubric.parse_truth(t_raw, t_labels)

  rec["truth_ok"] = t_parsed["ok"]
  rec["truth_parse_error"] = t_parsed["parse_error"]
  rec["truth_retried"] = t_retried
  if not t_parsed["ok"]:
    # FAIL CLOSED. Ranking replies whose truth is unknown would silently
    # reinstate exactly the r3 behaviour this stage exists to remove.
    rec.update({
        "ok": False,
        "parse_error": f"truth stage: {t_parsed['parse_error']}",
        "clamped": False,
        "retried": t_retried,
        "label_to_candidate": {},
        "scores": [],
        "best_cand_idx": None,
        "margin": 0,
        "judge_incoherent": False,
        "judge_best_cand_idx": None,
        "truth_flags": [None] * n,
        "truth_checks": {},
        "raw": (t_raw or "")[:4000],
    })
    return rec

  # Map the per-label verdicts back onto ORIGINAL candidate indices.
  local = judge_rubric.truth_flags(t_parsed["checks"], t_map, len(survivors))
  flags: list[Optional[str]] = [None] * n
  for local_i, orig_i in enumerate(survivors):
    flags[orig_i] = local[local_i]
  rec["truth_flags"] = flags
  rec["truth_checks"] = {
      str(survivors[t_map[l]]): v for l, v in t_parsed["checks"].items()
  }

  ranked = [i for i in survivors if flags[i] == judge_rubric.TRUTH_OK]
  if not ranked:
    rec.update({
        "ok": True,
        "parse_error": "",
        "clamped": False,
        "retried": t_retried,
        "label_to_candidate": {},
        "scores": [None] * n,
        "best_cand_idx": None,
        "margin": 0,
        "judge_incoherent": False,
        "judge_best_cand_idx": None,
    })
    return rec

  # ── Stage 3: rank the survivors ───────────────────────────────────────────
  messages, local_map = judge_rubric.build_judge_messages(
      prefix, [candidates[i] for i in ranked], perm_seed
  )
  labels = sorted(local_map)
  raw = endpoint.chat(messages)
  parsed = judge_rubric.parse_verdicts(raw, labels)
  retried = False
  if not parsed["ok"]:
    retried = True
    raw = retry_endpoint.chat(messages)
    parsed = judge_rubric.parse_verdicts(raw, labels)

  mapping = {l: ranked[local_map[l]] for l in local_map}
  rec.update({
      "ok": parsed["ok"],
      "parse_error": parsed["parse_error"],
      "clamped": parsed.get("clamped", False),
      "retried": retried or t_retried,
      "label_to_candidate": mapping,
  })
  if parsed["ok"]:
    scored = judge_rubric.score_candidates(parsed, local_map, len(ranked))
    # Re-index onto the full candidate list so `scores[i]` lines up with
    # `candidates[i]` for every consumer (select, report, dump_judged).
    full: list[Any] = [None] * n
    for local_i, orig_i in enumerate(ranked):
      full[orig_i] = scored["scores"][local_i]
    best = scored["best_cand_idx"]
    jb = scored["judge_best_cand_idx"]
    rec.update({
        "scores": full,
        "best_cand_idx": ranked[best] if best is not None else None,
        "margin": scored["margin"],
        "judge_incoherent": scored["judge_incoherent"],
        "judge_best_cand_idx": ranked[jb] if jb is not None else None,
    })
  else:
    rec.update({
        "scores": [],
        "best_cand_idx": None,
        "margin": 0,
        "judge_incoherent": False,
        "judge_best_cand_idx": None,
        # Only kept on FAILURE: the raw reply is the only way to debug a
        # contract problem, and keeping it on every row would multiply the file
        # size for nothing.
        "raw": (raw or "")[:4000],
    })
  return rec


def judge_one_grounded(
    prefix: dict[str, Any],
    cand_rec: dict[str, Any],
    endpoint: ChatEndpoint,
    retry_endpoint: ChatEndpoint,
    perm_seed: int = 0,
) -> dict[str, Any]:
  """Score one GROUNDED-arm prefix: two programmatic vetoes, then ONE ranking call.

  Shape differs from ``judge_one`` in one structural way: rubric g1 has NO LLM
  veto stage. Both vetoes -- writing code, and ending the conversation before the
  agent has shown a complete function -- are mechanical facts that ``env_spec``
  already enforces during a real rollout, so a model is never asked to re-derive
  them and can never overrule them. That also halves the calls per prefix.

  Args:
    prefix: one grounded ``collect_prefixes`` record.
    cand_rec: the matching ``collect_candidates`` record.
    endpoint: the judge at its sampling temperature.
    retry_endpoint: the SAME judge at temperature 0, for the one content retry.
    perm_seed: run-level permutation seed.

  Returns:
    The judged JSONL record, written whether or not parsing succeeded.

  Raises:
    ChatCallFailedError: transient API exhaustion; the caller must leave the
      row unwritten so a resume re-pays for it.
    ChatCallFatalError: non-retryable, aborts the batch.
    ChatCallRefusedError: this prompt is permanently refused.
  """
  candidates = cand_rec["candidates"]
  n = len(candidates)
  rec: dict[str, Any] = {
      "prefix_id": prefix["prefix_id"],
      "task_index": prefix["task_index"],
      "task_id": prefix["task_id"],
      "episode_idx": prefix["episode_idx"],
      "turn_idx": prefix["turn_idx"],
      "sim_user_sha16": prefix["sim_user_sha16"],
      "candidates": candidates,
      "dup_counts": cand_rec.get("dup_counts", []),
      "n_empty": cand_rec.get("n_empty", 0),
      "rubric_version": prompts.GROUNDED_JUDGE_RUBRIC_VERSION,
      "harness_version": JUDGE_HARNESS_VERSION,
      "judge_model": endpoint.model,
      "perm_seed": perm_seed,
      "sim_conditioning": prefix.get("sim_conditioning", "grounded"),
      # Carried so a reader of the dump can see WHY a termination was vetoed
      # without joining back to the prefix file.
      "terminate_allowed": bool(prefix.get("terminate_allowed", False)),
  }
  _empty = {
      "clamped": False,
      "retried": False,
      "label_to_candidate": {},
      "best_cand_idx": None,
      "margin": 0,
      "judge_incoherent": False,
      "judge_best_cand_idx": None,
  }
  if not candidates:
    rec.update(_empty)
    rec.update({"ok": False, "parse_error": "no candidates", "scores": [],
                "code_vetoed": [], "term_vetoed": [], "form_vetoed": []})
    return rec

  # ── Vetoes: all three PROGRAMMATIC, no API call ───────────────────────────
  code_vetoed, term_vetoed, form_vetoed = judge_rubric.grounded_vetoes(
      prefix, candidates
  )
  rec["code_vetoed"] = code_vetoed
  rec["term_vetoed"] = term_vetoed
  rec["form_vetoed"] = form_vetoed
  vetoed = set(code_vetoed) | set(term_vetoed) | set(form_vetoed)
  ranked = [i for i in range(n) if i not in vetoed]
  if not ranked:
    # Nothing to rank and nothing to pay for. `ok` is True because the pipeline
    # ran correctly; `select` drops this as all_vetoed.
    rec.update(_empty)
    rec.update({"ok": True, "parse_error": "", "scores": [None] * n})
    return rec

  # ── The single ranking call ───────────────────────────────────────────────
  messages, local_map = judge_rubric.build_grounded_judge_messages(
      prefix, [candidates[i] for i in ranked], perm_seed
  )
  labels = sorted(local_map)
  def _parse(text):
    return judge_rubric.parse_verdicts(
        text, labels, dims=judge_rubric.GROUNDED_GRADED_DIMS,
        require_code_leak=False,
    )

  raw = endpoint.chat(messages)
  parsed = _parse(raw)
  retried = False
  if not parsed["ok"]:
    retried = True
    raw = retry_endpoint.chat(messages)
    parsed = _parse(raw)

  rec.update({
      "ok": parsed["ok"],
      "parse_error": parsed["parse_error"],
      "clamped": parsed.get("clamped", False),
      "retried": retried,
      "label_to_candidate": {l: ranked[local_map[l]] for l in local_map},
  })
  if parsed["ok"]:
    scored = judge_rubric.score_candidates(parsed, local_map, len(ranked))
    full: list[Any] = [None] * n
    for local_i, orig_i in enumerate(ranked):
      full[orig_i] = scored["scores"][local_i]
    best = scored["best_cand_idx"]
    jb = scored["judge_best_cand_idx"]
    rec.update({
        "scores": full,
        "best_cand_idx": ranked[best] if best is not None else None,
        "margin": scored["margin"],
        "judge_incoherent": scored["judge_incoherent"],
        "judge_best_cand_idx": ranked[jb] if jb is not None else None,
    })
  else:
    rec.update({
        "scores": [],
        "best_cand_idx": None,
        "margin": 0,
        "judge_incoherent": False,
        "judge_best_cand_idx": None,
        "raw": (raw or "")[:4000],
    })
  return rec


def existing_prefix_ids(
    path: str,
    judge_model: str = "",
    allow_mixed: bool = False,
    rubric_version: str = "",
) -> set[str]:
  """``prefix_id``s already judged, so a resume never re-pays for one.

  Also REFUSES a resume that would mix judges or scoring standards in one file.
  Resume matches on ``prefix_id`` only, so without this check, pointing a
  different judge at an existing ``--out`` silently does one of two harmful
  things: on a COMPLETE file every row reads as "already done" and the new judge
  is never called at all (which looks like perfect agreement), and on a PARTIAL
  file -- the normal state, since a bulk pass stops on the spend cap -- the new
  judge fills in the gaps and the file becomes an undeclared mixture that
  ``build_sft_parquet`` and ``eval_sim_sft`` will then treat as one standard.
  Same argument as the rubric version keying the filename, one level down.

  Args:
    path: the judged JSONL; a missing file reads as empty.
    judge_model: the judge about to run; empty skips the identity check.
    allow_mixed: proceed despite a mismatch (records stay self-describing --
      every row carries its own judge/rubric/harness, so a deliberate mixture
      is still analysable; it just must not happen by accident).

  Returns:
    The set of prefix ids already on disk.

  Raises:
    SystemExit: the file on disk was judged under a different judge model,
      rubric version or harness version.
  """
  rows = [r for r in read_jsonl(path) if "prefix_id" in r]
  rubric_version = rubric_version or prompts.JUDGE_RUBRIC_VERSION
  if rows and not allow_mixed:
    want = (judge_model, rubric_version, JUDGE_HARNESS_VERSION)
    seen = {
        (r.get("judge_model", ""), r.get("rubric_version", ""),
         r.get("harness_version", ""))
        for r in rows
    }
    if judge_model and seen != {want}:
      have = "; ".join(
          f"judge={j or '?'} rubric={r or '?'} harness={h or '?'}"
          for j, r, h in sorted(seen)
      )
      raise SystemExit(
          f"[judge_candidates] {path} holds {len(rows)} rows judged under "
          f"[{have}], but this run is judge={judge_model} "
          f"rubric={rubric_version} "
          f"harness={JUDGE_HARNESS_VERSION}.\n"
          "Resume matches on prefix_id ALONE, so continuing would silently mix "
          "scoring standards in one dataset. Write the new judge to its own "
          "--out (put the judge in the filename), or pass --allow_mixed_judge "
          "if the mixture is deliberate."
      )
  return {r["prefix_id"] for r in rows}


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the judge.

  Returns:
    The configured parser.
  """
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--prefixes", required=True)
  ap.add_argument("--candidates", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument(
      "--max_prefixes",
      type=int,
      default=None,
      help="Judge only the first N prefixes -- the PILOT slice (~50).",
  )
  ap.add_argument(
      "--perm_seed",
      type=int,
      default=0,
      help="Candidate-order permutation seed (recorded on every row).",
  )
  ap.add_argument(
      "--judge_base_url", default="https://api.openai.com/v1"
  )
  ap.add_argument("--judge_model", required=True)
  ap.add_argument(
      "--truth_model",
      default="",
      help=(
          "Model for STAGE 2, the truth veto. Empty = the same as "
          "--judge_model. Split out because stage 2 is the one that has to "
          "read the hidden code, and it is cheaper than the ranker (one "
          "verdict per candidate, no scores), so it is the one worth paying "
          "up for. Always runs at temperature 0."
      ),
  )
  ap.add_argument("--judge_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
  ap.add_argument(
      "--judge_api_key_file",
      default=os.environ.get("GEN_API_KEY_FILE", ""),
      help="Read the key from a file (keeps it out of argv and logs).",
  )
  ap.add_argument(
      "--judge_vendor", choices=["vllm", "openai"], default="openai"
  )
  ap.add_argument(
      "--arm",
      choices=["gt", "grounded"],
      default="gt",
      help="Which rubric to apply. 'gt' = r* (hidden-code sim; LLM truth veto "
      "then rank). 'grounded' = g* (GT source + plot; BOTH vetoes programmatic, "
      "then ONE ranking call on gt_adherence/plot_adherence/not_overhelpful). "
      "Must match the arm the prefixes were collected on.",
  )
  ap.add_argument(
      "--allow_mixed_judge",
      action="store_true",
      help="Permit resuming a judged file written by a DIFFERENT judge model, "
      "rubric or harness version. Off by default because resume matches on "
      "prefix_id alone and would otherwise mix scoring standards silently.",
  )
  ap.add_argument("--temperature", type=float, default=0.0)
  ap.add_argument("--top_p", type=float, default=1.0)
  ap.add_argument("--max_tokens", type=int, default=4096)
  ap.add_argument("--retries", type=int, default=6)
  ap.add_argument(
      "--timeout",
      type=float,
      default=600.0,
      help="Generous on purpose: abandoning a slow generation still pays for "
      "it, then pays again on retry.",
  )
  ap.add_argument("--backoff_base", type=float, default=2.0)
  ap.add_argument("--backoff_cap", type=float, default=60.0)
  ap.add_argument(
      "--service_tier",
      default=None,
      choices=["auto", "default", "flex", "scale", "priority"],
      help="'flex' is ~half price for extra latency; remember to halve "
      "--price_in/--price_out.",
  )
  ap.add_argument("--price_in", type=float, default=0.0)
  ap.add_argument("--price_out", type=float, default=0.0)
  ap.add_argument(
      "--max_cost",
      type=float,
      default=0.0,
      help="HARD spend cap in USD (0 = unlimited); needs both prices.",
  )
  ap.add_argument("--concurrency", type=int, default=16)
  ap.add_argument("--flush_every", type=int, default=20)
  return ap


def main(argv: Optional[list[str]] = None) -> None:
  """Judge every prefix not already present in --out, under a spend cap."""
  args = build_arg_parser().parse_args(argv)

  prefixes = {p["prefix_id"]: p for p in read_prefixes(args.prefixes)}
  cands = {
      c["prefix_id"]: c
      for c in read_jsonl(args.candidates)
      if "prefix_id" in c
  }
  done = existing_prefix_ids(
      args.out, args.judge_model, args.allow_mixed_judge,
      rubric_version_for(args.arm),
  )
  # Deterministic order so --max_prefixes always names the SAME pilot batch,
  # which is the whole point of re-judging one batch across rubric rewordings.
  pairs = [
      (prefixes[pid], cands[pid])
      for pid in sorted(cands, key=lambda k: prefixes[k]["prefix_id"])
      if pid in prefixes
  ]
  missing = [pid for pid in cands if pid not in prefixes]
  if missing:
    raise SystemExit(
        f"[judge_candidates] {len(missing)} candidate rows have no matching "
        f"prefix (first: {missing[0]}). Mismatched files?"
    )
  for p, c in pairs:
    if p["sim_user_sha16"] != c["sim_user_sha16"]:
      raise SystemExit(
          f"[judge_candidates] prefix {p['prefix_id']}: candidates were drawn "
          "from a DIFFERENT prompt than this prefix file holds."
      )
  if args.max_prefixes is not None:
    pairs = pairs[: args.max_prefixes]
  todo = [(p, c) for p, c in pairs if p["prefix_id"] not in done]
  print(
      f"[judge_candidates] arm={args.arm} rubric={rubric_version_for(args.arm)} "
      f"harness={JUDGE_HARNESS_VERSION} judge={args.judge_model}\n"
      f"[judge_candidates] {len(pairs)} prefixes in scope, {len(done)} already "
      f"judged, {len(todo)} to judge -> {args.out}",
      flush=True,
  )
  if not todo:
    return
  if args.max_cost and not (args.price_in or args.price_out):
    raise SystemExit(
        "[judge_candidates] --max_cost needs --price_in/--price_out to mean "
        "anything."
    )

  api_key = args.judge_api_key
  if args.judge_api_key_file:
    with open(
        os.path.expanduser(args.judge_api_key_file), encoding="utf-8"
    ) as f:
      api_key = f.read().strip()
  usage = {
      "attempts": 0,
      "calls": 0,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "reasoning_tokens": 0,
      "cached_tokens": 0,
  }
  common = dict(
      base_url=args.judge_base_url,
      model=args.judge_model,
      api_key=api_key,
      vendor=args.judge_vendor,
      top_p=args.top_p,
      max_tokens=args.max_tokens,
      retries=args.retries,
      backoff_base=args.backoff_base,
      backoff_cap=args.backoff_cap,
      timeout=args.timeout,
      service_tier=args.service_tier,
      usage=usage,
      # A written row counts as done on resume, so a row whose CALL failed must
      # not be persisted -- leave the gap and let a resume re-author it.
      raise_on_exhausted=True,
  )
  endpoint = ChatEndpoint(temperature=args.temperature, **common)
  retry_endpoint = ChatEndpoint(temperature=0.0, **common)
  # Stage 2 may run on a stronger model than the ranker: it is the stage that
  # needs the hidden code actually comprehended, and it is the cheaper of the
  # two (no scores, one verdict per candidate). Defaults to the same model so
  # that switching the PIPELINE and switching the MODEL stay separate
  # experiments -- the r4 split is worth measuring on its own first.
  truth_common = dict(common, model=args.truth_model or args.judge_model)
  truth_endpoint = ChatEndpoint(temperature=0.0, **truth_common)
  truth_retry_endpoint = truth_endpoint

  def _spent() -> float:
    """Dollars billed so far, from the server-reported token counts.

    Returns:
      Cost in USD at the per-million ``--price_in`` / ``--price_out`` rates.
    """
    return (usage["prompt_tokens"] / 1e6) * args.price_in + (
        usage["completion_tokens"] / 1e6
    ) * args.price_out

  t0 = time.time()
  buf: list[dict[str, Any]] = []
  n_done = n_ok = n_defer = n_refused = n_retried = 0
  fatal = None
  over_budget = False
  with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
    def _submit(p, c):
      # The grounded arm has NO LLM veto stage, so it takes neither truth
      # endpoint; both of its vetoes are programmatic.
      if args.arm == "grounded":
        return pool.submit(
            judge_one_grounded, p, c, endpoint, retry_endpoint, args.perm_seed
        )
      return pool.submit(
          judge_one, p, c, endpoint, retry_endpoint, args.perm_seed,
          truth_endpoint, truth_retry_endpoint,
      )

    futs = {_submit(p, c): (p, c) for p, c in todo}
    try:
      for fut in as_completed(futs):
        p, _ = futs[fut]
        try:
          rec = fut.result()
        except ChatCallFatalError as e:
          fatal = e
          for f in futs:
            f.cancel()
          break
        except ChatCallFailedError:
          n_defer += 1  # deliberately unwritten -> retried on the next resume
          continue
        except ChatCallRefusedError as e:
          # Permanent for THIS prompt: RECORD it rather than deferring, or the
          # run can never converge.
          n_refused += 1
          buf.append({
              "prefix_id": p["prefix_id"],
              "task_index": p["task_index"],
              "task_id": p["task_id"],
              "turn_idx": p["turn_idx"],
              "candidates": [],
              "scores": [],
              "ok": False,
              "parse_error": "refused",
              "refused": True,
              "refusal_reason": str(e)[:400],
              "rubric_version": rubric_version_for(args.arm),
              "harness_version": JUDGE_HARNESS_VERSION,
          })
          continue
        buf.append(rec)
        n_done += 1
        n_ok += int(bool(rec.get("ok")))
        n_retried += int(bool(rec.get("retried")))
        if len(buf) >= args.flush_every:
          append_jsonl(args.out, buf)
          buf = []
        if n_done % args.flush_every == 0:
          spent = (
              f", ${_spent():.2f}" if (args.price_in or args.price_out) else ""
          )
          print(
              f"[judge_candidates] {n_done}/{len(todo)} judged ({n_ok} parsed "
              f"ok, {n_retried} needed a temp-0 retry, {n_defer} deferred"
              f"{spent}) in {time.time() - t0:.0f}s",
              flush=True,
          )
        if args.max_cost and _spent() >= args.max_cost:
          over_budget = True
          for f in futs:
            f.cancel()
          break
    finally:
      # Always persist what we already paid for, even on Ctrl-C / crash.
      if buf:
        append_jsonl(args.out, buf)
        buf = []

  fail_rate = 1.0 - (n_ok / n_done) if n_done else 0.0
  print(
      f"[judge_candidates] DONE {n_done} judged, {n_ok} parsed ok, "
      f"{n_defer} deferred, {n_refused} refused, {time.time() - t0:.0f}s "
      f"-> {args.out}\n"
      f"[judge_candidates] judge_parse_fail_rate={fail_rate:.4f} "
      f"({n_retried} rows needed the temp-0 retry)"
  )
  if fail_rate > 0.02:
    print(
        "[judge_candidates] WARNING: judge_parse_fail_rate > 0.02. That is an "
        "AMBIGUOUS OUTPUT CONTRACT, not bad luck -- fix the rubric text in "
        "colbench/prompts.py and bump JUDGE_RUBRIC_VERSION.",
        flush=True,
    )
  if usage["calls"]:
    print(
        f"[judge_candidates] USAGE: {usage['attempts']} requests issued / "
        f"{usage['calls']} returned | prompt {usage['prompt_tokens']:,} "
        f"(cached {usage['cached_tokens']:,}) | completion "
        f"{usage['completion_tokens']:,} "
        f"(reasoning {usage['reasoning_tokens']:,})"
    )
    if args.price_in or args.price_out:
      print(
          f"[judge_candidates] EST COST: ${_spent():.2f} at "
          f"${args.price_in}/M in + ${args.price_out}/M out (completion "
          "tokens INCLUDE hidden reasoning)"
      )
  if n_defer:
    print(
        f"[judge_candidates] {n_defer} rows were NOT written (transient API "
        "errors). Re-run the SAME command to judge only those."
    )
  if over_budget:
    print(
        f"[judge_candidates] SPEND CAP REACHED (${_spent():.2f} >= "
        f"${args.max_cost:.2f}). {n_done} rows saved. Raise --max_cost and "
        "re-run; judged rows are never re-paid for.",
        flush=True,
    )
    # Exit 4 = budget stop. A harness must NOT start another pass: each pass
    # gets a fresh counter, so looping would spend the cap again per pass.
    raise SystemExit(4)
  if fatal is not None:
    print(
        f"[judge_candidates] ABORTED on a non-retryable API error: {fatal}\n"
        f"[judge_candidates] {n_done} rows saved; fix the cause and re-run.",
        flush=True,
    )
    # Exit 3 = "fatal, do not retry".
    raise SystemExit(3)


if __name__ == "__main__":
  main()

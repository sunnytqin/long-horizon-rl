r"""Stage 3a: draw K candidate replies per collected prefix.

No env, no parquet, no solver, no dialogue simulation -- Stage 1 materialized
the rendered simulator prompt, so this script is a flat map over
``prefixes.*.jsonl``: send ``(sim_system, sim_user)`` to the base simulator K
times, finalize each draw exactly as the rollout would, dedupe, write.

Two details that are easy to get wrong and impossible to notice afterwards:

  * Every draw goes through ``env.finalize_sim_reply`` -- the SAME <think>-strip
    and ``SIM_CHAR_LIMIT`` slice the training rollout applies. Reimplementing
    that transform here is exactly the byte-drift the materialized prompt exists
    to prevent, which is why it was promoted to module level in env.py rather
    than copied.
  * Sampling defaults are the PRODUCTION sim config (temp 0.7 / top_p 0.8 /
    top_k 20 / min_p 0, max_tokens 256), so the candidate pool is drawn from the
    same distribution the frozen sim serves during training. Judging a pool
    drawn at some other temperature would answer a question nobody asked.

Dedupe is on the exact finalized string. Duplicates carry no extra information
for the judge but would be paid for K times over, so only uniques are sent on;
``dup_count`` keeps the multiplicity visible (a prefix whose 8 draws collapse to
2 uniques is a low-entropy prefix, and that is worth seeing in the report).

Example:
    python -m colbench.simtrain.collect_candidates \
        --prefixes $SIMTRAIN_ROOT/gs200/prefixes.train.fence.c1.jsonl \
        --out      $SIMTRAIN_ROOT/gs200/candidates.train.fence.c1.jsonl \
        --sim_base_url http://127.0.0.1:30001/v1 --sim_model colbench-sim --k 8
"""

# pylint: disable=g-importing-member
import argparse
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import threading
import time
from typing import Any
from typing import Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench import templates
from colbench.env import finalize_sim_reply
from colbench.selfplay.dataio import append_jsonl
from colbench.selfplay.dataio import read_jsonl
from colbench.selfplay.llm_client import ChatEndpoint
from colbench.simtrain import COLLECTOR_VERSION
from colbench.simtrain.collect_prefixes import read_prefixes
from colbench.simtrain.collect_prefixes import require_sim_char_limit
from colbench.simtrain.collect_prefixes import sha16

_WRITE_LOCK = threading.Lock()

# Replies that carry no signal and must not reach the judge. "No response." is
# ``env.openai_sim_backend``'s retry-exhausted fallback, i.e. an infrastructure
# artifact rather than a simulator behaviour; judging it would score the network.
_DEAD_REPLIES = ("", "No response.")


# Generation-side system-prompt variants for candidate drawing. The prefix's
# recorded `sim_system` (the production "You are a helpful assistant.") is the
# default and the only one the served simulator ever sees; the others are
# TEACHERS whose replies get trained against the production prompt. See the
# comment block above SIM_ROLE_SYSTEM_PROMPT in colbench/prompts.py.
SIM_SYSTEM_VARIANTS = ("default", "role", "role_restraint")


def is_dead(reply: str) -> bool:
  """Is this draw an empty/fallback reply that must not be judged?

  Args:
    reply: a finalized candidate.

  Returns:
    True for an empty (or whitespace-only) reply and for the sim backend's
    ``"No response."`` fallback.
  """
  return (reply or "").strip() in _DEAD_REPLIES


def draw_candidates(
    prefix: dict[str, Any],
    endpoint: ChatEndpoint,
    k: int,
    system_variant: str = "default",
) -> dict[str, Any]:
  """Draw, finalize and dedupe K replies for one prefix.

  Args:
    prefix: one ``collect_prefixes`` record.
    endpoint: the base simulator.
    k: draws to take.
    system_variant: key into ``SIM_SYSTEM_VARIANTS``. ``"default"`` sends the
      prefix's recorded ``sim_system`` verbatim, which is what the served
      simulator uses. Anything else replaces ONLY the system message -- the
      user message, carrying the problem, the hidden GT and the dialogue, is
      always the materialized one, so the candidates still answer the exact
      prompt the production simulator was asked.

  Returns:
    A candidates record: ``{prefix_id, candidates, dup_counts, n_draws,
    n_empty, n_unique, sim_system_variant, ...}``. ``candidates`` holds the
    unique, non-dead replies in first-seen order.
  """
  # "default" sends the prefix's OWN recorded system string rather than
  # re-resolving it, so a prefix collected under some other arm still replays
  # exactly what it was asked.
  override = (
      None
      if system_variant == "default"
      else templates.resolve_sim_system(system_variant)
  )
  messages = [
      {"role": "system", "content": override or prefix["sim_system"]},
      {"role": "user", "content": prefix["sim_user"]},
  ]
  uniques: list[str] = []
  dup_counts: list[int] = []
  seen: dict[str, int] = {}
  n_empty = 0
  for _ in range(k):
    reply = finalize_sim_reply(endpoint.chat(messages))
    if is_dead(reply):
      n_empty += 1
      continue
    if reply in seen:
      dup_counts[seen[reply]] += 1
      continue
    seen[reply] = len(uniques)
    uniques.append(reply)
    dup_counts.append(1)
  return {
      "prefix_id": prefix["prefix_id"],
      "task_index": prefix["task_index"],
      "task_id": prefix["task_id"],
      "episode_idx": prefix["episode_idx"],
      "turn_idx": prefix["turn_idx"],
      # Fingerprint of the prompt these were drawn from. Downstream stages
      # assert against it, so a candidates file can never be paired with a
      # prefix file it was not drawn from.
      "sim_user_sha16": prefix["sim_user_sha16"],
      "candidates": uniques,
      "dup_counts": dup_counts,
      "n_draws": k,
      "n_empty": n_empty,
      "n_unique": len(uniques),
      "sim_system_variant": system_variant,
      "collector_version": COLLECTOR_VERSION,
  }


def existing_prefix_ids(path: str) -> set[str]:
  """``prefix_id``s already drawn for, so a resume pays for nothing twice.

  Args:
    path: the candidates JSONL; a missing file reads as empty.

  Returns:
    The set of prefix ids already on disk.
  """
  return {r["prefix_id"] for r in read_jsonl(path) if "prefix_id" in r}


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the candidate collector.

  Returns:
    The configured parser.
  """
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--prefixes", required=True, help="prefixes.*.jsonl.")
  ap.add_argument("--out", required=True, help="candidates.*.jsonl.")
  ap.add_argument(
      "--k", type=int, default=8, help="Draws per prefix (before dedupe)."
  )
  ap.add_argument(
      "--max_prefixes",
      type=int,
      default=None,
      help="Take only the first N prefixes (pilot slice).",
  )
  ap.add_argument("--sim_base_url", required=True)
  ap.add_argument("--sim_model", required=True)
  ap.add_argument("--sim_api_key", default="EMPTY")
  ap.add_argument("--sim_temperature", type=float, default=0.7)
  ap.add_argument("--sim_top_p", type=float, default=0.8)
  ap.add_argument("--sim_top_k", type=int, default=20)
  ap.add_argument("--sim_min_p", type=float, default=0.0)
  ap.add_argument("--sim_max_tokens", type=int, default=256)
  ap.add_argument(
      "--sim_system",
      choices=list(SIM_SYSTEM_VARIANTS),
      default="default",
      help="System-prompt variant for GENERATION only. 'default' is the "
      "production prompt the served sim uses; 'role'/'role_restraint' are "
      "teachers whose replies are trained against the production prompt "
      "(context distillation). Recorded on every row.",
  )
  ap.add_argument("--vendor", choices=["vllm", "openai"], default="vllm")
  ap.add_argument("--timeout", type=float, default=300.0)
  ap.add_argument("--retries", type=int, default=3)
  ap.add_argument(
      "--concurrency",
      type=int,
      default=8,
      help="Parallel PREFIXES; each one issues --k sequential draws.",
  )
  ap.add_argument("--flush_every", type=int, default=20)
  return ap


def main(argv: Optional[list[str]] = None) -> None:
  """Draw candidates for every prefix not already present in --out."""
  args = build_arg_parser().parse_args(argv)
  char_limit = require_sim_char_limit()

  prefixes = read_prefixes(args.prefixes)
  if args.max_prefixes is not None:
    prefixes = prefixes[: args.max_prefixes]
  done = existing_prefix_ids(args.out)
  todo = [p for p in prefixes if p["prefix_id"] not in done]
  print(
      f"[collect_candidates] {len(prefixes)} prefixes, {len(done)} already "
      f"drawn, {len(todo)} to draw x k={args.k}\n"
      f"[collect_candidates] SIM_CHAR_LIMIT={char_limit} "
      f"sim_system={args.sim_system} -> {args.out}",
      flush=True,
  )
  if not todo:
    return
  # A prompt fingerprint mismatch means the prefix file was rewritten under a
  # resume. Catch it here rather than letting judged scores attach to the wrong
  # prompt.
  for p in todo:
    # The role/role_restraint teachers are GT-PATH prompts: they carry no GT
    # source and no plot. Applying one to a spec/grounded prefix would silently
    # replace the simulator's entire conditioning and draw candidates from a
    # model that cannot see what it is supposed to be answering from -- garbage
    # that still looks like a valid candidate file.
    conditioning = p.get("sim_conditioning", "gt")
    if conditioning != "gt" and args.sim_system != "default":
      raise SystemExit(
          f"[collect_candidates] prefix {p['prefix_id']} was collected on the "
          f"'{conditioning}' arm, but --sim_system={args.sim_system} is a "
          "GT-path teacher prompt with no ground truth and no plot in it. "
          "Use --sim_system default, which replays the prefix's own recorded "
          "system string."
      )
    if sha16(p["sim_user"]) != p["sim_user_sha16"]:
      raise SystemExit(
          f"[collect_candidates] prefix {p['prefix_id']}: sim_user_sha16 does "
          "not match sim_user -- the prefix file has been modified."
      )

  endpoint = ChatEndpoint(
      base_url=args.sim_base_url,
      model=args.sim_model,
      api_key=args.sim_api_key,
      vendor=args.vendor,
      temperature=args.sim_temperature,
      top_p=args.sim_top_p,
      top_k=args.sim_top_k,
      min_p=args.sim_min_p,
      max_tokens=args.sim_max_tokens,
      retries=args.retries,
      timeout=args.timeout,
  )
  provenance = {
      "sim_model": args.sim_model,
      "sim_system_variant": args.sim_system,
      "sim_sampling": {
          "temperature": args.sim_temperature,
          "top_p": args.sim_top_p,
          "top_k": args.sim_top_k,
          "min_p": args.sim_min_p,
          "max_tokens": args.sim_max_tokens,
      },
      "sim_char_limit": char_limit,
  }

  t0 = time.time()
  buf: list[dict[str, Any]] = []
  n_done = n_uniq = n_empty = 0
  with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
    futs = {
        pool.submit(draw_candidates, p, endpoint, args.k, args.sim_system): p
        for p in todo
    }
    try:
      for fut in as_completed(futs):
        p = futs[fut]
        try:
          rec = fut.result()
        except Exception as e:  # pylint: disable=broad-exception-caught
          print(
              f"[collect_candidates] prefix {p['prefix_id']} FAILED, deferred "
              f"to resume: {e!r}",
              flush=True,
          )
          continue
        rec.update(provenance)
        with _WRITE_LOCK:
          buf.append(rec)
          n_done += 1
          n_uniq += rec["n_unique"]
          n_empty += rec["n_empty"]
          if len(buf) >= args.flush_every:
            append_jsonl(args.out, buf)
            buf = []
        if n_done % (args.flush_every * 5) == 0:
          print(
              f"[collect_candidates] {n_done}/{len(todo)} prefixes, "
              f"{n_uniq / max(1, n_done):.2f} uniques/prefix, {n_empty} dead "
              f"draws, {time.time() - t0:.0f}s",
              flush=True,
          )
    finally:
      # Always persist what has already been generated, even on Ctrl-C.
      with _WRITE_LOCK:
        if buf:
          append_jsonl(args.out, buf)

  print(
      f"[collect_candidates] DONE {n_done} prefixes, "
      f"{n_uniq / max(1, n_done):.2f} uniques/prefix (k={args.k}), {n_empty} "
      f"dead draws, {time.time() - t0:.0f}s -> {args.out}",
      flush=True,
  )


if __name__ == "__main__":
  main()

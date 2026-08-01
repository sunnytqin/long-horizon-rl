r"""Author one natural-language spec per ColBench task, OFFLINE.

Phase-0, Deliverable 1.

For each task the spec author sees the public (under-specified) problem AND the
hidden GT code, and writes a persona + scenario + complete requirements spec
(see ``spec_templates``). The author is any OpenAI-compatible endpoint:

  * ``--backend strong``   -- an external teacher model (faithfulness ceiling).
  * ``--backend selfplay`` -- the trained model's FROZEN BASE checkpoint (no
    external model).

``--backend`` is a label recorded on every row and, by default, embedded in the
output path so the strong / self-gen caches never collide. Generation is
concurrent and RESUMABLE: rows already present in the output JSONL are skipped,
so re-running continues where it left off.

Example (self-play against a served frozen base):
    python -m colbench.selfplay.generate_specs \
        --data_file ~/data/colbench/train.parquet --max_rows 100 \
        --backend selfplay \
        --gen_base_url http://127.0.0.1:30000/v1 --gen_model colbench-base \
        --out ~/data/colbench/specs/train.selfplay.jsonl
"""

# This tree imports names directly (``from colbench.env import
# ColBenchUserSimEnv``) rather than the enclosing module, matching how the
# rest of verl is written; call sites read on the bare name throughout.
# pylint: disable=g-importing-member
import argparse
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
from typing import Any

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# The module-level setup above (env vars, sys.path) has to run
# before these imports resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench.selfplay import spec_templates
from colbench.selfplay.dataio import append_jsonl
from colbench.selfplay.dataio import existing_indices
from colbench.selfplay.dataio import read_tasks
from colbench.selfplay.llm_client import ChatCallFailedError
from colbench.selfplay.llm_client import ChatCallFatalError
from colbench.selfplay.llm_client import ChatCallRefusedError
from colbench.selfplay.llm_client import ChatEndpoint


def _author_one(
    endpoint: ChatEndpoint,
    task: dict[str, Any],
    backend_label: str,
    mode: str = "static",
) -> dict[str, Any]:
  """Author a spec for one task.

  Returns a JSONL record (always, even on failure).

  ``mode="static"``  -> persona/scenario/requirements (the complete-requirements
  spec). ``mode="plot"`` -> persona/scenario/requirements (the full intent the
  simulator must convey) plus ``plot`` (a tailored, high-level direction for how
  this conversation naturally unfolds; the simulator improvises the actual turns
  from it -- it is NOT a turn-by-turn script).

  Args:
    endpoint: the author model's endpoint.
    task: ``{index, problem_description, ground_truth, test_cases}``.
    backend_label: recorded on the row so specs from different authors stay
      distinguishable.
    mode: ``"static"`` or ``"plot"``, as described above.

  Returns:
    The JSONL record for this task, written even when authoring failed
    (``ok=False``).
  """
  rec = {
      "index": task["index"],
      "backend": backend_label,
      "mode": mode,
      "problem_description": task["problem_description"],
  }
  if mode == "plot":
    raw = endpoint.chat(
        spec_templates.build_plot_author_messages(
            task["problem_description"], task["ground_truth"]
        )
    )
    spec = spec_templates.parse_plot_spec(raw)
    rec.update(
        {
            "persona": spec["persona"],
            "scenario": spec["scenario"],
            "requirements": spec["requirements"],
            "plot": spec["plot"],
            "ok": spec["ok"],
            "raw": spec["raw"],
        }
    )
  else:
    raw = endpoint.chat(
        spec_templates.build_author_messages(
            task["problem_description"], task["ground_truth"]
        )
    )
    spec = spec_templates.parse_spec(raw)
    rec.update(
        {
            "persona": spec["persona"],
            "scenario": spec["scenario"],
            "requirements": spec["requirements"],
            "ok": spec["ok"],
            "raw": spec["raw"],
        }
    )
  return rec


def _default_out(data_file: str, backend: str, mode: str = "static") -> str:
  """Default spec-cache path for a run, derived from its inputs.

  Args:
    data_file: the source parquet; its stem and directory seed the path.
    backend: the authoring backend label, e.g. ``"selfplay"``.
    mode: authoring mode; anything but ``"static"`` is appended to the tag so
      static and plot specs never share a cache file.

  Returns:
    ``<data_file dir>/specs/<stem>.<tag>.jsonl``.
  """
  stem = os.path.splitext(os.path.basename(data_file))[0]
  d = os.path.join(
      os.path.dirname(os.path.abspath(os.path.expanduser(data_file))), "specs"
  )
  tag = backend if mode == "static" else f"{backend}.{mode}"
  return os.path.join(d, f"{stem}.{tag}.jsonl")


def main():
  """Author specs for every task in the parquet, resuming any partial run."""
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument(
      "--data_file",
      default=os.path.expanduser("~/data/colbench/train.parquet"),
      help="ColBench parquet (raw InfoPO or preprocessed schema).",
  )
  ap.add_argument(
      "--max_rows", type=int, default=None, help="Limit #tasks (debug slice)."
  )
  ap.add_argument(
      "--backend",
      choices=["strong", "selfplay"],
      required=True,
      help="Author role label (also picks the default output path).",
  )
  ap.add_argument(
      "--mode",
      choices=["static", "plot"],
      default="static",
      help="static: complete-requirements spec. plot: requirements + a"
      " tailored, "
      "high-level plot the simulator improvises around (not a script).",
  )
  ap.add_argument(
      "--out",
      default=None,
      help="Output JSONL (default:"
      " <data_dir>/specs/<stem>.<backend>[.plot].jsonl).",
  )
  # Author endpoint.
  ap.add_argument(
      "--gen_base_url",
      default=os.environ.get("GEN_BASE_URL", "http://127.0.0.1:30000/v1"),
  )
  ap.add_argument("--gen_model", default=os.environ.get("GEN_MODEL", ""))
  ap.add_argument(
      "--gen_api_key", default=os.environ.get("GEN_API_KEY", "EMPTY")
  )
  ap.add_argument(
      "--gen_api_key_file",
      default=os.environ.get("GEN_API_KEY_FILE", ""),
      help=(
          "Read the API key from this file (keeps it out of argv/logs)."
          " Overrides "
          "--gen_api_key."
      ),
  )
  ap.add_argument(
      "--gen_vendor",
      choices=["vllm", "openai"],
      default=os.environ.get("GEN_VENDOR", "vllm"),
      help=(
          "'vllm' local server (top_k/min_p extras) or 'openai' vanilla API "
          "(no extras)."
      ),
  )
  ap.add_argument(
      "--temperature",
      type=float,
      default=0.7,
      help="Author sampling temperature (diversity).",
  )
  ap.add_argument("--top_p", type=float, default=0.8)
  ap.add_argument("--top_k", type=int, default=20)
  ap.add_argument("--min_p", type=float, default=0.0)
  ap.add_argument("--max_tokens", type=int, default=4096)
  ap.add_argument(
      "--enable_thinking",
      choices=["true", "false"],
      default=None,
      help="Set the SGLang enable_thinking kwarg (default: send nothing).",
  )
  ap.add_argument(
      "--retries",
      type=int,
      default=3,
      help="Attempts per row before the row is DEFERRED (left unwritten,"
      " retried "
      "on "
      "resume). Raise it for a rate-limited metered API, e.g. 8.",
  )
  ap.add_argument(
      "--backoff_base",
      type=float,
      default=2.0,
      help=(
          "Backoff seconds: uniform(0, min(cap, base*2**attempt)); Retry-After "
          "wins."
      ),
  )
  ap.add_argument(
      "--backoff_cap", type=float, default=60.0, help="Max backoff sleep (s)."
  )
  ap.add_argument(
      "--price_in",
      type=float,
      default=0.0,
      help=(
          "USD per 1M input tokens; set it to get a cost line in the run "
          "summary."
      ),
  )
  ap.add_argument(
      "--price_out", type=float, default=0.0, help="USD per 1M output tokens."
  )
  ap.add_argument(
      "--max_cost",
      type=float,
      default=0.0,
      help="HARD SPEND CAP in USD (0 = unlimited). Uses server-reported usage "
      "and "
      "--price_in/--price_out; stops the run once exceeded, keeping every row "
      "already authored. Needs both prices to be meaningful.",
  )
  ap.add_argument(
      "--service_tier",
      default=None,
      choices=["auto", "default", "flex", "scale", "priority"],
      help="OpenAI service tier. 'flex' is ~half price for extra latency, "
      "which fits "
      "offline authoring; it needs a LONG --timeout and retries 429 "
      "resource_unavailable (capacity) errors. Remember to halve"
      " --price_in/out.",
  )
  ap.add_argument(
      "--timeout",
      type=float,
      default=600.0,
      help="Per-request timeout (s). On a metered API keep this GENEROUS: "
      "abandoning "
      "a slow generation still pays for it, then pays again on retry.",
  )
  ap.add_argument(
      "--concurrency", type=int, default=16, help="Parallel author calls."
  )
  ap.add_argument(
      "--flush_every",
      type=int,
      default=20,
      help="Append to disk every N completions.",
  )
  args = ap.parse_args()

  out = os.path.expanduser(
      args.out or _default_out(args.data_file, args.backend, args.mode)
  )
  tasks = read_tasks(args.data_file, args.max_rows)
  done = existing_indices(out)
  todo = [t for t in tasks if t["index"] not in done]
  print(
      f"[generate_specs] mode={args.mode} {len(tasks)} tasks, {len(done)} "
      f"already done, "
      f"{len(todo)} to author -> {out}",
      flush=True,
  )
  if not todo:
    return

  # Server-reported token usage for THIS process, so a metered run reports what
  # it actually spent instead of leaving it to be reconstructed from a billing
  # dashboard afterwards.
  usage = {
      "attempts": 0,
      "calls": 0,
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "reasoning_tokens": 0,
      "cached_tokens": 0,
  }

  api_key = args.gen_api_key
  if args.gen_api_key_file:
    with open(os.path.expanduser(args.gen_api_key_file), encoding="utf-8") as f:
      api_key = f.read().strip()
  endpoint = ChatEndpoint(
      base_url=args.gen_base_url,
      model=args.gen_model,
      api_key=api_key,
      vendor=args.gen_vendor,
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=args.top_k,
      min_p=args.min_p,
      max_tokens=args.max_tokens,
      enable_thinking=None
      if args.enable_thinking is None
      else (args.enable_thinking == "true"),
      retries=args.retries,
      backoff_base=args.backoff_base,
      backoff_cap=args.backoff_cap,
      timeout=args.timeout,
      service_tier=args.service_tier,
      usage=usage,
      # A row that reaches disk counts as DONE on resume, so on a metered API we
      # must NOT persist a row whose call failed -- leave the gap and let a
      # resume re-author it.
      raise_on_exhausted=True,
  )

  if args.max_cost and not (args.price_in or args.price_out):
    raise SystemExit(
        (
            "[generate_specs] --max_cost needs --price_in/--price_out to mean "
            "anything."
        )
    )

  def _spent() -> float:
    """Dollars billed so far, from the server-reported token counts.

    Returns:
      Cost in USD at the per-million ``--price_in`` / ``--price_out`` rates.
    """
    return (usage["prompt_tokens"] / 1e6) * args.price_in + (
        usage["completion_tokens"] / 1e6
    ) * args.price_out

  t0 = time.time()
  buf, n_done, n_ok, n_defer, n_refused = [], 0, 0, 0, 0
  fatal = None
  over_budget = False
  with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
    futs = {
        pool.submit(_author_one, endpoint, t, args.backend, args.mode): t
        for t in todo
    }
    try:
      for fut in as_completed(futs):
        try:
          rec = fut.result()
        except ChatCallFatalError as e:
          # Bad key / no model access: every remaining row would fail the same
          # way.
          fatal = e
          for f in futs:
            f.cancel()
          break
        except ChatCallFailedError:
          n_defer += 1  # deliberately unwritten -> retried on the next resume
          continue
        except ChatCallRefusedError as e:
          # Content-safety refusal of this specific prompt: permanent, so RECORD
          # it (ok=False) rather than deferring. A deferred row is retried on
          # every resume, which would stop the run from ever converging.
          # preprocess drops ok=False rows.
          task = futs[fut]
          n_refused += 1
          buf.append(
              {
                  "index": task["index"],
                  "backend": args.backend,
                  "mode": args.mode,
                  "problem_description": task["problem_description"],
                  "persona": {},
                  "scenario": "",
                  "requirements": "",
                  "plot": "",
                  "ok": False,
                  "refused": True,
                  "refusal_reason": str(e)[:400],
                  "raw": "",
              }
          )
          if len(buf) >= args.flush_every:
            append_jsonl(out, buf)
            buf = []
          continue
        buf.append(rec)
        n_done += 1
        n_ok += int(rec["ok"])
        if len(buf) >= args.flush_every:
          append_jsonl(out, buf)
          buf = []
        if n_done % args.flush_every == 0:
          spent = (
              f", ${_spent():.2f}" if (args.price_in or args.price_out) else ""
          )
          print(
              f"[generate_specs] {n_done}/{len(todo)} authored ({n_ok} parsed "
              f"ok, "
              f"{n_defer} deferred, {n_refused} refused{spent}) in"
              f" {time.time() - t0:.0f}s",
              flush=True,
          )
        # Hard spend cap: stop as soon as the server-reported usage says we hit
        # it.
        if args.max_cost and _spent() >= args.max_cost:
          over_budget = True
          for f in futs:
            f.cancel()
          break
    finally:
      # Always persist what we already paid for, even on Ctrl-C / cancellation /
      # crash.
      if buf:
        append_jsonl(out, buf)
        buf = []
  print(
      f"[generate_specs] DONE {n_done} authored, {n_ok} parsed ok, {n_defer} "
      f"deferred, "
      f"{n_refused} refused-by-safety-filter, {time.time() - t0:.0f}s -> {out}"
  )
  if usage["calls"]:
    # BILLED calls, not authored rows: retried/timed-out attempts are billed
    # too, so this is the number that reconciles with an invoice.
    per_row = usage["attempts"] / max(1, n_done)
    print(
        f"[generate_specs] USAGE: {usage['attempts']} requests issued / "
        f"{usage['calls']} "
        f"returned ({per_row:.2f} requests per authored row -- >1.2 means "
        f"you are paying for "
        f"discarded generations) | prompt {usage['prompt_tokens']:,} "
        f"(cached {usage['cached_tokens']:,}) | completion "
        f"{usage['completion_tokens']:,} (reasoning"
        f" {usage['reasoning_tokens']:,})"
    )
    if args.price_in or args.price_out:
      cost = (usage["prompt_tokens"] / 1e6) * args.price_in + (
          usage["completion_tokens"] / 1e6
      ) * args.price_out
      if usage["attempts"] > usage["calls"]:
        lost = usage["attempts"] - usage["calls"]
        print(
            f"[generate_specs] WARNING: {lost} request(s) issued but never"
            f" returned "
            f"a "
            f"usage record (timeout/discarded). Those are billed but NOT in "
            f"the estimate "
            f"below -- raise --timeout rather than abandoning generations."
        )
      print(
          f"[generate_specs] EST COST: ${cost:.2f} at ${args.price_in}/M in + "
          f"${args.price_out}/M out (completion tokens INCLUDE hidden"
          f" reasoning)"
      )
  if n_defer:
    print(
        f"[generate_specs] {n_defer} rows were NOT written (transient API "
        f"errors). "
        f"Re-run the SAME command to author only those."
    )
  if over_budget:
    print(
        f"[generate_specs] SPEND CAP REACHED (${_spent():.2f} >="
        f" ${args.max_cost:.2f}). "
        f"{n_done} rows saved to {out}. Raise --max_cost and re-run to"
        f" continue; "
        f"already "
        f"authored rows are never re-paid for.",
        flush=True,
    )
    # Exit 4 = budget stop. The harness must NOT start another pass: each pass
    # gets a fresh counter, so looping would spend the cap again per pass.
    raise SystemExit(4)
  if fatal is not None:
    print(
        f"[generate_specs] ABORTED on a non-retryable API error: {fatal}\n"
        f"[generate_specs] {n_done} rows saved to {out}; fix the cause (e.g. "
        f"add credits) "
        f"and re-run the SAME command to author only what is missing.",
        flush=True,
    )
    # Exit 3 = "fatal, do not retry": the harness uses it to stop its mop-up
    # passes instead of re-running a command that cannot possibly succeed.
    raise SystemExit(3)


if __name__ == "__main__":
  main()

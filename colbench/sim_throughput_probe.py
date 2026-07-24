#!/usr/bin/env python3
"""Standalone throughput/health smoke test for the frozen ColBench user-simulator server.

Invoked by xcloud_setup/entrypoint_colbench.sh in SIM_SERVER_ONLY + SIM_SMOKE=True mode, AFTER
the SGLang server is healthy. It fires a concurrency sweep of chat-completion requests at the
LOCAL server (127.0.0.1, network-free -> a pure model-serving number) and prints aggregate
completion-token throughput + per-request latency percentiles. Use it to size a large sim (e.g.
Qwen3-235B-A22B vs Llama-3.3-70B) BEFORE committing to a full training run.

Deliberately depends only on `openai` (already a colbench dependency) + stdlib, so it runs in the
training container without extra installs and without importing sglang. Non-fatal by contract:
the entrypoint tolerates a non-zero exit and leaves the server up for manual inspection.
"""
import argparse
import asyncio
import os
import statistics
import time

from openai import AsyncOpenAI

# A sim turn is a smallish system+context prompt -> a short natural-language reply. We approximate
# the prompt length by padding filler text to a target token count (~4 chars/token is close enough
# for a sizing probe; exact tokenization does not change the throughput picture).
_FILLER_SENTENCE = "The user is collaborating with an assistant to refine a solution. "


def _make_prompt(input_tokens: int) -> str:
    target_chars = max(0, input_tokens - 32) * 4
    reps = max(1, target_chars // len(_FILLER_SENTENCE))
    return (
        _FILLER_SENTENCE * reps
        + "\n\nGiven the discussion so far, give one concise piece of feedback."
    )


async def _one_request(client, model, prompt, output_tokens, temperature):
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_tokens,
        temperature=temperature,
    )
    dt = time.perf_counter() - t0
    usage = getattr(resp, "usage", None)
    completion_toks = getattr(usage, "completion_tokens", 0) or 0
    return dt, completion_toks


async def _run_level(client, model, prompt, output_tokens, temperature, concurrency, num_prompts):
    sem = asyncio.Semaphore(concurrency)
    results = []
    errors = 0

    async def _bounded():
        nonlocal errors
        async with sem:
            try:
                results.append(await _one_request(client, model, prompt, output_tokens, temperature))
            except Exception as e:  # noqa: BLE001 - a probe should report, not crash
                errors += 1
                if errors <= 3:
                    print(f"      request error: {type(e).__name__}: {e}")

    wall0 = time.perf_counter()
    await asyncio.gather(*[_bounded() for _ in range(num_prompts)])
    wall = time.perf_counter() - wall0

    if not results:
        return {"concurrency": concurrency, "ok": 0, "errors": errors, "wall": wall}
    lats = sorted(r[0] for r in results)
    total_out = sum(r[1] for r in results)
    return {
        "concurrency": concurrency,
        "ok": len(results),
        "errors": errors,
        "wall": wall,
        "out_toks_per_s": total_out / wall if wall > 0 else 0.0,
        "req_per_s": len(results) / wall if wall > 0 else 0.0,
        "p50_ms": statistics.median(lats) * 1000,
        "p95_ms": lats[min(len(lats) - 1, int(0.95 * len(lats)))] * 1000,
    }


async def _main_async(args):
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30000/v1")
    model = os.environ.get("MULTITURN_MODEL_NAME", "colbench-sim")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)
    prompt = _make_prompt(args.input_tokens)

    print("=" * 78)
    print(f"Sim throughput probe -> {base_url} (model={model})")
    print(f"input~{args.input_tokens} tok, output={args.output_tokens} tok, temp={args.temperature}")
    print("=" * 78)
    header = f"{'conc':>5} {'ok':>5} {'err':>4} {'wall_s':>8} {'out_tok/s':>10} {'req/s':>7} {'p50_ms':>9} {'p95_ms':>9}"
    print(header)
    print("-" * len(header))
    for concurrency in args.concurrency:
        num_prompts = max(concurrency * args.prompts_per_conc, concurrency)
        r = await _run_level(
            client, model, prompt, args.output_tokens, args.temperature, concurrency, num_prompts
        )
        if r.get("ok"):
            print(
                f"{r['concurrency']:>5} {r['ok']:>5} {r['errors']:>4} {r['wall']:>8.1f} "
                f"{r['out_toks_per_s']:>10.1f} {r['req_per_s']:>7.2f} {r['p50_ms']:>9.0f} {r['p95_ms']:>9.0f}"
            )
        else:
            print(f"{r['concurrency']:>5} {'0':>5} {r['errors']:>4} {r['wall']:>8.1f}  (all failed)")
    print("=" * 78)


def _parse_int_list(s):
    return [int(x) for x in s.replace(",", " ").split()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=_parse_int_list, default=[1, 8, 32, 64],
                   help="Space/comma-separated concurrency levels to sweep, e.g. '1 8 32 64'.")
    p.add_argument("--input-tokens", type=int, default=1024, help="Approx prompt length in tokens.")
    p.add_argument("--output-tokens", type=int, default=256, help="max_tokens per request.")
    p.add_argument("--prompts-per-conc", type=int, default=4,
                   help="Requests per level = concurrency * this (bounds wall time).")
    p.add_argument("--temperature", type=float, default=0.7, help="Mirror the sim's non-greedy sampling.")
    p.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (s).")
    args = p.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()

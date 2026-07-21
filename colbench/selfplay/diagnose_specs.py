# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Phase-0, Deliverable 2: the full-spec solve-rate diagnostic.

Hand a solver the ENTIRE authored spec in a SINGLE turn (no clarification dialogue), extract
the code, and grade it against the UNCHANGED GT ``test_cases`` via the existing exec sidecar
(``colbench.reward.grade``). Run it on one or more spec files (e.g. strong-gen and self-gen)
and it prints a solve rate per file -- the ceiling the Phase-1 dialogue rollout could reach.

Interpretation: a low self-gen number that TRACKS a low strong-gen number => the solver/task
is the bottleneck (the spec is fine); a self-gen number well BELOW strong-gen => self-authored
spec quality is the bottleneck. This is a diagnostic, not a filter: nothing is dropped.

Example:
    python -m colbench.selfplay.diagnose_specs \
        --data_file ~/data/colbench/train.parquet --max_rows 100 \
        --specs ~/data/colbench/specs/train.strong.jsonl ~/data/colbench/specs/train.selfplay.jsonl \
        --solver_base_url http://127.0.0.1:30000/v1 --solver_model colbench-base \
        --n_samples 1 --out ~/data/colbench/specs/diagnostic.json
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from colbench import reward, templates
from colbench.selfplay import spec_templates
from colbench.selfplay.dataio import read_jsonl, read_tasks
from colbench.selfplay.llm_client import ChatEndpoint


def _solve_and_grade(endpoint: ChatEndpoint, task: dict, spec: dict, reward_time_limit: float) -> dict:
    """One sample: the authored ``requirements`` -> solver code -> grade against GT test_cases.

    This is the full-spec faithfulness check: can a solver reconstruct GT behavior from the
    authored requirements alone. (The plot only shapes the Phase-1 dialogue; it is not a
    self-contained spec and is not graded here.)
    """
    messages = spec_templates.build_full_spec_solver_messages(task["problem_description"], spec)
    raw = endpoint.chat(messages)
    code = templates.extract_code_answer(templates.strip_think(raw))
    result = reward.grade(code, task["ground_truth"], task["test_cases"], time_limit=reward_time_limit)
    return {
        "index": task["index"],
        "pass_rate": float(result.get("pass_rate", 0.0)),
        "all_pass": bool(result.get("all_pass", False)),
        "n": int(result.get("n", 0)),
    }


def _aggregate(rows: list[dict], n_samples: int) -> dict:
    """Aggregate per-sample rows into solve-rate metrics.

    solve_rate    = mean(all_pass) over ALL samples (per-sample correctness),
    pass_at_n     = fraction of TASKS with >=1 fully-correct sample,
    mean_pass_rate= mean fractional pass-rate over all samples.
    """
    if not rows:
        return {"n_tasks": 0, "n_samples_total": 0, "solve_rate": 0.0, "pass_at_n": 0.0, "mean_pass_rate": 0.0}
    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r["index"], []).append(r)
    solve_rate = sum(r["all_pass"] for r in rows) / len(rows)
    mean_pass = sum(r["pass_rate"] for r in rows) / len(rows)
    pass_at_n = sum(any(s["all_pass"] for s in samples) for samples in by_task.values()) / len(by_task)
    return {
        "n_tasks": len(by_task),
        "n_samples_total": len(rows),
        "solve_rate": solve_rate,
        "pass_at_n": pass_at_n,
        "mean_pass_rate": mean_pass,
    }


def diagnose_specs_file(specs_path: str, tasks_by_index: dict, endpoint: ChatEndpoint,
                        n_samples: int, reward_time_limit: float, concurrency: int) -> dict:
    """Run the requirements full-spec diagnostic for one spec file. Returns {label, metrics,
    rows}. Specs with no ``requirements`` text are skipped (counted)."""
    specs = read_jsonl(specs_path)
    backend = specs[0].get("backend") if specs and specs[0].get("backend") else os.path.basename(specs_path)
    label = backend
    jobs, n_missing = [], 0
    for spec in specs:
        task = tasks_by_index.get(spec.get("index"))
        if task is None or not task["test_cases"]:
            continue  # no matching task or nothing to grade against
        if not str(spec.get("requirements", "") or "").strip():
            n_missing += 1
            continue  # nothing to solve from
        for _ in range(n_samples):
            jobs.append((task, spec))

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = [pool.submit(_solve_and_grade, endpoint, task, spec, reward_time_limit)
                for task, spec in jobs]
        for fut in as_completed(futs):
            rows.append(fut.result())
    return {"label": label, "path": specs_path, "n_missing_field": n_missing,
            "metrics": _aggregate(rows, n_samples), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_file", default=os.path.expanduser("~/data/colbench/train.parquet"),
                    help="The SAME parquet the specs were authored from (for GT + test_cases).")
    ap.add_argument("--max_rows", type=int, default=None, help="Limit #tasks loaded (must cover the specs' indices).")
    ap.add_argument("--specs", nargs="+", required=True, help="One or more spec JSONL files to diagnose.")
    ap.add_argument("--out", default=None, help="Optional JSON path to dump per-file metrics + rows.")
    # Solver endpoint.
    ap.add_argument("--solver_base_url", default=os.environ.get("SOLVER_BASE_URL", "http://127.0.0.1:30000/v1"))
    ap.add_argument("--solver_model", default=os.environ.get("SOLVER_MODEL", ""))
    ap.add_argument("--solver_api_key", default=os.environ.get("SOLVER_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--enable_thinking", choices=["true", "false"], default=None)
    ap.add_argument("--n_samples", type=int, default=1, help="Solver samples per spec (pass@n).")
    ap.add_argument("--reward_time_limit", type=float, default=6.0, help="Per-case GT exec timeout (s).")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("CODECONTEST_EXEC_CONCURRENCY", "16")),
                    help="Parallel solve+grade workers.")
    args = ap.parse_args()

    tasks = read_tasks(args.data_file, args.max_rows)
    tasks_by_index = {t["index"]: t for t in tasks}
    endpoint = ChatEndpoint(
        base_url=args.solver_base_url, model=args.solver_model, api_key=args.solver_api_key,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
        max_tokens=args.max_tokens,
        enable_thinking=None if args.enable_thinking is None else (args.enable_thinking == "true"),
    )

    t0 = time.time()
    results = []
    for specs_path in args.specs:
        r = diagnose_specs_file(specs_path, tasks_by_index, endpoint, args.n_samples,
                                args.reward_time_limit, args.concurrency)
        results.append(r)
        m = r["metrics"]
        print(f"[diagnose] {r['label']:<20} tasks={m['n_tasks']:<5} "
              f"solve_rate={m['solve_rate']:.3f}  pass@{args.n_samples}={m['pass_at_n']:.3f}  "
              f"mean_pass_rate={m['mean_pass_rate']:.3f}  (missing_field={r['n_missing_field']})")

    print(f"[diagnose] done in {time.time() - t0:.0f}s")
    print("\n=== full-spec solve-rate diagnostic ===")
    for r in results:
        print(f"  {r['label']:<20} solve_rate={r['metrics']['solve_rate']:.3f}  ({r['path']})")

    if args.out:
        out = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump([{"label": r["label"], "path": r["path"],
                        "n_missing_field": r["n_missing_field"], "metrics": r["metrics"],
                        "rows": r["rows"]} for r in results], f, indent=2)
        print(f"[diagnose] wrote per-file metrics -> {out}")


if __name__ == "__main__":
    main()

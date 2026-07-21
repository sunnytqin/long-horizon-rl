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
"""Phase-0, Deliverable 1: author one natural-language spec per ColBench task, OFFLINE.

For each task the spec author sees the public (under-specified) problem AND the hidden GT code,
and writes a persona + scenario + complete requirements spec (see ``spec_templates``). The
author is any OpenAI-compatible endpoint:

  * ``--backend strong``   -- an external teacher model (faithfulness ceiling).
  * ``--backend selfplay`` -- the trained model's FROZEN BASE checkpoint (no external model).

``--backend`` is a label recorded on every row and, by default, embedded in the output path so
the strong / self-gen caches never collide. Generation is concurrent and RESUMABLE: rows
already present in the output JSONL are skipped, so re-running continues where it left off.

Example (self-play against a served frozen base):
    python -m colbench.selfplay.generate_specs \
        --data_file ~/data/colbench/train.parquet --max_rows 100 \
        --backend selfplay \
        --gen_base_url http://127.0.0.1:30000/v1 --gen_model colbench-base \
        --out ~/data/colbench/specs/train.selfplay.jsonl
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from colbench.selfplay import spec_templates
from colbench.selfplay.dataio import append_jsonl, existing_indices, read_tasks
from colbench.selfplay.llm_client import ChatEndpoint


def _author_one(endpoint: ChatEndpoint, task: dict, backend_label: str, mode: str = "static") -> dict:
    """Author a spec for one task; returns a JSONL record (always, even on failure).

    ``mode="static"``  -> persona/scenario/requirements (the complete-requirements spec).
    ``mode="plot"`` -> persona/scenario/requirements (the full intent the simulator must convey)
    plus ``plot`` (a tailored, high-level direction for how this conversation naturally unfolds;
    the simulator improvises the actual turns from it -- it is NOT a turn-by-turn script).
    """
    rec = {"index": task["index"], "backend": backend_label, "mode": mode,
           "problem_description": task["problem_description"]}
    if mode == "plot":
        raw = endpoint.chat(
            spec_templates.build_plot_author_messages(task["problem_description"], task["ground_truth"]))
        spec = spec_templates.parse_plot_spec(raw)
        rec.update({
            "persona": spec["persona"], "scenario": spec["scenario"],
            "requirements": spec["requirements"], "plot": spec["plot"],
            "ok": spec["ok"], "raw": spec["raw"],
        })
    else:
        raw = endpoint.chat(
            spec_templates.build_author_messages(task["problem_description"], task["ground_truth"]))
        spec = spec_templates.parse_spec(raw)
        rec.update({
            "persona": spec["persona"], "scenario": spec["scenario"],
            "requirements": spec["requirements"], "ok": spec["ok"], "raw": spec["raw"],
        })
    return rec


def _default_out(data_file: str, backend: str, mode: str = "static") -> str:
    stem = os.path.splitext(os.path.basename(data_file))[0]
    d = os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(data_file))), "specs")
    tag = backend if mode == "static" else f"{backend}.{mode}"
    return os.path.join(d, f"{stem}.{tag}.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_file", default=os.path.expanduser("~/data/colbench/train.parquet"),
                    help="ColBench parquet (raw InfoPO or preprocessed schema).")
    ap.add_argument("--max_rows", type=int, default=None, help="Limit #tasks (debug slice).")
    ap.add_argument("--backend", choices=["strong", "selfplay"], required=True,
                    help="Author role label (also picks the default output path).")
    ap.add_argument("--mode", choices=["static", "plot"], default="static",
                    help="static: complete-requirements spec. plot: requirements + a tailored, "
                         "high-level plot the simulator improvises around (not a script).")
    ap.add_argument("--out", default=None, help="Output JSONL (default: <data_dir>/specs/<stem>.<backend>[.plot].jsonl).")
    # Author endpoint.
    ap.add_argument("--gen_base_url", default=os.environ.get("GEN_BASE_URL", "http://127.0.0.1:30000/v1"))
    ap.add_argument("--gen_model", default=os.environ.get("GEN_MODEL", ""))
    ap.add_argument("--gen_api_key", default=os.environ.get("GEN_API_KEY", "EMPTY"))
    ap.add_argument("--gen_api_key_file", default=os.environ.get("GEN_API_KEY_FILE", ""),
                    help="Read the API key from this file (keeps it out of argv/logs). Overrides --gen_api_key.")
    ap.add_argument("--gen_vendor", choices=["vllm", "openai"], default=os.environ.get("GEN_VENDOR", "vllm"),
                    help="'vllm' local server (top_k/min_p extras) or 'openai' vanilla API (no extras).")
    ap.add_argument("--temperature", type=float, default=0.7, help="Author sampling temperature (diversity).")
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--enable_thinking", choices=["true", "false"], default=None,
                    help="Set the SGLang enable_thinking kwarg (default: send nothing).")
    ap.add_argument("--concurrency", type=int, default=16, help="Parallel author calls.")
    ap.add_argument("--flush_every", type=int, default=20, help="Append to disk every N completions.")
    args = ap.parse_args()

    out = os.path.expanduser(args.out or _default_out(args.data_file, args.backend, args.mode))
    tasks = read_tasks(args.data_file, args.max_rows)
    done = existing_indices(out)
    todo = [t for t in tasks if t["index"] not in done]
    print(f"[generate_specs] mode={args.mode} {len(tasks)} tasks, {len(done)} already done, "
          f"{len(todo)} to author -> {out}")
    if not todo:
        return

    api_key = args.gen_api_key
    if args.gen_api_key_file:
        with open(os.path.expanduser(args.gen_api_key_file)) as f:
            api_key = f.read().strip()
    endpoint = ChatEndpoint(
        base_url=args.gen_base_url, model=args.gen_model, api_key=api_key, vendor=args.gen_vendor,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
        max_tokens=args.max_tokens,
        enable_thinking=None if args.enable_thinking is None else (args.enable_thinking == "true"),
    )

    t0 = time.time()
    buf, n_done, n_ok = [], 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futs = [pool.submit(_author_one, endpoint, t, args.backend, args.mode) for t in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            buf.append(rec)
            n_done += 1
            n_ok += int(rec["ok"])
            if len(buf) >= args.flush_every:
                append_jsonl(out, buf)
                buf = []
            if n_done % args.flush_every == 0:
                print(f"[generate_specs] {n_done}/{len(todo)} authored ({n_ok} parsed ok) "
                      f"in {time.time() - t0:.0f}s")
    if buf:
        append_jsonl(out, buf)
    print(f"[generate_specs] DONE {n_done} authored, {n_ok} parsed ok, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

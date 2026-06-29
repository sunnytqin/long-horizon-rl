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
"""Standalone multi-turn validation / inspection harness for the CodeContests solver.

Runs the SAME oracle code-refinement conversation as training
(``codecontest.code_refine_agent.CodeRefineAgentLoop``) on the validation/test set,
but as an *offline* SGLang batch job with freely tunable inference hyper-parameters
(temperature, top_p, number of turns, per-turn token budget, etc.). The point is to
manually examine what a trained checkpoint actually does, so every trajectory is dumped
in human-readable "conversation" form to a JSON file.

Uses SGLang's offline ``Engine`` (same inference backend as the training rollout), so
it runs in the very same SGLang container as training -- no vLLM dependency.

The heavy lifting is reused verbatim from the training path:
  - ``codecontest.env.GTOracleEnv``      -- grade latest code vs GT tests, build feedback
  - ``codecontest.templates``            -- prompts / oracle-feedback formatting
  - ``codecontest.exec_client``          -- sandbox-backed (or in-process) code grading
  - ``codecontest.local_exec.extract_code`` -- pull the last ```python block

The conversation flow mirrors ``CodeRefineAgentLoop.run`` exactly:
  turn 0      : solver writes code from the problem statement
  turn 1..N-1 : env grades the code; on failure injects failing cases as a user turn
                and the solver refines
  termination : all GT tests pass (early stop), max assistant turns reached, or the
                cumulative response-token budget would overflow.
Final reward is binary: 1.0 iff the final code passes all GT tests, else 0.0.

The one intentional difference vs the RL loop: we re-render the full message list with
the chat template each turn (clean, inspectable conversations) instead of appending raw
token ids. The model sees an equivalent prompt; there are no train-time masks to keep
aligned here.

Grading backend (identical semantics to training):
  - Sidecar sandbox: export CODECONTEST_EXEC_URL=http://host:8088   (preferred)
  - In-process fallback (dev/smoke, no sidecar): export CODECONTEST_ALLOW_INPROCESS=1

Example (inside the verl SGLang container, run from the repo root):
    export CODECONTEST_ALLOW_INPROCESS=1   # or CODECONTEST_EXEC_URL=...
    PYTHONPATH=$(pwd) python codecontest/validate_codecontest.py \
        --model /path/to/merged_hf_checkpoint \
        --val_file ~/data/codecontests/test.parquet \
        --out runs/eval_step120.json \
        --max_problems 64 --n_samples 4 \
        --temperature 0.8 --top_p 0.95 \
        --max_assistant_turns 4 --max_new_tokens_per_turn 4096
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import sglang as sgl
from transformers import AutoTokenizer

from codecontest import templates
from codecontest.env import GTOracleEnv
from codecontest.local_exec import extract_code

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----------------------------------------------------------------------------- #
# Per-trajectory state. One of these per (problem, sample) pair; it carries the
# growing conversation, its GT oracle env, and the running token-budget bookkeeping.
# ----------------------------------------------------------------------------- #
class Trajectory:
    def __init__(self, row_index, task_id, sample_idx, messages, env, response_budget):
        self.row_index = row_index
        self.task_id = task_id
        self.sample_idx = sample_idx
        # Full conversation: starts with [system, user(problem)] from the parquet.
        self.messages = list(messages)
        self.env = env
        self.response_budget = response_budget  # max cumulative response tokens (assistant+feedback)

        self.response_tokens = 0   # cumulative response-side tokens used so far
        self.assistant_turns = 0
        self.user_turns = 0
        self.solved = False
        self.solved_at_turn = -1
        self.overflow = False
        self.done = False
        self.final_code = None
        # Per-turn audit trail (mirrors the StepResult fields the env returns).
        self.turn_records = []

    def to_dict(self):
        return {
            "row_index": self.row_index,
            "task_id": self.task_id,
            "sample_idx": self.sample_idx,
            "solved": self.solved,
            "solved_at_turn": self.solved_at_turn,
            "num_assistant_turns": self.assistant_turns,
            "num_user_turns": self.user_turns,
            "overflow": self.overflow,
            "response_tokens": self.response_tokens,
            "reward": 1.0 if self.solved else 0.0,
            "final_code": self.final_code,
            "turn_records": self.turn_records,
            "messages": self.messages,
        }


def build_trajectories(val_df, n_samples, eval_args):
    """Expand each validation row into ``n_samples`` independent trajectories."""
    trajs = []
    for row_index, row in val_df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        gt = extra_info.get("ground_truth")
        if gt is None:
            gt = (row.get("reward_model", {}) or {}).get("ground_truth", {})
        task_id = extra_info.get("task_id", row_index)
        # parquet stores the chat prompt as a list/array of {role, content} dicts.
        messages = [dict(m) for m in row["prompt"]]
        for s in range(n_samples):
            env = GTOracleEnv(
                test_input=list(gt["test_input"]),
                test_output=list(gt["test_output"]),
                test_time_limit=float(gt.get("test_time_limit", 6.0)),
                max_failures_shown=eval_args.max_failures_shown,
                max_gt_test=eval_args.max_gt_test,
                max_feedback_chars=eval_args.max_feedback_chars,
                # Match the training loop's per-sample seed (env uses it to sample which
                # failing cases to show). Vary by sample so reruns are reproducible.
                seed=int(row_index) * 1000 + s,
            )
            trajs.append(
                Trajectory(
                    row_index=int(row_index),
                    task_id=int(task_id) if task_id is not None else int(row_index),
                    sample_idx=s,
                    messages=messages,
                    env=env,
                    response_budget=eval_args.max_response_length,
                )
            )
    return trajs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model / data
    ap.add_argument("--model", required=True, help="HF model dir or name (a merged checkpoint).")
    ap.add_argument("--val_file", default=os.path.expanduser("~/data/codecontests/test.parquet"),
                    help="Validation/test parquet (VERL schema from preprocess_codecontests.py).")
    ap.add_argument("--out", required=True, help="Output JSON path for the conversation dump.")
    ap.add_argument("--max_problems", type=int, default=None, help="Limit #problems (debug). None = all.")
    ap.add_argument("--n_samples", type=int, default=1, help="Trajectories sampled per problem (pass@k).")

    # Inference hyper-parameters (the whole point of this script -- tune freely).
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1, help="-1 disables top-k.")
    ap.add_argument("--seed", type=int, default=0, help="SGLang engine random seed.")
    ap.add_argument("--max_assistant_turns", type=int, default=4, help="Total solver attempts (1 = single-turn).")
    ap.add_argument("--max_new_tokens_per_turn", type=int, default=4096)
    ap.add_argument("--max_response_length", type=int, default=16384,
                    help="Cumulative response-token budget (assistant + feedback) before overflow stop.")
    ap.add_argument("--max_prompt_length", type=int, default=4096,
                    help="Initial-prompt cap; also derives the default feedback char budget.")

    # Oracle / feedback knobs (must match training to reproduce its grading).
    ap.add_argument("--max_failures_shown", type=int, default=3)
    ap.add_argument("--max_gt_test", type=int, default=20)
    ap.add_argument("--max_feedback_chars", type=int, default=0,
                    help="0 => derive from --max_prompt_length (same rule as the agent loop).")

    # SGLang engine
    ap.add_argument("--tensor_parallel_size", type=int, default=int(os.getenv("ROLLOUT_TP", "1")))
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                    help="SGLang mem_fraction_static (fraction of GPU mem for weights+KV cache).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--grade_concurrency", type=int, default=int(os.getenv("CODECONTEST_EXEC_CONCURRENCY", "32")),
                    help="Threads used to grade a turn's batch in parallel (env.step is blocking).")
    args = ap.parse_args()

    # Derive the feedback char budget the same way CodeRefineAgentLoop does, so the
    # injected feedback text is identical to what the model saw in training.
    if args.max_feedback_chars <= 0:
        args.max_feedback_chars = int(args.max_prompt_length * 3.0 * 0.5)

    if not os.environ.get("CODECONTEST_EXEC_URL") and os.environ.get("CODECONTEST_ALLOW_INPROCESS") != "1":
        raise SystemExit(
            "No code-exec backend configured. Set CODECONTEST_EXEC_URL=http://host:8088 (sidecar) "
            "or CODECONTEST_ALLOW_INPROCESS=1 (in-process dev/smoke) before running."
        )

    # ---- load data ----
    val_df = pd.read_parquet(args.val_file)
    if args.max_problems is not None:
        val_df = val_df.iloc[: args.max_problems].copy()
    val_df = val_df.reset_index(drop=True)
    print(f"[validate] loaded {len(val_df)} problems from {args.val_file}; "
          f"{args.n_samples} sample(s) each => {len(val_df) * args.n_samples} trajectories")

    # ---- tokenizer + engine ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    max_model_len = args.max_prompt_length + args.max_response_length
    llm = sgl.Engine(
        model_path=args.model,
        dtype=args.dtype,
        tp_size=args.tensor_parallel_size,
        mem_fraction_static=args.gpu_memory_utilization,
        context_length=max_model_len,
        trust_remote_code=args.trust_remote_code,
        random_seed=args.seed,
    )
    # SGLang sampling params (dict, applied to every request in a batch). Note the
    # name is ``max_new_tokens`` here, and top_k=-1 means "use the whole vocabulary".
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens_per_turn,
    }

    trajs = build_trajectories(val_df, args.n_samples, args)
    grade_pool = ThreadPoolExecutor(max_workers=max(1, args.grade_concurrency))

    t0 = time.time()
    # ---- multi-turn loop: every turn processes ALL still-active trajectories as one
    #      SGLang batch, then grades the batch in parallel. ----
    for turn in range(args.max_assistant_turns):
        active = [t for t in trajs if not t.done]
        if not active:
            break

        # Respect the cumulative response-token budget per trajectory (overflow = stop).
        gen_batch = []
        for t in active:
            if t.response_tokens >= t.response_budget:
                t.overflow = True
                t.done = True
            else:
                gen_batch.append(t)
        if not gen_batch:
            break

        prompts = [
            tokenizer.apply_chat_template(t.messages, add_generation_prompt=True, tokenize=False)
            for t in gen_batch
        ]
        print(f"[validate] turn {turn}: generating for {len(gen_batch)} active trajectories")
        outputs = llm.generate(prompt=prompts, sampling_params=sampling_params)

        # Append assistant turns, then grade in parallel.
        for t, out in zip(gen_batch, outputs):
            text = out["text"]
            t.messages.append({"role": "assistant", "content": text})
            t.assistant_turns += 1
            t.response_tokens += int(out["meta_info"].get("completion_tokens", 0))
            t.final_code = extract_code(text)

        def grade(t):
            return t, t.env.step(t.messages[-1]["content"])

        results = list(grade_pool.map(grade, gen_batch))

        is_last_turn = turn == args.max_assistant_turns - 1
        for t, step in results:
            t.turn_records.append({
                "turn": turn,
                "solved": step.solved,
                "had_code": step.had_code,
                "num_failures_shown": step.num_failures_shown,
            })
            if step.solved:
                t.solved = True
                t.solved_at_turn = turn
                t.done = True
                continue
            if is_last_turn:
                t.done = True
                continue
            # Inject oracle feedback as the next user turn (and charge its tokens to the
            # response budget, matching the agent loop's accounting).
            feedback_ids = tokenizer.encode(step.feedback, add_special_tokens=False)
            if t.response_tokens + len(feedback_ids) >= t.response_budget:
                t.overflow = True
                t.done = True
                continue
            t.messages.append({"role": "user", "content": step.feedback})
            t.user_turns += 1
            t.response_tokens += len(feedback_ids)

    grade_pool.shutdown(wait=True)
    llm.shutdown()
    elapsed = time.time() - t0

    # ---- aggregate metrics ----
    by_problem = {}
    for t in trajs:
        by_problem.setdefault(t.row_index, []).append(t)
    n_problems = len(by_problem)
    n_traj = len(trajs)
    traj_solved = sum(1 for t in trajs if t.solved)
    pass_at_1 = sum(1 for grp in by_problem.values() if grp[0].solved) / max(1, n_problems)
    pass_at_k = sum(1 for grp in by_problem.values() if any(t.solved for t in grp)) / max(1, n_problems)
    solved_trajs = [t for t in trajs if t.solved]
    avg_turns_to_solve = (
        sum(t.solved_at_turn + 1 for t in solved_trajs) / len(solved_trajs) if solved_trajs else None
    )
    overflow_rate = sum(1 for t in trajs if t.overflow) / max(1, n_traj)

    summary = {
        "model": args.model,
        "val_file": args.val_file,
        "n_problems": n_problems,
        "n_samples_per_problem": args.n_samples,
        "n_trajectories": n_traj,
        "trajectory_solve_rate": traj_solved / max(1, n_traj),
        "pass@1": pass_at_1,
        f"pass@{args.n_samples}": pass_at_k,
        "avg_turns_to_solve": avg_turns_to_solve,
        "overflow_rate": overflow_rate,
        "elapsed_sec": round(elapsed, 1),
        "inference": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_assistant_turns": args.max_assistant_turns,
            "max_new_tokens_per_turn": args.max_new_tokens_per_turn,
            "max_response_length": args.max_response_length,
            "max_prompt_length": args.max_prompt_length,
            "max_gt_test": args.max_gt_test,
            "max_failures_shown": args.max_failures_shown,
            "max_feedback_chars": args.max_feedback_chars,
            "seed": args.seed,
        },
    }
    print("[validate] summary:")
    print(json.dumps(summary, indent=2))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "trajectories": [t.to_dict() for t in trajs]}, f, indent=2)
    print(f"[validate] wrote {n_traj} conversations -> {args.out}")


if __name__ == "__main__":
    main()

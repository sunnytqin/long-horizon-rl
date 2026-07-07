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

Runs the SAME multi-turn code-refinement conversation as training on the validation/test
set, but as an *offline* SGLang batch job with freely tunable inference hyper-parameters
(temperature, top_p, number of turns, per-turn token budget, etc.). The point is to
manually examine what a trained checkpoint actually does, so every trajectory is dumped
in human-readable "conversation" form to a JSON file.

``--feedback_mode`` selects which training loop to mirror, so a checkpoint is always
evaluated in the regime it was trained in (apples-to-apples):
  oracle         -- raw failing cases injected as the next user turn
                    (``code_refine_agent.CodeRefineAgentLoop``).
  model_feedback -- a SECOND policy call (the SAME model, as a "user model") reads the
                    failing cases and writes a diagnosis; only that diagnosis is injected
                    and the raw cases are hidden (``model_feedback_agent.ModelFeedbackAgentLoop``).
The between-turns feedback is the ONLY thing that differs; grading and reward are identical.
Both modes share the number-affecting transforms with training by importing the same
``codecontest.templates`` helpers (feedback formatting, ``normalize_diagnosis``).

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

Sweep several temperatures in ONE submission (engine loaded once). Each temperature
writes its own JSON with the config tagged into the name (turns / n_samples / temp), so
--out runs/eval_step120.json with --max_assistant_turns 4 --n_samples 8 --temperatures 0.0 0.8
produces runs/eval_step120_turns4_n1_t0.json and runs/eval_step120_turns4_n8_t0.8.json
(t=0 is forced to a single sample, hence _n1):
    PYTHONPATH=$(pwd) python codecontest/validate_codecontest.py \
        --model /path/to/merged_hf_checkpoint --out runs/eval_step120.json \
        --n_samples 8 --temperatures 0.0 0.8
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from math import comb

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
    def __init__(self, row_index, task_id, sample_idx, messages, env, response_budget, problem_text=""):
        self.row_index = row_index
        self.task_id = task_id
        self.sample_idx = sample_idx
        # Full conversation: starts with [system, user(problem)] from the parquet.
        self.messages = list(messages)
        self.env = env
        self.response_budget = response_budget  # max cumulative response tokens (assistant+feedback)
        # Initial problem-statement user turn, reused as the `problem` field of the
        # user-model feedback prompt in --feedback_mode model_feedback (mirrors the agent loop,
        # which grabs the last user turn from raw_prompt). Unused in oracle mode.
        self.problem_text = problem_text

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
        # Model-feedback bookkeeping (mirrors ModelFeedbackAgentLoop's extra_fields).
        # Empty/unused in oracle mode.
        self.feedback_turn_lengths = []   # user-model diagnosis lengths (tokens), per feedback turn
        self.feedback_empty = 0           # diagnoses that came back empty (fell back to the fixed string)

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
            "feedback_turn_lengths": self.feedback_turn_lengths,
            "feedback_empty": self.feedback_empty,
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
        # Problem text for the user-model feedback prompt (model mode): the last user
        # turn, which wraps the problem statement -- exactly what the agent loop uses.
        problem_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
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
                    problem_text=problem_text,
                )
            )
    return trajs


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Kulal et al. / HumanEval): probability that at least one
    of k samples drawn WITHOUT replacement from n total (c of which succeed) is a success.

        pass@k = 1 - C(n-c, k) / C(n, k)

    Returns None when k > n (not enough samples to define it). Uses exact integer binomials
    (n is tiny here, so no overflow concern).
    """
    if k > n:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def per_turn_pass_at_k(by_problem, max_turns):
    """Per-turn pass@k surface for the multi-turn eval.

    For each turn cutoff ``t`` (0..max_turns-1) a sample counts as a success iff it solved
    *by* turn t, i.e. ``traj.solved and traj.solved_at_turn <= t``. NOTE we deliberately use
    ``solved_at_turn`` rather than ``turn_records[t].solved`` because an early-solving
    trajectory stops and has no records past its solve turn -- the cutoff form is monotonic
    and correct. Each problem contributes its own (n, c_t); we average the unbiased pass@k
    over problems, reporting the full k=1..n curve (per-problem n; smaller groups skip larger k).

    per_turn[0]["pass@k"]["1"] is the sample-averaged single-turn solve rate -- the number
    that must match a matched-n single-turn run's pass@1.
    """
    per_turn = []
    for t in range(max_turns):
        # Collect (n, c_t) per problem for this cutoff.
        groups = []
        max_n = 0
        for grp in by_problem.values():
            n = len(grp)
            c = sum(1 for tr in grp if tr.solved and tr.solved_at_turn <= t)
            groups.append((n, c))
            max_n = max(max_n, n)
        pass_k = {}
        for k in range(1, max_n + 1):
            vals = [pass_at_k(n, c, k) for (n, c) in groups if k <= n]
            if vals:
                pass_k[str(k)] = sum(vals) / len(vals)
        per_turn.append({"turn": t, "pass@k": pass_k})
    return per_turn


_MODE_TAG = {"oracle": "oracle", "model_feedback": "modelfb"}


def eval_tagged_path(base_out, turns, n_samples, temperature, feedback_mode):
    """Insert a '_turns<N>_n<K>_t<temp>_<mode>' tag before the extension so each eval
    config gets its own self-documenting file (and can never clobber a different config).

    ``n_samples`` is the temperature-adjusted value actually used (t=0 is forced to 1),
    so the tag faithfully records what was run. ``feedback_mode`` (oracle|model_feedback) is
    tagged too (as oracle|modelfb) so an oracle eval and a model-feedback eval of the SAME
    checkpoint never collide.

    e.g. ("runs/eval_step120.json", 4, 8, 0.8, "model_feedback")
         -> "runs/eval_step120_turns4_n8_t0.8_modelfb.json"
    """
    root, ext = os.path.splitext(base_out)
    mode_tag = _MODE_TAG[feedback_mode]
    return f"{root}_turns{turns}_n{n_samples}_t{temperature:g}_{mode_tag}{ext or '.json'}"


def build_next_user_turns(pending, feedback_mode, llm, tokenizer, args, max_model_len, sampling_params):
    """Resolve the next user-turn text for each failed-but-continuing trajectory.

    ``pending`` is a list of ``(traj, StepResult)`` for trajectories that were graded this
    turn, did not solve, and are not on the last turn. Returns a list of user-turn strings
    aligned to ``pending``.

    - oracle mode: the env already built the failing-case feedback -> inject ``step.feedback``.
    - model mode : a SECOND policy call (the "user model") writes the diagnosis, exactly like
      ``ModelFeedbackAgentLoop``. Trajectories that surfaced failing cases are batched into a
      single ``llm.generate`` (throughput), their diagnoses normalized by the SHARED
      ``templates.normalize_diagnosis`` (identical <think>-strip + empty fallback as training)
      and wrapped by ``templates.build_model_feedback_user_message``. Trajectories with no
      failing cases (no parseable code) fall back to the env's fixed instruction, matching the
      agent loop's ``else`` branch. Per-trajectory feedback bookkeeping is mutated in place.
    """
    if feedback_mode == "oracle":
        return [step.feedback for (_t, step) in pending]

    # ---- model mode: build the user-model prompts, batch-generate, normalize, wrap. ----
    contents = [None] * len(pending)
    fb_prompts, fb_params, fb_slots = [], [], []
    engine_ctx = max_model_len  # prompt_length + response_length, the SGLang context window
    # Small margin so input+completion stays STRICTLY under engine_ctx: SGLang rejects a
    # request whose input_len + max_new_tokens EQUALS the context length (not just exceeds),
    # and our token count can drift a little from the engine's. Matches the main loop's guard.
    CTX_MARGIN = 8
    for i, (t, step) in enumerate(pending):
        if not step.failures:
            # No code / nothing to diagnose: use the env's fixed message (agent-loop parity).
            contents[i] = step.feedback
            continue
        fb_messages = templates.build_feedback_model_messages(
            step.failures,
            problem=t.problem_text,
            code=step.code,
            max_total_chars=(args.max_feedback_chars or None),
        )
        # Throwaway feedback prompt: rendered UNCAPPED (no solver prompt_length truncation,
        # which would drop the problem), bounded only by the engine context below.
        fb_prompt_text = tokenizer.apply_chat_template(fb_messages, add_generation_prompt=True, tokenize=False)
        fb_input_len = len(tokenizer.encode(fb_prompt_text, add_special_tokens=False))
        # Cap the diagnosis (mirrors the agent loop): <= max_feedback_tokens, <= remaining
        # response budget (it lands in the response tail), and never overflow the engine ctx.
        fb_remaining = t.response_budget - t.response_tokens
        fb_cap = min(args.max_feedback_tokens, fb_remaining, engine_ctx - fb_input_len - CTX_MARGIN)
        if fb_cap < 1:
            # The problem+code+failures prompt nearly fills the context (or the response
            # budget is spent), leaving no room to generate a diagnosis. Skip the user-model
            # call and inject the generic fallback -- keeps model-feedback semantics (no raw
            # cases leaked); the later overflow check still handles the response budget.
            contents[i] = templates.build_model_feedback_user_message(templates.EMPTY_DIAGNOSIS_FALLBACK)
            t.feedback_empty += 1
            continue
        sp = {**sampling_params, "max_new_tokens": fb_cap}
        fb_prompts.append(fb_prompt_text)
        fb_params.append(sp)
        fb_slots.append(i)

    if fb_prompts:
        fb_outputs = llm.generate(prompt=fb_prompts, sampling_params=fb_params)
        for i, out in zip(fb_slots, fb_outputs):
            t, _step = pending[i]
            analysis, was_empty = templates.normalize_diagnosis(out["text"])
            t.feedback_turn_lengths.append(int(out["meta_info"].get("completion_tokens", 0)))
            if was_empty:
                t.feedback_empty += 1
            contents[i] = templates.build_model_feedback_user_message(analysis)
    return contents


def run_eval(llm, tokenizer, val_df, temperature, n_samples, args, out_path, max_model_len):
    """Run the full multi-turn oracle eval at a single temperature and dump one JSON.

    The (expensive) SGLang engine is loaded once by the caller and shared across every
    temperature; only the per-request sampling params and the fresh trajectory state
    differ between calls. ``n_samples`` is the (possibly temperature-adjusted) number of
    trajectories per problem -- greedy decoding (temperature 0) is forced to 1 sample.
    """
    # SGLang sampling params (dict, applied to every request in a batch). Note the
    # name is ``max_new_tokens`` here, and top_k=-1 means "use the whole vocabulary".
    sampling_params = {
        "temperature": temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens_per_turn,
    }

    trajs = build_trajectories(val_df, n_samples, args)
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

        # Assemble each prompt and clamp its completion budget to the room left in the
        # model's context window. The cumulative-response check above only bounds
        # response_tokens; it does NOT bound (initial_prompt + accumulated turns +
        # max_new_tokens_per_turn) against max_model_len, so on later turns a fixed
        # max_new_tokens request can exceed context and make SGLang raise ValueError,
        # killing the whole run. Guard per-request instead: shrink the completion, or
        # (if there's no room to generate at all) stop the trajectory as overflow.
        prompts, per_req_params, kept = [], [], []
        # Small safety margin for tokenization drift between our count and SGLang's.
        CTX_MARGIN = 8
        for t in gen_batch:
            prompt_text = tokenizer.apply_chat_template(
                t.messages, add_generation_prompt=True, tokenize=False
            )
            input_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            room = max_model_len - input_len - CTX_MARGIN
            if room <= 0:
                t.overflow = True
                t.done = True
                continue
            sp = dict(sampling_params)
            sp["max_new_tokens"] = min(args.max_new_tokens_per_turn, room)
            prompts.append(prompt_text)
            per_req_params.append(sp)
            kept.append(t)
        if not kept:
            continue
        clamped = sum(1 for sp in per_req_params if sp["max_new_tokens"] < args.max_new_tokens_per_turn)
        print(f"[validate] t={temperature:g} turn {turn}: generating for {len(kept)} active trajectories"
              f" ({clamped} with context-clamped max_new_tokens)")
        outputs = llm.generate(prompt=prompts, sampling_params=per_req_params)

        # Append assistant turns, then grade in parallel.
        gen_batch = kept
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
        # First pass: record each grade and classify. Trajectories that failed but can
        # still refine are collected for feedback (built in one batch below).
        pending = []  # (traj, StepResult) needing an injected user turn this turn
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
            pending.append((t, step))

        # Resolve the next user turn per mode: oracle injects the env's failing-case
        # feedback; model runs a batched "user model" diagnosis call (see the helper).
        user_contents = build_next_user_turns(
            pending, args.feedback_mode, llm, tokenizer, args, max_model_len, sampling_params
        )

        # Second pass: inject each user turn, charging its tokens to the response budget
        # (matching the agent loop's accounting; overflow => stop unsolved).
        for (t, step), user_content in zip(pending, user_contents):
            feedback_ids = tokenizer.encode(user_content, add_special_tokens=False)
            if t.response_tokens + len(feedback_ids) >= t.response_budget:
                t.overflow = True
                t.done = True
                continue
            t.messages.append({"role": "user", "content": user_content})
            t.user_turns += 1
            t.response_tokens += len(feedback_ids)

    grade_pool.shutdown(wait=True)
    elapsed = time.time() - t0

    # ---- aggregate metrics ----
    by_problem = {}
    for t in trajs:
        by_problem.setdefault(t.row_index, []).append(t)
    n_problems = len(by_problem)
    n_traj = len(trajs)
    traj_solved = sum(1 for t in trajs if t.solved)
    pass_at_1 = sum(1 for grp in by_problem.values() if grp[0].solved) / max(1, n_problems)
    # "final" pass@n: fraction of problems solved by ANY of the n samples at end-of-conversation
    # (naive any-solved rate, kept for backward-compat; the unbiased curve lives in per_turn).
    pass_at_n = sum(1 for grp in by_problem.values() if any(t.solved for t in grp)) / max(1, n_problems)
    solved_trajs = [t for t in trajs if t.solved]
    avg_turns_to_solve = (
        sum(t.solved_at_turn + 1 for t in solved_trajs) / len(solved_trajs) if solved_trajs else None
    )
    overflow_rate = sum(1 for t in trajs if t.overflow) / max(1, n_traj)
    # Per-turn pass@k surface: for each turn cutoff, the sample-averaged unbiased pass@k
    # over problems. per_turn[0]["pass@k"]["1"] is the single-turn (turn-0) solve rate that
    # should match a matched-n single-turn run's pass@1.
    per_turn = per_turn_pass_at_k(by_problem, args.max_assistant_turns)

    summary = {
        "model": args.model,
        "val_file": args.val_file,
        "feedback_mode": args.feedback_mode,
        "n_problems": n_problems,
        "n_samples_per_problem": n_samples,
        "n_trajectories": n_traj,
        "trajectory_solve_rate": traj_solved / max(1, n_traj),
        "pass@1": pass_at_1,
        f"pass@{n_samples}": pass_at_n,
        "per_turn": per_turn,
        "avg_turns_to_solve": avg_turns_to_solve,
        "overflow_rate": overflow_rate,
        "elapsed_sec": round(elapsed, 1),
        "inference": {
            "temperature": temperature,
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
    # Model-feedback diagnostics (user-model diagnosis length + empty-diagnosis rate),
    # mirroring ModelFeedbackAgentLoop's feedback_resp_len_mean / feedback_empty. Only
    # meaningful in model mode; omitted in oracle mode where no diagnosis is generated.
    if args.feedback_mode == "model_feedback":
        summary["inference"]["max_feedback_tokens"] = args.max_feedback_tokens
        fb_means = [
            sum(t.feedback_turn_lengths) / len(t.feedback_turn_lengths)
            for t in trajs if t.feedback_turn_lengths
        ]
        summary["feedback_resp_len_mean"] = (sum(fb_means) / len(fb_means)) if fb_means else 0.0
        summary["feedback_empty_total"] = sum(t.feedback_empty for t in trajs)
    print(f"[validate] summary (t={temperature:g}):")
    print(json.dumps(summary, indent=2))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "trajectories": [t.to_dict() for t in trajs]}, f, indent=2)
    print(f"[validate] wrote {n_traj} conversations -> {out_path}")
    return summary


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
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="Single sampling temperature (used only when --temperatures is not given).")
    ap.add_argument("--temperatures", type=float, nargs="+", default=None,
                    help="Sweep several temperatures in ONE submission (engine loaded once). Each "
                         "writes its own JSON tagged with the temp: '<out_stem>_t<temp>.json'. "
                         "Overrides --temperature. Example: --temperatures 0.0 0.8")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1, help="-1 disables top-k.")
    ap.add_argument("--seed", type=int, default=0, help="SGLang engine random seed.")
    ap.add_argument("--max_assistant_turns", type=int, default=4, help="Total solver attempts (1 = single-turn).")
    ap.add_argument("--max_new_tokens_per_turn", type=int, default=4096)
    ap.add_argument("--max_response_length", type=int, default=8192,
                    help="Cumulative response-token budget (assistant + feedback) before overflow "
                         "stop. Default 8192 matches MAX_RESPONSE_LENGTH in the training launchers.")
    ap.add_argument("--max_prompt_length", type=int, default=4096,
                    help="Initial-prompt cap; also derives the default feedback char budget.")

    # Oracle / feedback knobs (must match training to reproduce its grading).
    ap.add_argument("--feedback_mode", choices=["oracle", "model_feedback"], default="oracle",
                    help="Between-turns feedback. 'oracle' (default) injects the raw failing "
                         "cases (matches run_oracle_codecontest_grpo.sh / CodeRefineAgentLoop). "
                         "'model_feedback' runs the SAME policy as a 'user model' to write a "
                         "diagnosis, hiding the raw cases (matches "
                         "run_model_feedback_codecontest_grpo.sh / ModelFeedbackAgentLoop). Use "
                         "'model_feedback' to eval a model-feedback-trained checkpoint apples-to-apples.")
    ap.add_argument("--max_failures_shown", type=int, default=3)
    ap.add_argument("--max_gt_test", type=int, default=20)
    ap.add_argument("--max_feedback_chars", type=int, default=0,
                    help="Combined char budget for the failing-case fields. oracle mode: budget "
                         "for the injected feedback turn. model mode: budget inside the user-model "
                         "prompt. 0 => derive from --max_prompt_length (same rule as the agent loop).")
    ap.add_argument("--max_feedback_tokens", type=int, default=2048,
                    help="model mode only: hard cap on the user-model diagnosis length (tokens). "
                         "Mirrors +codecontest.max_feedback_tokens. Shares --max_response_length "
                         "with the solver turns, so keep it modest (see run_model_feedback_*.sh).")

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
          f"up to {args.n_samples} sample(s) each (temperature 0 forced to 1)")

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
    # Resolve the temperature sweep: --temperatures wins, else the single --temperature.
    # De-dup while preserving order so a stray repeat doesn't run (and overwrite) twice.
    temps = args.temperatures if args.temperatures else [args.temperature]
    _seen = set()
    temps = [t for t in temps if not (t in _seen or _seen.add(t))]
    print(f"[validate] temperature sweep: {temps} (engine loaded once, reused across all)")

    # Each temperature gets its own tagged JSON, e.g. eval_step120.json -> eval_step120_t0.8.json.
    results_index = []
    for temperature in temps:
        # Greedy decoding produces identical samples, so pass@k is degenerate -- force 1
        # sample at temperature 0 (no point spending compute on duplicate trajectories).
        n_samples = 1 if temperature == 0.0 else args.n_samples
        if temperature == 0.0 and args.n_samples > 1:
            print(f"[validate] temperature=0 is greedy; forcing n_samples 1 (was {args.n_samples}).")
        out_path = eval_tagged_path(args.out, args.max_assistant_turns, n_samples, temperature, args.feedback_mode)
        summary = run_eval(llm, tokenizer, val_df, temperature, n_samples, args, out_path, max_model_len)
        results_index.append({"temperature": temperature, "out": out_path,
                              "pass@1": summary["pass@1"], f"pass@{n_samples}": summary[f"pass@{n_samples}"]})

    llm.shutdown()

    print("[validate] sweep complete:")
    print(json.dumps(results_index, indent=2))


if __name__ == "__main__":
    main()

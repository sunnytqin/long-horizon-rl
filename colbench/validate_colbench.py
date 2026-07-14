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
"""Standalone multi-turn validation / inspection harness for the ColBench solver.

Runs the SAME solver<->frozen-simulator conversation as training on the validation/test
set, but as an *offline* SGLang batch job with freely tunable inference hyper-parameters
(temperature, number of turns, per-turn token budget, ...). The point is to manually examine
what a trained checkpoint actually does, so a random sample of trajectories is dumped in
human-readable "conversation" form to a JSON file.

The rollout logic is reused VERBATIM from the training path -- this harness drives the exact
same env seams the agent loop (``colbench.colbench_agent.ColBenchAgentLoop``) drives:
  - ``colbench.env.ColBenchUserSimEnv.is_answer``          -- did the solver submit this turn?
  - ``colbench.env.ColBenchUserSimEnv.generate_user_turn`` -- the frozen sim's next reply
    (HTTP to the sim server via OPENAI_BASE_URL / MULTITURN_MODEL_NAME); the hidden GT is
    passed ONLY inside this call and never enters the solver's message list.
  - ``colbench.env.ColBenchUserSimEnv.score``              -- fractional GT pass-rate reward.
So prompts, marker/code extraction (``colbench.templates``), sim prompt, and grading
(``colbench.reward``) are byte-identical to training.

Uses SGLang's offline ``Engine`` for the SOLVER (same inference backend as the training
rollout), so it runs in the very same SGLang container as training -- no vLLM dependency.
The frozen user simulator is a SEPARATE SGLang OpenAI server (the entrypoint brings it up and
exports OPENAI_BASE_URL), exactly as in training.

The one intentional difference vs the RL loop (shared with codecontest/validate_codecontest.py):
we re-render the full message list with the chat template each turn (clean, inspectable
conversations) instead of appending raw token ids. The model sees an equivalent prompt; there
are no train-time masks to keep aligned here.

Reward convention: the trajectory reward is the final submission's FRACTIONAL pass-rate in
[0, 1] (mean per-case functional equivalence vs the hidden GT), matching training.

Grading backend (identical semantics to training):
  - Sidecar sandbox: export CODECONTEST_EXEC_URL=http://host:8088   (preferred)
  - In-process fallback (dev/smoke, no sidecar): export CODECONTEST_ALLOW_INPROCESS=1

Example (inside the verl SGLang container, run from the repo root, sim server already up):
    export CODECONTEST_ALLOW_INPROCESS=1                 # or CODECONTEST_EXEC_URL=...
    export OPENAI_BASE_URL=http://127.0.0.1:30000/v1     # frozen sim server
    export MULTITURN_MODEL_NAME=colbench-sim
    PYTHONPATH=$(pwd) python colbench/validate_colbench.py \
        --model /path/to/merged_hf_checkpoint \
        --val_file ~/data/colbench/test.parquet \
        --out runs/eval_step120.json \
        --max_problems 64 --n_samples 4 --temperature 0.6 \
        --max_assistant_turns 10 --max_new_tokens_per_turn 1024
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from colbench import templates
from colbench.env import ColBenchUserSimEnv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Small safety margin so input+completion stays STRICTLY under the engine context window:
# SGLang rejects a request whose input_len + max_new_tokens EQUALS the context length (not
# just exceeds), and our token count can drift a little from the engine's. Matches the
# codecontest validator's guard.
CTX_MARGIN = 8


# ----------------------------------------------------------------------------- #
# Per-trajectory state. One of these per (problem, sample) pair; it carries the growing
# solver conversation, the separate GT-free sim dialogue, its env, and token bookkeeping.
# ----------------------------------------------------------------------------- #
class Trajectory:
    def __init__(self, row_index, task_id, sample_idx, messages, problem_text, env, response_budget):
        self.row_index = row_index
        self.task_id = task_id
        self.sample_idx = sample_idx
        # Full SOLVER conversation: starts with [system, user(problem)] from the parquet.
        self.messages = [dict(m) for m in messages]
        # The SIMULATOR's running dialogue (problem + solver turns + user replies). Contains
        # NO ground truth -- byte-identical to what ColBenchAgentLoop.run builds and passes to
        # env.generate_user_turn (which injects the hidden GT only inside its own prompt).
        self.sim_dialogue = [{"role": "user", "content": problem_text}]
        self.env = env
        self.response_budget = response_budget  # max cumulative response tokens (solver + sim replies)

        self.response_tokens = 0   # cumulative response-side tokens used so far
        self.assistant_turns = 0
        self.user_turns = 0
        self.answered = False
        self.answered_at_turn = -1
        self.overflow = False
        self.done = False
        self.reward = 0.0          # fractional GT pass-rate of the final submission
        self.all_pass = False
        self.num_test_cases = 0
        self.final_answer = None
        self.final_code = None
        # Simulation-failure category: the sim could not produce a code-free reply within the
        # rejection-sampling budget, so the conversation was terminated. This is a THIRD
        # outcome on top of pass/fail -- excluded from the pass-rate denominator so a bad sim
        # is never scored as a solver failure. (Stays False when rejection sampling is off.)
        self.sim_failed = False
        self.sim_failure_turn = -1
        # Per-user-turn rejection audit: one entry per injected sim reply,
        # {"turn": int, "tries": int, "reasons": [str, ...]}. tries==1 & empty reasons means
        # the first sample was clean (no rejection needed).
        self.sim_reject_events = []
        # Per-turn audit trail.
        self.turn_records = []

    def to_dict(self):
        return {
            "row_index": self.row_index,
            "task_id": self.task_id,
            "sample_idx": self.sample_idx,
            "answered": self.answered,
            "answered_at_turn": self.answered_at_turn,
            "num_assistant_turns": self.assistant_turns,
            "num_user_turns": self.user_turns,
            "overflow": self.overflow,
            "response_tokens": self.response_tokens,
            "reward": self.reward,
            "pass_rate": self.reward,
            "all_pass": self.all_pass,
            "num_test_cases": self.num_test_cases,
            "final_answer": self.final_answer,
            "final_code": self.final_code,
            "sim_failed": self.sim_failed,
            "sim_failure_turn": self.sim_failure_turn,
            "sim_reject_events": self.sim_reject_events,
            "turn_records": self.turn_records,
            "messages": self.messages,
        }


def build_trajectories(val_df, n_samples, args, sim_backend=None):
    """Expand each validation row into ``n_samples`` independent trajectories.

    ``sim_backend`` is injectable (default None -> the real HTTP frozen-sim backend); CPU
    tests pass a stub so no server / openai SDK is needed.
    """
    trajs = []
    for row_index, row in val_df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        gt = extra_info.get("ground_truth")
        if gt is None:
            gt = (row.get("reward_model", {}) or {}).get("ground_truth", {})
        task_id = extra_info.get("task_id", extra_info.get("index", row_index))
        # parquet stores the chat prompt as a list/array of {role, content} dicts.
        messages = [dict(m) for m in row["prompt"]]
        # The initial (public) problem turn = the last user message of the prompt; it seeds
        # the simulator's dialogue and carries NO ground truth (mirrors ColBenchAgentLoop).
        problem_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        # verl (HF datasets) hands test_cases as a plain list; a pandas reader gives np.ndarray.
        # Convert via an explicit None check to stay safe under both (per env.py / reward.py).
        _tc = gt.get("test_cases")
        for s in range(n_samples):
            env = ColBenchUserSimEnv(
                problem_description=gt.get("problem_description", problem_text),
                ground_truth=gt["ground_truth"],
                test_cases=list(_tc) if _tc is not None else [],
                max_steps=args.max_assistant_turns,
                reward_time_limit=args.reward_time_limit,
                sim_backend=sim_backend,
            )
            trajs.append(
                Trajectory(
                    row_index=int(row_index),
                    task_id=task_id if task_id is not None else int(row_index),
                    sample_idx=s,
                    messages=messages,
                    problem_text=problem_text,
                    env=env,
                    response_budget=args.max_response_length,
                )
            )
    return trajs


def eval_tagged_path(base_out, turns, n_samples, temperature):
    """Insert a '_turns<N>_n<K>_t<temp>' tag before the extension so each eval config gets
    its own self-documenting file (and can never clobber a different config).

    ``n_samples`` is the temperature-adjusted value actually used (t=0 is forced to 1), so
    the tag faithfully records what was run. e.g.
      ("runs/eval_step120.json", 10, 4, 0.6) -> "runs/eval_step120_turns10_n4_t0.6.json".
    """
    root, ext = os.path.splitext(base_out)
    return f"{root}_turns{turns}_n{n_samples}_t{temperature:g}{ext or '.json'}"


def run_eval(llm, tokenizer, val_df, temperature, n_samples, args, out_path, max_model_len, sim_backend=None):
    """Run the full multi-turn eval at a single temperature and dump one JSON.

    The (expensive) SGLang engine is loaded once by the caller and shared across every
    temperature; only the per-request sampling params and the fresh trajectory state differ
    between calls. ``n_samples`` is the (possibly temperature-adjusted) number of trajectories
    per problem -- greedy decoding (temperature 0) is forced to 1 sample.

    ``llm`` needs a ``.generate(prompt=[...], sampling_params=[...])`` method returning, per
    request, a dict with ``"text"`` and ``"meta_info"["completion_tokens"]`` -- i.e. an
    ``sgl.Engine`` (or a test stub). ``sim_backend`` is forwarded to each env (None = real).
    """
    # SGLang sampling params (dict, applied to every request in a batch). Note the name is
    # ``max_new_tokens`` here, and top_k=-1 means "use the whole vocabulary".
    sampling_params = {
        "temperature": temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens_per_turn,
    }

    trajs = build_trajectories(val_df, n_samples, args, sim_backend=sim_backend)
    # Rejection-sample the sim reply this run? (>0 tries). Constant across turns; defined here
    # so it is always bound for the aggregate section even if the turn loop breaks early.
    reject = bool(args.sim_reject_max_tries and args.sim_reject_max_tries > 0)
    # One pool, reused for the two blocking env calls (grading + sim HTTP turns).
    pool = ThreadPoolExecutor(max_workers=max(1, args.grade_concurrency))

    t0 = time.time()
    # ---- multi-turn loop: every turn generates for ALL still-active trajectories as one
    #      SGLang batch, then grades / advances the sim in parallel. Mirrors the per-turn
    #      structure of ColBenchAgentLoop.run. ----
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

        # Assemble each prompt and clamp its completion budget to the room left in the model's
        # context window (a fixed max_new_tokens on later turns could exceed context and make
        # SGLang raise, killing the run). Same guard as validate_codecontest.py.
        prompts, per_req_params, kept = [], [], []
        for t in gen_batch:
            prompt_text = tokenizer.apply_chat_template(t.messages, add_generation_prompt=True, tokenize=False)
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

        is_last_turn = turn == args.max_assistant_turns - 1

        # Append assistant turns; classify (answered / last-turn / continue).
        pending = []  # trajectories that did not answer and can still get a sim reply
        for t, out in zip(kept, outputs):
            text = out["text"]
            t.messages.append({"role": "assistant", "content": text})
            t.sim_dialogue.append({"role": "assistant", "content": text})
            t.assistant_turns += 1
            t.response_tokens += int(out["meta_info"].get("completion_tokens", 0))

            has_answer, ans = t.env.is_answer(text, episode_done=is_last_turn)
            t.turn_records.append({"turn": turn, "answered": has_answer, "is_last": is_last_turn})
            if has_answer:
                t.answered = True
                t.answered_at_turn = turn
                t.final_answer = ans
                t.final_code = templates.extract_code_answer(ans)
                t._grade_pending = True   # graded in the parallel pass below
            elif is_last_turn:
                # Ran out of turns without a usable submission -> reward stays 0.
                t.done = True
            else:
                pending.append(t)

        # ---- Grade every answered trajectory in parallel (blocking exec sidecar). ----
        to_grade = [t for t in kept if getattr(t, "_grade_pending", False)]

        def _grade(t):
            return t, t.env.score(t.final_answer)

        for t, result in pool.map(_grade, to_grade):
            t._grade_pending = False
            t.reward = float(result.get("pass_rate", 0.0))
            t.all_pass = bool(result.get("all_pass", False))
            t.num_test_cases = int(result.get("n", 0))
            t.done = True

        # ---- Advance the frozen simulator for each still-open trajectory (blocking HTTP,
        #      in parallel). With --sim_reject_max_tries>0 we REJECTION-SAMPLE the sim reply
        #      (resample until it contains no leaked code); otherwise it's the single-shot
        #      turn byte-identical to training via env.generate_user_turn.
        def _sim(t):
            if reject:
                return t, t.env.generate_user_turn_checked(
                    list(t.sim_dialogue), max_tries=args.sim_reject_max_tries,
                    ngram_n=args.sim_reject_ngram_n, min_operators=args.sim_reject_min_ops,
                )
            reply = t.env.generate_user_turn(list(t.sim_dialogue))
            return t, {"reply": reply, "tries": 1, "accepted": True, "reasons": []}

        for t, res in pool.map(_sim, pending):
            if not res["accepted"]:
                # Simulation failure: the sim only ever produced code within the budget. End the
                # conversation and mark it as its own outcome (NOT a solver failure / reward 0).
                t.sim_failed = True
                t.sim_failure_turn = turn
                t.sim_reject_events.append({"turn": turn, "tries": res["tries"],
                                            "reasons": res["reasons"], "accepted": False})
                t.done = True
                continue
            user_content = res["reply"]
            t.sim_reject_events.append({"turn": turn, "tries": res["tries"],
                                        "reasons": res["reasons"], "accepted": True})
            feedback_ids = tokenizer.encode(user_content, add_special_tokens=False)
            # Need room for the user turn AND at least one response token next turn.
            if t.response_tokens + len(feedback_ids) >= t.response_budget:
                t.overflow = True
                t.done = True
                continue
            t.messages.append({"role": "user", "content": user_content})
            t.sim_dialogue.append({"role": "user", "content": user_content})
            t.user_turns += 1
            t.response_tokens += len(feedback_ids)

    pool.shutdown(wait=True)
    elapsed = time.time() - t0

    # ---- aggregate metrics ----
    # Simulation failures (sim never produced a code-free reply within the rejection budget)
    # are a THIRD outcome: they are excluded from the pass-rate denominator so a bad frozen sim
    # is never scored as a solver failure. With rejection off there are none, so `valid` ==
    # `trajs` and every metric below is byte-identical to the pre-rejection behavior.
    n_traj = len(trajs)
    valid = [t for t in trajs if not t.sim_failed]
    n_valid = len(valid)
    n_sim_failures = n_traj - n_valid

    by_problem = {}
    for t in trajs:
        by_problem.setdefault(t.row_index, []).append(t)
    n_problems = len(by_problem)
    valid_by_problem = {}
    for t in valid:
        valid_by_problem.setdefault(t.row_index, []).append(t)

    answered_trajs = [t for t in valid if t.answered]
    mean_pass_rate = sum(t.reward for t in valid) / max(1, n_valid)          # primary (val-core analog)
    all_pass_rate = sum(1 for t in valid if t.all_pass) / max(1, n_valid)     # fully correct
    answered_rate = len(answered_trajs) / max(1, n_valid)
    overflow_rate = sum(1 for t in valid if t.overflow) / max(1, n_valid)
    avg_turns_to_answer = (
        sum(t.answered_at_turn + 1 for t in answered_trajs) / len(answered_trajs)
        if answered_trajs else None
    )
    # Per-problem best (max over VALID samples) -> mean. The pass@n analog for a fractional
    # reward. Problems with no valid sample (all sim-failed) drop out of the denominator.
    mean_best_pass_rate = (
        sum(max(t.reward for t in grp) for grp in valid_by_problem.values())
        / max(1, len(valid_by_problem))
    )
    # First-(valid-)sample metric (comparable to a greedy / single-sample run).
    pass_at_1 = (
        sum(grp[0].reward for grp in valid_by_problem.values()) / max(1, len(valid_by_problem))
    )

    # ---- rejection-sampling audit (aggregated over every injected sim turn) ----
    all_events = [ev for t in trajs for ev in t.sim_reject_events]
    accepted_events = [ev for ev in all_events if ev.get("accepted")]
    n_turns_retried = sum(1 for ev in accepted_events if ev["tries"] > 1)
    total_samples = sum(ev["tries"] for ev in all_events)
    reason_counts: dict = {}
    for ev in all_events:
        for r in ev["reasons"]:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    tries_hist: dict = {}
    for ev in accepted_events:
        tries_hist[ev["tries"]] = tries_hist.get(ev["tries"], 0) + 1
    rejection = {
        "enabled": reject,
        "max_tries": args.sim_reject_max_tries,
        "ngram_n": args.sim_reject_ngram_n,
        "min_operators": args.sim_reject_min_ops,
        "n_sim_failures": n_sim_failures,
        "sim_failure_rate": n_sim_failures / max(1, n_traj),
        "n_user_turns_accepted": len(accepted_events),
        "n_user_turns_with_retry": n_turns_retried,
        "retry_rate": n_turns_retried / max(1, len(accepted_events)),
        "mean_samples_per_turn": round(total_samples / max(1, len(all_events)), 3),
        "max_tries_used": max((ev["tries"] for ev in all_events), default=0),
        "tries_histogram": dict(sorted(tries_hist.items())),
        "reject_reason_counts": reason_counts,
    }

    summary = {
        "model": args.model,
        "val_file": args.val_file,
        "n_problems": n_problems,
        "n_samples_per_problem": n_samples,
        "n_trajectories": n_traj,
        "n_scored_trajectories": n_valid,        # excludes simulation failures
        "n_sim_failures": n_sim_failures,        # third outcome (not pass, not fail)
        "mean_pass_rate": mean_pass_rate,
        "mean_best_pass_rate": mean_best_pass_rate,
        "pass@1_mean_pass_rate": pass_at_1,
        "all_pass_rate": all_pass_rate,
        "answered_rate": answered_rate,
        "avg_turns_to_answer": avg_turns_to_answer,
        "overflow_rate": overflow_rate,
        "elapsed_sec": round(elapsed, 1),
        "rejection_sampling": rejection,
        "inference": {
            "temperature": temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_assistant_turns": args.max_assistant_turns,
            "max_new_tokens_per_turn": args.max_new_tokens_per_turn,
            "max_response_length": args.max_response_length,
            "max_prompt_length": args.max_prompt_length,
            "reward_time_limit": args.reward_time_limit,
            "seed": args.seed,
        },
    }
    print(f"[validate] summary (t={temperature:g}):")
    print(json.dumps(summary, indent=2))

    # ---- Save a RANDOM sample of full conversations (storage-bounded); metrics above are
    #      over ALL trajectories. Seeded so the selection is reproducible. ----
    rng = random.Random(args.seed)
    if args.max_saved_convos is not None and 0 <= args.max_saved_convos < n_traj:
        saved = rng.sample(trajs, args.max_saved_convos)
    else:
        saved = list(trajs)
    # Stable order (problem, sample) for readability.
    saved.sort(key=lambda t: (t.row_index, t.sample_idx))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "num_saved_conversations": len(saved),
            "trajectories": [t.to_dict() for t in saved],
        }, f, indent=2)
    print(f"[validate] wrote {len(saved)}/{n_traj} conversations -> {out_path}")
    return summary


def _load_engine(args, max_model_len):
    """Load the SGLang solver engine (imported lazily so CPU tests never need sglang)."""
    import sglang as sgl
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm = sgl.Engine(
        model_path=args.model,
        dtype=args.dtype,
        tp_size=args.tensor_parallel_size,
        mem_fraction_static=args.gpu_memory_utilization,
        context_length=max_model_len,
        trust_remote_code=args.trust_remote_code,
        random_seed=args.seed,
    )
    return llm, tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Model / data
    ap.add_argument("--model", required=True, help="HF model dir or name (a merged checkpoint).")
    ap.add_argument("--val_file", default=os.path.expanduser("~/data/colbench/test.parquet"),
                    help="Validation/test parquet (VERL schema from preprocess_colbench.py).")
    ap.add_argument("--out", required=True, help="Output JSON path stem for the conversation dump.")
    ap.add_argument("--max_problems", type=int, default=None, help="Limit #problems (debug). None = all.")
    ap.add_argument("--n_samples", type=int, default=1, help="Trajectories sampled per problem.")
    ap.add_argument("--max_saved_convos", type=int, default=1000,
                    help="Cap on how many (randomly sampled) full conversations are written to "
                         "JSON (each carries its per-turn sim_reject_events audit). Metrics are "
                         "still computed over ALL trajectories. <0 = save all.")

    # User-simulator rejection sampling (EVAL only). Resample the sim's reply until it contains
    # no leaked code (templates.detect_code_leak), so the frozen sim can't just hand the solver
    # the solution. Off by default (0 tries) => single-shot turns, byte-identical to training.
    ap.add_argument("--sim_reject_max_tries", type=int, default=0,
                    help="Max sim resamples per user turn (0 disables rejection sampling). On "
                         "exhaustion the conversation is marked a 'simulation failure' (a third "
                         "outcome, excluded from the pass-rate denominator).")
    ap.add_argument("--sim_reject_ngram_n", type=int, default=0,
                    help="Detector (D): reject if a symbol-aware n-gram of this length is shared "
                         "with the hidden GT source (and contains >= --sim_reject_min_ops operators). "
                         "0 (default) DISABLES (D) -- held as a future consideration; A/B still run.")
    ap.add_argument("--sim_reject_min_ops", type=int, default=2,
                    help="Detector (D): min code operators required within the matched n-gram, so "
                         "it fires on copied EXPRESSIONS, not prose that shares identifiers.")

    # Inference hyper-parameters (defaults match run_colbench_grpo.sh so a checkpoint is
    # evaluated with the same budgets it trained under).
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Single sampling temperature (used only when --temperatures is not given).")
    ap.add_argument("--temperatures", type=float, nargs="+", default=None,
                    help="Sweep several temperatures in ONE submission (engine loaded once). Each "
                         "writes its own JSON tagged '<out_stem>_turns<N>_n<K>_t<temp>.json'. "
                         "Overrides --temperature. Example: --temperatures 0.0 0.6")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1, help="-1 disables top-k.")
    ap.add_argument("--seed", type=int, default=0, help="SGLang engine + saved-sample RNG seed.")
    ap.add_argument("--max_assistant_turns", type=int, default=10, help="Total solver turns (clarify + submit).")
    ap.add_argument("--max_new_tokens_per_turn", type=int, default=1024)
    ap.add_argument("--max_response_length", type=int, default=14336,
                    help="Cumulative response-token budget (solver turns + injected sim replies) "
                         "before overflow stop. Default 14336 matches MAX_RESPONSE_LENGTH in "
                         "run_colbench_grpo.sh.")
    ap.add_argument("--max_prompt_length", type=int, default=2048,
                    help="Initial-prompt cap; with --max_response_length sets the engine context.")
    ap.add_argument("--reward_time_limit", type=float, default=6.0,
                    help="Per-case GT exec timeout (seconds) for grading (matches training).")

    # SGLang engine
    ap.add_argument("--tensor_parallel_size", type=int, default=int(os.getenv("ROLLOUT_TP", "1")))
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                    help="SGLang mem_fraction_static (fraction of GPU mem for weights+KV cache).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--grade_concurrency", type=int, default=int(os.getenv("CODECONTEST_EXEC_CONCURRENCY", "32")),
                    help="Threads for the blocking env calls (grading + sim HTTP turns) per turn.")
    args = ap.parse_args()

    if not os.environ.get("CODECONTEST_EXEC_URL") and os.environ.get("CODECONTEST_ALLOW_INPROCESS") != "1":
        raise SystemExit(
            "No code-exec backend configured. Set CODECONTEST_EXEC_URL=http://host:8088 (sidecar) "
            "or CODECONTEST_ALLOW_INPROCESS=1 (in-process dev/smoke) before running."
        )
    if not os.environ.get("OPENAI_BASE_URL"):
        print("[validate] WARNING: OPENAI_BASE_URL is unset; the frozen sim server must be "
              "reachable for generate_user_turn (multi-turn eval will otherwise degrade to "
              "'No response.' replies).")

    # ---- load data ----
    val_df = pd.read_parquet(args.val_file)
    if args.max_problems is not None:
        val_df = val_df.iloc[: args.max_problems].copy()
    val_df = val_df.reset_index(drop=True)
    print(f"[validate] loaded {len(val_df)} problems from {args.val_file}; "
          f"up to {args.n_samples} sample(s) each (temperature 0 forced to 1)")

    # ---- tokenizer + engine ----
    max_model_len = args.max_prompt_length + args.max_response_length
    llm, tokenizer = _load_engine(args, max_model_len)

    # Resolve the temperature sweep: --temperatures wins, else the single --temperature.
    # De-dup while preserving order so a stray repeat doesn't run (and overwrite) twice.
    temps = args.temperatures if args.temperatures else [args.temperature]
    _seen = set()
    temps = [t for t in temps if not (t in _seen or _seen.add(t))]
    print(f"[validate] temperature sweep: {temps} (engine loaded once, reused across all)")

    results_index = []
    for temperature in temps:
        # Greedy decoding produces identical samples, so multiple samples are degenerate.
        n_samples = 1 if temperature == 0.0 else args.n_samples
        if temperature == 0.0 and args.n_samples > 1:
            print(f"[validate] temperature=0 is greedy; forcing n_samples 1 (was {args.n_samples}).")
        out_path = eval_tagged_path(args.out, args.max_assistant_turns, n_samples, temperature)
        summary = run_eval(llm, tokenizer, val_df, temperature, n_samples, args, out_path, max_model_len)
        results_index.append({"temperature": temperature, "out": out_path,
                              "mean_pass_rate": summary["mean_pass_rate"],
                              "all_pass_rate": summary["all_pass_rate"]})

    llm.shutdown()

    print("[validate] sweep complete:")
    print(json.dumps(results_index, indent=2))


if __name__ == "__main__":
    main()

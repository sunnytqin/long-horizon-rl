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
"""Full solver<->sim simulation + eval for the SPEC path (Phase 1).

The spec-path sibling of ``colbench.validate_colbench``. It drives a REAL multi-turn
conversation: the solver (assistant) and the user-simulator are BOTH served models, and the
episode ends when the *sim* emits ``[TERMINATE]`` (user-driven termination) -- we then grade the
last function the solver showed against the hidden GT (unchanged grading). This is the harness we
use to (a) eyeball whether the spec sim honors its plot in a genuine back-and-forth, and (b)
measure the extraction-through-dialogue solve rate.

Two solver backends, ONE termination loop:
  * ``--solver_backend openai`` -- the solver is an OpenAI-API call to a served model (reuses
    ``selfplay.llm_client.ChatEndpoint``). Run BOTH roles against the SAME served base model in
    the ``openrlhf`` conda env (no container, in-process grading) for a cheap first eyeball.
  * ``--solver_backend sglang`` -- the solver is an offline ``sgl.Engine`` (a merged checkpoint),
    the production eval inside the VERL/SGLang container. Same loop, same dumps.

The frozen user-simulator is always the HTTP server ``colbench.env_spec.ColBenchSpecUserSimEnv``
calls (OPENAI_BASE_URL / MULTITURN_MODEL_NAME) -- in the openai path that's the same server the
solver uses. The GT source is passed ONLY inside the sim prompt and never enters the solver's
message list; there is no code leak possible here (the sim conditions on the NL spec, not code).

Grading backend (identical to training / the GT validator):
  - Sidecar sandbox: export CODECONTEST_EXEC_URL=http://host:8088   (preferred)
  - In-process fallback (dev/eyeball, no sidecar): export CODECONTEST_ALLOW_INPROCESS=1

Example (conda env, base Qwen3-4B on both roles, one vLLM server on :30000):
    export CODECONTEST_ALLOW_INPROCESS=1
    export OPENAI_BASE_URL=http://127.0.0.1:30000/v1 MULTITURN_MODEL_NAME=colbench-sim
    PYTHONPATH=$(pwd) python colbench/validate_colbench_spec.py \
        --solver_backend openai --base_url http://127.0.0.1:30000/v1 --served_model colbench-sim \
        --model /path/to/Qwen3-4B-Instruct-2507 \
        --val_file ~/data/colbench_spec/selfplay_cond30.parquet \
        --out runs/spec_eval_cond30.json --n_samples 2 --temperature 0.6
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from colbench import templates
from colbench.env_spec import ColBenchSpecUserSimEnv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Same strict-under-context margin as the GT validator (SGLang rejects input+max_new==ctx).
CTX_MARGIN = 8


# ----------------------------------------------------------------------------- #
# Per-trajectory state (one per (problem, sample)). Carries the growing solver conversation,
# the GT-free sim dialogue, its spec env, and the user-driven-termination bookkeeping.
# ----------------------------------------------------------------------------- #
class Trajectory:
    def __init__(self, row_index, task_id, sample_idx, messages, problem_text, env, response_budget):
        self.row_index = row_index
        self.task_id = task_id
        self.sample_idx = sample_idx
        self.messages = [dict(m) for m in messages]           # solver view: [system, user(problem), ...]
        self.sim_dialogue = [{"role": "user", "content": problem_text}]  # sim view (no GT)
        self.env = env
        self.response_budget = response_budget

        self.response_tokens = 0
        self.assistant_turns = 0
        self.user_turns = 0
        self.code_proposals = 0
        self.showed_code = False
        self.last_code = ""
        self.first_code = ""          # the assistant's FIRST code proposal (for the lift diagnostic)
        self.first_code_reward = 0.0  # pass_rate of first_code, graded post-loop
        self.first_code_all_pass = False
        self.done = False
        self.overflow = False
        # How the episode ended: "user" (sim [TERMINATE]) | "code_cap" | "turn_cap" | "no_code"
        # (terminated/ran out with the solver never having shown a function -> reward 0).
        self.terminated_by = None
        self.reward = 0.0
        self.all_pass = False
        self.num_test_cases = 0
        self.turn_records = []
        self.sim_code_rejected = 0   # count of code-writing sim replies discarded (rejection sampling)
        self._grade_pending = False

    def to_dict(self):
        return {
            "row_index": self.row_index, "task_id": self.task_id, "sample_idx": self.sample_idx,
            "terminated_by": self.terminated_by, "num_assistant_turns": self.assistant_turns,
            "num_user_turns": self.user_turns, "code_proposals": self.code_proposals,
            "showed_code": self.showed_code, "overflow": self.overflow,
            "sim_code_rejected": self.sim_code_rejected,
            "response_tokens": self.response_tokens, "reward": self.reward, "pass_rate": self.reward,
            "all_pass": self.all_pass, "num_test_cases": self.num_test_cases,
            "first_code_pass_rate": self.first_code_reward, "first_code_all_pass": self.first_code_all_pass,
            "first_code": self.first_code,
            "final_code": self.last_code, "turn_records": self.turn_records, "messages": self.messages,
        }


# ----------------------------------------------------------------------------- #
# Solver backends -- both expose generate(message_lists, tokenizer, max_model_len, sampling)
# -> list of {"text": str, "tokens": int}. The termination loop is backend-agnostic.
# ----------------------------------------------------------------------------- #
class OpenAISolver:
    """Solver = OpenAI-API chat calls to a served model (reuses ChatEndpoint), threaded per req."""

    def __init__(self, base_url, model, temperature, top_p, top_k, max_new_tokens,
                 enable_thinking, concurrency):
        from colbench.selfplay.llm_client import ChatEndpoint
        self._ep = ChatEndpoint(
            base_url=base_url, model=model, temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_new_tokens, enable_thinking=enable_thinking, vendor="vllm",
        )
        self._pool = ThreadPoolExecutor(max_workers=max(1, concurrency))

    def generate(self, message_lists, tokenizer, max_model_len):
        texts = list(self._pool.map(self._ep.chat, message_lists))
        out = []
        for t in texts:
            n = len(tokenizer.encode(t, add_special_tokens=False)) if tokenizer is not None else 0
            out.append({"text": t, "tokens": n})
        return out

    def shutdown(self):
        self._pool.shutdown(wait=True)


class SGLangSolver:
    """Solver = offline sgl.Engine (a merged checkpoint), batched, with per-turn context clamp."""

    def __init__(self, engine, template_kwargs, temperature, top_p, top_k, max_new_tokens):
        self._llm = engine
        self._tk = template_kwargs
        self._sp = {"temperature": temperature, "top_p": top_p, "top_k": top_k,
                    "max_new_tokens": max_new_tokens}
        self._max_new = max_new_tokens

    def generate(self, message_lists, tokenizer, max_model_len):
        prompts, params = [], []
        for msgs in message_lists:
            ptext = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **self._tk)
            room = max_model_len - len(tokenizer.encode(ptext, add_special_tokens=False)) - CTX_MARGIN
            sp = dict(self._sp)
            sp["max_new_tokens"] = max(1, min(self._max_new, room))
            prompts.append(ptext)
            params.append(sp)
        outs = self._llm.generate(prompt=prompts, sampling_params=params)
        return [{"text": o["text"], "tokens": int(o["meta_info"].get("completion_tokens", 0))} for o in outs]

    def shutdown(self):
        self._llm.shutdown()


def _solver_template_kwargs():
    """apply_chat_template kwargs for the SOLVER from SOLVER_ENABLE_THINKING (see GT validator)."""
    v = os.environ.get("SOLVER_ENABLE_THINKING", "").strip().lower()
    if v in ("true", "1"):
        return {"enable_thinking": True}
    if v in ("false", "0"):
        return {"enable_thinking": False}
    return {}


def _solver_thinking():
    v = os.environ.get("SOLVER_ENABLE_THINKING", "").strip().lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


def build_trajectories(val_df, n_samples, args, sim_backend=None):
    """Expand each spec row into ``n_samples`` independent trajectories (spec env per traj)."""
    trajs = []
    for row_index, row in val_df.iterrows():
        extra_info = row.get("extra_info", {}) or {}
        gt = extra_info.get("ground_truth")
        if gt is None:
            gt = (row.get("reward_model", {}) or {}).get("ground_truth", {})
        spec = extra_info.get("spec", {}) or {}
        task_id = extra_info.get("index", row_index)
        messages = [dict(m) for m in row["prompt"]]
        problem_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        _tc = gt.get("test_cases")
        for s in range(n_samples):
            env = ColBenchSpecUserSimEnv(
                problem_description=gt.get("problem_description", problem_text),
                spec=spec, ground_truth=gt["ground_truth"],
                test_cases=list(_tc) if _tc is not None else [],
                max_steps=args.max_assistant_turns, reward_time_limit=args.reward_time_limit,
                sim_backend=sim_backend, sim_max_tries=args.sim_max_tries,
            )
            trajs.append(Trajectory(int(row_index), task_id if task_id is not None else int(row_index),
                                    s, messages, problem_text, env, args.max_response_length))
    return trajs


def eval_tagged_path(base_out, turns, n_samples, temperature, max_code):
    """'_turns<N>_n<K>_t<temp>_cc<C>' tag so each config gets its own self-documenting file."""
    root, ext = os.path.splitext(base_out)
    return f"{root}_turns{turns}_n{n_samples}_t{temperature:g}_cc{max_code}{ext or '.json'}"


def run_eval(solver, tokenizer, val_df, temperature, n_samples, args, out_path, max_model_len,
             sim_backend=None):
    """Run the full spec simulation at one temperature and dump one JSON.

    The termination loop mirrors ``colbench.tests.test_env_spec.drive`` (the pinned contract) and
    the eventual ``colbench_spec_agent``: solver turn -> track last code / count proposals ->
    turn cap -> code cap -> else sim reply -> [TERMINATE]. Grade the last shown function; reward 0
    (terminated_by 'no_code') if the solver never showed code.
    """
    trajs = build_trajectories(val_df, n_samples, args, sim_backend=sim_backend)
    pool = ThreadPoolExecutor(max_workers=max(1, args.grade_concurrency))
    t0 = time.time()

    for turn in range(args.max_assistant_turns):
        active = [t for t in trajs if not t.done]
        if not active:
            break
        # Budget check before generating.
        gen_batch = []
        for t in active:
            if t.response_tokens >= t.response_budget:
                t.overflow = True
                t.done = True
                t.terminated_by = t.terminated_by or ("turn_cap" if t.showed_code else "no_code")
                if t.showed_code:
                    t._grade_pending = True
            else:
                gen_batch.append(t)
        if not gen_batch:
            # still grade any just-marked overflow trajectories below
            pass
        else:
            print(f"[validate_spec] t={temperature:g} turn {turn}: generating for {len(gen_batch)} active")
            outs = solver.generate([t.messages for t in gen_batch], tokenizer, max_model_len)
            is_last = turn == args.max_assistant_turns - 1
            pending = []
            for t, out in zip(gen_batch, outs):
                text = out["text"]
                t.messages.append({"role": "assistant", "content": text})
                t.sim_dialogue.append({"role": "assistant", "content": text})
                t.assistant_turns += 1
                t.response_tokens += int(out.get("tokens", 0))
                showed = templates.contains_code(text)
                if showed:
                    t.showed_code = True
                    t.last_code = templates.extract_last_code(t.sim_dialogue)
                    t.code_proposals += 1
                    if not t.first_code:            # remember the assistant's FIRST proposal
                        t.first_code = t.last_code
                t.turn_records.append({"turn": turn, "showed_code": showed})
                if is_last:
                    t.done = True
                    t.terminated_by = "turn_cap" if t.showed_code else "no_code"
                    t._grade_pending = t.showed_code
                elif t.code_proposals >= args.max_code_proposals:
                    t.done = True
                    t.terminated_by = "code_cap"
                    t._grade_pending = True
                else:
                    pending.append(t)

            # ---- Advance the frozen sim for each still-open trajectory (parallel HTTP). ----
            def _sim(t):
                reply = t.env.generate_user_turn(list(t.sim_dialogue))
                return (t, reply, t.env.last_sim_raw, t.env.last_sim_code_rejected,
                        t.env.last_sim_code_reject_exhausted)

            for t, reply, raw, code_rejected, exhausted in pool.map(_sim, pending):
                t.sim_code_rejected += code_rejected
                if exhausted:
                    # Every retry still wrote code -> abort this conversation for inspection.
                    # Save the offending (uncapped) reply so it shows in the transcript dump.
                    t.messages.append({"role": "user", "content": raw})
                    t.done = True
                    t.terminated_by = "sim_code_reject"
                    t._grade_pending = t.showed_code
                    continue
                if templates.sim_terminated(raw):
                    t.done = True
                    t.terminated_by = "user" if t.showed_code else "no_code"
                    t._grade_pending = t.showed_code
                    continue
                feedback_tokens = len(tokenizer.encode(reply, add_special_tokens=False)) if tokenizer else len(reply) // 4
                if t.response_tokens + feedback_tokens >= t.response_budget:
                    t.overflow = True
                    t.done = True
                    t.terminated_by = "turn_cap" if t.showed_code else "no_code"
                    t._grade_pending = t.showed_code
                    continue
                t.messages.append({"role": "user", "content": reply})
                t.sim_dialogue.append({"role": "user", "content": reply})
                t.user_turns += 1
                t.response_tokens += feedback_tokens

        # ---- Grade every trajectory that finished THIS turn with a shown function (parallel). ----
        to_grade = [t for t in trajs if t._grade_pending]

        def _grade(t):
            return t, t.env.score(t.last_code)

        for t, result in pool.map(_grade, to_grade):
            t._grade_pending = False
            t.reward = float(result.get("pass_rate", 0.0))
            t.all_pass = bool(result.get("all_pass", False))
            t.num_test_cases = int(result.get("n", 0))

    # ---- Grade each trajectory's FIRST code proposal (the lift diagnostic: how much the sim's
    #      feedback improved the final code over the assistant's first attempt). ----
    first_to_grade = [t for t in trajs if t.first_code]

    def _grade_first(t):
        return t, t.env.score(t.first_code)

    for t, result in pool.map(_grade_first, first_to_grade):
        t.first_code_reward = float(result.get("pass_rate", 0.0))
        t.first_code_all_pass = bool(result.get("all_pass", False))

    pool.shutdown(wait=True)
    elapsed = time.time() - t0

    summary = _aggregate(trajs, temperature, n_samples, args, elapsed)
    print(f"[validate_spec] summary (t={temperature:g}):")
    print(json.dumps(summary, indent=2))

    rng = random.Random(args.seed)
    n_traj = len(trajs)
    if args.max_saved_convos is not None and 0 <= args.max_saved_convos < n_traj:
        saved = rng.sample(trajs, args.max_saved_convos)
    else:
        saved = list(trajs)
    saved.sort(key=lambda t: (t.row_index, t.sample_idx))
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "num_saved_conversations": len(saved),
                   "trajectories": [t.to_dict() for t in saved]}, f, indent=2)
    print(f"[validate_spec] wrote {len(saved)}/{n_traj} conversations -> {out_path}")

    # Always dump the sim-code-reject ABORTS (all rejection tries wrote code) to a sidecar file so
    # this fidelity failure is tracked every run -- these are the conversations to read/fix.
    aborts = [t for t in trajs if t.terminated_by == "sim_code_reject"]
    aborts_path = os.path.splitext(out_path)[0] + ".aborts.txt"
    with open(aborts_path, "w") as f:
        f.write(f"{len(aborts)} sim_code_reject aborts / {n_traj} trajectories "
                f"(sim={summary.get('sim_model')}, sim_max_tries={args.sim_max_tries})\n")
        for i, t in enumerate(sorted(aborts, key=lambda t: (t.row_index, t.sample_idx)), 1):
            f.write("\n" + "#" * 100 + "\n")
            f.write(f"ABORT {i}/{len(aborts)}  row={t.row_index} sample={t.sample_idx}  "
                    f"turns={t.assistant_turns} proposals={t.code_proposals} "
                    f"sim_code_rejected={t.sim_code_rejected}  final_pass={t.reward}\n" + "#" * 100 + "\n")
            for m in t.messages:
                if m.get("role") == "system":
                    continue
                f.write(f"\n----------[{m['role'].upper()}]----------\n{m['content']}\n")
    print(f"[validate_spec] {len(aborts)} sim_code_reject aborts -> {aborts_path}")
    return summary


def _aggregate(trajs, temperature, n_samples, args, elapsed):
    n = len(trajs)
    by_problem = {}
    for t in trajs:
        by_problem.setdefault(t.row_index, []).append(t)

    term_counts = {}
    for t in trajs:
        term_counts[t.terminated_by] = term_counts.get(t.terminated_by, 0) + 1

    mean_pass = sum(t.reward for t in trajs) / max(1, n)
    all_pass = sum(1 for t in trajs if t.all_pass) / max(1, n)
    showed = [t for t in trajs if t.showed_code]
    showed_rate = len(showed) / max(1, n)

    # Lift diagnostic: FIRST-code vs FINAL-code pass rate. Computed both over ALL trajectories
    # (no-code counts 0, apples-to-apples with mean_pass_rate) and over the showed-code subset
    # (isolates the trajectories where iteration could actually help). mean_pass - first = the
    # improvement attributable to the sim's feedback + the assistant's revisions.
    mean_first_pass = sum(t.first_code_reward for t in trajs) / max(1, n)
    first_all_pass = sum(1 for t in trajs if t.first_code_all_pass) / max(1, n)
    mean_first_pass_showed = (sum(t.first_code_reward for t in showed) / len(showed)) if showed else None
    mean_final_pass_showed = (sum(t.reward for t in showed) / len(showed)) if showed else None
    mean_code_proposals = sum(t.code_proposals for t in trajs) / max(1, n)
    mean_turns = sum(t.assistant_turns for t in trajs) / max(1, n)
    overflow_rate = sum(1 for t in trajs if t.overflow) / max(1, n)
    # Fidelity guard: how often the sim tried to write code (an ordinary user shouldn't). Total
    # discarded code replies + the fraction of trajectories that hit >=1 rejection.
    total_code_rejected = sum(t.sim_code_rejected for t in trajs)
    sim_code_reject_traj_rate = sum(1 for t in trajs if t.sim_code_rejected > 0) / max(1, n)
    sim_code_reject_aborted = sum(1 for t in trajs if t.terminated_by == "sim_code_reject")

    # The imperfect-signal cost: among USER-terminated trajectories (the sim decided it was
    # satisfied), how often did the graded code NOT fully pass GT. High -> the sim is quitting on
    # wrong code; low -> user-termination tracks correctness well enough.
    user_term = [t for t in trajs if t.terminated_by == "user"]
    false_terminate_rate = (sum(1 for t in user_term if not t.all_pass) / len(user_term)) if user_term else None

    mean_best = sum(max(t.reward for t in grp) for grp in by_problem.values()) / max(1, len(by_problem))
    pass_at_1 = sum(grp[0].reward for grp in by_problem.values()) / max(1, len(by_problem))

    return {
        "model": args.model, "solver_backend": args.solver_backend, "val_file": args.val_file,
        "sim_backend": args.sim_backend,
        "sim_model": (args.sim_model if args.sim_backend == "openai" else os.environ.get("MULTITURN_MODEL_NAME", "")),
        "n_problems": len(by_problem), "n_samples_per_problem": n_samples, "n_trajectories": n,
        "mean_pass_rate": mean_pass, "mean_best_pass_rate": mean_best, "pass@1_mean_pass_rate": pass_at_1,
        "all_pass_rate": all_pass, "showed_code_rate": showed_rate,
        # First-vs-final code lift (how much the sim's feedback + revisions helped):
        "mean_first_code_pass_rate": mean_first_pass, "first_code_all_pass_rate": first_all_pass,
        "feedback_lift_pass_rate": round(mean_pass - mean_first_pass, 4),
        "mean_first_code_pass_rate_showed": (round(mean_first_pass_showed, 4) if mean_first_pass_showed is not None else None),
        "mean_final_code_pass_rate_showed": (round(mean_final_pass_showed, 4) if mean_final_pass_showed is not None else None),
        "mean_code_proposals": round(mean_code_proposals, 3), "mean_assistant_turns": round(mean_turns, 3),
        "overflow_rate": overflow_rate,
        "sim_code_rejected_total": total_code_rejected,
        "sim_code_reject_traj_rate": round(sim_code_reject_traj_rate, 3),
        "sim_code_reject_aborted": sim_code_reject_aborted,   # conversations aborted (all retries wrote code)
        "terminated_by_counts": dict(sorted(term_counts.items(), key=lambda kv: str(kv[0]))),
        "false_terminate_rate": false_terminate_rate,   # user-terminated AND GT-failed
        "elapsed_sec": round(elapsed, 1),
        "inference": {
            "temperature": temperature, "top_p": args.top_p, "top_k": args.top_k,
            "max_assistant_turns": args.max_assistant_turns, "max_code_proposals": args.max_code_proposals,
            "max_new_tokens_per_turn": args.max_new_tokens_per_turn,
            "max_response_length": args.max_response_length, "reward_time_limit": args.reward_time_limit,
            "seed": args.seed,
        },
    }


def _load_tokenizer(args):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    except Exception as e:  # noqa: BLE001 - openai mode can proceed without exact token counts
        print(f"[validate_spec] WARNING: could not load tokenizer from {args.model!r} ({e}); "
              "token budgets will be approximate.")
        return None


def build_solver(args, tokenizer, max_model_len):
    thinking = _solver_thinking()
    if args.solver_backend == "openai":
        return OpenAISolver(
            base_url=args.base_url, model=args.served_model or args.model,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens_per_turn, enable_thinking=thinking,
            concurrency=args.solver_concurrency,
        )
    import sglang as sgl
    engine = sgl.Engine(
        model_path=args.model, dtype=args.dtype, tp_size=args.tensor_parallel_size,
        mem_fraction_static=args.gpu_memory_utilization, context_length=max_model_len,
        trust_remote_code=args.trust_remote_code, random_seed=args.seed,
    )
    return SGLangSolver(engine, _solver_template_kwargs(), args.temperature, args.top_p,
                        args.top_k, args.max_new_tokens_per_turn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver_backend", choices=["openai", "sglang"], default="openai",
                    help="openai = API solver (conda/eyeball, both roles same server); sglang = "
                         "offline Engine (container/production, a merged checkpoint).")
    ap.add_argument("--model", required=True,
                    help="HF dir: the sglang engine path AND the tokenizer (openai mode uses it "
                         "only for token counting).")
    ap.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30000/v1"),
                    help="openai solver endpoint (usually the same server as the sim).")
    ap.add_argument("--served_model", default=os.environ.get("MULTITURN_MODEL_NAME", ""),
                    help="openai served-model alias for the solver (defaults to MULTITURN_MODEL_NAME).")
    ap.add_argument("--val_file", default=os.path.expanduser("~/data/colbench_spec/selfplay_cond30.parquet"),
                    help="Spec parquet from preprocess_colbench_spec.py.")
    ap.add_argument("--out", required=True, help="Output JSON path stem.")
    ap.add_argument("--max_problems", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--max_saved_convos", type=int, default=1000)

    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--temperatures", type=float, nargs="+", default=None,
                    help="Sweep several temperatures in one run (engine loaded once).")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_assistant_turns", type=int, default=10)
    ap.add_argument("--max_code_proposals", type=int, default=2,
                    help="Guardrail: force-terminate after this many solver code proposals.")
    ap.add_argument("--max_new_tokens_per_turn", type=int, default=1024)
    ap.add_argument("--max_response_length", type=int, default=14336)
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--reward_time_limit", type=float, default=6.0)

    ap.add_argument("--solver_concurrency", type=int, default=int(os.getenv("SOLVER_CONCURRENCY", "16")),
                    help="Threads for the openai solver (per-turn parallel requests).")
    ap.add_argument("--grade_concurrency", type=int, default=int(os.getenv("CODECONTEST_EXEC_CONCURRENCY", "32")),
                    help="Threads for grading + sim HTTP turns.")
    # sglang engine (ignored in openai mode)
    ap.add_argument("--tensor_parallel_size", type=int, default=int(os.getenv("ROLLOUT_TP", "1")))
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--trust_remote_code", action="store_true")
    # ── Frozen user-simulator backend ────────────────────────────────────────
    # 'vllm': the sim hits OPENAI_BASE_URL/MULTITURN_MODEL_NAME (the local Qwen server; in openai
    #   solver mode that's the SAME server as the solver -- the default self-play setup).
    # 'openai': the sim hits a REAL OpenAI endpoint (--sim_base_url/--sim_model, OPENAI_API_KEY),
    #   DECOUPLED from the solver -> Qwen solver vs GPT sim comparison. No vLLM sampling extras sent.
    ap.add_argument("--sim_backend", choices=["vllm", "openai"], default="vllm",
                    help="Frozen user-simulator backend: 'vllm' (local server) or 'openai' (hosted GPT).")
    ap.add_argument("--sim_model", default=os.environ.get("SIM_OPENAI_MODEL", "gpt-5.4-mini"),
                    help="Model name for --sim_backend openai (e.g. gpt-5.4-mini, gpt-4o-mini).")
    ap.add_argument("--sim_base_url", default=os.environ.get("SIM_OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    help="Base URL for --sim_backend openai.")
    ap.add_argument("--sim_temperature", type=float, default=float(os.environ.get("SIM_TEMPERATURE", "1.0")))
    ap.add_argument("--sim_top_p", type=float, default=float(os.environ.get("SIM_TOP_P", "1.0")))
    ap.add_argument("--sim_max_tries", type=int, default=int(os.environ.get("SIM_MAX_TRIES", "8")),
                    help="Rejection-sampling tries when the sim writes code; if all fail, the "
                         "conversation is aborted (terminated_by 'sim_code_reject') for inspection.")
    ap.add_argument("--sim_max_tokens", type=int, default=int(os.environ.get("SIM_MAX_TOKENS", "256")),
                    help="Generation-time token bound on each user (sim) turn. Replaces the old "
                         "post-hoc character truncation; the vllm backend reads SIM_MAX_TOKENS too.")
    args = ap.parse_args()

    if not os.environ.get("CODECONTEST_EXEC_URL") and os.environ.get("CODECONTEST_ALLOW_INPROCESS") != "1":
        raise SystemExit("No code-exec backend: set CODECONTEST_EXEC_URL or CODECONTEST_ALLOW_INPROCESS=1.")
    # The sim always reads OPENAI_BASE_URL / MULTITURN_MODEL_NAME; in openai mode default them from
    # the solver args so BOTH roles hit the same server without extra env plumbing.
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if not os.environ.get("MULTITURN_MODEL_NAME") and args.served_model:
        os.environ["MULTITURN_MODEL_NAME"] = args.served_model
    if not os.environ.get("MULTITURN_MODEL_NAME"):
        print("[validate_spec] WARNING: MULTITURN_MODEL_NAME unset; the sim server call needs a "
              "served model name (pass --served_model or export MULTITURN_MODEL_NAME).")

    val_df = pd.read_parquet(os.path.expanduser(args.val_file))
    if args.max_problems is not None:
        val_df = val_df.iloc[: args.max_problems].copy()
    val_df = val_df.reset_index(drop=True)
    print(f"[validate_spec] loaded {len(val_df)} spec problems from {args.val_file}; "
          f"solver_backend={args.solver_backend}")

    max_model_len = args.max_prompt_length + args.max_response_length
    tokenizer = _load_tokenizer(args)
    solver = build_solver(args, tokenizer, max_model_len)

    # sim_backend=None -> the env default (env.openai_sim_backend, reads OPENAI_BASE_URL). For the
    # 'openai' sim we build a decoupled hosted-GPT backend so the solver stays on the Qwen server.
    sim_backend = None
    if args.sim_backend == "openai":
        from colbench.env_spec import make_openai_sim_backend
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("[validate_spec] --sim_backend openai needs OPENAI_API_KEY exported.")
        print(f"[validate_spec] SIM = OpenAI '{args.sim_model}' @ {args.sim_base_url} "
              f"(temp={args.sim_temperature}, top_p={args.sim_top_p}); SOLVER stays on "
              f"{args.solver_backend}.")
        sim_backend = make_openai_sim_backend(
            base_url=args.sim_base_url, model=args.sim_model, api_key=api_key,
            temperature=args.sim_temperature, top_p=args.sim_top_p, max_tokens=args.sim_max_tokens,
        )
    # The vllm sim backend (env.openai_sim_backend) reads SIM_MAX_TOKENS from the env; mirror the
    # CLI value there so both backends honor the same bound.
    os.environ["SIM_MAX_TOKENS"] = str(args.sim_max_tokens)

    temps = args.temperatures if args.temperatures else [args.temperature]
    _seen = set()
    temps = [t for t in temps if not (t in _seen or _seen.add(t))]

    results_index = []
    for temperature in temps:
        n_samples = 1 if temperature == 0.0 else args.n_samples
        # NOTE: openai solver sampling is fixed at construction; for a temperature sweep the openai
        # backend would need rebuilding per temp. sglang re-uses the engine with per-request temp.
        if args.solver_backend == "openai" and len(temps) > 1:
            args.temperature = temperature
            solver.shutdown()
            solver = build_solver(args, tokenizer, max_model_len)
        out_path = eval_tagged_path(args.out, args.max_assistant_turns, n_samples, temperature, args.max_code_proposals)
        summary = run_eval(solver, tokenizer, val_df, temperature, n_samples, args, out_path, max_model_len,
                           sim_backend=sim_backend)
        results_index.append({"temperature": temperature, "out": out_path,
                              "mean_pass_rate": summary["mean_pass_rate"],
                              "false_terminate_rate": summary["false_terminate_rate"]})

    solver.shutdown()
    print("[validate_spec] sweep complete:")
    print(json.dumps(results_index, indent=2))


if __name__ == "__main__":
    main()

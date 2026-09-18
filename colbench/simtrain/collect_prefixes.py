r"""Stage 1: run a HACKING partner assistant against the frozen sim and record
every user-turn prompt it produces.

The output is the "task dataset" for simulator training: one record per user
turn, each carrying the EXACT rendered simulator prompt (system + user) that the
frozen sim was asked at that point in the dialogue. Everything downstream
(candidate draws, judging, SFT rows, the base-vs-SFT eval) reads that prompt
verbatim, so the four stages can never disagree about what the sim was asked --
there is no parquet join anywhere after this script.

WHY A NEW SCRIPT AND NOT AN EXTENSION OF ``validate_colbench.py``: that file is
the GT-arm YARDSTICK whose numbers back every RL run. Its output is
episode-granular, it never serializes ``ground_truth`` / ``problem_description``
/ ``sim_dialogue``, it subsamples to ``--max_saved_convos``, and it writes once
at the end. Different writer, different resume key -- and this collector needs
no grading, no exec sidecar and no in-process GPU engine at all, so sharing the
file would mean bolting a second mode onto the one thing that must stay stable.

Both roles are served HTTP endpoints, so the shape is one worker THREAD per
episode with a sequential turn loop; the servers do the batching. Each completed
episode is flushed, so a ``scancel`` costs at most one episode per worker.

Two output files:
  * ``prefixes.<tag>.jsonl``  -- one record per user turn (the dataset).
  * ``episodes.<tag>.jsonl``  -- one record per episode. THIS is the resume key:
    an episode where the solver submits on turn 0 yields ZERO prefixes, so
    resuming off the prefix file alone would re-run it forever.

Example (both roles on locally served endpoints):
    python -m colbench.simtrain.collect_prefixes \
        --data_file $DATA_ROOT/colbench/train.fence.parquet --end 600 \
        --solver_base_url http://127.0.0.1:30000/v1 --solver_model partner-gs200 \
        --sim_base_url    http://127.0.0.1:30001/v1 --sim_model    colbench-sim \
        --out_dir $SIMTRAIN_ROOT/gs200
"""

# This tree imports names directly (``from colbench.env import
# ColBenchUserSimEnv``) rather than the enclosing module, matching how the
# rest of verl is written; call sites read on the bare name throughout.
# pylint: disable=g-importing-member
import argparse
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import threading
import time
from typing import Any
from typing import Optional

import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# The module-level setup above (sys.path) has to run before these imports
# resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench import templates
from colbench.env import ColBenchUserSimEnv
from colbench.env_spec import ColBenchSpecUserSimEnv
from colbench.selfplay.dataio import append_jsonl
from colbench.selfplay.dataio import read_jsonl
from colbench.selfplay.llm_client import ChatEndpoint
from colbench.simtrain import COLLECTOR_VERSION

# One lock for BOTH output files. Worker threads finish out of order, and the
# two appends for one episode (prefixes then the episode record) must not be
# interleaved with another worker's -- otherwise a truncating kill can leave a
# prefix block split around a foreign episode record.
_WRITE_LOCK = threading.Lock()


def sha16(text: str) -> str:
  """First 16 hex chars of the sha256 of ``text``.

  Used as a cheap fingerprint of the rendered sim prompt, so a downstream stage
  can assert it is scoring the bytes this collector actually asked.

  Args:
    text: any string.

  Returns:
    16 lowercase hex characters.
  """
  return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def read_rows(
    data_file: str, start: int = 0, end: Optional[int] = None
) -> list[dict[str, Any]]:
  """Load ColBench parquet rows as task payloads, INCLUDING the solver prompt.

  Deliberately not ``selfplay.dataio.read_tasks``: that returns the minimal
  ``{index, problem_description, ground_truth, test_cases}`` payload and drops
  the parquet's ``prompt`` column. The prompt column is the solver's
  ``[system, user]`` opening verbatim, and using anything else here would make
  the collected dialogues differ from training on the first token.

  Extraction mirrors ``validate_colbench.build_trajectories``.

  Args:
    data_file: the preprocessed ColBench parquet.
    start: first row position to take.
    end: one past the last row position; ``None`` takes everything.

  Returns:
    One ``{task_index, task_id, prompt_messages, problem_text,
    problem_description, ground_truth, test_cases}`` per row.
  """
  df = pd.read_parquet(os.path.expanduser(data_file))
  end = len(df) if end is None else min(end, len(df))
  rows = []
  for pos in range(start, end):
    row = df.iloc[pos]
    extra_info = row.get("extra_info", {}) or {}
    gt = extra_info.get("ground_truth")
    if gt is None:
      gt = (row.get("reward_model", {}) or {}).get("ground_truth", {})
    task_id = extra_info.get("task_id", extra_info.get("index", pos))
    messages = [dict(m) for m in row["prompt"]]
    # The initial (public) problem turn = the last user message of the prompt.
    # It seeds the simulator's dialogue and carries NO ground truth.
    problem_text = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    # verl (HF datasets) hands test_cases as a plain list; pandas gives an
    # np.ndarray. Convert via an explicit None check to stay safe under both.
    _tc = gt.get("test_cases")
    rows.append({
        "task_index": pos,
        "task_id": int(task_id) if task_id is not None else pos,
        "prompt_messages": messages,
        "problem_text": problem_text,
        "problem_description": gt.get("problem_description", problem_text),
        "ground_truth": gt["ground_truth"],
        "test_cases": list(_tc) if _tc is not None else [],
        # Spec-path only. The GROUNDED sim reads the GT source plus `plot`;
        # `plot` lives in extra_info["spec"], which the GT path never touches.
        # Absent on a GT parquet -> {} -> an empty plot, which the grounded
        # prompt renders as no plot at all.
        "spec": extra_info.get("spec") or {},
    })
  return rows


def make_sim_backend(endpoint: ChatEndpoint):
  """Adapt a ``ChatEndpoint`` to ``env.SimBackend``'s (system, user) signature.

  Args:
    endpoint: the served simulator.

  Returns:
    A ``(system_content, user_content) -> raw reply`` callable the env can use
    in place of its default ``openai_sim_backend``.
  """

  def backend(system_content: str, user_content: str) -> str:
    return endpoint.chat([
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ])

  return backend


def run_episode(
    task: dict[str, Any],
    episode_idx: int,
    solver_chat,
    sim_backend,
    max_assistant_turns: int,
    provenance: dict[str, Any],
    sim_prompt: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Play one full partner-vs-sim episode, materializing a prefix per user turn.

  The turn loop is byte-equivalent to ``validate_colbench``'s and to
  ``ColBenchAgentLoop.run``, minus grading and the token-budget clamp: the
  solver answers, ``env.is_answer`` decides whether that turn was a submission,
  and if not (and turns remain) the frozen sim replies SINGLE-SHOT -- matching
  ``SIM_REJECT_MAX_TRIES=0`` in run_colbench_grpo.sh, which is how every
  checkpoint on disk was trained.

  There is no ``max_response_length`` accounting here. Training ends an episode
  at 14336 cumulative response tokens, which at 10 turns x 1024 solver tokens +
  256 sim tokens is only reachable by the very longest dialogues; adding a
  tokenizer to this script to catch that tail would cost more than it buys.
  Episodes therefore end on submission, the turn cap, or an empty reply from
  either side, and ``episode_terminated_by`` records which.

  Args:
    task: one ``read_rows`` payload.
    episode_idx: which sample of this task; part of the resume key and of
      ``prefix_id``.
    solver_chat: ``messages -> assistant text`` (the partner assistant).
    sim_backend: ``(system, user) -> raw reply`` for the frozen simulator.
    max_assistant_turns: solver turn cap (10 in training).
    provenance: model/sampling identification stamped on every record.
    sim_prompt: WHICH system prompt the simulator gets, as an arm label
      (``""``/``auto`` = the stock one; ``role``/``role_restraint`` = the
      client-role variants). MUST match the arm the production run serves --
      episodes generated against one simulator are off-distribution context for
      another, so this is not a cosmetic knob.

  Returns:
    ``(prefix_records, episode_record)``. The prefix list is empty when the
    solver submits on its first turn, which is exactly why the episode record
    is the resume key.
  """
  env = ColBenchUserSimEnv(
      problem_description=task["problem_description"],
      ground_truth=task["ground_truth"],
      test_cases=task["test_cases"],
      max_steps=max_assistant_turns,
      sim_prompt=sim_prompt,
      sim_backend=sim_backend,
  )
  messages = [dict(m) for m in task["prompt_messages"]]
  # The SIMULATOR's running dialogue (problem + solver turns + user replies).
  # Contains NO ground truth -- the GT is injected only inside the sim prompt.
  sim_dialogue: list[dict[str, str]] = [
      {"role": "user", "content": task["problem_text"]}
  ]

  prefixes: list[dict[str, Any]] = []
  terminated_by = "turn_cap"
  assistant_turns = 0
  t0 = time.time()

  for turn in range(max_assistant_turns):
    raw_assistant = solver_chat(messages)
    assistant_text = templates.strip_think(raw_assistant)
    if not (assistant_text or "").strip():
      terminated_by = "solver_empty"
      break
    # Training appends the RAW turn (the rollout's own tokens); is_answer
    # strips <think> internally. Keep both shapes: raw in the dialogue, the
    # stripped text on the record.
    messages.append({"role": "assistant", "content": raw_assistant})
    sim_dialogue.append({"role": "assistant", "content": raw_assistant})
    assistant_turns += 1

    is_last_turn = turn == max_assistant_turns - 1
    has_answer, _ = env.is_answer(raw_assistant, episode_done=is_last_turn)
    if has_answer:
      terminated_by = "submit"
      break
    if is_last_turn:
      terminated_by = "turn_cap"
      break

    # ── THE DESIGN POINT: materialize the rendered sim prompt BEFORE drawing ──
    # the reply, so the recorded bytes are provably the bytes that produced it.
    # Built from the PUBLIC builder rather than env._build_sim_prompt: the
    # private method delegates to exactly this call, so byte-identity is both
    # guaranteed and assertable (see tests/test_collect_prefixes.py).
    sim_system = templates.resolve_sim_system(sim_prompt)
    sim_user = templates.build_sim_user_message(
        task["problem_description"], task["ground_truth"], sim_dialogue
    )
    assert (sim_system, sim_user) == env._build_sim_prompt(sim_dialogue), (  # pylint: disable=protected-access
        "materialized sim prompt diverged from env._build_sim_prompt"
    )

    reply = env.generate_user_turn(list(sim_dialogue))

    prefixes.append({
        "prefix_id": f"{task['task_index']}-{episode_idx}-{turn}",
        "collector_version": COLLECTOR_VERSION,
        "task_index": task["task_index"],
        "task_id": task["task_id"],
        "episode_idx": episode_idx,
        "turn_idx": turn,
        # The rendered simulator prompt. Everything downstream reads THESE two
        # strings; nothing re-renders.
        "sim_system": sim_system,
        "sim_user": sim_user,
        "sim_user_sha16": sha16(sim_user),
        "sim_system_variant": sim_prompt or "default",
        # Carried so the judge (and a human reading the dump) can see what the
        # sim was allowed to know without joining back to the parquet.
        "problem_description": task["problem_description"],
        "ground_truth": task["ground_truth"],
        "test_cases": task["test_cases"],
        "sim_dialogue": [dict(m) for m in sim_dialogue],
        # The assistant turn this prefix must be answered against. `_raw` keeps
        # any <think> block that `partner_reply` drops.
        "partner_reply": assistant_text,
        "partner_reply_raw": raw_assistant,
        # The frozen BASE sim's own single-shot draw at this prefix -- the reply
        # that actually continued the episode. A free per-prefix baseline
        # sample, and the anchor the K candidates of Stage 3 are compared to.
        "sim_reply": reply,
        "sim_reply_leak_reason": templates.detect_code_leak(
            reply, task["ground_truth"], ngram_n=0
        ),
        "sim_reply_leak_reason_strict": templates.detect_code_leak(
            reply, task["ground_truth"], ngram_n=10, min_operators=2
        ),
        **provenance,
    })

    if not (reply or "").strip():
      terminated_by = "sim_empty"
      break
    messages.append({"role": "user", "content": reply})
    sim_dialogue.append({"role": "user", "content": reply})

  # terminated_by is only known here, so stamp it on the prefixes now rather
  # than leaving downstream stages to join against the episode file.
  for rec in prefixes:
    rec["episode_terminated_by"] = terminated_by

  episode = {
      "task_index": task["task_index"],
      "task_id": task["task_id"],
      "episode_idx": episode_idx,
      "collector_version": COLLECTOR_VERSION,
      "n_prefixes": len(prefixes),
      "n_assistant_turns": assistant_turns,
      "episode_terminated_by": terminated_by,
      "seconds": round(time.time() - t0, 2),
      **provenance,
  }
  return prefixes, episode


def run_episode_spec(
    task: dict[str, Any],
    episode_idx: int,
    solver_chat,
    sim_backend,
    max_assistant_turns: int,
    provenance: dict[str, Any],
    grounded: bool = True,
    sim_max_tries: int = 8,
    max_code_proposals: int = 2,
    sim_code_leak_detector: str = "auto",
    reward_time_limit: int = 6,
    early_term_guard: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Play one spec/grounded episode, materializing a prefix per user turn.

  A SEPARATE function from ``run_episode`` rather than a flag inside it,
  because the loop genuinely differs rather than being parameterised:
  termination is USER-driven (the sim emits ``[TERMINATE]``, there is no
  ``env.is_answer`` submit), there is a ``max_code_proposals`` cap, and the
  sim's right to end the episode is GATED on the solver having shown a complete
  function. Folding all of that into the GT loop behind a boolean would make
  both paths harder to read than keeping them apart.

  Mirrors ``validate_colbench_spec``'s turn loop, minus grading: same
  ``contains_code`` / ``sim_terminated`` calls, the same
  ``allow_terminate=showed_code`` gate, and the same ``terminated_by`` labels,
  so collected episodes are drawn from the distribution eval reports on.

  THE GROUNDED ARM SEES THE GT CODE. Unlike the spec arm -- where a code leak
  is structurally impossible because the sim is conditioned on
  persona/scenario/requirements -- the grounded sim is handed the function
  source, so ``sim_wrote_code`` rejection is load-bearing here and
  ``detect_code_leak`` against the GT is a MEANINGFUL measurement rather than a
  vacuous one.

  Args:
    task: one ``read_rows`` payload (must carry ``spec`` for the plot).
    episode_idx: which sample of this task; part of the resume key.
    solver_chat: ``messages -> assistant text`` (the partner assistant).
    sim_backend: ``(system, user) -> raw reply`` for the frozen simulator.
    max_assistant_turns: solver turn cap (10 in training).
    provenance: model/sampling identification stamped on every record.
    grounded: True conditions the sim on GT source + plot (the arm we train
      against); False uses persona/scenario/requirements.
    sim_max_tries: rejection budget for code-writing / premature-termination
      draws. Matches the production env, so episodes advance the way training
      advances them.
    max_code_proposals: solver code-block cap before the episode is cut.
    sim_code_leak_detector: which leak policy the env rejects on.
    reward_time_limit: passed through to the env; no grading happens here.
    early_term_guard: whether the ENV rejection-samples a termination that
      arrives before the agent has shown a complete function. This is a POLICY
      and it must match the run being collected for -- `qwen3_4b_spec_rej8`
      launched with `--noearly_term_guard`, so collecting with the guard ON
      would produce episodes that never contain the premature terminations that
      run actually saw. Distinct from `terminate_allowed` on each prefix, which
      is the FACT of whether code is on the table and is recorded either way --
      the rubric's premature-termination veto reads that fact, so the veto keeps
      working regardless of how the env was configured.
  Returns:
    ``(prefix_records, episode_record)``.
  """
  env = ColBenchSpecUserSimEnv(
      problem_description=task["problem_description"],
      spec=task.get("spec") or {},
      ground_truth=task["ground_truth"],
      test_cases=task["test_cases"],
      max_steps=max_assistant_turns,
      reward_time_limit=reward_time_limit,
      sim_backend=sim_backend,
      sim_max_tries=sim_max_tries,
      grounded=grounded,
      sim_code_leak_detector=sim_code_leak_detector,
  )
  plot = (task.get("spec") or {}).get("plot", "")
  messages = [dict(m) for m in task["prompt_messages"]]
  sim_dialogue: list[dict[str, str]] = [
      {"role": "user", "content": task["problem_text"]}
  ]

  prefixes: list[dict[str, Any]] = []
  terminated_by = "turn_cap"
  assistant_turns = 0
  code_proposals = 0
  showed_code = False
  t0 = time.time()

  for turn in range(max_assistant_turns):
    raw_assistant = solver_chat(messages)
    assistant_text = templates.strip_think(raw_assistant)
    if not (assistant_text or "").strip():
      terminated_by = "solver_empty"
      break
    messages.append({"role": "assistant", "content": raw_assistant})
    sim_dialogue.append({"role": "assistant", "content": raw_assistant})
    assistant_turns += 1

    if templates.contains_code(assistant_text):
      showed_code = True
      code_proposals += 1

    is_last_turn = turn == max_assistant_turns - 1
    if is_last_turn:
      terminated_by = "turn_cap" if showed_code else "no_code"
      break
    if code_proposals >= max_code_proposals:
      terminated_by = "code_cap"
      break

    # Materialize the rendered sim prompt BEFORE drawing the reply, so the
    # recorded bytes are provably the bytes that produced it (same invariant as
    # the GT path, asserted against the env's own private builder).
    sim_system, sim_user = env._build_sim_prompt(sim_dialogue)  # pylint: disable=protected-access
    if grounded:
      assert (sim_system, sim_user) == templates.build_grounded_sim_messages(
          task["problem_description"], task["ground_truth"], plot, sim_dialogue
      ), "materialized grounded prompt diverged from env._build_sim_prompt"

    # The prompt's MINIMUM bar ("until you have seen a complete function you
    # MUST NOT end the conversation"), enforced in code -- but only when the
    # run being collected for enabled that guard. `terminate_allowed` is
    # recorded below EITHER WAY, because it is the FACT the rubric's
    # premature-termination veto reads, and a fact about the PREFIX rather than
    # a judgement call must never cost an LLM call.
    allow_terminate = showed_code or not early_term_guard
    reply = env.generate_user_turn(
        list(sim_dialogue), allow_terminate=allow_terminate
    )
    raw = env.last_sim_raw

    prefixes.append({
        "prefix_id": f"{task['task_index']}-{episode_idx}-{turn}",
        "collector_version": COLLECTOR_VERSION,
        "task_index": task["task_index"],
        "task_id": task["task_id"],
        "episode_idx": episode_idx,
        "turn_idx": turn,
        "sim_system": sim_system,
        "sim_user": sim_user,
        "sim_user_sha16": sha16(sim_user),
        "sim_system_variant": "grounded" if grounded else "spec",
        "sim_conditioning": "grounded" if grounded else "spec",
        "problem_description": task["problem_description"],
        "ground_truth": task["ground_truth"],
        "test_cases": task["test_cases"],
        # The plot is the referent for plot-adherence grading; carried so the
        # judge and a human reading the dump never join back to the parquet.
        "plot": plot,
        "spec": task.get("spec") or {},
        "sim_dialogue": [dict(m) for m in sim_dialogue],
        "partner_reply": assistant_text,
        "partner_reply_raw": raw_assistant,
        # PROGRAMMATIC facts the rubric needs and must not pay a judge for:
        # whether ending is even permitted here, and how far the solver has got.
        "terminate_allowed": showed_code,
        "code_proposals_so_far": code_proposals,
        "sim_reply": reply,
        "sim_reply_raw": raw,
        "sim_reply_terminated": templates.sim_terminated(raw),
        # The grounded sim DOES see the GT, so overlap against it is meaningful
        # here (it is not on the spec arm). ngram_n=0 is the syntactic policy
        # the production env rejects on; the strict form is the stricter probe.
        "sim_reply_leak_reason": templates.detect_code_leak(
            reply, task["ground_truth"], ngram_n=0
        ),
        "sim_reply_leak_reason_strict": templates.detect_code_leak(
            reply, task["ground_truth"], ngram_n=10, min_operators=2
        ),
        "sim_code_rejected": env.last_sim_code_rejected,
        "sim_early_term_rejected": env.last_sim_early_term_rejected,
        # THE DRAFTS THE SAMPLER THREW AWAY. `sim_reply` is what the rejection
        # sampler settled on; these are what the sim wanted to say first, which
        # is its MODAL behaviour and therefore the honest measurement. Keeping
        # them also yields (rejected draft, accepted reply) pairs for free, and
        # records what a guard actually suppressed rather than leaving it to be
        # inferred from a count.
        "sim_code_reject_samples": list(env.last_sim_code_reject_samples),
        "sim_early_term_samples": list(env.last_sim_early_term_samples),
        # The gate in force AT THIS TURN, so a reader never has to reconstruct
        # it from the flags: terminate_allowed above is the FACT (has code been
        # shown), this is the POLICY that was applied.
        "terminate_permitted": allow_terminate,
        **provenance,
    })

    if env.last_sim_code_reject_exhausted:
      terminated_by = "sim_code_reject"
      break
    if templates.sim_terminated(raw):
      terminated_by = "user" if showed_code else "no_code"
      break
    if not (reply or "").strip():
      terminated_by = "sim_empty"
      break
    messages.append({"role": "user", "content": reply})
    sim_dialogue.append({"role": "user", "content": reply})

  for rec in prefixes:
    rec["episode_terminated_by"] = terminated_by

  episode = {
      "task_index": task["task_index"],
      "task_id": task["task_id"],
      "episode_idx": episode_idx,
      "collector_version": COLLECTOR_VERSION,
      "n_prefixes": len(prefixes),
      "n_assistant_turns": assistant_turns,
      "n_code_proposals": code_proposals,
      "showed_code": showed_code,
      "sim_conditioning": "grounded" if grounded else "spec",
      "early_term_guard": early_term_guard,
      "episode_terminated_by": terminated_by,
      "seconds": round(time.time() - t0, 2),
      **provenance,
  }
  return prefixes, episode


def existing_episodes(path: str) -> set[tuple[int, int]]:
  """``(task_index, episode_idx)`` pairs already collected, for resume.

  Args:
    path: the episodes JSONL; a missing file reads as empty.

  Returns:
    The set of episode keys already on disk.
  """
  return {
      (int(r["task_index"]), int(r["episode_idx"]))
      for r in read_jsonl(path)
      if "task_index" in r and "episode_idx" in r
  }


def read_prefixes(path: str) -> list[dict[str, Any]]:
  """Read a prefix file, keeping the LAST record for any repeated ``prefix_id``.

  Duplicates are possible by construction: an episode's prefixes are appended
  before its episode record, so a kill in that window leaves prefixes on disk
  for an episode the resume key says is unfinished, and the resumed run writes
  them again. Deduping on read is cheaper and safer than trying to make two
  appends atomic; ``prefix_id`` is deterministic
  (``<task_index>-<episode_idx>-<turn_idx>``) precisely so this works.

  Args:
    path: a ``prefixes.*.jsonl`` written by this collector.

  Returns:
    The records, deduped by ``prefix_id``, in first-seen order.
  """
  out: dict[str, dict[str, Any]] = {}
  for rec in read_jsonl(path):
    out[rec["prefix_id"]] = rec
  return list(out.values())


def require_sim_char_limit() -> int:
  """Hard-require ``SIM_CHAR_LIMIT``, and warn about the knobs that do nothing.

  With a ``ChatEndpoint`` sim backend injected, ``SIM_TEMPERATURE`` /
  ``SIM_TOP_P`` / ``SIM_TOP_K`` / ``SIM_MIN_P`` / ``SIM_MAX_TOKENS`` are NOT
  read: those live in ``env.openai_sim_backend``, which this script replaces.
  Sampling comes from the CLI instead. But ``SIM_CHAR_LIMIT`` IS still read, per
  call, inside ``env._finalize_reply``, and it defaults to 400 -- so forgetting
  to export ``SIM_CHAR_LIMIT=0`` silently applies a 400-character slice that
  training (which sets 0) does not, to every reply in the dataset.

  Returns:
    The configured limit.

  Raises:
    SystemExit: ``SIM_CHAR_LIMIT`` is unset, which would silently default to a
      slice the training runs do not apply.
  """
  raw = os.environ.get("SIM_CHAR_LIMIT")
  if raw is None:
    raise SystemExit(
        "[collect_prefixes] SIM_CHAR_LIMIT is unset. env._finalize_reply would "
        "default it to "
        f"{templates.HUMAN_RESPONSE_CHARACTER_LIMIT}, silently truncating every "
        "collected reply -- while run_colbench_grpo.sh exports "
        "SIM_CHAR_LIMIT=0. Export it explicitly (0 to match training)."
    )
  ignored = [
      k
      for k in (
          "SIM_TEMPERATURE",
          "SIM_TOP_P",
          "SIM_TOP_K",
          "SIM_MIN_P",
          "SIM_MAX_TOKENS",
      )
      if k in os.environ
  ]
  if ignored:
    print(
        f"[collect_prefixes] NOTE: {', '.join(ignored)} set but IGNORED -- this "
        "script injects a ChatEndpoint sim backend, so sim sampling comes from "
        "--sim_temperature/--sim_top_p/--sim_top_k/--sim_min_p/--sim_max_tokens.",
        flush=True,
    )
  return int(raw)


def build_arg_parser() -> argparse.ArgumentParser:
  """CLI for the collector.

  Returns:
    The configured parser.
  """
  ap = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--data_file", required=True, help="ColBench parquet.")
  ap.add_argument("--start", type=int, default=0, help="First row position.")
  ap.add_argument(
      "--end",
      type=int,
      default=600,
      help="One past the last row position (600 = the planned PoC slice).",
  )
  ap.add_argument(
      "--episodes_per_task",
      type=int,
      default=1,
      help="A second episode on the SAME task yields near-duplicate prefixes, "
      "so 1 by default.",
  )
  ap.add_argument(
      "--out_dir", required=True, help="Directory for prefixes/episodes JSONL."
  )
  ap.add_argument(
      "--tag",
      default="",
      help="Filename tag (default: the parquet stem + collector version).",
  )
  ap.add_argument(
      "--max_assistant_turns",
      type=int,
      default=10,
      help="Solver turn cap; 10 matches training.",
  )
  ap.add_argument("--concurrency", type=int, default=16)
  # ── Partner (solver) endpoint. Defaults = run_colbench_grpo.sh's rollout ──
  ap.add_argument("--solver_base_url", required=True)
  ap.add_argument("--solver_model", required=True)
  ap.add_argument("--solver_api_key", default="EMPTY")
  ap.add_argument("--solver_temperature", type=float, default=0.7)
  ap.add_argument("--solver_top_p", type=float, default=0.8)
  ap.add_argument("--solver_top_k", type=int, default=20)
  ap.add_argument("--solver_min_p", type=float, default=0.0)
  ap.add_argument(
      "--solver_max_tokens",
      type=int,
      default=1024,
      help="Per-turn solver cap; 1024 matches MAX_NEW_TOKENS_PER_TURN.",
  )
  # ── Simulator endpoint. Defaults = env._sim_sampling + SIM_MAX_TOKENS=256 ──
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
      choices=["default", "role", "role_restraint"],
      default="default",
      help="WHICH system prompt the simulator gets while the episodes are "
      "played. Must match the arm the production training run serves.",
  )
  # ONE flag selects the arm rather than a parallel script per path: the three
  # differ in which env and which sim conditioning they use, not in what the
  # collector is for.
  ap.add_argument(
      "--arm",
      choices=["gt", "spec", "grounded"],
      default="gt",
      help="Which simulator the episodes are played against. 'gt' = the "
      "hidden-code sim (env.ColBenchUserSimEnv). 'spec' = conditioned on "
      "persona/scenario/requirements. 'grounded' = conditioned on the GT "
      "source + plot (+colbench.grounded_sim). MUST match the arm the "
      "production run serves -- episodes collected against one simulator are "
      "off-distribution context for another.",
  )
  ap.add_argument(
      "--sim_max_tries",
      type=int,
      default=8,
      help="spec/grounded: the env's rejection budget for code-writing and "
      "premature-termination draws. 8 matches training, so episodes advance "
      "the way training advances them.",
  )
  ap.add_argument(
      "--max_code_proposals",
      type=int,
      default=2,
      help="spec/grounded: solver code-block cap before the episode is cut.",
  )
  ap.add_argument(
      "--noearly_term_guard",
      dest="early_term_guard",
      action="store_false",
      help="spec/grounded: do NOT reject sim terminations that arrive before "
      "the agent has shown a complete function. Match the run being collected "
      "for -- qwen3_4b_spec_rej8 launched with --noearly_term_guard, and "
      "collecting with the guard ON would drop exactly the premature "
      "terminations that run saw. The per-prefix `terminate_allowed` fact is "
      "recorded either way, so the rubric's veto is unaffected.",
  )
  ap.add_argument(
      "--sim_code_leak_detector",
      default="auto",
      help="spec/grounded: which leak policy the env rejects on; 'auto' "
      "resolves per arm (fence-only for spec/grounded).",
  )
  ap.add_argument(
      "--vendor",
      choices=["vllm", "openai"],
      default="vllm",
      help="ChatEndpoint sampling dialect for BOTH endpoints.",
  )
  ap.add_argument("--timeout", type=float, default=300.0)
  ap.add_argument("--retries", type=int, default=3)
  return ap


def main(argv: Optional[list[str]] = None) -> None:
  """Collect prefixes for every (task, episode) not already on disk."""
  args = build_arg_parser().parse_args(argv)
  char_limit = require_sim_char_limit()

  tag = args.tag or (
      f"{os.path.splitext(os.path.basename(args.data_file))[0]}."
      f"{COLLECTOR_VERSION}"
  )
  out_dir = os.path.expanduser(args.out_dir)
  os.makedirs(out_dir, exist_ok=True)
  prefix_path = os.path.join(out_dir, f"prefixes.{tag}.jsonl")
  episode_path = os.path.join(out_dir, f"episodes.{tag}.jsonl")

  rows = read_rows(args.data_file, args.start, args.end)
  done = existing_episodes(episode_path)
  todo = [
      (t, e)
      for t in rows
      for e in range(args.episodes_per_task)
      if (t["task_index"], e) not in done
  ]
  print(
      f"[collect_prefixes] {len(rows)} tasks x {args.episodes_per_task} "
      f"episodes; {len(done)} already collected, {len(todo)} to run\n"
      f"[collect_prefixes] SIM_CHAR_LIMIT={char_limit} "
      f"(0 = no slice, matching training)  arm={args.arm} "
      f"sim_system={args.sim_system}\n"
      f"[collect_prefixes] -> {prefix_path}\n"
      f"[collect_prefixes] -> {episode_path}",
      flush=True,
  )
  if not todo:
    return

  solver_ep = ChatEndpoint(
      base_url=args.solver_base_url,
      model=args.solver_model,
      api_key=args.solver_api_key,
      vendor=args.vendor,
      temperature=args.solver_temperature,
      top_p=args.solver_top_p,
      top_k=args.solver_top_k,
      min_p=args.solver_min_p,
      max_tokens=args.solver_max_tokens,
      retries=args.retries,
      timeout=args.timeout,
  )
  sim_ep = ChatEndpoint(
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
      "solver_model": args.solver_model,
      "solver_sampling": {
          "temperature": args.solver_temperature,
          "top_p": args.solver_top_p,
          "top_k": args.solver_top_k,
          "min_p": args.solver_min_p,
          "max_tokens": args.solver_max_tokens,
      },
      "sim_model": args.sim_model,
      "sim_sampling": {
          "temperature": args.sim_temperature,
          "top_p": args.sim_top_p,
          "top_k": args.sim_top_k,
          "min_p": args.sim_min_p,
          "max_tokens": args.sim_max_tokens,
      },
      "sim_char_limit": char_limit,
      "sim_system_variant": args.sim_system,
      "source_parquet": os.path.abspath(os.path.expanduser(args.data_file)),
      "max_assistant_turns": args.max_assistant_turns,
  }

  sim_backend = make_sim_backend(sim_ep)
  t0 = time.time()
  n_done = n_prefix = 0
  by_term: dict[str, int] = {}

  def _one(pair):
    task, episode_idx = pair
    if args.arm == "gt":
      return run_episode(
          task,
          episode_idx,
          solver_ep.chat,
          sim_backend,
          args.max_assistant_turns,
          provenance,
          "" if args.sim_system == "default" else args.sim_system,
      )
    return run_episode_spec(
        task,
        episode_idx,
        solver_ep.chat,
        sim_backend,
        args.max_assistant_turns,
        provenance,
        grounded=args.arm == "grounded",
        sim_max_tries=args.sim_max_tries,
        max_code_proposals=args.max_code_proposals,
        sim_code_leak_detector=args.sim_code_leak_detector,
        early_term_guard=args.early_term_guard,
    )

  with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
    futs = {pool.submit(_one, p): p for p in todo}
    for fut in as_completed(futs):
      task, episode_idx = futs[fut]
      try:
        prefixes, episode = fut.result()
      except Exception as e:  # pylint: disable=broad-exception-caught
        # One bad episode must not lose the ones already paid for. Leave it
        # UNWRITTEN so a resume retries it, exactly like a deferred row in
        # selfplay/generate_specs.
        print(
            f"[collect_prefixes] episode {task['task_index']}/{episode_idx} "
            f"FAILED, deferred to resume: {e!r}",
            flush=True,
        )
        continue
      with _WRITE_LOCK:
        # Prefixes FIRST, then the episode record: the episode record is the
        # resume key, so it must never land before the data it vouches for.
        # The reverse gap (prefixes written, episode record lost) is handled by
        # read_prefixes' dedupe.
        if prefixes:
          append_jsonl(prefix_path, prefixes)
        append_jsonl(episode_path, [episode])
      n_done += 1
      n_prefix += len(prefixes)
      by_term[episode["episode_terminated_by"]] = (
          by_term.get(episode["episode_terminated_by"], 0) + 1
      )
      if n_done % 20 == 0:
        print(
            f"[collect_prefixes] {n_done}/{len(todo)} episodes, {n_prefix} "
            f"prefixes ({n_prefix / max(1, n_done):.2f}/episode) in "
            f"{time.time() - t0:.0f}s  {json.dumps(by_term, sort_keys=True)}",
            flush=True,
        )

  print(
      f"[collect_prefixes] DONE {n_done} episodes, {n_prefix} prefixes "
      f"({n_prefix / max(1, n_done):.2f}/episode) in {time.time() - t0:.0f}s\n"
      f"[collect_prefixes] terminated_by: {json.dumps(by_term, sort_keys=True)}",
      flush=True,
  )
  if n_done < len(todo):
    print(
        f"[collect_prefixes] {len(todo) - n_done} episode(s) were NOT written "
        "(errors). Re-run the SAME command to collect only those.",
        flush=True,
    )


if __name__ == "__main__":
  main()

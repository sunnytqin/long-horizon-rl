"""Environment for the ColBench multi-turn loop: a user simulator + a GT-graded reward.

``ColBenchUserSimEnv`` is the pluggable "what happens between assistant turns" component
(the analog of ``codecontest.env.GTOracleEnv``). It holds the problem, the HIDDEN
ground-truth function source, and the GT call-strings, and exposes three seams the agent
loop drives:

  * ``is_answer(assistant_text, episode_done)`` -> ``(has_answer, answer_text)`` -- did the
    solver submit (marker, or a code-like final turn)?  (templates.final_answer)
  * ``generate_user_turn(messages)`` -> a <=400-char human-like reply. THE sim seam. Two
    backends exist: ``sim_backend`` (sync) = an HTTP call to a SEPARATE FROZEN sim server, and
    ``asim_backend`` (async, driven by ``agenerate_user_turn``) = generation on the TRAINING
    rollout engine itself, i.e. the LIVE policy weights, no frozen copy anywhere (the
    "same copy throughout training" arm -- see colbench_agent.py's ``sim_live``). Either way
    the GT source is passed ONLY inside this call's prompt -- it never enters the solver's
    message list, and the sim's own generated TOKENS never enter the solver trajectory.
  * ``score(answer_text)`` -> fractional pass-rate against the GT (reward.grade).

Reward convention: the trajectory reward is the final submission's fractional pass-rate in
[0,1]; the frozen simulator governs the DIALOGUE only and has zero influence on grading.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from colbench import reward, templates

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ── Conversation debug (mirrors codecontest.env's CODECONTEST_DEBUG_FEEDBACK gate) ──
# Env-var gated so the user can eyeball dialogues. Logged at WARNING because plain print()
# is swallowed by the Ray rollout workers. See colbench_agent.py for the per-turn / answer /
# reward dumps; here we dump the SIM side (prompt incl. the hidden GT, + raw reply).
_DEBUG_SIM = bool(int(os.getenv("COLBENCH_DEBUG_SIM", "0") or "0"))
_DEBUG_PREVIEW = int(os.getenv("COLBENCH_DEBUG_CONVO_PREVIEW", "400") or "400")


def _sim_extra_body():
  """Resolve the OpenAI `extra_body` for the sim call from SIM_ENABLE_THINKING.

  Default (unset) -> None: send NO thinking kwarg, which is safe for every model (Qwen2.5,
  Qwen3-Instruct-2507, non-Qwen). Only a HYBRID Qwen3 sim that would otherwise emit <think>
  needs SIM_ENABLE_THINKING=false; SIM_ENABLE_THINKING=true forces it on. This replaces the
  old brittle `"qwen3" in served_name` guard (the served name is a fixed alias, not the model
  family, so that check never fired).
  """
  v = os.environ.get("SIM_ENABLE_THINKING", "").strip().lower()
  if v in ("true", "1"):
    return {"enable_thinking": True}
  if v in ("false", "0"):
    return {"enable_thinking": False}
  return None


# A sim backend maps (system_content, user_content) -> raw reply text. The default is an
# OpenAI-compatible HTTP call to a separate FROZEN sim server; tests inject a stub (no server,
# no openai import).
SimBackend = Callable[[str, str], str]
# The async variant, used by the LIVE-weights arm: the agent loop injects a coroutine that
# generates on the TRAINING rollout engine (current policy), so there is no frozen copy and no
# second server. Set via ``asim_backend``; consumed by ``agenerate_user_turn(_checked)``.
AsyncSimBackend = Callable[[str, str], Awaitable[str]]


def _sim_sampling():
  """Resolve the sim's sampling params from env, defaulting to Qwen3-Instruct's recommended.

  IMPORTANT: default temperature is 0.7 (NOT greedy). The Qwen3 family is explicitly
  documented to degrade / repeat under greedy (temp 0) decoding, so a Qwen3 sim MUST sample.
  Defaults = Qwen3-*-Instruct-2507 recommendation (temp 0.7, top_p 0.8, top_k 20, min_p 0);
  for a Qwen3-32B (thinking) sim set SIM_TEMPERATURE=0.6 SIM_TOP_P=0.95. Returns
  (temperature, top_p, top_k, min_p).
  """
  return (
      float(os.environ.get("SIM_TEMPERATURE", "0.7")),
      float(os.environ.get("SIM_TOP_P", "0.8")),
      int(os.environ.get("SIM_TOP_K", "20")),
      float(os.environ.get("SIM_MIN_P", "0")),
  )


def openai_sim_backend(system_content: str, user_content: str) -> str:
  """Default Phase-1 sim backend: query the FROZEN sim server over the OpenAI API.

  Reads OPENAI_BASE_URL (e.g. http://localhost:<SIM_PORT>/v1), MULTITURN_MODEL_NAME, and
  OPENAI_API_KEY (default "EMPTY"), matching the entrypoint's exported env. Sampling comes
  from SIM_TEMPERATURE/SIM_TOP_P/SIM_TOP_K/SIM_MIN_P (see _sim_sampling -- defaults to Qwen3's
  recommended non-greedy sampling); max_tokens 4096; 3 retries then "No response." -- mirrors
  sweet_rl HumanInteractionEnv.invoke_model / InfoPO APIHumanSimulator.invoke_model.
  ``openai`` is imported lazily so CPU tests (which inject a stub) never need it installed.
  """
  from openai import OpenAI  # lazy: only the real sim path needs the SDK

  base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
  model = os.environ.get("MULTITURN_MODEL_NAME", "")
  api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
  # max_retries=0: the SDK otherwise adds its OWN exponentially-backed-off retries (default 2)
  # on top of our manual loop below, so a slow/busy sim could fan a single call out to ~9 requests
  # and stack timeouts into minutes. Keep our loop the SINGLE source of retry behavior.
  client = OpenAI(
      api_key=api_key,
      base_url=base_url,
      max_retries=int(os.environ.get("SIM_MAX_RETRIES", "0") or "0"),
  )

  messages = [
      {"role": "system", "content": system_content},
      {"role": "user", "content": user_content},
  ]
  temperature, top_p, top_k, min_p = _sim_sampling()
  # top_k / min_p are SGLang extensions -> extra_body.
  extra_body = {"top_k": top_k, "min_p": min_p}
  thinking = _sim_extra_body()
  if thinking is not None:
    # MUST be nested under `chat_template_kwargs`. A hybrid Qwen3 turns reasoning off in its
    # JINJA TEMPLATE (which pre-fills an empty `<think></think>`), so the flag has to reach the
    # template renderer. VERIFIED on the running SGLang sim 2026-07-30: a TOP-LEVEL
    # `enable_thinking: false` is accepted and SILENTLY IGNORED (reply still opens `<think>`),
    # while `chat_template_kwargs: {enable_thinking: false}` works. Same trap as vLLM, which
    # declares no such field and sets extra="allow".
    extra_body["chat_template_kwargs"] = thinking
  # SIM_MAX_TOKENS bounds the user turn at GENERATION time (default 4096 = unchanged). The spec
  # path sets it small so a brief human-like reply is not chopped post-hoc (it replaces the old
  # HUMAN_RESPONSE_CHARACTER_LIMIT slice, which truncated mid-sentence).
  sim_max_tokens = int(os.environ.get("SIM_MAX_TOKENS", "4096") or "4096")
  # Per-request timeout to the sim server. Default 180s (was a hardcoded 60s): a large or busy
  # frozen sim -- especially a big REMOTE model under bursty rollout concurrency where the sim is
  # slower than the assistant rollouts -- has long tail latency, and a timeout here does not just
  # slow the step, it POISONS the turn with the "No response." fallback below. Prefer a slow turn
  # over a poisoned one. SIM_TIMEOUT overrides (bump higher for a very large/saturated sim).
  sim_timeout = float(os.environ.get("SIM_TIMEOUT", "180") or "180")
  params = {
      "model": model,
      "messages": messages,
      "max_tokens": sim_max_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "extra_body": extra_body,
      "timeout": sim_timeout,
  }

  for _ in range(3):
    try:
      completion = client.chat.completions.create(**params)
      return completion.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001 - degrade to a default reply, never crash rollout
      logger.warning("[colbench] sim server call failed: %r", e)
  return "No response."


@dataclass
class ColBenchUserSimEnv:
  """User-simulator env holding the problem, hidden GT, and GT call-strings.

  Args:
      problem_description: the user's (public) problem statement.
      ground_truth: the HIDDEN ground-truth function source (never shown to the solver).
      test_cases: list of GT call-strings used for grading.
      max_steps: max solver turns before the episode is force-ended (sweet_rl default 10).
      reward_time_limit: per-case exec timeout (seconds) for grading.
      sim_backend: (system, user) -> raw reply. Defaults to the frozen-server HTTP call;
          tests inject their own. This is the ONE seam that sees the GT.
      asim_backend: async (system, user) -> raw reply. When set, the LIVE-weights path
          (``agenerate_user_turn``) uses it instead of ``sim_backend``: the agent loop
          generates the user turn on the training rollout engine, so the simulator IS the
          current policy (no frozen copy). ``sim_backend`` is left untouched, so the sync
          path (offline eval, tests) keeps working.
  """

  problem_description: str
  ground_truth: str
  test_cases: list
  max_steps: int = 10
  reward_time_limit: float = 6.0
  sim_backend: Optional[SimBackend] = None
  asim_backend: Optional[AsyncSimBackend] = None
  # Populated on the last generate_user_turn call, for the agent loop's debug dump.
  last_sim_reply: str = field(default="", repr=False)

  def __post_init__(self):
    if self.sim_backend is None:
      self.sim_backend = openai_sim_backend

  def is_answer(
      self, assistant_text: str, episode_done: bool
  ) -> tuple[bool, str]:
    """Did the solver submit a final answer this turn? Returns (has_answer, answer_text).

    ``assistant_text`` is <think>-stripped first so a marker inside a reasoning block is
    ignored. On the final turn a code-like response is accepted as the answer even without
    the marker (templates.final_answer).
    """
    clean = templates.strip_think(assistant_text)
    return templates.final_answer(clean, episode_done)

  def _build_sim_prompt(self, messages: list[dict]) -> tuple[str, str]:
    """(system_content, user_content) for one sim call. The ONE place the GT is injected.

    Kept separate from the sampling so the sync (frozen-server) and async (live-weights)
    paths build a BYTE-IDENTICAL prompt -- the two arms must differ only in which weights
    answer it.
    """
    return templates.SIM_SYSTEM_PROMPT, templates.build_sim_user_message(
        self.problem_description, self.ground_truth, messages
    )

  def _finalize_reply(self, raw: str, user_content: str) -> str:
    """<think>-strip, then apply the post-hoc character cap (and debug-dump it).

    The cap is sweet_rl's ``HUMAN_RESPONSE_CHARACTER_LIMIT`` (400) by default, so anything that
    does not set ``SIM_CHAR_LIMIT`` is byte-identical to stock ColBench. ``SIM_CHAR_LIMIT=0``
    DISABLES the slice, which is what the GT-vs-SPEC study needs: the spec path removed this
    slice entirely and bounds the user turn only at generation time (SIM_MAX_TOKENS), so with
    the slice live the GT arm's user could deliver ~400 chars/turn against the spec arm's
    ~700-1000 -- a 2.5x information asymmetry masquerading as an environment-quality result.
    With it off, both arms are bounded by the SAME knob (SIM_MAX_TOKENS) and the sim prompt's
    "IN TWO SENTENCES" instruction does the shaping. See run_colbench_grpo.sh.
    """
    limit = int(
        os.environ.get(
            "SIM_CHAR_LIMIT", templates.HUMAN_RESPONSE_CHARACTER_LIMIT
        )
    )
    reply = templates.strip_think(raw)
    if limit > 0:
      reply = reply[:limit]
    if _DEBUG_SIM:
      n = _DEBUG_PREVIEW
      logger.warning(
          "[COLBENCH_SIM] hidden_gt[:%d]=%r\n[COLBENCH_SIM] sim_user_prompt[:%d]=%r\n"
          "[COLBENCH_SIM] raw_reply[:%d]=%r\n[COLBENCH_SIM] capped_reply=%r",
          n,
          str(self.ground_truth)[:n],
          n,
          user_content[:n],
          n,
          str(raw)[:n],
          reply,
      )
    return reply

  def _sample_user_reply(self, messages: list[dict]) -> str:
    """One raw sim sample: build the prompt (with hidden GT), call the backend, clean up.

    The reply is <think>-stripped and hard-capped at 400 chars, so the solver's message
    list only ever receives that short reply. Shared by the single-shot training path
    (generate_user_turn) and the eval rejection loop (generate_user_turn_checked).
    """
    system_content, user_content = self._build_sim_prompt(messages)
    raw = self.sim_backend(system_content, user_content)
    return self._finalize_reply(raw, user_content)

  async def _asample_user_reply(self, messages: list[dict]) -> str:
    """Async twin of ``_sample_user_reply``, going through ``asim_backend``."""
    if self.asim_backend is None:
      raise RuntimeError(
          "asim_backend is not set: the live-weights sim path requires the agent loop to "
          "inject a same-engine backend (see colbench_agent.py sim_live)."
      )
    system_content, user_content = self._build_sim_prompt(messages)
    raw = await self.asim_backend(system_content, user_content)
    return self._finalize_reply(raw, user_content)

  def generate_user_turn(self, messages: list[dict]) -> str:
    """Produce the next user (simulator) reply. THE Phase-1/Phase-2 seam.

    ``messages`` is the running dialogue as ``[{role, content}, ...]`` (problem + solver
    turns + prior user replies) -- it contains NO ground truth. The GT source is injected
    as ``hidden_information`` into the sim prompt built here and passed only to the
    backend. Single-shot (no rejection sampling). Callers opt into rejection sampling via
    generate_user_turn_checked; this stays the path when it is disabled.
    """
    reply = self._sample_user_reply(messages)
    self.last_sim_reply = reply
    return reply

  def generate_user_turn_checked(
      self,
      messages: list[dict],
      max_tries: int = 32,
      ngram_n: int = 10,
      min_operators: int = 2,
  ) -> dict:
    """Rejection-sampled user turn: resample until the reply has no code leak.

    Keeps drawing sim replies (up to ``max_tries``) until one passes
    ``templates.detect_code_leak`` (against the hidden GT this env holds), preventing the
    frozen simulator from just handing the solver the solution. Returns a record::

        {"reply": str|None, "tries": int, "accepted": bool, "reasons": [str, ...]}

    On success ``reply`` is the accepted turn, ``tries`` the number of samples drawn, and
    ``reasons`` the leak reason of each REJECTED sample (len == tries-1). On exhaustion
    ``accepted`` is False, ``reply`` is None (a "simulation failure" -- the caller
    terminates the trajectory), and ``reasons`` has one entry per try.

    Both callers terminate on exhaustion, but score it differently: eval treats it as a
    THIRD outcome excluded from the pass-rate denominator, while training keeps the
    trajectory in the batch at reward 0 so the offending solver turn takes a negative
    advantage (see colbench_agent.py).
    """
    reasons: list[str] = []
    for i in range(1, max_tries + 1):
      reply = self._sample_user_reply(messages)
      reason = templates.detect_code_leak(
          reply, self.ground_truth, ngram_n, min_operators
      )
      if reason is None:
        self.last_sim_reply = reply
        return {
            "reply": reply,
            "tries": i,
            "accepted": True,
            "reasons": reasons,
        }
      reasons.append(reason)
    return {
        "reply": None,
        "tries": max_tries,
        "accepted": False,
        "reasons": reasons,
    }

  async def agenerate_user_turn(self, messages: list[dict]) -> str:
    """LIVE-weights twin of ``generate_user_turn`` (same prompt, current policy answers)."""
    reply = await self._asample_user_reply(messages)
    self.last_sim_reply = reply
    return reply

  async def agenerate_user_turn_checked(
      self,
      messages: list[dict],
      max_tries: int = 32,
      ngram_n: int = 10,
      min_operators: int = 2,
  ) -> dict:
    """LIVE-weights twin of ``generate_user_turn_checked`` (identical record contract).

    Rejection sampling matters MORE here than on the frozen path: a live simulator shares the
    solver's weights and its incentive-free view of the GT, so if the policy drifts toward
    dumping code the "user" drifts with it. Same accept/exhaust semantics as the sync method.
    """
    reasons: list[str] = []
    for i in range(1, max_tries + 1):
      reply = await self._asample_user_reply(messages)
      reason = templates.detect_code_leak(
          reply, self.ground_truth, ngram_n, min_operators
      )
      if reason is None:
        self.last_sim_reply = reply
        return {
            "reply": reply,
            "tries": i,
            "accepted": True,
            "reasons": reasons,
        }
      reasons.append(reason)
    return {
        "reply": None,
        "tries": max_tries,
        "accepted": False,
        "reasons": reasons,
    }

  def score(self, answer_text: str) -> dict:
    """Grade the submitted answer against the GT. Returns reward.grade's dict.

    The answer is fence-stripped to code, then compared to the GT function on every
    call-string via the sandboxed exec sidecar (functional equivalence).
    """
    code = templates.extract_code_answer(answer_text)
    return reward.grade(
        code,
        self.ground_truth,
        self.test_cases,
        time_limit=self.reward_time_limit,
    )

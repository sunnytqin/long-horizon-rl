"""Spec-path environment for the ColBench multi-turn loop (Phase 1).

The sibling of ``colbench.env.ColBenchUserSimEnv`` for the SPEC setting. The one
difference that matters: the frozen user-simulator is conditioned on a
natural-language **spec** (``persona/scenario/requirements/plot``) instead of
the hidden GT function source. Because the sim never sees code, a code leak is
structurally impossible here -- so there is NO ``detect_code_leak`` machinery in
this path (it exists only on the GT env); the rejection sampling that IS here
(``sim_wrote_code``) only keeps the sim in character.

EXCEPTION -- ``grounded=True`` (``+colbench.grounded_sim``, opt-in, default
off): the sim is conditioned on the hidden GT source + ``spec["plot"]`` instead,
to test whether the plot mechanism survives with a sim that has a reliable
artifact to read off. In that mode the GT IS in the sim's prompt, so "leak
impossible by construction" no longer holds and the ``sim_wrote_code`` rejection
sampling becomes the actual leak defense. Every other seam is unchanged.

Termination is USER-DRIVEN (see ``colbench.colbench_spec_agent`` /
``validate_colbench_spec``): the sim ends the conversation by emitting
``[TERMINATE]``; the loop grades the last function the solver showed. This env
only owns the two seams the loop drives:

  * ``generate_user_turn(messages)`` -> the spec-conditioned sim's next reply
    (the GT never enters the solver's message list -- only the spec does, inside
    this call's prompt).
  * ``score(answer_text)`` -> fractional GT pass-rate (reward.grade) --
    IDENTICAL to the GT env; grading is unchanged, still objective GT code +
    test_cases.

Note there is no ``is_answer`` seam: in the spec path the solver does not
"submit" with a marker (it proposes via a ```python block and the USER
terminates), so answer detection / grading-target selection lives in the loop
via ``templates.contains_code`` / ``templates.extract_last_code``.
"""

# This tree imports names directly (``from colbench.env import
# ColBenchUserSimEnv``) rather than the enclosing module, matching how the
# rest of verl is written; call sites read on the bare name throughout.
# pylint: disable=g-importing-member
from dataclasses import dataclass
from dataclasses import field
import logging
import os
from typing import Any
from typing import Optional

from colbench import reward
from colbench import templates

# Reuse the GT env's frozen-sim HTTP backend + sampling resolution verbatim --
# only the PROMPT built here differs (spec vs GT source). Keeps sim sampling /
# thinking-kwarg behavior identical.
from colbench.env import _sim_extra_body  # pylint: disable=unused-import
from colbench.env import _sim_sampling  # pylint: disable=unused-import
from colbench.env import openai_sim_backend  # pylint: disable=unused-import
from colbench.env import SimBackend  # pylint: disable=unused-import


def make_openai_sim_backend(
    base_url: str,
    model: str,
    api_key: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    timeout: float = 60.0,
) -> SimBackend:
  """Build a sim backend that queries a REAL OpenAI-API endpoint.

  For example api.openai.com.

  For comparison studies where the frozen user-simulator is a hosted GPT model
  instead of the local vLLM/SGLang Qwen server. Differs from
  ``env.openai_sim_backend`` in ways that matter for the genuine OpenAI API: (1)
  it takes its own base_url / model / key so the SOLVER and the SIM can sit on
  DIFFERENT endpoints (Qwen solver, GPT sim), (2) it sends NO ``extra_body``
  (``top_k``/``min_p`` are vLLM/SGLang extensions the real API rejects with a
  400), and (3) it is **schema-adaptive** across model families: the GPT-5 /
  reasoning family wants ``max_completion_tokens`` (not ``max_tokens``) and only
  accepts the DEFAULT ``temperature``/``top_p`` (1.0). On a 400 that names an
  unsupported/renamed parameter, it drops or renames that field and retries, so
  the same call works for gpt-4o-mini and gpt-5.4-mini alike. Degrades to "No
  response." on persistent error, never crashes the rollout.

  Args:
    base_url: the sim endpoint, independent of the solver's.
    model: served model name at that endpoint.
    api_key: key for that endpoint.
    temperature: sampling temperature; the reasoning family accepts only the
      default 1.0.
    top_p: nucleus mass; likewise pinned to 1.0 on the reasoning family.
    max_tokens: reply cap, renamed to ``max_completion_tokens`` when the model
      demands it.
    timeout: per-request socket timeout in seconds.

  Returns:
    A ``SimBackend`` closure mapping ``(system_content, user_content)`` to reply
    text, degrading to ``"No response."`` rather than raising.
  """
  # pylint: disable=g-import-not-at-top
  from openai import OpenAI  # lazy: only the real sim path needs the SDK

  client = OpenAI(api_key=api_key, base_url=base_url)

  def backend(system_content: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    # Start with the standard-chat schema; peel off fields the model rejects
    # (GPT-5/reasoning
    # models: token-limit param is renamed, sampling params are fixed at
    #         default).
    params = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    for _ in range(5):
      try:
        completion = client.chat.completions.create(**params)
        return completion.choices[0].message.content or ""
      except Exception as e:  # pylint: disable=broad-exception-caught  # degrade to a default, never crash rollout
        msg = str(e).lower()
        # Adapt the request to this model family's schema, then retry (no
        # attempt spent "for real" -- we only fall through to the warning once
        # nothing more can be pared).
        if "max_tokens" in msg and "max_tokens" in params:
          params["max_completion_tokens"] = params.pop("max_tokens")
          continue
        if "temperature" in msg and "temperature" in params:
          params.pop("temperature", None)
          params.pop(
              "top_p", None
          )  # same reasoning-model restriction covers both
          continue
        if "top_p" in msg and "top_p" in params:
          params.pop("top_p", None)
          continue
        logger.warning("[colbench_spec] OpenAI sim call failed: %r", e)
        return "No response."
    return "No response."

  return backend


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Conversation debug (mirrors env._DEBUG_SIM). Dumps the SPEC the sim conditions
# on + raw reply, so the user can eyeball that (a) the GT never enters the sim
# prompt and (b) the sim behaves.
_DEBUG_SIM = bool(int(os.getenv("COLBENCH_DEBUG_SIM", "0") or "0"))
_DEBUG_PREVIEW = int(os.getenv("COLBENCH_DEBUG_CONVO_PREVIEW", "400") or "400")


@dataclass
class ColBenchSpecUserSimEnv:
  """Spec-conditioned user-simulator env.

  Holds the problem, the spec, the hidden GT and the GT calls.

  Attributes:
    problem_description: the user's (public) problem statement.
    spec: the authored spec
      ``{persona{who,domain,python_skill,communication_style}, scenario,
      requirements, plot}`` the sim conditions on (NEVER the GT code).
    ground_truth: the HIDDEN GT function source -- used ONLY for grading
      (score), never shown to the sim or the solver (except in ``grounded``
      mode, where it conditions the sim -- still never the solver).
    test_cases: list of GT call-strings used for grading.
    max_steps: max solver turns before the episode is force-ended (loop
      guardrail).
    reward_time_limit: per-case exec timeout (seconds) for grading.
    sim_backend: (system, user) -> raw reply. Defaults to the frozen-server HTTP
      call; tests / Phase-2 inject their own.
    sim_max_tries: reject-sampling budget. The sim is re-queried up to this many
      times if it writes code (an ordinary user describes in words, never pastes
      a function); if ALL tries still contain a code fence the loop aborts the
      episode (see ``last_sim_code_reject_exhausted``) rather than injecting or
      stripping a bad reply.
    grounded: GROUNDED mode (``+colbench.grounded_sim``) -- condition the sim on
      the hidden GT function source + ``spec["plot"]`` instead of
      persona/scenario/requirements. Everything else about this env is
      unchanged. When True the GT IS in the sim's prompt, so the
      ``sim_max_tries`` rejection above becomes the load-bearing leak defense
      rather than a character guard.
    last_sim_reply: the reply from the last ``generate_user_turn`` call, kept
      for the loop's debug dump / audit.
    last_sim_raw: the most recent RAW (uncapped, but ``<think>``-stripped) sim
      reply, so the loop can string-match ``[TERMINATE]`` on the sim's true
      output before the injected turn is char-capped.
    last_sim_code_rejected: how many code-writing sim replies were discarded on
      the last ``generate_user_turn`` (diagnostic).
    last_sim_code_reject_exhausted: True iff EVERY try on the last
      ``generate_user_turn`` still wrote code -> the loop should abort this
      conversation (``terminated_by`` "sim_code_reject") so it can be read, NOT
      inject a bad turn.
  """

  problem_description: str
  spec: dict[str, Any]
  ground_truth: str
  test_cases: list[str]
  max_steps: int = 10
  reward_time_limit: float = 6.0
  sim_backend: Optional[SimBackend] = None
  sim_max_tries: int = 8
  grounded: bool = False
  last_sim_reply: str = field(default="", repr=False)
  last_sim_raw: str = field(default="", repr=False)
  last_sim_code_rejected: int = field(default=0, repr=False)
  last_sim_code_reject_exhausted: bool = field(default=False, repr=False)

  def __post_init__(self):
    if self.sim_backend is None:
      self.sim_backend = openai_sim_backend

  def generate_user_turn(self, messages: list[dict[str, str]]) -> str:
    """Produce the next spec-conditioned user (simulator) reply.

    ``messages`` is the running dialogue as ``[{role, content}, ...]`` (problem
    + solver turns + prior user replies) -- it carries NO ground truth. The spec
    (or, in ``grounded`` mode, the GT source + plot) is injected as the sim's
    SYSTEM message here and passed only to the backend -- it never enters
    ``messages``; the returned reply is ``<think>``-stripped and char-capped so
    the solver's message list only receives the short human-like turn.
    ``last_sim_raw`` keeps the stripped-but-uncapped reply so the loop can
    detect ``[TERMINATE]`` (the sentinel could otherwise fall past the char
    cap).

    Args:
      messages: the running dialogue as ``[{role, content}, ...]``; carries no
        GT.

    Returns:
      The next user reply: ``<think>``-stripped and char-capped.
      ``last_sim_raw`` holds the uncapped form for ``[TERMINATE]`` detection.
    """
    if self.grounded:
      system_content, user_content = templates.build_grounded_sim_messages(
          self.problem_description,
          self.ground_truth,
          (self.spec or {}).get("plot", ""),
          messages,
      )
    else:
      system_content, user_content = templates.build_spec_sim_messages(
          self.spec, messages
      )
    # Rejection sampling: an ordinary user never pastes code. If the sim writes
    # a code fence, re-query (sampling temperature makes retries differ). If
    # EVERY try still contains code, we do NOT strip or inject it (stripping
    # yields weird half-sentences) -- we flag exhaustion so the loop aborts the
    # conversation for the user to read.
    rejected = 0
    raw = ""
    stripped = ""
    exhausted = True
    for _ in range(max(1, self.sim_max_tries)):
      raw = self.sim_backend(system_content, user_content)
      stripped = templates.strip_think(raw)
      if not templates.sim_wrote_code(stripped):
        exhausted = False
        break
      rejected += 1
    self.last_sim_code_rejected = rejected
    self.last_sim_code_reject_exhausted = exhausted
    # No post-hoc character truncation: the old HUMAN_RESPONSE_CHARACTER_LIMIT
    # slice chopped verbose replies mid-sentence (the solver then saw
    # fragments). Brevity is enforced at the source instead -- by the "one or
    # two short sentences" instruction and the SIM_MAX_TOKENS generation bound
    # in the backend. The full (think-stripped) reply is injected as-is.
    reply = stripped
    self.last_sim_raw = stripped
    self.last_sim_reply = reply
    if _DEBUG_SIM:
      n = _DEBUG_PREVIEW
      logger.warning(
          "[COLBENCH_SPEC_SIM] spec_requirements[:%d]=%r\n"
          "[COLBENCH_SPEC_SIM] plot[:%d]=%r\n"
          "[COLBENCH_SPEC_SIM] sim_system[:%d]=%r\n"
          "[COLBENCH_SPEC_SIM] raw_reply[:%d]=%r\n"
          "[COLBENCH_SPEC_SIM] capped_reply=%r",
          n,
          str(self.spec.get("requirements"))[:n],
          n,
          str(self.spec.get("plot"))[:n],
          n,
          system_content[:n],
          n,
          str(raw)[:n],
          reply,
      )
    return reply

  def score(self, answer_text: str) -> dict[str, Any]:
    """Grade the submitted answer against the GT.

    Identical to the GT env: the answer is fence-stripped to code, then compared
    to the GT function on every call-string via the sandboxed exec sidecar
    (functional equivalence).

    Args:
      answer_text: the submitted answer, fence-stripped here.

    Returns:
      ``reward.grade``'s dict: ``pass_rate``, ``all_pass``, ``per_case``, ``n``.
    """
    code = templates.extract_code_answer(answer_text)
    return reward.grade(
        code,
        self.ground_truth,
        self.test_cases,
        time_limit=self.reward_time_limit,
    )

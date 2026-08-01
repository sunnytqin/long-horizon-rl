"""A configurable OpenAI-compatible chat client for the Phase-0 offline scripts.

Generalizes ``colbench.env.openai_sim_backend``: instead of reading a single
fixed endpoint from env vars, ``ChatEndpoint`` takes an explicit base_url /
model / sampling so the spec author (``strong`` teacher OR ``selfplay`` frozen
base) and the diagnostic solver can point at DIFFERENT served models in the same
run. Same retry-to-default and SGLang ``extra_body`` (top_k / min_p /
enable_thinking) handling as the sim backend, so behavior matches training.

``openai`` is imported lazily so CPU tests (which inject a stub callable) never
need the SDK.
"""

# This tree imports names directly (``from colbench.env import
# ColBenchUserSimEnv``) rather than the enclosing module, matching how the
# rest of verl is written; call sites read on the bare name throughout.
# pylint: disable=g-importing-member
from dataclasses import dataclass
from dataclasses import field
import logging
import os
import random
import threading
import time
from typing import Any
from typing import Callable
from typing import Optional

# Guards the shared usage counters, which are written from every worker thread.
_USAGE_LOCK = threading.Lock()

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# A raw chat backend maps a list[{role, content}] -> reply text. Default is the
# real HTTP call; tests inject a stub. Mirrors ``colbench.env.SimBackend`` but
# takes full messages.
ChatBackend = Callable[[list], str]


class ChatCallFailedError(RuntimeError):
  """Every retry was exhausted without a reply.

  Transient: rate limit, 5xx or timeout.

  Raised only when ``raise_on_exhausted=True``. Callers that PAY PER CALL use
  this to avoid persisting a failed row: a written row counts as done on resume,
  so swallowing the error would permanently burn that task's spend. See
  ``generate_specs``.
  """


class ChatCallFatalError(RuntimeError):
  """A non-retryable error (bad key, no access to the model, malformed request).

  Aborts the whole batch immediately rather than repeating a doomed call 10k
  times.
  """


class ChatCallRefusedError(RuntimeError):
  """The provider refused THIS prompt on content-safety grounds.

  Reported as ``invalid_prompt``.

  Permanent and row-specific: the same prompt will be refused every time, so it
  must neither be retried (pure waste) nor deferred (a deferred row is
  re-attempted on every resume, so the run could never converge). The caller
  records it as an unusable row and moves on.
  """


def _retry_after_seconds(exc) -> Optional[float]:
  """Honour a server-provided ``Retry-After`` (seconds) header when present.

  Args:
    exc: the exception raised by the API call; its ``response.headers`` are
      inspected if it has any.

  Returns:
    The header's value in seconds (never negative), or ``None`` when the
    header is absent or unparseable, in which case the caller backs off on its
    own schedule.
  """
  resp = getattr(exc, "response", None)
  hdrs = getattr(resp, "headers", None) or {}
  for key in ("retry-after", "Retry-After"):
    if key in hdrs:
      try:
        return max(0.0, float(hdrs[key]))
      except (TypeError, ValueError):
        return None
  return None


# Quota/billing exhaustion arrives as HTTP 429 with a RateLimitError --
# IDENTICAL in status and type to an ordinary rate limit, but it will NEVER
# clear on its own. Retrying it burns hours of walltime for nothing (observed
# 2026-07-30: a 10k run spun for 4h on `credit_balance_exhausted`), so it is
# matched on the error body and treated as fatal.
_QUOTA_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "exceeded your current quota",
    "no credits remaining",
    "billing_hard_limit_reached",
)


def _is_quota_exhausted(exc) -> bool:
  """Is this a billing/quota exhaustion rather than an ordinary rate limit?

  Args:
    exc: the exception raised by the API call; matched on its text, since the
      status and type are identical to a retryable 429.

  Returns:
    True if the error body carries one of ``_QUOTA_MARKERS``, meaning waiting
    can never clear it.
  """
  return any(m in str(exc).lower() for m in _QUOTA_MARKERS)


# A content-safety refusal of one specific prompt. Arrives as a 400 (same status
# as a malformed request), so it must be matched on the error code -- otherwise
# it is retried as a "maybe the params were wrong" case, which can never succeed
# for that row.
_REFUSAL_MARKERS = (
    "invalid_prompt",
    "content_policy_violation",
    "limited access to this content for safety reasons",
)


def _is_prompt_refused(exc) -> bool:
  """Did the provider refuse this specific prompt on content-safety grounds?

  Args:
    exc: the exception raised by the API call; matched on its text, since the
      status (400) is shared with a malformed request.

  Returns:
    True if the error body carries one of ``_REFUSAL_MARKERS``, meaning the
    row is unusable rather than retryable.
  """
  return any(m in str(exc).lower() for m in _REFUSAL_MARKERS)


def _classify(exc) -> str:
  """Decide how the caller should react to a failed API call.

  Args:
    exc: the exception raised by the call. Classified on HTTP status where one
      is available, else on the exception's type name, else on the error text
      (quota exhaustion and content refusals share a status with retryable
      errors, so they can only be told apart by body).

  Returns:
    One of ``"fatal"`` (abort the run -- a human must intervene),
    ``"refused"`` (skip this row permanently), ``"invalid"`` (retry with a
    more-minimal param set) or ``"transient"`` (back off and retry). Unknown
    errors are ``"transient"``, bounded by the caller's retry budget.
  """
  status = getattr(exc, "status_code", None) or getattr(
      exc, "http_status", None
  )
  if status is None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
  name = type(exc).__name__
  if status in (401, 403) or name in (
      "AuthenticationError",
      "PermissionDeniedError",
  ):
    return "fatal"  # bad/expired key or no access to this model -> stop now
  if status == 404 or name == "NotFoundError":
    return "fatal"  # unknown model id -> every row would fail identically
  if _is_quota_exhausted(exc):
    return "fatal"  # out of credits: a wait cannot fix it, only a human can
  if status == 429 or name == "RateLimitError":
    return "transient"  # a genuine rate limit -> back off and retry
  if status is not None and 500 <= status < 600:
    return "transient"
  if name in ("APITimeoutError", "APIConnectionError", "InternalServerError"):
    return "transient"
  if _is_prompt_refused(exc):
    # This prompt will never be accepted -> record it and skip the row.
    return "refused"
  if status == 400 or name in ("BadRequestError", "UnprocessableEntityError"):
    return (
        "invalid"  # try a more-minimal param set (model rejects a sampling arg)
    )
  return "transient"  # unknown -> treat as retryable, bounded by `retries`


@dataclass
class ChatEndpoint:
  """One served model behind an OpenAI-compatible API.

  With fixed sampling params.

  Attributes:
    base_url: e.g. ``http://127.0.0.1:30000/v1``.
    model: served model name/alias (the server's ``--served-model-name``).
    api_key: usually ``"EMPTY"`` for a local SGLang/vLLM server.
    temperature: sampling temperature.
    top_p: nucleus mass.
    top_k: top-k cutoff.
    min_p: minimum-probability cutoff. These four default to the Qwen3-Instruct
      recommendation, matching ``colbench.env._sim_sampling``; NB Qwen3 degrades
      under greedy decoding.
    max_tokens: completion cap.
    enable_thinking: None -> send no thinking kwarg (safe for all models);
      True/False routes the flag to the hybrid-Qwen3 chat template. NB the wire
      format is vendor-specific: vLLM has no top-level ``enable_thinking`` field
      and ``extra="allow"``, so a top-level key would be SILENTLY DROPPED (the
      model would think and emit ``<think>`` blocks the spec parser cannot use)
      -> it must go through ``chat_template_kwargs``.
    retries: per-call retry budget.
    timeout: socket timeout in seconds. On a METERED API a client timeout is
      the most expensive possible failure -- the server still generated (and
      billed) the completion, we throw it away, and then we pay to redo it.
      Waiting is strictly cheaper than abandoning, so this must be generous:
      far above the slowest expected generation, not a "responsiveness" value.
    backoff_base: base of the exponential backoff (with full jitter) between
      attempts. A server-sent ``Retry-After`` overrides the computed wait. Only
      matters for a metered API.
    backoff_cap: ceiling on that computed wait, in seconds.
    raise_on_exhausted: raise ``ChatCallFailedError`` instead of returning
      ``""`` when the retries run out, so a PAID caller can decline to persist
      the row and retry it on resume instead.
    usage: optional shared dict for SERVER-REPORTED token usage, accumulated
      across threads: ``{calls, prompt_tokens, completion_tokens,
      reasoning_tokens, cached_tokens}``. Without it a metered run has no idea
      what it actually spent -- counting the visible reply text undercounts a
      reasoning model, whose hidden thinking is billed as output.
    service_tier: OpenAI service tier. ``"flex"`` trades latency for ~half
      price, which suits offline authoring; NB flex returns 429
      ``resource_unavailable`` when capacity is short, which is TRANSIENT
      (retry with backoff) and NOT the quota exhaustion that aborts a run, and
      it needs a generous timeout.
    vendor: ``"vllm"``/``"sglang"`` local server (sends ``top_k``/``min_p`` via
      ``extra_body``) or ``"openai"`` vanilla API (no vendor extras; some
      models reject custom sampling, so params are tried progressively-minimal
      -- see ``_param_sets``).
    backend: injectable raw backend (tests). None -> the lazy OpenAI HTTP call.
  """

  base_url: str
  model: str
  api_key: str = "EMPTY"
  temperature: float = 0.7
  top_p: float = 0.8
  top_k: int = 20
  min_p: float = 0.0
  max_tokens: int = 4096
  enable_thinking: Optional[bool] = None
  retries: int = 3
  timeout: float = 120.0
  backoff_base: float = 2.0
  backoff_cap: float = 60.0
  raise_on_exhausted: bool = False
  usage: Optional[dict[str, Any]] = field(default=None, repr=False)
  service_tier: Optional[str] = None
  vendor: str = "vllm"
  backend: Optional[ChatBackend] = field(default=None, repr=False)

  def _extra_body(self) -> dict[str, Any]:
    """Vendor extensions to send outside the standard chat-completion params.

    Returns:
      ``{}`` for the public OpenAI API, which 400s on these; otherwise the
      vLLM/SGLang extras (``top_k``, ``min_p``, and the thinking flag).
    """
    if self.vendor == "openai":
      return (
          {}
      )  # top_k / min_p / enable_thinking are vLLM/SGLang extensions -> the
      # public OpenAI API 400s on them.
    eb = {"top_k": self.top_k, "min_p": self.min_p}
    if self.enable_thinking is not None:
      # MUST go through `chat_template_kwargs` (the form both vLLM and SGLang
      # document): Qwen3-hybrid disables reasoning in the JINJA TEMPLATE (it
      # pre-fills an empty `<think></think>` block), so the flag has to reach
      # the renderer. vLLM declares no top-level `enable_thinking` field and
      # allows extras -> that form is silently dropped.
      eb["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
    return eb

  def _param_sets(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Ordered param dicts to try.

    For the OpenAI API we degrade gracefully because newer models reject a
    custom ``temperature``/``top_p`` and require ``max_completion_tokens``
    instead of ``max_tokens``: try full sampling, then temperature-only, then
    bare.

    Args:
      messages: the chat messages to send.

    Returns:
      Request-parameter dicts to try in order, each one a further degradation of
      the last.
    """
    if self.vendor == "openai":
      base = {
          "model": self.model,
          "messages": messages,
          "max_completion_tokens": self.max_tokens,
          "timeout": self.timeout,
      }
      if self.service_tier:
        base["service_tier"] = self.service_tier
      return [
          {**base, "temperature": self.temperature, "top_p": self.top_p},
          {**base, "temperature": self.temperature},
          base,
      ]
    return [
        {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": self._extra_body(),
            "timeout": self.timeout,
        }
    ]

  def _record_usage(self, usage) -> None:
    """Accumulate SERVER-REPORTED token counts into ``usage``.

    Thread-safe: the ``usage`` dict is shared across workers.

    Records reasoning tokens separately: a reasoning model bills hidden thinking
    as output, so counting the visible reply understates the bill (and cost
    estimates built on it are wrong).

    Args:
      usage: the ``usage`` object from one chat completion. ``None`` (or an
        endpoint with no ``usage`` dict) is ignored, so callers need not check.
    """
    if self.usage is None or usage is None:
      return
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    pdetails = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(pdetails, "cached_tokens", 0) or 0
    with _USAGE_LOCK:
      self.usage["calls"] = self.usage.get("calls", 0) + 1
      self.usage["prompt_tokens"] = self.usage.get("prompt_tokens", 0) + (
          getattr(usage, "prompt_tokens", 0) or 0
      )
      self.usage["completion_tokens"] = self.usage.get(
          "completion_tokens", 0
      ) + (getattr(usage, "completion_tokens", 0) or 0)
      self.usage["reasoning_tokens"] = (
          self.usage.get("reasoning_tokens", 0) + reasoning
      )
      self.usage["cached_tokens"] = self.usage.get("cached_tokens", 0) + cached

  def _http_backend(self, messages: list[dict[str, str]]) -> str:
    # pylint: disable=g-import-not-at-top
    """Issue one chat completion and return its reply text.

    Args:
      messages: the chat messages to send.

    Returns:
      The assistant reply text, or ``""`` once the retries are spent and
      ``raise_on_exhausted`` is False.

    Raises:
      ChatCallFatalError: the error is non-retryable (bad key, unknown model,
        quota exhausted), so the whole batch should stop rather than repeat a
        doomed call.
      ChatCallRefusedError: the provider refused this specific prompt on
        content-safety grounds; the row is unusable and must be skipped, not
        retried or deferred.
      ChatCallFailedError: every retry was exhausted. Only raised when
        ``raise_on_exhausted`` is True, so a paid caller can avoid persisting
        the row.
    """
    from openai import OpenAI  # lazy: only the real path needs the SDK

    # max_retries=0: the SDK's own retry layer would silently re-send on
    # 429/timeout, hiding failures from the classifier below and (for a timeout)
    # paying for a generation twice. This module owns retrying, so it is the
    # ONLY retry layer.
    client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
    param_sets = self._param_sets(messages)
    last = None
    for attempt in range(self.retries):
      for (
          params
      ) in param_sets:  # fall to a more-minimal param set on invalid-request
        try:
          if self.usage is not None:
            # Count REQUESTS ISSUED, not just successful ones: a timed-out or
            # discarded generation is billed even though it returns no usage to
            # us. attempts vs authored rows is the multiplier that reconciles
            # with an invoice.
            with _USAGE_LOCK:
              self.usage["attempts"] = self.usage.get("attempts", 0) + 1
          completion = client.chat.completions.create(**params)
          self._record_usage(getattr(completion, "usage", None))
          return completion.choices[0].message.content or ""
        except Exception as e:  # pylint: disable=broad-exception-caught
          last, kind = e, _classify(e)
          if kind == "fatal":
            # Never retry: it would fail identically for every remaining row.
            raise ChatCallFatalError(f"non-retryable API error: {e!r}") from e
          if kind == "refused":
            # Permanent for THIS prompt only -- surface at once, keep the batch
            # going.
            raise ChatCallRefusedError(str(e)) from e
          if kind == "invalid":
            continue  # next (more minimal) param set, no sleep
          break  # transient -> stop trying param sets, back off instead
      if attempt < self.retries - 1:
        # Exponential backoff with full jitter, capped; a server-sent
        # Retry-After wins.
        wait = min(self.backoff_cap, self.backoff_base * (2**attempt))
        wait = random.uniform(0, wait)
        ra = _retry_after_seconds(last)
        if ra is not None:
          wait = min(self.backoff_cap, max(wait, ra))
        logger.warning(
            "[selfplay] %s -> retry %d/%d in %.1fs (%r)",
            self.base_url,
            attempt + 1,
            self.retries,
            wait,
            last,
        )
        time.sleep(wait)
    if self.raise_on_exhausted:
      raise ChatCallFailedError(f"exhausted {self.retries} attempts: {last!r}")
    logger.warning(
        "[selfplay] giving up on %s after %d attempts: %r",
        self.base_url,
        self.retries,
        last,
    )
    return ""

  def chat(self, messages: list[dict[str, str]]) -> str:
    """Reply text for ``messages``; retries, then "" if exhausted.

    Args:
      messages: the chat messages to send.

    Returns:
      The assistant's reply text, or ``""`` once the retry budget is spent.
    """
    backend = self.backend or self._http_backend
    return backend(messages)

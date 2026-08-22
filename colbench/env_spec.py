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

# How many discarded premature-termination replies to keep per turn for the eval
# dump. Bounded so a pathological sim cannot bloat the trajectory record.
_EARLY_TERM_SAMPLES = 3

# Legal values of `ColBenchSpecUserSimEnv.sim_code_leak_detector`. Public and
# module-level so the agent loop can validate `+colbench.sim_code_leak_detector`
# at LAUNCH against the same set the env enforces per episode, instead of
# keeping a second copy that can drift.
SIM_CODE_LEAK_DETECTORS = frozenset({"auto", "fence_only", "a0_strict"})


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
    grounded: BACK-COMPAT alias for ``sim_prompt="grounded"`` -- kept so
      ``+colbench.grounded_sim``, validate_colbench_spec's ``--grounded`` and
      every existing run record keep working unchanged. Setting it is equivalent
      to passing ``sim_prompt="grounded"``; an explicit ``sim_prompt`` wins.
    sim_prompt: WHICH user-simulator prompt conditions the sim. The one knob that
      selects an arm:
        "spec"     -- authored persona/scenario/requirements/plot, no GT.
        "grounded" -- hidden GT source + ``spec["plot"]``, character role-play.
        "grounded_v0" -- the historical pre-guard grounded prompt + plot.
        "grounded_v0_no_plot" -- the same historical grounded prompt, but
                                  with the plot omitted. This is the clean
                                  plot ablation for the V0 reference arm.
        "codeonly" -- A1: the NAIVE arm's stock answerer on the GT, no plot, no
                      role-play. Byte-identical to the naive arm's sim call, so
                      it is the null-delta CONTROL for the ladder.
        "plot"     -- A2: "codeonly" + the authored ``spec["plot"]``.
      "codeonly"/"plot" are meant to run at ``max_code_proposals=1``, which
      removes the sim's judging and termination roles entirely (the loop grades
      on the first proposal and breaks before the sim's next turn). In every mode
      that puts the GT in the sim's prompt -- grounded, grounded_v0,
      grounded_v0_no_plot, codeonly, plot -- the
      ``sim_max_tries`` rejection above is the load-bearing leak defense rather
      than a character guard.
    sim_code_leak_detector: WHICH detector screens a sim draw for code, as an
      explicit experiment axis independent of ``sim_prompt``:
        "auto"       -- per-arm legacy behavior: strict for codeonly/plot,
                        fence-only for spec/grounded/grounded_v0/
                        grounded_v0_no_plot. BACKWARD
                        COMPATIBLE: every run before this knob existed is
                        reproduced byte-for-byte by the default.
        "fence_only" -- always ``templates.sim_wrote_code`` (triple-backtick
                        fence only).
        "a0_strict"  -- always the NAIVE arm's
                        ``templates.detect_code_leak(..., ngram_n=0)``: bare
                        ``def name(`` OR fence.
      A string, not a bool, so a run record says which policy was in force
      rather than leaving "strict=False" to be read against a per-arm default.
      NB turning this on for grounded/spec makes the exhaustion path live there:
      eight bare-def draws must end the episode as ``sim_code_reject``, which is
      why the attribution below uses THIS predicate and not a fixed detector.
    last_sim_reply: the reply from the last ``generate_user_turn`` call, kept
      for the loop's debug dump / audit.
    last_sim_raw: the most recent RAW (uncapped, but ``<think>``-stripped) sim
      reply, so the loop can string-match ``[TERMINATE]`` on the sim's true
      output before the injected turn is char-capped.
    last_sim_code_rejected: how many code-writing sim replies were discarded on
      the last ``generate_user_turn`` (diagnostic).
    last_sim_code_reject_exhausted: True iff the last ``generate_user_turn``
      ran out of tries AND its final draw still wrote code -> the loop should
      abort this conversation (``terminated_by`` "sim_code_reject") so it can be
      read, NOT inject a bad turn.
    last_sim_early_term_rejected: how many PREMATURE-termination sim replies
      were discarded on the last ``generate_user_turn`` -- only ever nonzero
      when the caller passed ``allow_terminate=False`` (diagnostic).
    last_sim_early_term_exhausted: True iff the last ``generate_user_turn`` ran
      out of tries while rejecting premature terminations, i.e. the sim insisted
      on ending before the solver had shown any code. The reply is returned
      anyway (the loop then ends the episode exactly as it did before this
      guard existed), so this flag is what tells you the guard was overruled.
    last_sim_early_term_samples: up to ``_EARLY_TERM_SAMPLES`` of the discarded
      premature-termination replies, so eval can dump WHY they were discarded.
    last_sim_raw_attempts: how many RAW draws the last ``generate_user_turn``
      pulled from the backend (accepted + rejected). The denominator for every
      per-draw rate below; without it a rejection count is not a rate at all.
    last_sim_raw_fenced_code: raw draws containing a code fence. No separate
      "rejected_fenced" counter exists because every current policy rejects a
      fence, so the two would be identical by construction.
    last_sim_raw_bare_target_def: raw draws where the A0-strict detector fires
      but fence-only does NOT -- i.e. an unfenced ``def name(``. This is the ONE
      population the two policies disagree about, so it is the census that makes
      a strict-vs-fence comparison readable.
    last_sim_raw_early_termination: raw draws that are termination-like while
      ``allow_terminate=False``. Counted BEFORE the code branch short-circuits,
      so a draw that both leaks and terminates lands in both raw counters (the
      *rejected* counters, which follow the loop, attribute it to code only).
    last_sim_rejected_bare_target_def: bare-target-def draws discarded on the
      CODE ground -- nonzero only under a strict policy.
    last_sim_accepted_bare_target_def: bare-target-def draws that were ACCEPTED
      and therefore injected into the solver's context, GT and all -- nonzero
      only under a fence-only policy. Raw minus rejected minus accepted is the
      residual: bare-def draws a fence-only policy let past the code branch and
      the early-term guard then discarded.
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
  sim_prompt: str = ""
  sim_code_leak_detector: str = "auto"
  last_sim_reply: str = field(default="", repr=False)
  last_sim_raw: str = field(default="", repr=False)
  last_sim_code_rejected: int = field(default=0, repr=False)
  last_sim_code_reject_exhausted: bool = field(default=False, repr=False)
  last_sim_early_term_rejected: int = field(default=0, repr=False)
  last_sim_early_term_exhausted: bool = field(default=False, repr=False)
  last_sim_early_term_samples: list[str] = field(
      default_factory=list, repr=False
  )
  last_sim_raw_attempts: int = field(default=0, repr=False)
  last_sim_raw_fenced_code: int = field(default=0, repr=False)
  last_sim_raw_bare_target_def: int = field(default=0, repr=False)
  last_sim_raw_early_termination: int = field(default=0, repr=False)
  last_sim_rejected_bare_target_def: int = field(default=0, repr=False)
  last_sim_accepted_bare_target_def: int = field(default=0, repr=False)

  # Modes that condition the sim on the hidden GT source. In all of them the
  # leak invariant is enforced by sim_wrote_code rejection sampling, NOT by
  # construction as in "spec".
  _GT_CONDITIONED = frozenset(
      {"grounded", "grounded_v0", "grounded_v0_no_plot", "codeonly", "plot"}
  )
  # Rejection-policy axis (see `sim_code_leak_detector`). "auto" is the legacy
  # per-arm split; the other two override it in the same direction for every
  # arm, so an experiment can vary the detector while holding the prompt fixed.
  _LEAK_DETECTORS = SIM_CODE_LEAK_DETECTORS

  def __post_init__(self):
    if self.sim_backend is None:
      self.sim_backend = openai_sim_backend
    # Resolve the arm ONCE, here, so generate_user_turn just dispatches. An
    # explicit sim_prompt wins; otherwise the legacy `grounded` bool decides, so
    # existing callers (+colbench.grounded_sim, validate's --grounded, the older
    # tests) keep their exact behavior.
    if not (self.sim_prompt or "").strip():
      self.sim_prompt = "grounded" if self.grounded else "spec"
    if self.sim_prompt not in {"spec"} | self._GT_CONDITIONED:
      raise ValueError(
          f"unknown sim_prompt {self.sim_prompt!r}; expected one of "
          f"spec, {', '.join(sorted(self._GT_CONDITIONED))}"
      )
    # Keep the alias truthful for anything that still reads `.grounded`.
    self.grounded = self.sim_prompt in (
        "grounded",
        "grounded_v0",
        "grounded_v0_no_plot",
    )
    # Fail loudly on a typo'd detector: silently falling back to "auto" would
    # run the arm under the OPPOSITE policy from the one the run record claims,
    # and the two differ only in how often the sim is resampled -- nothing in
    # the logs would look wrong.
    self.sim_code_leak_detector = (
        str(self.sim_code_leak_detector or "auto").strip().lower()
    )
    if self.sim_code_leak_detector not in self._LEAK_DETECTORS:
      raise ValueError(
          "unknown sim_code_leak_detector"
          f" {self.sim_code_leak_detector!r}; expected one of"
          f" {', '.join(sorted(self._LEAK_DETECTORS))}"
      )

  def _fenced_code(self, reply: str) -> bool:
    """True iff the reply contains a triple-backtick fence (the WEAK detector).

    Args:
      reply: the candidate sim reply, already ``<think>``-stripped.

    Returns:
      ``templates.sim_wrote_code(reply)``.
    """
    return templates.sim_wrote_code(reply)

  def _a0_strict_code(self, reply: str) -> bool:
    """True iff the NAIVE arm's detector fires (bare ``def name(`` OR fence).

    Calls ``templates.detect_code_leak`` with the naive arm's exact arguments
    (``ngram_n=0``, default ``min_operators``) -- deliberately the same function
    object the GT env screens with, NOT a re-derived regex, so the two arms
    cannot drift.

    Args:
      reply: the candidate sim reply, already ``<think>``-stripped.

    Returns:
      True iff a leak reason was found.
    """
    return templates.detect_code_leak(reply, self.ground_truth, 0) is not None

  def _bare_target_def(self, reply: str) -> bool:
    """True iff the reply is the population the two detectors DISAGREE about.

    That is: an unfenced target-function definition -- A0-strict rejects it,
    fence-only accepts it. Every other draw shape is scored identically by both
    policies, so this is the only counter that separates them.

    Args:
      reply: the candidate sim reply, already ``<think>``-stripped.

    Returns:
      True iff strict fires and fence-only does not.
    """
    return (not self._fenced_code(reply)) and self._a0_strict_code(reply)

  def _strict_detection(self) -> bool:
    """Resolve ``sim_code_leak_detector`` (+ the arm, under "auto") to a policy.

    Returns:
      True iff draws should be screened with the A0-strict detector.
    """
    if self.sim_code_leak_detector == "a0_strict":
      return True
    if self.sim_code_leak_detector == "fence_only":
      return False
    return self.sim_prompt in ("codeonly", "plot")

  def _leaked_code(self, reply: str) -> bool:
    """Screen a candidate sim reply for code, under the configured policy.

    ``sim_code_leak_detector`` picks the detector; "auto" (the default, and the
    behavior of every run predating that field) resolves it PER ARM, and it is
    that per-arm split the rest of this docstring describes:

    * ``codeonly`` / ``plot`` (the A1/A2 ladder) use ``detect_code_leak`` with
      ``ngram_n=0`` -- the bare ``def name(`` regex PLUS the fence -- which is
      byte-for-byte what the NAIVE arm screens with
      (``colbench_agent`` -> ``env.generate_user_turn_checked``). These arms
      exist to reproduce the naive arm, and ``def name(`` is the dominant leak
      shape there (~60% of user turns at weak checkpoints), so an unfenced
      ``def f(x, y): return x + y`` must be rejected here exactly as it is
      there.
    * ``spec`` / ``grounded`` keep ``sim_wrote_code`` (any triple-backtick
      fence) -- the DEFAULT there, not a constraint: every completed
      spec/grounded run was produced under it, so "auto" keeps them comparable.
      ``sim_code_leak_detector="a0_strict"`` is how a new grounded arm opts into
      the strict screen deliberately. ``spec`` also has no GT in the sim's
      prompt at all, so a leak is impossible there by construction.

    The strict detector is not free -- ``def f(x, y):`` is also what a
    legitimate reply looks like when the user answers "what signature do you
    want?", so it resamples some good turns away. The naive arm pays that cost
    too, which is the point: matching it matters more than being right.

    Args:
      reply: the candidate sim reply, already ``<think>``-stripped.

    Returns:
      True iff the reply should be rejected and resampled.
    """
    if self._strict_detection():
      return self._a0_strict_code(reply)
    return self._fenced_code(reply)

  def generate_user_turn(
      self, messages: list[dict[str, str]], allow_terminate: bool = True
  ) -> str:
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
      allow_terminate: whether a terminating reply is admissible THIS turn. The
        caller owns that fact (it tracks whether the solver has shown code), and
        passes False before the first code proposal -- the prompt's own MINIMUM
        bar. A premature termination is then rejected and resampled out of the
        SAME ``sim_max_tries`` budget as a code-writing reply. Default True =
        pre-existing behavior.

    Returns:
      The next user reply: ``<think>``-stripped and char-capped.
      ``last_sim_raw`` holds the uncapped form for ``[TERMINATE]`` detection.
    """
    plot = (self.spec or {}).get("plot", "")
    if self.sim_prompt in ("grounded", "grounded_v0", "grounded_v0_no_plot"):
      system_content, user_content = templates.build_grounded_sim_messages(
          self.problem_description,
          self.ground_truth,
          "" if self.sim_prompt == "grounded_v0_no_plot" else plot,
          messages,
          version=(
              "v0"
              if self.sim_prompt in ("grounded_v0", "grounded_v0_no_plot")
              else "v1"
          ),
      )
    elif self.sim_prompt in ("codeonly", "plot"):
      # A1 passes plot="" -> the naive arm's sim call, byte for byte.
      system_content, user_content = templates.build_minimal_sim_messages(
          self.problem_description,
          self.ground_truth,
          plot if self.sim_prompt == "plot" else "",
          messages,
      )
    else:
      system_content, user_content = templates.build_spec_sim_messages(
          self.spec, messages
      )
    # Rejection sampling, on TWO grounds, sharing one try budget:
    #  (a) an ordinary user never pastes code. If the sim writes a code fence,
    #      re-query (sampling temperature makes retries differ). If EVERY try
    #      still contains code, we do NOT strip or inject it (stripping yields
    #      weird half-sentences) -- we flag exhaustion so the loop aborts the
    #      conversation for the user to read.
    #  (b) a user cannot be satisfied by code that does not exist yet. When the
    #      caller says a termination is inadmissible (no code shown yet), a
    #      terminating draw is discarded the same way. This enforces in CODE the
    #      MINIMUM bar the sim prompt already states in prose, and it covers
    #      BOTH ways such a reply arises -- a genuine premature quit and a
    #      passing MENTION of the sentinel that `sim_terminated` cannot tell
    #      apart (see templates.TERMINATE_MARKER).
    rejected = 0
    early_term_rejected = 0
    early_term_samples: list[str] = []
    raw = ""
    stripped = ""
    accepted = False
    # RAW per-draw census, scored with BOTH detectors regardless of which one is
    # in force. Counting only what the active policy rejected would make the two
    # policies incomparable: under fence_only nothing records that a bare def
    # was ever drawn, so "strict rejects more" could not be told apart from "the
    # sim stopped writing bare defs". These are the numerators the guard
    # hypothesis needs; last_sim_raw_attempts is the denominator.
    raw_attempts = 0
    raw_fenced_code = 0
    raw_bare_target_def = 0
    raw_early_termination = 0
    rejected_bare_target_def = 0
    accepted_bare_target_def = 0
    for _ in range(max(1, self.sim_max_tries)):
      raw = self.sim_backend(system_content, user_content)
      stripped = templates.strip_think(raw)
      raw_attempts += 1
      fenced = self._fenced_code(stripped)
      # Same predicate as _bare_target_def, inlined to reuse `fenced` rather
      # than re-running the fence regex on every draw.
      bare_target_def = (not fenced) and self._a0_strict_code(stripped)
      raw_fenced_code += int(fenced)
      raw_bare_target_def += int(bare_target_def)
      # BEFORE the code branch below, which short-circuits: a draw that both
      # leaks and terminates is a real premature termination and must show up in
      # the raw census even though the loop attributes its REJECTION to code.
      if not allow_terminate and templates.sim_terminated(stripped):
        raw_early_termination += 1
      if self._leaked_code(stripped):
        rejected += 1
        rejected_bare_target_def += int(bare_target_def)
        # Per-DRAW dump, mirroring the GT env (whose _DEBUG_SIM sits inside
        # _finalize_reply and therefore fires on every sample). Without this the
        # spec path dumps only the FINAL reply, so the rejected drafts -- the
        # whole point of inspecting a rejection-sampling asymmetry -- are
        # invisible and the two arms cannot be compared draw for draw.
        if _DEBUG_SIM:
          logger.warning(
              "[COLBENCH_SPEC_SIM] REJECTED draw %d (code) reply[:%d]=%r",
              rejected,
              _DEBUG_PREVIEW,
              stripped[:_DEBUG_PREVIEW],
          )
        continue
      if not allow_terminate and templates.sim_terminated(stripped):
        early_term_rejected += 1
        if len(early_term_samples) < _EARLY_TERM_SAMPLES:
          early_term_samples.append(stripped)
        continue
      accepted = True
      # An ACCEPTED bare def is the failure this whole axis is about: the reply
      # is injected into the solver's context with the GT function in it.
      accepted_bare_target_def += int(bare_target_def)
      break
    self.last_sim_code_rejected = rejected
    self.last_sim_early_term_rejected = early_term_rejected
    self.last_sim_early_term_samples = early_term_samples
    self.last_sim_raw_attempts = raw_attempts
    self.last_sim_raw_fenced_code = raw_fenced_code
    self.last_sim_raw_bare_target_def = raw_bare_target_def
    self.last_sim_raw_early_termination = raw_early_termination
    self.last_sim_rejected_bare_target_def = rejected_bare_target_def
    self.last_sim_accepted_bare_target_def = accepted_bare_target_def
    # Attribute exhaustion to the FINAL rejection ground, not to whichever
    # ground happened most often. Crucially, use the SAME detector that drove
    # the retry loop: a strict policy (codeonly/plot under "auto", or ANY arm
    # under sim_code_leak_detector="a0_strict") rejects an unfenced
    # ``def name(`` as well as a fence, while sim_wrote_code is fence-only.
    # Using sim_wrote_code here would let eight rejected bare-def leaks be
    # mislabeled as an early-term exhaustion and injected into the solver
    # context -- the 2026-08-07 bug. It is load-bearing for a0_strict on
    # grounded/spec for exactly the same reason, so this line must keep calling
    # _leaked_code and never a fixed detector.
    final_rejected_for_code = self._leaked_code(stripped)
    self.last_sim_code_reject_exhausted = (
        (not accepted) and final_rejected_for_code
    )
    self.last_sim_early_term_exhausted = (
        (not accepted) and not final_rejected_for_code
    )
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
      # WHAT GETS DUMPED, and why it changed (2026-08-06). The previous version
      # logged `sim_system` + `spec_requirements` + `plot`. In the codeonly/plot
      # modes that is exactly backwards: the system message is the 28-char
      # constant SIM_SYSTEM_PROMPT (Ray's log dedup even collapses it to a
      # single "[repeated Nx]" line), `requirements` is not in the prompt at
      # all, and everything that DOES vary -- the problem, the hidden GT, the
      # optional plot clause, the rendered dialogue -- lives in `user_content`,
      # which was never dumped. The GT arm logs `sim_user_prompt`, so the one
      # comparison the dump exists to support (is A1's sim call byte-identical
      # to A0's?) was the one it could not answer. Log the same field, plus the
      # resolved mode so a dump self-identifies which rung of the ladder it is.
      logger.warning(
          "[COLBENCH_SPEC_SIM] sim_prompt_mode=%r\n"
          "[COLBENCH_SPEC_SIM] sim_system[:%d]=%r\n"
          "[COLBENCH_SPEC_SIM] sim_user_prompt[:%d]=%r\n"
          "[COLBENCH_SPEC_SIM] raw_reply[:%d]=%r\n"
          "[COLBENCH_SPEC_SIM] capped_reply=%r",
          self.sim_prompt,
          n,
          system_content[:n],
          n,
          user_content[:n],
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

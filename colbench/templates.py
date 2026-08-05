"""Prompt BUILDERS + answer/code extraction for the ColBench multi-turn loop.

The prompt TEXT itself moved to ``colbench/prompts.py``; what remains here are
the number-affecting transforms (marker extraction, code fence-strip,
``<think>`` strip, leak/termination detection) and the small functions that
format a prompt for one turn. Both live in one module per concern so the
training rollout (``colbench_agent`` / ``colbench_spec_agent``) and the offline
validators apply byte-identical text handling.

Ported from ``sweet_rl``
(``prompts/{llm_agent_code_prompt,human_simulator_code_prompt}.txt``,
``utils/code_utils.check_correctness`` fence-strip) and InfoPO's
``run_simulate_api.py`` (``check_and_extract_answer`` flexible marker patterns).
"""

# The long lines in this file are prose comments recording WHY a transform is
# shaped the way it is (the prompt text they refer to now lives in prompts.py).
# Re-wrapping them would churn blame across the whole file for no reader
# benefit, so the line-length limit stays disabled file-wide.
# pylint: disable=line-too-long

import re
from typing import Any
from typing import Optional

# ── Prompts ───────────────────────────────────────────────────────────────
# Every literal prompt string + protocol marker lives in colbench/prompts.py --
# this module keeps only the text TRANSFORMS. They are re-exported here (and NOT
# used directly by every call site) so `templates.X` keeps resolving for the
# agent loops, the validators, preprocess_* and the tests; prompts.py is the
# source of truth and prompt edits belong there.
# pylint: disable=unused-import
from colbench.prompts import ANSWER_MARKER
from colbench.prompts import COLBENCH_AGENT_SYSTEM_PROMPT
from colbench.prompts import COLBENCH_SPEC_AGENT_SYSTEM_PROMPT
from colbench.prompts import GROUNDED_SIM_SYSTEM_PROMPT
from colbench.prompts import HUMAN_SIMULATOR_PROMPT
from colbench.prompts import MINIMAL_SIM_PROMPT_WITH_PLOT
from colbench.prompts import SIM_SYSTEM_PROMPT
from colbench.prompts import SPEC_SIM_SYSTEM_PROMPT
from colbench.prompts import TERMINATE_MARKER
# pylint: enable=unused-import

# Cap on the simulator's reply, mirroring sweet_rl
# HUMAN_RESPONSE_CHARACTER_LIMIT. A brief, human-like reply -- also bounds how
# much a single user turn can cost the solver's budget.
HUMAN_RESPONSE_CHARACTER_LIMIT = 400



# ── <think> stripping ─────────────────────────────────────────────────────────
# Qwen3 (and other reasoning models) may emit a <think>...</think> block. We strip it from
# BOTH the solver text (before searching for the answer marker) and the sim reply (before the
# char cap) so reasoning never leaks into the extracted answer or the injected user turn. For
# a plain Instruct model this is a defensive no-op.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# An UNTERMINATED block: the generation hit its token cap mid-reasoning, so
# `</think>` never arrives. This is not a rare edge case -- the spec sim runs at
# SIM_MAX_TOKENS=256, far too few for a hybrid Qwen3 to finish thinking, so
# EVERY reply is truncated this way if thinking is on. Matching only the closed
# form let raw chain-of-thought through as the user's dialogue turn.
_THINK_OPEN_UNCLOSED = re.compile(r"<think>(?!.*</think>).*\Z", re.DOTALL)


def strip_think(text: str) -> str:
  """Remove ``<think>...</think>`` blocks, INCLUDING a trailing unterminated one.

  Callers treat an empty reply as a failed turn rather than injecting reasoning
  fragments into the conversation.

  Args:
    text: raw model output, possibly containing reasoning blocks.

  Returns:
    ``text`` with every ``<think>`` block removed and stripped; ``""`` if
    nothing but reasoning remained.
  """
  out = _THINK_BLOCK.sub("", text or "")
  out = _THINK_OPEN_UNCLOSED.sub("", out)
  return out.strip()


# ── Answer-marker extraction (ported from InfoPO run_simulate_api.check_and_extract_answer) ──
# Accept the several marker spellings observed in rollouts. Case-insensitive match; the answer
# is everything AFTER the marker.
_ANSWER_PATTERNS = [
    "I WANT TO ANSWER:",  # standard
    "I WANT_TO_ANSWER:",  # underscore
    "I WANT_TO ANSWER:",  # mixed
    "i want to answer:",  # lowercase
    "i want_to_answer:",  # lowercase + underscore
]


def check_and_extract_answer(response: str) -> tuple[bool, str]:
  """Return ``(has_answer, answer_text)``.

  ``has_answer`` is True iff any accepted spelling of the answer marker is
  present; the answer text is everything after the (first-matched) marker,
  stripped. Byte-identical to InfoPO's ``check_and_extract_answer`` so training
  and offline eval agree.

  Args:
    response: one assistant turn, already ``<think>``-stripped.
  """
  if not response:
    return False, ""
  response_lower = response.lower()
  for pattern in _ANSWER_PATTERNS:
    pattern_lower = pattern.lower()
    if pattern_lower in response_lower:
      if pattern in response:
        idx = response.find(pattern)
        return True, response[idx + len(pattern) :].strip()
      idx_lower = response_lower.find(pattern_lower)
      return True, response[idx_lower + len(pattern) :].strip()
  return False, ""


# ── Code fence-strip (ported from sweet_rl code_utils.check_correctness) ──────
# The ONLY piece of sweet_rl's in-process "safety" preprocessing we keep: pull the code out
# of a ```python ... ``` (or bare ``` ... ```) fence if the model wrapped it. sweet_rl's
# keyword blocklist (import os/sys, sudo, exit(, argparse, ...) is dropped -- it was a
# poor-man's substitute for a sandbox because sweet_rl exec'd in-process; our container
# sidecar supersedes it.
def extract_code_answer(answer_text: str) -> str:
  r"""Strip an answer MARKER and/or a ```python / ``` code fence, returning the code to grade.

  Mirrors sweet_rl's fence handling: prefer a ```python block, else the first ``` block, else
  the raw text. Returns the code string to grade (stripped).

  An ``I WANT TO ANSWER:`` marker is stripped ONLY IN THE UNFENCED CASE (no ``` anywhere). On
  the GT path that is a NO-OP -- ``final_answer`` already split at the marker before calling
  here. It matters when a policy TRAINED on the GT protocol is graded by the SPEC path, which
  has no marker convention and grades whatever ``extract_last_code`` finds on the WHOLE
  assistant turn. The GT solver prompt says "Directly output the raw python code after
  'I WANT TO ANSWER:'", so such a turn arrives UNFENCED and would reach the sandbox as
  ``"I WANT TO ANSWER:\ndef f(...)"`` -- a SyntaxError, i.e. a zero scored for a protocol
  reason and indistinguishable from a capability result in the cross-arm comparison.

  The no-fence guard is load-bearing, not tidiness. Stripping unconditionally REGRESSES a turn
  that shows a fenced function and mentions the marker AFTERWARDS ("```python...``` I WANT TO
  ANSWER: that's it"): splitting at the marker discards the fence and grades the
  trailing prose. A fence, when present, is always the more reliable signal, so
  it wins outright.

  Args:
    answer_text: the submitted answer text, marker and/or fence included.

  Returns:
    The code to hand to the grader, stripped.
  """
  text = answer_text or ""
  if "```" not in text:
    has_marker, after_marker = check_and_extract_answer(text)
    if has_marker:
      text = after_marker
  if "```python" in text:
    text = text.split("```python", 1)[1].split("```", 1)[0]
  elif "```" in text:
    # First fenced block (``` ... ```).
    parts = text.split("```")
    if len(parts) >= 3:
      text = parts[1]
  return text.strip()


# ── Code-leak detection (for the eval user-simulator rejection sampling) ──────
# The frozen sim is meant to answer in plain English from the hidden GT; at weak checkpoints
# it often just PASTES the solution instead (see the qwen3_4b step-200 study: ~60% of user
# turns leak a `def`). These detectors flag a candidate sim reply that gives out code so the
# eval harness can reject and resample. Three signals, byte-identical to the offline study:
#   (A) a python function signature `def name(`     -> "def"      (the dominant leak shape)
#   (B) a ```python fenced block                    -> "fenced"
#   (D) a >=ngram_n symbol-aware token run shared with the GT source whose matched span holds
#       >= min_operators code operators             -> "ngram"    (inline formula / expression)
# (C) whole-line overlap from the study is intentionally dropped (0 hits,
#     redundant with A).
# (D) is symbol-aware and operator-gated on PURPOSE: a word-only n-gram also
#     fires on legitimate natural-language behavior specs (e.g. matching
#     platform names / version strings), which are exactly the good sim turns we
#     must NOT reject. Requiring operators in the matched span keeps the real
#     expression leaks and spares prose.
_DEF_SIGNATURE_RE = re.compile(r"\bdef\s+\w+\s*\(")
_PY_FENCE_RE = re.compile(r"```python", re.IGNORECASE)
# Arithmetic / comparison / bracket operators. Deliberately excludes ',' '.' ':'
# (common in prose) so the operator gate keys on expression structure, not
# punctuation.
_CODE_OPERATORS = frozenset("+-*/%()[]=<>")


def _code_tokens(text: str) -> list[str]:
  r"""Symbol-aware tokenizer: each word OR each individual operator/punctuation char.

  Unlike a word-only (``\w+``) split this preserves operators, so a matched
  n-gram can be required to contain them -- the knob that separates a copied
  CODE expression from a prose behavior description that merely shares
  identifiers with the GT.

  Args:
    text: any source or prose fragment.

  Returns:
    One token per word and one per operator/punctuation character, in order.
  """
  return re.findall(r"\w+|[^\w\s]", text or "")


def detect_code_leak(
    text: str, ground_truth: str, ngram_n: int = 10, min_operators: int = 2
) -> Optional[str]:
  """Return a short reason string if ``text`` leaks code, else ``None``.

  ``ground_truth`` is the hidden GT source the simulator sees (n-gram overlap is
  computed against it, NOT the agent's own output -- that substitution was an
  offline-study convenience only). Reasons: ``"def"`` / ``"fenced"`` /
  ``"ngram"`` (see the module comment above). Checked in that order and
  short-circuits on the first hit.

  ``ngram_n <= 0`` DISABLES detector (D) (the default in the eval harness for
  now): the operator-gated n-gram check is a solid idea but held back as a
  FUTURE CONSIDERATION while we validate on the A/B leaks that dominate. (A)/(B)
  always run.

  Args:
    text: the simulator reply being screened.
    ground_truth: the hidden GT source the simulator was shown; n-gram overlap
      is measured against this.
    ngram_n: n-gram width for detector (D); ``<= 0`` disables it.
    min_operators: operators an n-gram match must contain to count, which is
      what separates copied code from prose sharing identifiers.
  """
  if not text:
    return None
  if _DEF_SIGNATURE_RE.search(text):
    return "def"
  if _PY_FENCE_RE.search(text):
    return "fenced"
  if ngram_n <= 0:
    return None
  gt_toks = _code_tokens(ground_truth)
  tx_toks = _code_tokens(text)
  if len(gt_toks) >= ngram_n and len(tx_toks) >= ngram_n:
    gt_ngrams = {
        tuple(gt_toks[i : i + ngram_n])
        for i in range(len(gt_toks) - ngram_n + 1)
    }
    for i in range(len(tx_toks) - ngram_n + 1):
      ng = tuple(tx_toks[i : i + ngram_n])
      if (
          ng in gt_ngrams
          and sum(tk in _CODE_OPERATORS for tk in ng) >= min_operators
      ):
        return "ngram"
  return None


# A ```python block, closed or left UNTERMINATED by the per-turn token cap. The solver's
# max_new_tokens_per_turn is 1024, so a long function can be cut off before its closing fence
# arrives; requiring the close would make that turn "not a submission", and the sim would then
# reply to half a function. Same failure shape as the truncated-<think> bug.
_PY_FENCE_CLOSED_RE = re.compile(
    r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
_PY_FENCE_UNCLOSED_RE = re.compile(
    r"```python\s*(.*)\Z", re.DOTALL | re.IGNORECASE
)


def fenced_function(text: str) -> Optional[str]:
  """Return the body of the first ```python block that DEFINES a function, else ``None``.

  The ``def`` requirement is the whole point: this is the GT path's SUBMIT
  signal, and a submission ends the episode after a single shot.
  ``contains_code`` (used on the spec path to pick a grading target) also fires
  on a bare ``def`` anywhere in prose -- harmless there, since showing code just
  invites a user reply and there is a 2-proposal budget, but fatal here, where
  "something like def parse(rows), is that right?" mid-clarification would
  force-submit the trajectory. Requiring a fenced block that actually contains a
  definition restores the deliberateness the "I WANT TO ANSWER:" marker used to
  provide.

  Args:
    text: one assistant turn.
  """
  for m in _PY_FENCE_CLOSED_RE.finditer(text or ""):
    body = m.group(1)
    if _DEF_SIGNATURE_RE.search(body):
      return body.strip()
  m = _PY_FENCE_UNCLOSED_RE.search(text or "")
  if m and _DEF_SIGNATURE_RE.search(m.group(1)):
    return m.group(1).strip()
  return None


def final_answer(assistant_text: str, episode_done: bool) -> tuple[bool, str]:
  """Resolve the solver's final answer text from one assistant turn.

  Resolution order:
    1. A ```python block defining a function -> that block IS the submission (the live protocol;
       see COLBENCH_AGENT_SYSTEM_PROMPT bullet 3).
    2. Else the legacy ``I WANT TO ANSWER:`` marker -> the text after it. Still
       accepted so that checkpoints and parquets predating 2026-07-31 keep
       working against this code: the marker prompt shipped in the GT dataset
       for months, and a fence-only detector would leave such a run silently
       unable to submit, every episode grinding to the turn cap. Accepting BOTH
       is also why no submit-protocol toggle is needed anywhere in the stack.
    3. Else, on the FINAL turn (``episode_done``), fall back to the whole
       response when it looks like code (``def``/``import``/``return``/`=` ...)
       or is non-trivial, so an episode that ran out of turns still submits the
       model's last attempt.
    4. Else no answer yet (keep interacting).
  ``assistant_text`` should already be ``strip_think``-ed by the caller.

  Args:
    assistant_text: one assistant turn, already ``strip_think``-ed by the
      caller.
    episode_done: True on the final turn, which enables the whole-response
      fallback (case 3).

  Returns:
    ``(has_answer, answer_text)``. ``has_answer`` is False when the solver
    should keep interacting, in which case ``answer_text`` is ``""``.
  """
  fenced = fenced_function(assistant_text)
  if fenced is not None:
    return True, fenced
  has_marker, answer_text = check_and_extract_answer(assistant_text)
  if has_marker:
    return True, answer_text
  if episode_done:
    last = (assistant_text or "").strip()
    if len(last) > 20 and any(
        k in last for k in ("def ", "import ", "class ", "return ", "=")
    ):
      return True, last
    if len(last) > 10:
      return True, last
  return False, ""


def str_dialogue_history(messages: list[dict[str, str]]) -> str:
  """Render the running dialogue as the sim-prompt ``{dialogue_history}`` string.

  Byte-identical to sweet_rl HumanInteractionEnv.str_dialogue_history:
  ``"<role>:<content>"`` per turn separated by four newlines, terminated with a
  trailing ``"agent:"`` cue so the simulator answers as the human to the agent's
  latest turn.

  Args:
    messages: the running dialogue as ``{role, content}`` dicts.

  Returns:
    The ``{dialogue_history}`` substring: ``"<role>:<content>"`` per turn joined
    by four newlines, with a trailing ``"agent:"`` cue.
  """
  result = ""
  for d in messages:
    result += str(d.get("role")) + ":"
    result += str(d.get("content")) + "\n\n\n\n"
  return result + "agent:"


def build_sim_user_message(
    problem_description: str,
    hidden_information: str,
    messages: list[dict[str, str]],
) -> str:
  """Format the user-simulator prompt for one turn (system stays SIM_SYSTEM_PROMPT)."""
  return HUMAN_SIMULATOR_PROMPT.format(
      problem_description=problem_description,
      hidden_information=hidden_information,
      dialogue_history=str_dialogue_history(messages),
  )


# Injected as the very first solver user turn wrapping the problem statement.
# ColBench's problem_description already reads as a direct user request ("Create
# a python function ..."), so -- matching sweet_rl HumanInteractionEnv.reset,
# which seeds the dialogue with the raw problem_description as the first user
# turn -- we pass it through unwrapped.
def build_initial_user_message(problem_description: str) -> str:
  return str(problem_description)


# ═════════════════════════════════════════════════════════════════════════════
# SPEC PATH (Phase 1) -- the builders below. Their prompt text (the solver
# prompt, the spec sim, the GROUNDED sim, the [TERMINATE] sentinel and the
# rationale comments for all of them) is in colbench/prompts.py.
# ═════════════════════════════════════════════════════════════════════════════



def build_spec_sim_messages(
    spec: dict[str, Any], messages: list[dict[str, str]]
) -> tuple[str, str]:
  """Build the spec sim's (system, user) messages for one turn.

  ``spec`` carries ``persona{who,domain,python_skill,communication_style},
  scenario, requirements, plot``. The system message conditions the sim on that
  spec (NEVER the GT code); the user message is the running dialogue
  (``str_dialogue_history``) -- the same seam split as the GT path's
  ``build_sim_user_message``.

  Args:
    spec: ``{persona{who,domain,python_skill,communication_style}, scenario,
      requirements, plot}``.
    messages: the running dialogue as ``{role, content}`` dicts.

  Returns:
    ``(system_content, user_content)`` for the simulator call.
  """
  persona = spec.get("persona", {}) or {}
  system = SPEC_SIM_SYSTEM_PROMPT.format(
      who=persona.get("who", "a person"),
      domain=persona.get("domain", ""),
      python_skill=persona.get("python_skill", ""),
      communication_style=persona.get("communication_style", ""),
      scenario=spec.get("scenario", ""),
      requirements=spec.get("requirements", ""),
      plot=spec.get("plot", ""),
  )
  return system, str_dialogue_history(messages)


def build_grounded_sim_messages(
    problem_description: str,
    ground_truth: str,
    plot: str,
    messages: list[dict[str, str]],
) -> tuple[str, str]:
  """Build the GROUNDED sim's (system, user) messages for one turn.

  Same seam split as ``build_spec_sim_messages`` (spec -> system, dialogue ->
  user), but the sim is conditioned on the hidden GT function source + the plot
  instead of the authored spec's persona/scenario/requirements. Kept as a
  SEPARATE function rather than a flag on ``build_spec_sim_messages`` so the
  pure-spec path stays branch-free and byte-identical.

  NB: this is the ONE place the GT source enters a sim prompt on the spec path
  -- the leak invariant (GT never reaches the solver's message list) is enforced
  downstream by the env's ``sim_wrote_code`` rejection sampling, not by
  construction as in the spec mode.

  Args:
    problem_description: the public, under-specified ask.
    ground_truth: the hidden GT function source the sim is grounded on.
    plot: the authored plot the sim improvises the conversation around.
    messages: the running dialogue as ``{role, content}`` dicts.

  Returns:
    ``(system_content, user_content)`` for the simulator call.
  """
  system = GROUNDED_SIM_SYSTEM_PROMPT.format(
      problem_description=problem_description,
      ground_truth=ground_truth,
      plot=plot or "",
  )
  return system, str_dialogue_history(messages)


def build_minimal_sim_messages(
    problem_description: str,
    ground_truth: str,
    plot: str,
    messages: list[dict[str, str]],
) -> tuple[str, str]:
  """Build the MINIMAL sim's (system, user) messages -- the A1/A2 rungs.

  The naive (GT-path) simulator, reachable from the spec loop. The sim is the
  stock sweet_rl answerer conditioned on the hidden GT source, with NO character
  role-play, NO judging role and NO termination role -- at
  ``max_code_proposals=1`` the loop grades on the first proposal and breaks
  before the sim's next turn, so it has neither.

  THE INVARIANT THAT MAKES A1 A CONTROL: with ``plot`` empty this returns
  EXACTLY what the naive arm sends -- ``SIM_SYSTEM_PROMPT`` as system and
  ``build_sim_user_message(...)`` as user, byte for byte. A1 is therefore the
  naive arm's environment reached through the spec loop, and any A1-vs-naive gap
  is a residual spec-path difference, not a prompt difference. Guarded by a test;
  do not "tidy" the empty-plot path into a shared format() call that would let
  the bytes drift.

  Args:
    problem_description: the user's public, under-specified ask.
    ground_truth: the hidden GT function source the sim answers from.
    plot: the authored ``spec["plot"]``. EMPTY = A1 ``codeonly``; non-empty = A2
      ``plot``.
    messages: the running dialogue as ``{role, content}`` dicts.

  Returns:
    ``(system_content, user_content)`` for the simulator call.
  """
  if not (plot or "").strip():
    # A1: byte-identical to the naive arm. Same call, same function.
    return SIM_SYSTEM_PROMPT, build_sim_user_message(
        problem_description, ground_truth, messages
    )
  user = MINIMAL_SIM_PROMPT_WITH_PLOT.format(
      problem_description=problem_description,
      hidden_information=ground_truth,
      plot=plot,
      dialogue_history=str_dialogue_history(messages),
  )
  return SIM_SYSTEM_PROMPT, user


def sim_terminated(reply: str) -> bool:
  """True iff the (``<think>``-stripped) sim reply contains the ``[TERMINATE]`` sentinel.

  Case-insensitive so a stray lowercasing by the sim still ends the episode.

  Args:
    reply: the sim reply, already ``<think>``-stripped.

  Returns:
    True iff the sentinel is present, so the episode should end.
  """
  return TERMINATE_MARKER.lower() in strip_think(reply).lower()


# Punctuation/emphasis a sim may wrap the bare sentinel in ("**[TERMINATE]**",
# "[TERMINATE].") and that still reads as a standalone signal.
_STANDALONE_TRIM = " \t\n*`\"'.!"


def sim_terminate_standalone(reply: str) -> bool:
  """True iff the reply is JUST the sentinel -- the form the prompt asks for.

  DIAGNOSTIC ONLY -- deliberately NOT part of any termination decision (see the
  note by ``TERMINATE_MARKER``). Recorded per trajectory so we can tell a real
  hand-off ("[TERMINATE]") from an episode killed by a passing mention ("I
  shouldn't say [TERMINATE] yet, so ..."), which ``sim_terminated`` cannot
  distinguish.

  Args:
    reply: the sim reply, ``<think>``-stripped or not.

  Returns:
    True iff the sentinel, ignoring surrounding whitespace/emphasis, is the
    entire reply.
  """
  clean = strip_think(reply).strip().strip(_STANDALONE_TRIM).strip()
  return clean.upper() == TERMINATE_MARKER


def contains_code(text: str) -> bool:
  """True iff ``text`` looks like it proposes a function (a ```python fence or a ``def`` sig).

  Reuses the same signals as the leak detector's (A)/(B); here they mark the
  solver's OWN proposed code (the grading target), not a sim leak.

  Args:
    text: one assistant turn.

  Returns:
    True iff the turn proposes a function.
  """
  clean = strip_think(text)
  return bool(_PY_FENCE_RE.search(clean) or _DEF_SIGNATURE_RE.search(clean))


# Any triple-backtick fence -- an ordinary user speaking never wraps a reply in a code block, so
# this is the signal that the SIM slipped out of character and wrote code (```python def ... or a
# bare ``` block). Distinct from contains_code (which also fires on a lone "def" mention in prose).
_ANY_FENCE_RE = re.compile(r"```")


def sim_wrote_code(reply: str) -> bool:
  """True iff the (``<think>``-stripped) sim reply contains a fenced code block.

  The spec sim is an ordinary user: it must describe things in words, never paste code. A strong
  coder model (e.g. GPT-5) will sometimes "correct" the solver by writing the function itself,
  which spoon-feeds structure and breaks character. We reject-sample on this signal in the spec
  env. Keyed on a triple-backtick fence (``` / ```python) -- normal prose never contains one.

  Args:
    reply: the sim reply, already ``<think>``-stripped.

  Returns:
    True iff the reply contains a fenced code block, which the spec env
    reject-samples on.
  """
  return bool(_ANY_FENCE_RE.search(strip_think(reply)))


def extract_last_code(messages: list[dict[str, str]]) -> str:
  """Return the code from the most recent assistant turn that proposed a function, else ``""``.

  Scans ``messages`` newest-first for an assistant turn with ``contains_code``
  and returns ``extract_code_answer`` of it -- the "last function the solver
  showed", which the spec path grades on termination.

  Args:
    messages: the full conversation, oldest first.
  """
  for m in reversed(messages):
    if m.get("role") == "assistant" and contains_code(m.get("content", "")):
      return extract_code_answer(strip_think(m.get("content", "")))
  return ""

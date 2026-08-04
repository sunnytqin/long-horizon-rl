"""Prompts + answer/code extraction for the ColBench multi-turn loop.

Ported from ``sweet_rl``
(``prompts/{llm_agent_code_prompt,human_simulator_code_prompt}.txt``,
``utils/code_utils.check_correctness`` fence-strip) and InfoPO's
``run_simulate_api.py`` (``check_and_extract_answer`` flexible marker patterns).
The number-affecting transforms (marker extraction, code fence-strip,
``<think>`` strip) live here so the training rollout (``colbench_agent``) and
the offline validator apply byte-identical text handling.
"""

# The long lines in this file are prompt text inside string literals.
# Re-wrapping them would change the exact bytes sent to the model and break
# comparability with completed runs, so the line-length limit is disabled
# file-wide rather than reflowed. A per-line disable is not an option here: the
# comment would land inside the prompt and be sent to the model.
# pylint: disable=line-too-long

import re
from typing import Any
from typing import Optional

# ── Solver (agent) system prompt ──────────────────────────────────────────────
# Byte-identical to sweet_rl/prompts/llm_agent_code_prompt.txt. Kept as the PROVENANCE record
# only -- the live prompt is COLBENCH_AGENT_SYSTEM_PROMPT below, which diverges from this in
# exactly two documented places.
_AGENT_PROMPT_RAW = """You are a helpful LLM agent.
Your task is to help a human user to resolve their problem, in particular python programming.
1) Note that the problem is highly personalized so you need to explicitly gather information
by asking questions to the human user about some hidden information and implicit constraints.
YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.
2) Note that you should not ask human users complicated questions as they will only answer questions briefly in two sentences.
3) When you have gathered enough information to answer, say "I WANT TO ANSWER:" in the beginning of your response and provide your final answer.
4) Note that you can only interact with the human users WITHIN 10 back-and-forth rounds and you have to provide your final answer before the conversation ends.
5) You should be as concise as possible in your response to human.


"I WANT TO ANSWER:" should be included in your response to human if you think that you have gathered enough information for addressing this problem.
Directly output the raw python code after "I WANT TO ANSWER:".

Complete only the immediate agent response in this dialogue:
{dialogue_history}"""

# The solver's LIVE system prompt (used by the agent loop +
# preprocess_colbench). Bullets 1, 2, 4 and 5 are verbatim from the sweet_rl
# original above; two things deliberately differ:
#
#  (a) The trailing "{dialogue_history}" placeholder is gone. sweet_rl formatted
#      the whole conversation into it and called a COMPLETION endpoint; we use a
#      real CHAT template and let the actual message turns carry the history
#      (same as InfoPO's run_simulate_api.py).
#
#  (b) 2026-07-31: bullet 3's "I WANT TO ANSWER:" submit marker is replaced by a ```python code
#      block, matching the SPEC path's submission syntax. The golden spec eval is the shared
#      yardstick for the GT-vs-spec-vs-grounded study and it grades whatever `extract_last_code`
#      finds on the raw turn -- so a GT arm RL'd onto a marker protocol would be scored partly on
#      protocol conformance rather than capability. Aligning the syntax kills that confound at the
#      source instead of teaching the extractor to be bilingual.
#
#      What is NOT changed is the TERMINATION CONTROL FLOW, which stays
#      intentionally different between the arms: here the solver's own
#      submission ends the episode (one shot, no reaction to its code), while
#      the spec path lets the user react and terminate.
#
#      The trailing paragraph mirrors sweet_rl's own two sentences almost word-for-word with the
#      mechanism swapped, plus one clause: "Showing this code block indicates you are submitting
#      your final answer." That clause restores SEMANTICS the marker had for free -- "I WANT TO
#      ANSWER:" announces itself as an act of submission, whereas a ```python block is something
#      models emit constantly while explaining, so nothing about it says "this is my submission".
#      It is deliberately phrased as what the act MEANS, not as an instruction about what to do.
#
#      Note the coupling this introduces: under the marker, showing code and
#      submitting were separate acts, so the solver could sketch a snippet
#      mid-clarification for free. Now it cannot. Whether that costs anything is
#      UNMEASURED. Watch `num_assistant_turns` / `answered_at_turn` in the first
#      ~20 steps: a collapse to 1-turn episodes means the rule is not landing,
#      and the fix would be in the prompt, not the detector.
COLBENCH_AGENT_SYSTEM_PROMPT = """You are a helpful LLM agent.
Your task is to help a human user to resolve their problem, in particular python programming.
1) Note that the problem is highly personalized so you need to explicitly gather information
by asking questions to the human user about some hidden information and implicit constraints.
YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.
2) Note that you should not ask human users complicated questions as they will only answer questions briefly in two sentences.
3) When you have gathered enough information to answer, output the COMPLETE python function inside a ```python code block.
4) Note that you can only interact with the human users WITHIN 10 back-and-forth rounds and you have to provide your final answer before the conversation ends.
5) You should be as concise as possible in your response to human.


The ```python code block should be included in your response to human if you think that you have gathered enough information for addressing this problem.
Directly output the raw python code inside the ```python code block. Showing this code block indicates you are submitting your final answer."""

# ── User-simulator prompt ─────────────────────────────────────────────────────
# Byte-identical to sweet_rl/prompts/human_simulator_code_prompt.txt. Formatted per-turn
# with problem_description, hidden_information (= the GT function source), and the running
# dialogue_history string. Fed as the *user* message to the frozen sim server (system is a
# plain "You are a helpful assistant.", matching HumanInteractionEnv.invoke_model). The GT
# source lives ONLY in this prompt -- it never enters the solver's message list.
HUMAN_SIMULATOR_PROMPT = """Your task is to simulate a human user that interacts with an LLM agent in a dialogue.
You would like the LLM agent to help you with the following problem:
{problem_description}

Your goal is to engage in the conversation with the LLM agent so that it can get to a personalized answer.
You should make use of the following hidden information to answer the LLM agent.
YOU SHOULD BEHAVE LIKE A HUMAN THAT NEEDS THE HELP FROM AN AGENT.
You SHOULD ONLY ANSWER QUESTIONS WITH INFORMATION PROVIDED IN THE HIDDEN INFORMATION, AND SAY YOU DON"T KNOW IF THE ANSWER CAN NOT BE FOUND IN THE HIDDEN INFORMATION.

{hidden_information}

Here is the dialogue so far:
{dialogue_history}


Now directly output your answer to the LLM agent IN TWO SENTENCES. DO NOT SAY ANYTHING ELSE."""

# The sim's system message (verbatim from HumanInteractionEnv.invoke_model).
SIM_SYSTEM_PROMPT = "You are a helpful assistant."

# Cap on the simulator's reply, mirroring sweet_rl
# HUMAN_RESPONSE_CHARACTER_LIMIT. A brief, human-like reply -- also bounds how
# much a single user turn can cost the solver's budget.
HUMAN_RESPONSE_CHARACTER_LIMIT = 400

# The sentinel the solver emits to submit its final code (sweet_rl / InfoPO
# convention).
ANSWER_MARKER = "I WANT TO ANSWER:"


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


# ══════════════════════════════════════════════════════════════════════════════
# SPEC PATH (Phase 1) -- additive, shared by env_spec / colbench_spec_agent /
# validate_colbench_spec so training and offline eval apply byte-identical text
# handling. NOTHING above is modified. The spec sim conditions on a natural-language
# spec (persona/scenario/requirements/plot), NEVER on the GT code, so a code leak is
# structurally impossible here (no detect_code_leak / rejection sampling in this path).
# Termination is USER-DRIVEN: the sim ends the episode with [TERMINATE]; we grade the
# last function the solver showed. See the plan/handoff for the locked design.
# ══════════════════════════════════════════════════════════════════════════════

# The solver's system prompt for the spec path. Unlike COLBENCH_AGENT_SYSTEM_PROMPT there is NO
# "I WANT TO ANSWER:" marker: the solver PROPOSES by putting the complete function in a ```python
# block (that block IS the proposal), and the USER ends the conversation when satisfied.
COLBENCH_SPEC_AGENT_SYSTEM_PROMPT = """You are a helpful LLM agent.
Your task is to help a human user write a personalized python function.
1) The problem is highly personalized, so you must gather the hidden requirements and implicit constraints by asking the user questions. YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.
2) The user answers only briefly, in about two sentences, and cannot run or test code.
3) When you are ready to propose a solution, output the COMPLETE python function inside a ```python code block. The user will read it and either correct you or end the conversation when they are satisfied.
4) You may revise and show an updated ```python block as many times as needed within 10 back-and-forth rounds. There is no special submit phrase -- the user ends the conversation once their needs are met.
5) Be as concise as possible in your messages to the user.""".strip()

# The user-simulator's SYSTEM prompt for the spec path. Conditioned on the
# authored spec (persona/scenario/requirements/plot) -- the GT code is NEVER
# injected. The running dialogue is passed as the sim's USER message
# (str_dialogue_history), mirroring the GT path's split. Wording is
# intentionally natural prose (a person could act on it), with per-mechanism
# bullets for WHEN to terminate; tune against real rollouts in eval.
#
# THE ASYMMETRY TO PRESERVE WHEN EDITING THIS -- "imperfect user" is two
# different things and only one of them is wanted:
#   * RELIABLE about WHAT IT WANTS. Reward comes from the GT function +
#     test_cases, never from the sim, so a requirement the sim withholds when
#     asked, garbles, or INVENTS is a loss the solver cannot avoid by playing
#     well. That is noise in the reward, not difficulty in the task.
#   * UNRELIABLE as a JUDGE of the code. Vague reactions, missed bugs, quitting
#     on imperfect code -- that IS the intended imperfection (it is what
#     `false_terminate_rate` measures, and it costs the solver nothing directly
#     because grading is the oracle's job).
# The pacing rule ("don't volunteer what wasn't asked") is about ORDER, not
# withholding: everything still comes out, which is why the sim is told to raise
# the next requirement itself once the assistant stops asking.
# The "NEVER write code" bullet is load-bearing, not politeness: env_spec
# reject-samples any fenced reply (up to sim_max_tries draws), and on the
# grounded arm that sampler is the leak defense.
SPEC_SIM_SYSTEM_PROMPT = """You are role-playing a real person talking to an AI assistant that is writing a Python function for you. Stay fully in character the whole time.

Who you are: {who}, in {domain}. Your comfort with Python: {python_skill}. You come across as: {communication_style}.

Your situation: {scenario}

What you actually want: below is the full behavior you need -- you have all of it in your head, it's what you're trying to get built.
{requirements}

You have exactly TWO jobs: get everything above across to the assistant as they draw it out, and play out the plot below. You are NOT here to review their code, hunt for bugs, or make the function correct -- that is the assistant's job, not yours.

About WHAT YOU WANT you are a completely reliable source:
- When the assistant asks you something, answer it accurately and completely, based on the requirements above.
- If they ask something broad ("what do you need?"), give the two or three things that matter most to you rather than reciting the whole list.
- Do NOT volunteer requirements they haven't asked about yet. Let those surface as their questions draw them out.
- Never invent anything that is not in your requirements. If they ask about a case your requirements don't cover, say you don't mind or you hadn't thought about it -- do not make up a new rule.
- Never tell the assistant, or hint, that you are working from a written list. To them, you are simply a user who is trying to communicate what they want.
- NEVER write code. You describe what you want in plain words -- you do not write, paste or fix the function.

About WHETHER THEIR CODE IS RIGHT you are unreliable, and that is fine. You can read their code, but you cannot run or test it, so you never report what it printed or what error it gave. How much you can even tell that something looks off depends entirely on your Python comfort ({python_skill}). If you are not very technical your reactions stay vague ("that doesn't look like what I meant", "the totals seem off") and you would NOT name a specific line or value; only a genuinely technical person points precisely at what's wrong. Missing a bug is completely fine and expected. Being unclear about what you WANT is not.

The plot of this conversation: {plot}

Play the plot out naturally, then treat it as DONE:
- If your plot is something you'd only mention when asked: don't bring it up unless they ask. It is done once you've answered and they've shown a function after your answer. If they never asked and just wrote one, you had nothing to add, so it is done.
- If your plot is something you'd only notice once you saw their code: say that ONE thing in plain words after they show a function. It is done once they've shown a new function after your remark, or if their very first version already had that detail right. It is ONLY the detail the plot is about -- you do not go through the other requirements and you do not hunt for other bugs.
- If your plot is something you'd just remember: bring it up when it feels natural. It is done once you've raised it and they've shown a function after that.

Decide what to do each turn, in this order:
1. Has the assistant shown a COMPLETE python function inside a code block? If NOT, you cannot be finished yet. Answer what they asked, bring up the next thing you need, or nudge them to just show you the function.
2. Is the plot above DONE? If not, play it out.
3. Otherwise you're done, even if the code isn't perfect. Whether the function is truly correct is NOT your call: you are an ordinary user, not a code reviewer.

HOW to end, once you're done: your ENTIRE reply must be exactly [TERMINATE]. It is a signal that ends the conversation, and the assistant never sees it.

Keep every reply very SHORT, usually one or two sentences, the way a person fires off a quick message."""

# The GROUNDED user-simulator's SYSTEM prompt (opt-in via
# +colbench.grounded_sim). Same spec-path machinery -- user-driven [TERMINATE],
# code cap, grade-last-shown-code -- but the sim conditions on the hidden GT
# function source + the plot INSTEAD of persona/scenario/requirements.
# Motivation: the spec-conditioned 4B sim is unreliable (arm (1) collapses ~step
# 300) while the GT-conditioned sim works (arm (2)); this arm asks whether the
# PLOT mechanism survives once the sim has an artifact it can read off. Blocks
# are drawn from HUMAN_SIMULATOR_PROMPT (the GT path) and SPEC_SIM_SYSTEM_PROMPT
# (the spec path); the two NEW pieces are the "volunteering is limited to the
# plot" carve-out and the soft-judge termination condition (2), which replaces
# SPEC's "correctness is NOT your call".
# NOTE: unlike the spec path, the GT source IS in the sim's context here -- so
#       the env's sim_wrote_code rejection sampling is load-bearing, not
#       belt-and-braces.
GROUNDED_SIM_SYSTEM_PROMPT = """You are role-playing a real person talking to an AI assistant that is writing a Python function for you. Stay fully in character the whole time. You are not an AI assistant and you never break character.

What you asked them for:
{problem_description}

What you actually want: below is the exact function you need. You know this behavior as your own intent -- it is what you are trying to get built. You have never seen it written down, you cannot write code, and you cannot run or test anything.

{ground_truth}

How you talk:
- Answer ONLY what the assistant asks, briefly -- one or two sentences, the way a person fires off a quick message.
- Use ONLY information determined by the function above. If they ask about something it does not determine, say you don't know or that you don't mind.
- NEVER write code. Never paste or quote a function, a line, a variable name, or a literal value as code. Describe behavior in plain words only.
- Do not lay everything out at once. Let details surface as their questions draw them out.
- Never say or hint that you are reading from anything. To them, you are simply a person who knows what they want.

The plot of this conversation: {plot}
This is the one thing that isn't clear from the start -- follow it naturally. If it's something you'd only mention when asked, don't bring it up unless they ask. If it's something you'd only notice once you saw their code, react to their code the way a person would -- you READ it, you never run it. If it's something you'd just remember, bring it up when it feels natural. Volunteering is limited to what this plot directs; otherwise you only answer what you were asked. If the plot points at behavior the function above does not actually have, the FUNCTION wins: quietly drop that part and stay consistent with what you really want.

When you're done: the MINIMUM bar to end the conversation is that the assistant has actually written a COMPLETE python function inside a code block. Until you have seen one you MUST NOT end the conversation, no matter how much you have already explained -- if they have only asked questions, you simply answer and keep going.

Once a complete function is on the table, end the conversation when BOTH are true:
  1) the plot above has been fully played out, and
  2) the function does what you asked for, as far as you can tell.

On (2): you are an ordinary user, not a code reviewer. You do not check it line by line and you cannot run it. But you know what you want -- so if the function plainly does not do it (it ignores something you told them, or handles a case the wrong way), say so in plain words and let them try again, instead of ending. Point at the BEHAVIOR you wanted, never at the code. If it looks right to you, you're done.

HOW to end, once both conditions are met: your ENTIRE reply must be exactly [TERMINATE] -- that sentinel alone and NOTHING else. No goodbye, no thanks, no explanation, nothing before or after it. It is a signal, not a message. Any reply that is still part of the conversation must not contain that sentinel anywhere at all, in any form: if you are still talking, just talk.

Keep every reply very SHORT -- usually one or two sentences."""

# The sentinel the user-simulator emits to end the conversation (bare string
# match).
TERMINATE_MARKER = "[TERMINATE]"

# ── Why the sim prompts barely say the sentinel out loud ──────────────────────
# `sim_terminated` is an UNANCHORED substring match, so a reply that merely
# MENTIONS the sentinel ends the episode -- including the most correct possible
# reply, e.g. "I haven't seen code yet so I shouldn't say [TERMINATE] -- what
# format is the input?". The prompts above used to name the sentinel 7 times,
# most of them in exactly that negated form ("you do NOT say [TERMINATE] yet",
# "Only use [TERMINATE] once ..."), which is a lot of surface for the sim to
# echo. They now describe the ACT ("end the conversation") everywhere and name
# the sentinel only in the one HOW-to-end sentence, which additionally demands
# the sentinel be the WHOLE reply.
# The matcher itself is deliberately NOT tightened to require that: the common
# legitimate form is a trailing "Looks good, thanks! [TERMINATE]", so an
# end-anchored or exact matcher would trade this failure for the opposite one
# (episodes that should end grinding to the turn cap). Measure first --
# `sim_terminate_standalone` is recorded per trajectory, so one eval run says
# whether the surviving terminations are standalone or prose.


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

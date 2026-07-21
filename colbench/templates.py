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
"""Prompts + answer/code extraction for the ColBench multi-turn loop.

Ported from ``sweet_rl`` (``prompts/{llm_agent_code_prompt,human_simulator_code_prompt}.txt``,
``utils/code_utils.check_correctness`` fence-strip) and InfoPO's ``run_simulate_api.py``
(``check_and_extract_answer`` flexible marker patterns). The number-affecting transforms
(marker extraction, code fence-strip, ``<think>`` strip) live here so the training rollout
(``colbench_agent``) and the offline validator apply byte-identical text handling.
"""

import re
from typing import Optional

# ── Solver (agent) system prompt ──────────────────────────────────────────────
# Byte-identical to sweet_rl/prompts/llm_agent_code_prompt.txt EXCEPT the trailing
# "{dialogue_history}" placeholder: sweet_rl string-formats the whole dialogue into it and
# calls a completion endpoint, whereas we use a CHAT template (system + real message turns).
# So, exactly like InfoPO's run_simulate_api.py, we drop the placeholder and let the actual
# conversation turns carry the history. Kept as the raw file text below; the placeholder is
# stripped in COLBENCH_AGENT_SYSTEM_PROMPT.
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

# System prompt used by the solver agent loop + preprocess. Placeholder removed (see above).
COLBENCH_AGENT_SYSTEM_PROMPT = _AGENT_PROMPT_RAW.replace("{dialogue_history}", "").strip()

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

# Cap on the simulator's reply, mirroring sweet_rl HUMAN_RESPONSE_CHARACTER_LIMIT. A brief,
# human-like reply -- also bounds how much a single user turn can cost the solver's budget.
HUMAN_RESPONSE_CHARACTER_LIMIT = 400

# The sentinel the solver emits to submit its final code (sweet_rl / InfoPO convention).
ANSWER_MARKER = "I WANT TO ANSWER:"


# ── <think> stripping ─────────────────────────────────────────────────────────
# Qwen3 (and other reasoning models) may emit a <think>...</think> block. We strip it from
# BOTH the solver text (before searching for the answer marker) and the sim reply (before the
# char cap) so reasoning never leaks into the extracted answer or the injected user turn. For
# a plain Instruct model this is a defensive no-op.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove any ``<think>...</think>`` block and surrounding whitespace."""
    return _THINK_BLOCK.sub("", text or "").strip()


# ── Answer-marker extraction (ported from InfoPO run_simulate_api.check_and_extract_answer) ──
# Accept the several marker spellings observed in rollouts. Case-insensitive match; the answer
# is everything AFTER the marker.
_ANSWER_PATTERNS = [
    "I WANT TO ANSWER:",   # standard
    "I WANT_TO_ANSWER:",   # underscore
    "I WANT_TO ANSWER:",   # mixed
    "i want to answer:",   # lowercase
    "i want_to_answer:",   # lowercase + underscore
]


def check_and_extract_answer(response: str) -> tuple[bool, str]:
    """Return ``(has_answer, answer_text)``.

    ``has_answer`` is True iff any accepted spelling of the answer marker is present; the
    answer text is everything after the (first-matched) marker, stripped. Byte-identical to
    InfoPO's ``check_and_extract_answer`` so training and offline eval agree.
    """
    if not response:
        return False, ""
    response_lower = response.lower()
    for pattern in _ANSWER_PATTERNS:
        pattern_lower = pattern.lower()
        if pattern_lower in response_lower:
            if pattern in response:
                idx = response.find(pattern)
                return True, response[idx + len(pattern):].strip()
            idx_lower = response_lower.find(pattern_lower)
            return True, response[idx_lower + len(pattern):].strip()
    return False, ""


# ── Code fence-strip (ported from sweet_rl code_utils.check_correctness) ──────
# The ONLY piece of sweet_rl's in-process "safety" preprocessing we keep: pull the code out
# of a ```python ... ``` (or bare ``` ... ```) fence if the model wrapped it. sweet_rl's
# keyword blocklist (import os/sys, sudo, exit(, argparse, ...) is dropped -- it was a
# poor-man's substitute for a sandbox because sweet_rl exec'd in-process; our container
# sidecar supersedes it.
def extract_code_answer(answer_text: str) -> str:
    """Strip a ```python / ``` code fence from the submitted answer, if present.

    Mirrors sweet_rl's fence handling: prefer a ```python block, else the first ``` block,
    else the raw text. Returns the code string to grade (stripped).
    """
    text = answer_text or ""
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
# (C) whole-line overlap from the study is intentionally dropped (0 hits, redundant with A).
# (D) is symbol-aware and operator-gated on PURPOSE: a word-only n-gram also fires on
# legitimate natural-language behavior specs (e.g. matching platform names / version strings),
# which are exactly the good sim turns we must NOT reject. Requiring operators in the matched
# span keeps the real expression leaks and spares prose.
_DEF_SIGNATURE_RE = re.compile(r"\bdef\s+\w+\s*\(")
_PY_FENCE_RE = re.compile(r"```python", re.IGNORECASE)
# Arithmetic / comparison / bracket operators. Deliberately excludes ',' '.' ':' (common in
# prose) so the operator gate keys on expression structure, not punctuation.
_CODE_OPERATORS = frozenset("+-*/%()[]=<>")


def _code_tokens(text: str) -> list:
    """Symbol-aware tokenizer: each word OR each individual operator/punctuation char.

    Unlike a word-only (``\\w+``) split this preserves operators, so a matched n-gram can be
    required to contain them -- the knob that separates a copied CODE expression from a prose
    behavior description that merely shares identifiers with the GT.
    """
    return re.findall(r"\w+|[^\w\s]", text or "")


def detect_code_leak(
    text: str, ground_truth: str, ngram_n: int = 10, min_operators: int = 2
) -> Optional[str]:
    """Return a short reason string if ``text`` leaks code, else ``None``.

    ``ground_truth`` is the hidden GT source the simulator sees (n-gram overlap is computed
    against it, NOT the agent's own output -- that substitution was an offline-study
    convenience only). Reasons: ``"def"`` / ``"fenced"`` / ``"ngram"`` (see the module comment
    above). Checked in that order and short-circuits on the first hit.

    ``ngram_n <= 0`` DISABLES detector (D) (the default in the eval harness for now): the
    operator-gated n-gram check is a solid idea but held back as a FUTURE CONSIDERATION while
    we validate on the A/B leaks that dominate. (A)/(B) always run.
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
        gt_ngrams = {tuple(gt_toks[i:i + ngram_n]) for i in range(len(gt_toks) - ngram_n + 1)}
        for i in range(len(tx_toks) - ngram_n + 1):
            ng = tuple(tx_toks[i:i + ngram_n])
            if ng in gt_ngrams and sum(tk in _CODE_OPERATORS for tk in ng) >= min_operators:
                return "ngram"
    return None


def final_answer(assistant_text: str, episode_done: bool) -> tuple[bool, str]:
    """Resolve the solver's final answer text from one assistant turn.

    Returns ``(has_marker, answer_text)``. Mirrors InfoPO's extraction (sweet_rl step +
    ``extract_answer_from_env`` fallbacks):
      1. If the answer marker is present, use the text after it.
      2. Else, on the FINAL turn (``episode_done``), fall back to the whole response when it
         looks like code (``def``/``import``/``return``/`=` ...) or is non-trivial, so an
         episode that ran out of turns still submits the model's last attempt.
      3. Else no answer yet (keep interacting).
    ``assistant_text`` should already be ``strip_think``-ed by the caller.
    """
    has_marker, answer_text = check_and_extract_answer(assistant_text)
    if has_marker:
        return True, answer_text
    if episode_done:
        last = (assistant_text or "").strip()
        if len(last) > 20 and any(k in last for k in ("def ", "import ", "class ", "return ", "=")):
            return True, last
        if len(last) > 10:
            return True, last
    return False, ""


def str_dialogue_history(messages: list[dict]) -> str:
    """Render the running dialogue as the sim-prompt ``{dialogue_history}`` string.

    Byte-identical to sweet_rl HumanInteractionEnv.str_dialogue_history: ``"<role>:<content>"``
    per turn separated by four newlines, terminated with a trailing ``"agent:"`` cue so the
    simulator answers as the human to the agent's latest turn.
    """
    result = ""
    for d in messages:
        result += str(d.get("role")) + ":"
        result += str(d.get("content")) + "\n\n\n\n"
    return result + "agent:"


def build_sim_user_message(problem_description: str, hidden_information: str, messages: list[dict]) -> str:
    """Format the user-simulator prompt for one turn (system stays SIM_SYSTEM_PROMPT)."""
    return HUMAN_SIMULATOR_PROMPT.format(
        problem_description=problem_description,
        hidden_information=hidden_information,
        dialogue_history=str_dialogue_history(messages),
    )


# Injected as the very first solver user turn wrapping the problem statement. ColBench's
# problem_description already reads as a direct user request ("Create a python function ..."),
# so -- matching sweet_rl HumanInteractionEnv.reset, which seeds the dialogue with the raw
# problem_description as the first user turn -- we pass it through unwrapped.
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

# The user-simulator's SYSTEM prompt for the spec path. Conditioned on the authored spec
# (persona/scenario/requirements/plot) -- the GT code is NEVER injected. The running dialogue is
# passed as the sim's USER message (str_dialogue_history), mirroring the GT path's split. Wording
# is intentionally natural prose (a person could act on it), with per-mechanism bullets for WHEN
# to terminate; tune against real rollouts in eval.
SPEC_SIM_SYSTEM_PROMPT = """You are role-playing a real person talking to an AI assistant that is writing a Python function for you. Stay fully in character the whole time.

Who you are: {who}, in {domain}. Your comfort with Python: {python_skill}. You come across as: {communication_style}.

Your situation: {scenario}

What you actually want: Below is the full behavior you need -- you have all of it in your head, it's what you're trying to get built. Talk about it in your own words, never as code, the way this person naturally would. If the assistant asks you something, answer ONLY what they asked -- briefly and in character -- without volunteering additional information. Never tell the assistant, or hint, that you are working from a full list of requirements: to them, you are simply a person who knows what they want. Act like a real user, letting each requirement surface naturally as the assistant's questions draw it out, rather than laying everything out at once.
{requirements}

What your job is (and isn't): Your job is to TELL the assistant what you want and to react as this person would -- it is NOT to make the code correct, and NOT to review it line by line. You never write code yourself and you never paste a function back to them, not even to fix a mistake -- you only describe things in plain words. You are not the judge of whether the code is correct; the assistant's job is to get it right from what you tell them. How much you can even tell that something looks wrong depends entirely on your Python comfort ({python_skill}): if you are not very technical, your reactions stay vague ("that doesn't look like what I meant", "the totals seem off") and you would NOT spot or name a specific line, value, or edge case; only a genuinely technical person would point precisely at what's wrong. It's completely fine to be an imperfect, ordinary user who misses bugs.

The plot of this conversation: {plot}
This plot is the one thing that isn't clear from the start -- follow it naturally. If it's something you'd only mention when asked, don't bring it up unless the assistant asks. If it's something you'd only notice once you saw their code, react to their code the way a person would -- you READ it, you never run or test it. If it's something you'd just remember, bring it up when it feels natural.

When you're done: Your ultimate goal is to walk away with the function you need, so the MINIMUM bar to end the conversation is that the assistant has actually written a COMPLETE python function inside a code block. Until you have seen such a code block, you MUST NOT end the conversation -- no matter how much you have already explained. If the assistant has only asked you questions and not yet shown any code, you simply answer and keep going; you do NOT say [TERMINATE] yet.

Once code is on the table, whether you're finished ALSO depends on how the plot of the conversation was set up -- play out the plot as described above first:
- If your plot was to clarify something only when asked -- you're done once you've answered it and they've shown a new function after your answer. After that, if you are satisfied with the function, you can end with [TERMINATE]. If the assistant just went ahead and wrote a function without ever asking, that's fine too -- you had nothing to add, so once a function is on the table you can end with [TERMINATE].
- If your plot was to correct the code when it got a detail wrong -- this is ONLY about the one specific detail your plot is about, not every possible bug and not all the other requirements. You point out that one thing in plain words (never by writing code, and only as precisely as your Python comfort allows), and you're done once you've said it and they've shown a new function after it -- or their very first version already had that detail right (nothing to correct, so you're done). You do NOT keep hunting for other problems. After that, once a complete function is on the table, you can end with [TERMINATE].
- If your plot was to remember something and bring it up yourself -- you're done once you've played out the plot of bringing up the forgotten part and they've shown a function after that. After that, once a complete function is on the table, you can end with [TERMINATE].

So only reply with [TERMINATE] when BOTH are true: (1) your plot above has been fully played out, and (2) a complete python function is on the table. Whether that function is actually correct is NOT your call and NOT a condition -- you are an ordinary user, not its judge. If either condition is missing, keep talking instead.

You're an ordinary user, not a code reviewer: you don't check every line, you can't test anything, and you don't keep hunting for new problems. Once both conditions above are satisfied you're done -- even if the code isn't perfect. You might casually mention something else that looks off, but you don't have to.

Keep every reply very SHORT -- usually one or two sentences, the way a person fires off a quick message. Do not explain everything at once or recite all your requirements in one go. Only use [TERMINATE] once both conditions above are met."""

# The sentinel the user-simulator emits to end the conversation (bare string match).
TERMINATE_MARKER = "[TERMINATE]"


def build_spec_sim_messages(spec: dict, messages: list[dict]) -> tuple[str, str]:
    """Build the spec sim's (system, user) messages for one turn.

    ``spec`` carries ``persona{who,domain,python_skill,communication_style}, scenario,
    requirements, plot``. The system message conditions the sim on that spec (NEVER the GT
    code); the user message is the running dialogue (``str_dialogue_history``) -- the same
    seam split as the GT path's ``build_sim_user_message``.
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


def sim_terminated(reply: str) -> bool:
    """True iff the (``<think>``-stripped) sim reply contains the ``[TERMINATE]`` sentinel.

    Case-insensitive so a stray lowercasing by the sim still ends the episode.
    """
    return TERMINATE_MARKER.lower() in strip_think(reply).lower()


def contains_code(text: str) -> bool:
    """True iff ``text`` looks like it proposes a function (a ```python fence or a ``def`` sig).

    Reuses the same signals as the leak detector's (A)/(B); here they mark the solver's OWN
    proposed code (the grading target), not a sim leak.
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
    """
    return bool(_ANY_FENCE_RE.search(strip_think(reply)))


def extract_last_code(messages: list[dict]) -> str:
    """Return the code from the most recent assistant turn that proposed a function, else ``""``.

    Scans ``messages`` newest-first for an assistant turn with ``contains_code`` and returns
    ``extract_code_answer`` of it -- the "last function the solver showed", which the spec path
    grades on termination.
    """
    for m in reversed(messages):
        if m.get("role") == "assistant" and contains_code(m.get("content", "")):
            return extract_code_answer(strip_think(m.get("content", "")))
    return ""

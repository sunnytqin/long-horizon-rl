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

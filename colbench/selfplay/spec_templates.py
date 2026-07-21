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
"""Prompts + parsing for the self-play spec setting (Phase 0).

Two prompts live here, kept separate from ``colbench.templates`` (the GT-code path) so the two
settings never entangle:

  * ``SPEC_AUTHOR_*`` -- asks a model to author a realistic **spec** from the public problem +
    the hidden GT code. The spec is split into ``persona`` / ``scenario`` / ``requirements``.
    The persona is the DIVERSITY lever: who would realistically ask for THIS function, and at
    what Python skill level -- which flavors *how* the requirements are later voiced by the
    simulator, WITHOUT changing their completeness (the person knows their own numbers even if
    they cannot write code). The requirements must capture every constant/branch/edge-case the
    GT implements, or the task becomes unsolvable regardless of solver skill.

  * ``FULL_SPEC_SOLVER_*`` -- the diagnostic solver prompt: hand the solver the ENTIRE spec in
    one turn and ask for the function. Grading (``colbench.reward``) is unchanged.

Parsing (``parse_spec``) is tolerant of models that wrap the JSON in prose or fences.
"""

import json
import re
from typing import Any


# ── Spec-author prompt ────────────────────────────────────────────────────────
SPEC_AUTHOR_SYSTEM = (
    "You are building a dataset of realistic Python programming-help requests. For each task "
    "you invent the believable human who would ask for it and write, in that person's own "
    "voice, the complete description of what they want. You are meticulous: you never drop a "
    "detail of the intended behavior, and you never write code."
)

# {problem_description} = the public, under-specified ask (carries the required signature).
# {ground_truth} = the hidden reference implementation (the source of the exact behavior).
SPEC_AUTHOR_USER = """A user needs a Python function. Here is their initial, rough request (this is all the \
agent helping them will see up front):

--- PUBLIC REQUEST ---
{problem_description}
--- END PUBLIC REQUEST ---

Here is the REFERENCE IMPLEMENTATION that captures exactly the behavior the user actually \
wants (the user has this behavior in mind; the agent will NOT see it):

--- REFERENCE IMPLEMENTATION ---
{ground_truth}
--- END REFERENCE IMPLEMENTATION ---

Your job: write a realistic, self-contained spec for this request, as a JSON object with \
three fields.

1) "persona": WHO would realistically ask for this exact function? Choose the single most \
believable person for THIS task, not a generic developer. Include their role/domain, WHY they \
need it, and their Python skill level on this scale:
   - "non-coder": a domain expert who cannot really program and describes things in plain, \
sometimes imprecise words;
   - "hobbyist": can write small scripts, informal vocabulary;
   - "analyst": scripts regularly (e.g. data/finance/science), semi-technical;
   - "engineer": a fluent professional who talks precisely.
Pick whatever is most natural for the task -- the mix across tasks should feel like real, \
diverse users. Give the persona as an object with keys "who", "domain", "python_skill" (one \
of the four labels), and "communication_style".

2) "scenario": 2-4 sentences of story -- the concrete situation that makes this person need \
the function right now.

3) "requirements": the COMPLETE behavior the function must have, written as this persona would \
describe their own needs, in their voice and vocabulary. This is the most important field and \
MUST be exhaustive:
   - Include EVERY specific rule, number, threshold, branch, default, rounding, ordering, and \
edge case that the reference implementation exhibits. Keep the exact numbers and constants.
   - Describe behavior in WORDS, never code. A non-coder would say "subtract the 126 people \
who were laid off" rather than "x - 126"; an engineer may be more precise -- but either way \
every number and rule from the reference must be present and unambiguous.
   - Do NOT paste or transcribe the reference code, and do NOT include a function body. \
Describe intent, not implementation.
The persona changes the TONE and vocabulary of this text; it must never change how COMPLETE it \
is. Someone reading only "requirements" (plus the public request's signature) should be able \
to reproduce the reference behavior exactly.

Output ONLY the JSON object, nothing else:
{{"persona": {{"who": "...", "domain": "...", "python_skill": "...", "communication_style": "..."}}, \
"scenario": "...", "requirements": "..."}}"""


# ── Full-spec diagnostic solver prompt ────────────────────────────────────────
FULL_SPEC_SOLVER_SYSTEM = (
    "You are an expert Python programmer. The user gives you a complete description of the "
    "function they need. Write a single, self-contained Python function that satisfies it, "
    "matching the requested name and signature exactly. Output ONLY the code in a single "
    "```python code block, with no explanation."
)

FULL_SPEC_SOLVER_USER = """{problem_description}

Here is exactly what I need.

Who I am: {persona}
My situation: {scenario}

What the function must do:
{requirements}"""


# ── Plot spec-author prompt (variant) ─────────────────────────────────────────
# The author writes the COMPLETE requirements (the substance the simulator must eventually get
# across) PLUS a short, tailored "plot": a direction for how a conversation about THIS specific
# function could naturally not be clear from the start, so the assistant has to draw the details
# out. Crucially the plot is a DIRECTION, not a turn-by-turn script -- the Phase-1 simulator
# improvises the actual turns from (requirements + plot + persona), which is what gives the group
# its rollout diversity. We do NOT tell the simulator what to say at each turn.
PLOT_AUTHOR_SYSTEM = (
    "You are building a dataset of realistic Python programming-help requests for a multi-turn "
    "setting. For each task you invent the believable person who needs the function, write down "
    "the complete behavior they want, and then -- looking at that specific behavior -- imagine "
    "ONE natural way a conversation about it might not be perfectly clear from the start, so the "
    "assistant has to draw the details out. You describe that only as a short PLOT (a direction "
    "for how the user reveals things), never as scripted dialogue, and you never write code."
)

PLOT_AUTHOR_USER = """A user needs a Python function. Here is their initial, rough request (this is all the \
assistant helping them will see up front):

--- PUBLIC REQUEST ---
{problem_description}
--- END PUBLIC REQUEST ---

Here is the REFERENCE IMPLEMENTATION that captures exactly the behavior the user actually \
wants (the user has this behavior in mind; the assistant will NOT see it):

--- REFERENCE IMPLEMENTATION ---
{ground_truth}
--- END REFERENCE IMPLEMENTATION ---

Your job: write a realistic, self-contained spec for this request as a JSON object with four \
fields.

1) "persona": WHO would realistically ask for this exact function? Choose the single most \
believable person for THIS task, not a generic developer. Give it as an object with keys \
"who", "domain", "python_skill" (one of "non-coder", "hobbyist", "analyst", "engineer"), and \
"communication_style". Pick whatever is most natural -- the mix across tasks should feel like \
real, diverse users.

2) "scenario": 2-4 sentences -- the concrete situation that makes this person need the function \
right now.

3) "requirements": the COMPLETE behavior the function must have, described as THE USER'S NEEDS \
in the THIRD PERSON ("The user needs the function to...", "The user wants it to..."), in this \
persona's vocabulary -- consistent with the "plot" field below, which is also written about "the \
user". This is the MOST IMPORTANT field and MUST be exhaustive and unambiguous:
   - Include EVERY specific rule, number, threshold, branch, default, rounding, ordering, return \
shape/type, and edge case the reference exhibits. Keep the EXACT constants. State each \
comparison precisely (strictly-greater vs at-least, before vs on-or-after a cutoff), and state \
each calculation/formula so there is only ONE way to read it (e.g. "principal times (the \
federal funds rate minus the interest rate)", not a looser paraphrase).
   - Describe behavior in WORDS, never code. A non-coder would say "subtract the 126 people who \
were laid off" rather than "x - 126"; an engineer may be more precise -- but either way every \
number and rule from the reference must be present and unambiguous.
   - If something is an INPUT PARAMETER (it appears in the signature), describe how that input \
is USED; do NOT replace it with a fixed example value or assume a specific number for it.
   - Do NOT paste or transcribe the reference code, and do NOT include a function body.
The persona changes only the TONE and vocabulary; it must never reduce how COMPLETE or exact \
this is. Someone reading only "requirements" plus the public request's signature must be able to \
reproduce the reference behavior EXACTLY.

4) "plot": This field is an INSTRUCTION to the user-simulator -- the model that will role-play \
this user in the conversation. Write it in that tone, telling the simulator how to behave: \
"the user would first...", "the user would...", "if the assistant asks..., the user would...". \
Now LOOK at the requirements you just wrote and invent ONE natural, interesting way THIS \
particular conversation could unfold so the assistant has to work a little to get the full \
intent -- a DIRECTION, not a script.

First choose WHICH single detail is not clear up front, and in WHAT WAY (pick what best fits \
this function): the user forgets an edge case; states one specific value or piece of the logic \
WRONG; is VAGUE about a threshold/default; or knows it but would not think to say it unprompted.

Then describe HOW it comes out -- but ONLY in terms of the USER's behavior, made CONDITIONAL on \
what the assistant does. The assistant is a separate agent we do NOT control, so you must NEVER \
assume it asks or says anything. Do not write "the assistant notices the gap and asks..."; write \
"IF the assistant asks about X, the user...; if it doesn't, the user...". Draw on these -- mix \
them, or add your own, as fits this function and persona:
   - the user forgot or is unsure of the detail and reveals it ONLY IF the assistant asks about \
it: if the assistant asks, the user answers (a non-technical user answers in their own plain, \
non-technical words); if the assistant never asks, the user never brings it up and it stays \
hidden;
   - the user mis-states, forgets, or misspeaks one requirement up front and would correct it \
ONLY IF the assistant produces code (or an explanation) reflecting that wrong understanding -- \
reading it (never running it), the user reacts "no, that's not quite what I meant..."; if the \
assistant's code already happens to be right, there is nothing to correct;
   - the user simply remembers the detail on their own and volunteers it in a later message, \
unprompted (this one does not depend on the assistant).
IMPORTANT: the user NEVER runs, tests, or executes the code -- they cannot. The gap surfaces \
ONLY through the conversation (answering the assistant's question, reacting to code the \
assistant shows, or just remembering), never through the user testing it and seeing a failure.

Write 1-3 sentences AS AN INSTRUCTION TO THE USER-SIMULATOR ("the user would...", "the user \
first...") describing the user's CONDITIONAL behavior at a high level: which specific \
requirement is initially missing / wrong / vague, and under what condition (if the assistant \
asks / if the assistant shows code / on the user's own) it gets clarified. Phrase the \
assistant's part hypothetically ("if the assistant..."). Do NOT write the user's exact words or \
lay out turn-by-turn dialogue. Do not force drama -- a single small, realistic wrinkle is ideal. \
The user's true intent is ALWAYS the full requirements; the plot only shapes the ORDER and \
MANNER in which they come out, and must stay consistent with the requirements.

Output ONLY the JSON object, nothing else:
{{"persona": {{"who": "...", "domain": "...", "python_skill": "...", "communication_style": "..."}}, \
"scenario": "...", "requirements": "...", "plot": "..."}}"""


# ── Parsing ───────────────────────────────────────────────────────────────────
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _persona_to_text(persona: Any) -> str:
    """Render the persona field (object or string) to a one-line human description."""
    if isinstance(persona, dict):
        who = str(persona.get("who", "")).strip()
        domain = str(persona.get("domain", "")).strip()
        skill = str(persona.get("python_skill", "")).strip()
        style = str(persona.get("communication_style", "")).strip()
        bits = [b for b in [who, f"domain: {domain}" if domain else "",
                            f"Python skill: {skill}" if skill else "",
                            f"style: {style}" if style else ""] if b]
        return "; ".join(bits)
    return str(persona or "").strip()


def parse_spec(raw: str) -> dict:
    """Parse an author model's reply into ``{persona, scenario, requirements, raw, ok}``.

    Tolerant: strips prose/fences around the JSON object. If no JSON parses, we fall back to
    treating the whole reply as free-text requirements (``ok=False``) so a malformed generation
    is still usable/inspectable rather than lost. ``persona`` is kept as-authored (dict or str);
    use ``_persona_to_text`` when composing prompt text.
    """
    text = (raw or "").strip()
    obj = None
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001 - tolerate malformed JSON, fall through to raw
            obj = None
    if isinstance(obj, dict):
        return {
            "persona": obj.get("persona", ""),
            "scenario": str(obj.get("scenario", "") or "").strip(),
            "requirements": str(obj.get("requirements", "") or "").strip(),
            "raw": raw,
            "ok": bool(str(obj.get("requirements", "")).strip()),
        }
    return {"persona": "", "scenario": "", "requirements": text, "raw": raw, "ok": False}


def build_author_messages(problem_description: str, ground_truth: str) -> list[dict]:
    """Chat messages for the spec-author call."""
    return [
        {"role": "system", "content": SPEC_AUTHOR_SYSTEM},
        {"role": "user", "content": SPEC_AUTHOR_USER.format(
            problem_description=problem_description, ground_truth=ground_truth)},
    ]


def build_full_spec_solver_messages(problem_description: str, spec: dict) -> list[dict]:
    """Chat messages for the full-spec diagnostic solver call.

    Includes the public request (for the exact signature/name) plus the full authored spec.
    ``spec`` is a parsed dict from ``parse_spec``.
    """
    return [
        {"role": "system", "content": FULL_SPEC_SOLVER_SYSTEM},
        {"role": "user", "content": FULL_SPEC_SOLVER_USER.format(
            problem_description=problem_description,
            persona=_persona_to_text(spec.get("persona", "")),
            scenario=spec.get("scenario", "") or "(not specified)",
            requirements=spec.get("requirements", ""),
        )},
    ]


def parse_plot_spec(raw: str) -> dict:
    """Parse a plot author reply into ``{persona, scenario, requirements, plot, raw, ok}``.

    ``requirements`` is the full intent the simulator must eventually convey; ``plot`` is the
    tailored, high-level direction for how this conversation naturally unfolds (NOT a script).
    Tolerant like ``parse_spec``: ``ok`` requires BOTH ``requirements`` and ``plot``. On
    malformed JSON the whole reply becomes free-text requirements with ``ok=False``.
    """
    text = (raw or "").strip()
    obj = None
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001 - tolerate malformed JSON, fall through to raw
            obj = None
    if isinstance(obj, dict):
        requirements = str(obj.get("requirements", "") or "").strip()
        plot = str(obj.get("plot", "") or "").strip()
        return {
            "persona": obj.get("persona", ""),
            "scenario": str(obj.get("scenario", "") or "").strip(),
            "requirements": requirements,
            "plot": plot,
            "raw": raw,
            "ok": bool(requirements and plot),
        }
    return {"persona": "", "scenario": "", "requirements": text, "plot": "", "raw": raw, "ok": False}


def build_plot_author_messages(problem_description: str, ground_truth: str) -> list[dict]:
    """Chat messages for the plot spec-author call."""
    return [
        {"role": "system", "content": PLOT_AUTHOR_SYSTEM},
        {"role": "user", "content": PLOT_AUTHOR_USER.format(
            problem_description=problem_description, ground_truth=ground_truth)},
    ]

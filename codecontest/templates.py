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
"""Prompts and oracle-feedback formatting for the CodeContests multi-turn loop.

Ported/adapted from the tunix harness (``template.py`` / ``code_util.py``). The
solver writes a complete stdin->stdout Python program in a ```python block; the
oracle environment runs it against ground-truth tests and, on failure, feeds back
a few failing cases formatted by ``format_oracle_feedback``.
"""

from typing import Optional

SOLVER_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. You will be given a problem "
    "statement. Write a correct, efficient Python 3 solution that reads from "
    "standard input and writes to standard output. Think step by step, then put "
    "your final solution in a single ```python ... ``` code block. The program "
    "must be self-contained and runnable as-is."
)

CODE_PROMPT_TEMPLATE = """Solve the following competitive programming problem in Python 3.

Your program must read input from standard input (stdin) and write the answer to
standard output (stdout). End your response with the complete solution in a single
```python ... ``` code block.

Problem:
{problem}
"""

# Wraps the failing-case block shown to the solver after an incorrect submission.
# Reflection-style: ask the solver to diagnose the previous attempt before rewriting,
# which tends to produce better refinements than a bare "fix it" instruction.
SOLVER_ORACLE_REFLECTION_FEEDBACK_TEMPLATE = (
    "The following test cases were run against your code and produced incorrect"
    " output. These test cases are guaranteed to be"
    " correct.\n\n{feedback_block}Before writing an improved solution, please analyze "
    "the code and explain:\n"
    "1. What approach or algorithm the previous solution used\n"
    "2. Why it might produce incorrect results\n"
    "3. What conceptual changes are needed to fix it\n\n"
    "Be concise. After your analysis, write the improved solution. "
    "Please make sure you write a solution based on the reflection you made and put the new complete code in a python block.\n"
)

# Shown when the previous submission passed all ground-truth tests.
SOLVER_CORRECT_MESSAGE = (
    "Your solution passed all of the tests. You are done; restate your final "
    "solution in a single ```python ... ``` code block."
)


def _indent(s: str) -> str:
    return "\n".join("    " + line for line in s.rstrip("\n").splitlines())


def _waterfill_cap(lengths: list[int], budget: int) -> Optional[int]:
    """Largest per-field cap ``c`` with ``sum(min(len, c)) <= budget``.

    This is the "truncate the longest, leave the rest intact" policy: fields shorter
    than ``c`` are untouched; only fields above ``c`` are clipped, all to the same ``c``.
    When a single field dominates (the usual case: one giant test input), ``c`` lands
    just below it and only that one field is clipped. Returns ``None`` when everything
    already fits (no clipping needed).
    """
    if budget <= 0 or not lengths:
        return None
    remaining = budget
    for i, length in enumerate(sorted(lengths)):
        n_rest = len(lengths) - i  # fields not yet fixed below the cap (incl. this one)
        if length * n_rest <= remaining:
            remaining -= length  # this field fits in full; it sits below the cap
        else:
            return max(1, remaining // n_rest)  # cap the remaining (largest) fields here
    return None


def _clip(s: str, cap: Optional[int]) -> str:
    """Clip ``s`` to ``cap`` chars keeping head+tail, with an honest elision marker.

    The marker makes the truncation explicit so the model does not treat a partial
    field as the whole spec (e.g. a clipped input no longer matches its full expected
    output). Applied uniformly to inputs, the model's output, and expected output.
    """
    s = str(s)
    if cap is None or len(s) <= cap:
        return s
    head, tail = (cap * 2) // 3, cap // 3
    return f"{s[:head]}\n... [clipped {len(s) - cap} of {len(s)} chars] ...\n{s[-tail:]}"


def format_oracle_feedback(failures, max_total_chars: Optional[int] = None) -> str:
    """Format failing cases as feedback text (ported from code_util.format_oracle_feedback).

    Args:
        failures: list of (input, actual_output, expected_output) tuples.
        max_total_chars: optional combined budget (chars) across ALL fields of ALL
            shown cases. When the raw fields exceed it, a single water-filling cap is
            computed over every field and applied uniformly, so the truncation lands on
            whichever fields are actually large (input, output, or expected) and small
            fields are left intact. ``None`` disables clipping.

    Returns:
        A formatted, human-readable feedback block.
    """
    # Strip first so the budget reflects exactly what we emit; cap is then computed
    # jointly over all three fields of all cases (the "combined" budget).
    cases = [(str(inp), str(actual).strip(), str(expected).strip()) for inp, actual, expected in failures]
    cap = _waterfill_cap([len(f) for case in cases for f in case], max_total_chars) if max_total_chars else None

    lines = []
    for i, (inp, actual, expected) in enumerate(cases, 1):
        lines.append(f"Test {i}:")
        lines.append(f"  Input:\n{_indent(_clip(inp, cap))}")
        lines.append(f"  Your output:     {_clip(actual, cap)}")
        lines.append(f"  Expected output: {_clip(expected, cap)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_feedback_message(failures, max_total_chars: Optional[int] = None) -> str:
    """Build the full user-turn feedback string for a set of failing cases.

    ``max_total_chars`` bounds the combined size of the failing-case fields (see
    ``format_oracle_feedback``) so the injected user turn stays well under
    ``rollout.prompt_length`` and is never blindly tail-truncated downstream.
    """
    block = format_oracle_feedback(failures, max_total_chars=max_total_chars)
    return SOLVER_ORACLE_REFLECTION_FEEDBACK_TEMPLATE.format(feedback_block=block)

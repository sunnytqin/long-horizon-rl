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
SOLVER_ORACLE_FEEDBACK_TEMPLATE = """Your solution did not pass all of the tests. Here are some failing cases (the expected outputs are correct):

{feedback_block}
Revise your solution to fix these failures. Provide the corrected, complete program in a single ```python ... ``` code block."""

# Shown when the previous submission passed all ground-truth tests.
SOLVER_CORRECT_MESSAGE = (
    "Your solution passed all of the tests. You are done; restate your final "
    "solution in a single ```python ... ``` code block."
)


def _indent(s: str) -> str:
    return "\n".join("    " + line for line in s.rstrip("\n").splitlines())


def format_oracle_feedback(failures) -> str:
    """Format failing cases as feedback text (ported from code_util.format_oracle_feedback).

    Args:
        failures: list of (input, actual_output, expected_output) tuples.

    Returns:
        A formatted, human-readable feedback block.
    """
    lines = []
    for i, (inp, actual, expected) in enumerate(failures, 1):
        lines.append(f"Test {i}:")
        lines.append(f"  Input:\n{_indent(inp)}")
        lines.append(f"  Your output:     {str(actual).strip()}")
        lines.append(f"  Expected output: {str(expected).strip()}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_feedback_message(failures) -> str:
    """Build the full user-turn feedback string for a set of failing cases."""
    return SOLVER_ORACLE_FEEDBACK_TEMPLATE.format(feedback_block=format_oracle_feedback(failures))

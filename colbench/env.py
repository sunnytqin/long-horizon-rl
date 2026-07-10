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
"""Environment for the ColBench multi-turn loop: a user simulator + a GT-graded reward.

``ColBenchUserSimEnv`` is the pluggable "what happens between assistant turns" component
(the analog of ``codecontest.env.GTOracleEnv``). It holds the problem, the HIDDEN
ground-truth function source, and the GT call-strings, and exposes three seams the agent
loop drives:

  * ``is_answer(assistant_text, episode_done)`` -> ``(has_answer, answer_text)`` -- did the
    solver submit (marker, or a code-like final turn)?  (templates.final_answer)
  * ``generate_user_turn(messages)`` -> a <=400-char human-like reply. THE Phase-1/Phase-2
    seam: Phase-1 backend = an HTTP call to the FROZEN sim server (same base model); Phase-2
    co-training swaps in a same-engine backend and unmasks these turns. The GT source is
    passed ONLY inside this call's prompt -- it never enters the solver's message list.
  * ``score(answer_text)`` -> fractional pass-rate against the GT (reward.grade).

Reward convention: the trajectory reward is the final submission's fractional pass-rate in
[0,1]; the frozen simulator governs the DIALOGUE only and has zero influence on grading.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from colbench import reward, templates

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ── Conversation debug (mirrors codecontest.env's CODECONTEST_DEBUG_FEEDBACK gate) ──
# Env-var gated so the user can eyeball dialogues. Logged at WARNING because plain print()
# is swallowed by the Ray rollout workers. See colbench_agent.py for the per-turn / answer /
# reward dumps; here we dump the SIM side (prompt incl. the hidden GT, + raw reply).
_DEBUG_SIM = bool(int(os.getenv("COLBENCH_DEBUG_SIM", "0") or "0"))
_DEBUG_PREVIEW = int(os.getenv("COLBENCH_DEBUG_CONVO_PREVIEW", "400") or "400")


# A sim backend maps (system_content, user_content) -> raw reply text. Phase-1 default is an
# OpenAI-compatible HTTP call to the frozen sim server; tests inject a stub (no server, no
# openai import); Phase-2 co-training swaps in a same-engine backend.
SimBackend = Callable[[str, str], str]


def openai_sim_backend(system_content: str, user_content: str) -> str:
    """Default Phase-1 sim backend: query the FROZEN sim server over the OpenAI API.

    Reads OPENAI_BASE_URL (e.g. http://localhost:<SIM_PORT>/v1), MULTITURN_MODEL_NAME, and
    OPENAI_API_KEY (default "EMPTY"), matching the entrypoint's exported env. temp 0,
    max_tokens 4096; Qwen3 gets enable_thinking=False. 3 retries then "No response." --
    mirrors sweet_rl HumanInteractionEnv.invoke_model / InfoPO APIHumanSimulator.invoke_model.
    ``openai`` is imported lazily so CPU tests (which inject a stub) never need it installed.
    """
    from openai import OpenAI  # lazy: only the real sim path needs the SDK

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("MULTITURN_MODEL_NAME", "")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    params = {"model": model, "messages": messages, "max_tokens": 4096, "temperature": 0, "timeout": 60.0}
    if "qwen3" in model.lower():
        params["extra_body"] = {"enable_thinking": False}

    for _ in range(3):
        try:
            completion = client.chat.completions.create(**params)
            return completion.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - degrade to a default reply, never crash rollout
            logger.warning("[colbench] sim server call failed: %r", e)
    return "No response."


@dataclass
class ColBenchUserSimEnv:
    """User-simulator env holding the problem, hidden GT, and GT call-strings.

    Args:
        problem_description: the user's (public) problem statement.
        ground_truth: the HIDDEN ground-truth function source (never shown to the solver).
        test_cases: list of GT call-strings used for grading.
        max_steps: max solver turns before the episode is force-ended (sweet_rl default 10).
        reward_time_limit: per-case exec timeout (seconds) for grading.
        sim_backend: (system, user) -> raw reply. Defaults to the frozen-server HTTP call;
            tests / Phase-2 inject their own. This is the ONE seam that sees the GT.
    """

    problem_description: str
    ground_truth: str
    test_cases: list
    max_steps: int = 10
    reward_time_limit: float = 6.0
    sim_backend: Optional[SimBackend] = None
    # Populated on the last generate_user_turn call, for the agent loop's debug dump.
    last_sim_reply: str = field(default="", repr=False)

    def __post_init__(self):
        if self.sim_backend is None:
            self.sim_backend = openai_sim_backend

    def is_answer(self, assistant_text: str, episode_done: bool) -> tuple[bool, str]:
        """Did the solver submit a final answer this turn? Returns (has_answer, answer_text).

        ``assistant_text`` is <think>-stripped first so a marker inside a reasoning block is
        ignored. On the final turn a code-like response is accepted as the answer even without
        the marker (templates.final_answer).
        """
        clean = templates.strip_think(assistant_text)
        return templates.final_answer(clean, episode_done)

    def generate_user_turn(self, messages: list[dict]) -> str:
        """Produce the next user (simulator) reply. THE Phase-1/Phase-2 seam.

        ``messages`` is the running dialogue as ``[{role, content}, ...]`` (problem + solver
        turns + prior user replies) -- it contains NO ground truth. The GT source is injected
        as ``hidden_information`` into the sim prompt built here and passed only to the
        backend; the returned reply is <think>-stripped and hard-capped at 400 chars, so the
        solver's message list only ever receives that short reply.
        """
        user_content = templates.build_sim_user_message(
            self.problem_description, self.ground_truth, messages
        )
        raw = self.sim_backend(templates.SIM_SYSTEM_PROMPT, user_content)
        reply = templates.strip_think(raw)[: templates.HUMAN_RESPONSE_CHARACTER_LIMIT]
        self.last_sim_reply = reply
        if _DEBUG_SIM:
            n = _DEBUG_PREVIEW
            logger.warning(
                "[COLBENCH_SIM] hidden_gt[:%d]=%r\n[COLBENCH_SIM] sim_user_prompt[:%d]=%r\n"
                "[COLBENCH_SIM] raw_reply[:%d]=%r\n[COLBENCH_SIM] capped_reply=%r",
                n, str(self.ground_truth)[:n], n, user_content[:n], n, str(raw)[:n], reply,
            )
        return reply

    def score(self, answer_text: str) -> dict:
        """Grade the submitted answer against the GT. Returns reward.grade's dict.

        The answer is fence-stripped to code, then compared to the GT function on every
        call-string via the sandboxed exec sidecar (functional equivalence).
        """
        code = templates.extract_code_answer(answer_text)
        return reward.grade(
            code, self.ground_truth, self.test_cases, time_limit=self.reward_time_limit
        )

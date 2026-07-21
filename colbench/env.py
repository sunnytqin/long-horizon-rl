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


def _sim_extra_body():
    """Resolve the OpenAI `extra_body` for the sim call from SIM_ENABLE_THINKING.

    Default (unset) -> None: send NO thinking kwarg, which is safe for every model (Qwen2.5,
    Qwen3-Instruct-2507, non-Qwen). Only a HYBRID Qwen3 sim that would otherwise emit <think>
    needs SIM_ENABLE_THINKING=false; SIM_ENABLE_THINKING=true forces it on. This replaces the
    old brittle `"qwen3" in served_name` guard (the served name is a fixed alias, not the model
    family, so that check never fired).
    """
    v = os.environ.get("SIM_ENABLE_THINKING", "").strip().lower()
    if v in ("true", "1"):
        return {"enable_thinking": True}
    if v in ("false", "0"):
        return {"enable_thinking": False}
    return None


# A sim backend maps (system_content, user_content) -> raw reply text. Phase-1 default is an
# OpenAI-compatible HTTP call to the frozen sim server; tests inject a stub (no server, no
# openai import); Phase-2 co-training swaps in a same-engine backend.
SimBackend = Callable[[str, str], str]


def _sim_sampling():
    """Resolve the sim's sampling params from env, defaulting to Qwen3-Instruct's recommended.

    IMPORTANT: default temperature is 0.7 (NOT greedy). The Qwen3 family is explicitly
    documented to degrade / repeat under greedy (temp 0) decoding, so a Qwen3 sim MUST sample.
    Defaults = Qwen3-*-Instruct-2507 recommendation (temp 0.7, top_p 0.8, top_k 20, min_p 0);
    for a Qwen3-32B (thinking) sim set SIM_TEMPERATURE=0.6 SIM_TOP_P=0.95. Returns
    (temperature, top_p, top_k, min_p).
    """
    return (
        float(os.environ.get("SIM_TEMPERATURE", "0.7")),
        float(os.environ.get("SIM_TOP_P", "0.8")),
        int(os.environ.get("SIM_TOP_K", "20")),
        float(os.environ.get("SIM_MIN_P", "0")),
    )


def openai_sim_backend(system_content: str, user_content: str) -> str:
    """Default Phase-1 sim backend: query the FROZEN sim server over the OpenAI API.

    Reads OPENAI_BASE_URL (e.g. http://localhost:<SIM_PORT>/v1), MULTITURN_MODEL_NAME, and
    OPENAI_API_KEY (default "EMPTY"), matching the entrypoint's exported env. Sampling comes
    from SIM_TEMPERATURE/SIM_TOP_P/SIM_TOP_K/SIM_MIN_P (see _sim_sampling -- defaults to Qwen3's
    recommended non-greedy sampling); max_tokens 4096; 3 retries then "No response." -- mirrors
    sweet_rl HumanInteractionEnv.invoke_model / InfoPO APIHumanSimulator.invoke_model.
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
    temperature, top_p, top_k, min_p = _sim_sampling()
    # top_k / min_p are SGLang extensions -> extra_body; enable_thinking (if set) merges in.
    extra_body = {"top_k": top_k, "min_p": min_p}
    thinking = _sim_extra_body()
    if thinking is not None:
        extra_body.update(thinking)
    # SIM_MAX_TOKENS bounds the user turn at GENERATION time (default 4096 = unchanged). The spec
    # path sets it small so a brief human-like reply is not chopped post-hoc (it replaces the old
    # HUMAN_RESPONSE_CHARACTER_LIMIT slice, which truncated mid-sentence).
    sim_max_tokens = int(os.environ.get("SIM_MAX_TOKENS", "4096") or "4096")
    params = {
        "model": model, "messages": messages, "max_tokens": sim_max_tokens,
        "temperature": temperature, "top_p": top_p, "extra_body": extra_body, "timeout": 60.0,
    }

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

    def _sample_user_reply(self, messages: list[dict]) -> str:
        """One raw sim sample: build the prompt (with hidden GT), call the backend, clean up.

        The reply is <think>-stripped and hard-capped at 400 chars, so the solver's message
        list only ever receives that short reply. Shared by the single-shot training path
        (generate_user_turn) and the eval rejection loop (generate_user_turn_checked).
        """
        user_content = templates.build_sim_user_message(
            self.problem_description, self.ground_truth, messages
        )
        raw = self.sim_backend(templates.SIM_SYSTEM_PROMPT, user_content)
        reply = templates.strip_think(raw)[: templates.HUMAN_RESPONSE_CHARACTER_LIMIT]
        if _DEBUG_SIM:
            n = _DEBUG_PREVIEW
            logger.warning(
                "[COLBENCH_SIM] hidden_gt[:%d]=%r\n[COLBENCH_SIM] sim_user_prompt[:%d]=%r\n"
                "[COLBENCH_SIM] raw_reply[:%d]=%r\n[COLBENCH_SIM] capped_reply=%r",
                n, str(self.ground_truth)[:n], n, user_content[:n], n, str(raw)[:n], reply,
            )
        return reply

    def generate_user_turn(self, messages: list[dict]) -> str:
        """Produce the next user (simulator) reply. THE Phase-1/Phase-2 seam.

        ``messages`` is the running dialogue as ``[{role, content}, ...]`` (problem + solver
        turns + prior user replies) -- it contains NO ground truth. The GT source is injected
        as ``hidden_information`` into the sim prompt built here and passed only to the
        backend. Single-shot (no rejection sampling). Callers opt into rejection sampling via
        generate_user_turn_checked; this stays the path when it is disabled.
        """
        reply = self._sample_user_reply(messages)
        self.last_sim_reply = reply
        return reply

    def generate_user_turn_checked(
        self, messages: list[dict], max_tries: int = 32, ngram_n: int = 10, min_operators: int = 2
    ) -> dict:
        """Rejection-sampled user turn: resample until the reply has no code leak.

        Keeps drawing sim replies (up to ``max_tries``) until one passes
        ``templates.detect_code_leak`` (against the hidden GT this env holds), preventing the
        frozen simulator from just handing the solver the solution. Returns a record::

            {"reply": str|None, "tries": int, "accepted": bool, "reasons": [str, ...]}

        On success ``reply`` is the accepted turn, ``tries`` the number of samples drawn, and
        ``reasons`` the leak reason of each REJECTED sample (len == tries-1). On exhaustion
        ``accepted`` is False, ``reply`` is None (a "simulation failure" -- the caller
        terminates the trajectory), and ``reasons`` has one entry per try.

        Both callers terminate on exhaustion, but score it differently: eval treats it as a
        THIRD outcome excluded from the pass-rate denominator, while training keeps the
        trajectory in the batch at reward 0 so the offending solver turn takes a negative
        advantage (see colbench_agent.py).
        """
        reasons: list[str] = []
        for i in range(1, max_tries + 1):
            reply = self._sample_user_reply(messages)
            reason = templates.detect_code_leak(reply, self.ground_truth, ngram_n, min_operators)
            if reason is None:
                self.last_sim_reply = reply
                return {"reply": reply, "tries": i, "accepted": True, "reasons": reasons}
            reasons.append(reason)
        return {"reply": None, "tries": max_tries, "accepted": False, "reasons": reasons}

    def score(self, answer_text: str) -> dict:
        """Grade the submitted answer against the GT. Returns reward.grade's dict.

        The answer is fence-stripped to code, then compared to the GT function on every
        call-string via the sandboxed exec sidecar (functional equivalence).
        """
        code = templates.extract_code_answer(answer_text)
        return reward.grade(
            code, self.ground_truth, self.test_cases, time_limit=self.reward_time_limit
        )

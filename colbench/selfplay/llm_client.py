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
"""A configurable OpenAI-compatible chat client for the Phase-0 offline scripts.

Generalizes ``colbench.env.openai_sim_backend``: instead of reading a single fixed endpoint
from env vars, ``ChatEndpoint`` takes an explicit base_url / model / sampling so the spec
author (``strong`` teacher OR ``selfplay`` frozen base) and the diagnostic solver can point at
DIFFERENT served models in the same run. Same retry-to-default and SGLang ``extra_body``
(top_k / min_p / enable_thinking) handling as the sim backend, so behavior matches training.

``openai`` is imported lazily so CPU tests (which inject a stub callable) never need the SDK.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# A raw chat backend maps a list[{role, content}] -> reply text. Default is the real HTTP
# call; tests inject a stub. Mirrors ``colbench.env.SimBackend`` but takes full messages.
ChatBackend = Callable[[list], str]


@dataclass
class ChatEndpoint:
    """One served model behind an OpenAI-compatible API, with fixed sampling params.

    Args:
        base_url: e.g. ``http://127.0.0.1:30000/v1``.
        model: served model name/alias (the server's ``--served-model-name``).
        api_key: usually ``"EMPTY"`` for a local SGLang/vLLM server.
        temperature/top_p/top_k/min_p: sampling (defaults = Qwen3-Instruct recommendation,
            matching ``colbench.env._sim_sampling``; NB Qwen3 degrades under greedy).
        max_tokens: completion cap.
        enable_thinking: None -> send no thinking kwarg (safe for all models); True/False sets
            the SGLang ``enable_thinking`` extra_body for a hybrid Qwen3 model.
        retries/timeout: per-call retry budget and socket timeout.
        backend: injectable raw backend (tests). None -> the lazy OpenAI HTTP call.
    """

    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 4096
    enable_thinking: Optional[bool] = None
    retries: int = 3
    timeout: float = 120.0
    vendor: str = "vllm"  # "vllm"/"sglang" local server (sends top_k/min_p extra_body) or
                          # "openai" vanilla API (no vendor extras; some models reject custom
                          # sampling, so params are tried progressively-minimal -- see below).
    backend: Optional[ChatBackend] = field(default=None, repr=False)

    def _extra_body(self) -> dict:
        if self.vendor == "openai":
            return {}  # top_k / min_p / enable_thinking are vLLM/SGLang extensions -> the
            # public OpenAI API 400s on them.
        eb = {"top_k": self.top_k, "min_p": self.min_p}
        if self.enable_thinking is not None:
            eb["enable_thinking"] = self.enable_thinking
        return eb

    def _param_sets(self, messages: list) -> list:
        """Ordered param dicts to try. For the OpenAI API we degrade gracefully because newer
        models reject a custom ``temperature``/``top_p`` and require ``max_completion_tokens``
        instead of ``max_tokens``: try full sampling, then temperature-only, then bare."""
        if self.vendor == "openai":
            base = {"model": self.model, "messages": messages,
                    "max_completion_tokens": self.max_tokens, "timeout": self.timeout}
            return [
                {**base, "temperature": self.temperature, "top_p": self.top_p},
                {**base, "temperature": self.temperature},
                base,
            ]
        return [{
            "model": self.model, "messages": messages, "max_tokens": self.max_tokens,
            "temperature": self.temperature, "top_p": self.top_p,
            "extra_body": self._extra_body(), "timeout": self.timeout,
        }]

    def _http_backend(self, messages: list) -> str:
        from openai import OpenAI  # lazy: only the real path needs the SDK

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        param_sets = self._param_sets(messages)
        for _ in range(self.retries):
            for params in param_sets:  # fall to a more-minimal param set on invalid-request
                try:
                    completion = client.chat.completions.create(**params)
                    return completion.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001 - degrade to empty, never crash the batch
                    logger.warning("[selfplay] chat call to %s failed: %r", self.base_url, e)
        return ""

    def chat(self, messages: list) -> str:
        """Return the assistant reply text for ``messages`` (retries then ""/empty)."""
        backend = self.backend or self._http_backend
        return backend(messages)

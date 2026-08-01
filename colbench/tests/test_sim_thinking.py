"""Thinking suppression for a HYBRID Qwen3 user-simulator.

Confirmed against the running SGLang sim on 2026-07-30:

    no kwarg                                    -> reply opens `<think>`   (model default)
    top-level    enable_thinking: false         -> reply opens `<think>`   ACCEPTED + IGNORED
    chat_template_kwargs: {enable_thinking:...} -> "Four."                 works

The nesting is required because a hybrid Qwen3 disables reasoning in its JINJA
TEMPLATE (which pre-fills an empty `<think></think>`), so the flag must reach
the template renderer -- not the sampler. Two things follow, and both are pinned
here:

  1. The HTTP paths (SGLang / vLLM `extra_body`) MUST nest it.
  2. The TOKENIZER paths (`tokenizer.apply_chat_template(**kwargs)`) must NOT --
     there `enable_thinking` is a direct kwarg, and wrapping it would re-break
     thinking suppression in eval while looking like a fix.
"""

import pytest

from colbench.env import _sim_extra_body
from colbench.templates import strip_think


# ── 1. HTTP path: the flag must be NESTED ─────────────────────────────────────
def _sim_extra_body_payload(monkeypatch, value):
  """Rebuild the extra_body exactly as openai_sim_backend does.

  Without a server.
  """
  monkeypatch.setenv("SIM_ENABLE_THINKING", value)
  # pylint: disable=g-import-not-at-top
  from colbench.env import _sim_sampling

  _, _, top_k, min_p = _sim_sampling()
  extra_body = {"top_k": top_k, "min_p": min_p}
  thinking = _sim_extra_body()
  if thinking is not None:
    extra_body["chat_template_kwargs"] = thinking
  return extra_body


def test_thinking_flag_is_nested_under_chat_template_kwargs(monkeypatch):
  eb = _sim_extra_body_payload(monkeypatch, "false")
  assert eb["chat_template_kwargs"] == {"enable_thinking": False}
  # The top-level form is what SGLang silently ignores -- it must never be sent.
  assert (
      "enable_thinking" not in eb
  ), "top-level enable_thinking is accepted-and-ignored by SGLang"


def test_unset_sends_no_thinking_kwarg_at_all(monkeypatch):
  """Non-hybrid models (Qwen2.5, Instruct-2507) misbehave on the kwarg."""
  monkeypatch.delenv("SIM_ENABLE_THINKING", raising=False)
  eb = _sim_extra_body_payload(monkeypatch, "")
  assert "chat_template_kwargs" not in eb
  assert "enable_thinking" not in eb


def test_real_backend_payload_nests_it(monkeypatch):
  """End-to-end through openai_sim_backend.

  Inspect the kwargs actually sent to the SDK.
  """
  # pylint: disable=g-import-not-at-top
  import sys
  import types

  captured = {}

  class _Client:

    def __init__(self, *a, **k):
      self.chat = type(
          "C",
          (),
          {
              "completions": type(
                  "D",
                  (),
                  {
                      "create": staticmethod(
                          lambda **kw: (
                              captured.update(kw),
                              type(
                                  "R",
                                  (),
                                  {
                                      "choices": [
                                          type(
                                              "M",
                                              (),
                                              {
                                                  "message": type(
                                                      "Z",
                                                      (),
                                                      {"content": "Four."},
                                                  )()
                                              },
                                          )()
                                      ]
                                  },
                              )(),
                          )[1]
                      )
                  },
              )()
          },
      )()

  mod = types.ModuleType("openai")
  mod.OpenAI = _Client
  monkeypatch.setitem(sys.modules, "openai", mod)
  monkeypatch.setenv("SIM_ENABLE_THINKING", "false")

  from colbench.env import openai_sim_backend

  assert openai_sim_backend("sys", "user") == "Four."
  eb = captured["extra_body"]
  assert eb["chat_template_kwargs"] == {"enable_thinking": False}
  assert "enable_thinking" not in eb


# ── 2. TOKENIZER path: the flag must stay FLAT (guard against the wrong "fix") ────────────────
@pytest.mark.parametrize(
    "mod_name,fn_name",
    [
        ("colbench.validate_colbench", "_solver_template_kwargs"),
        ("colbench.validate_colbench_spec", "_solver_template_kwargs"),
    ],
)
def test_tokenizer_kwargs_stay_flat(monkeypatch, mod_name, fn_name):
  """These feed tokenizer.apply_chat_template(**kwargs).

  That call takes enable_thinking DIRECTLY.

  Wrapping them in chat_template_kwargs would pass the tokenizer an unknown
  argument and silently stop suppressing thinking in eval -- the same bug, newly
  introduced.
  """
  pytest.importorskip("transformers")
  mod = pytest.importorskip(mod_name)
  fn = getattr(mod, fn_name)
  monkeypatch.setenv("SOLVER_ENABLE_THINKING", "false")
  kw = fn()
  assert kw == {"enable_thinking": False}, "tokenizer kwargs must NOT be nested"
  monkeypatch.delenv("SOLVER_ENABLE_THINKING", raising=False)
  assert fn() == {}, "unset must send nothing (safe for non-hybrid models)"


# ── 3. strip_think must not fail open on a TRUNCATED block ────────────────────
def test_strip_think_closed_block():
  assert strip_think("<think>reasoning</think>Four.") == "Four."
  assert strip_think("no reasoning here") == "no reasoning here"
  assert strip_think("") == ""


def test_strip_think_truncated_block_is_removed():
  """SIM_MAX_TOKENS=256 cannot fit a hybrid Qwen3's reasoning.

  The `</think>` marker therefore never arrives.

  Before the fix the regex required a closing tag, so the raw monologue was
  injected into the conversation as the user's turn.
  """
  truncated = (
      "<think>Okay, the user is asking about a CSV parser. Let me consider "
      "whether they"
  )
  assert (
      strip_think(truncated) == ""
  ), "unterminated reasoning must not reach the dialogue"


def test_strip_think_keeps_text_before_a_truncated_block():
  assert (
      strip_think(
          "Sure, can you handle quotes?\n<think>wait, should I ask about"
      )
      == "Sure, can you handle quotes?"
  )


def test_strip_think_handles_closed_then_truncated():
  text = "<think>first</think>Real question here.<think>second one cut off"
  assert strip_think(text) == "Real question here."

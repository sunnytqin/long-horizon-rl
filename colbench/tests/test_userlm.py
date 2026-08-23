"""CPU tests for the USERLM sim protocol (colbench.userlm + env_spec wiring).

No server and no weights: the protocol seam (``sim_chat_backend``) is stubbed, so
what is pinned here is everything that can be wrong WITHOUT a GPU --

  * the dialogue reaches the simulator with its REAL roles (user = the simulated
    person, assistant = the solver), not flattened into one user message;
  * the task intent is the arm's OWN prompt, for every rung of the ladder, with
    the ``spec``/``grounded`` split (conditioning in the system message) and the
    ``codeonly``/``plot`` split (conditioning in the user message, dialogue cue
    stripped) both landing on the right text;
  * ``<|endconversation|>`` becomes ``[TERMINATE]``, so the loop's termination
    machinery (``sim_terminated``, the ``allow_terminate`` guard) fires on a user
    LM without any change to it;
  * the assistant protocol is left BYTE-IDENTICAL (the whole point of a
    default-off axis);
  * the prompt string the model will actually see, rendered by the real
    ``chat_template.jinja`` from microsoft/UserLM-8b -- pinned as a literal so a
    refactor here cannot silently drift from what the server does. That template
    appends a ``user`` generation header UNCONDITIONALLY; if a future SGLang
    served the model with its built-in llama-3 template instead, the header would
    say ``assistant`` and the sim would answer in the wrong role.
"""

import os

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

# The module-level setup above (env vars) has to run before these imports
# resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
import pytest

from colbench import templates
from colbench import userlm
from colbench.env_spec import ColBenchSpecUserSimEnv

GT = "def f(x, y):\n    return x + y\n"
CALLS = ["f(1, 2)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."
SPEC = {
    "persona": {
        "who": "an analyst",
        "domain": "ops",
        "python_skill": "beginner",
        "communication_style": "brief",
    },
    "scenario": "monthly reporting",
    "requirements": "add the two numbers",
    "plot": "you forgot to mention negatives",
}
DIALOGUE = [
    {"role": "user", "content": PROBLEM},
    {"role": "assistant", "content": "What should it do with negatives?"},
]


class _Recorder:
  """A stub ``ChatSimBackend`` that records its call and returns a fixed reply."""

  def __init__(self, reply="sure, just add them"):
    self.reply = reply
    self.calls = []

  def __call__(self, intent, dialogue):
    self.calls.append((intent, [dict(m) for m in dialogue]))
    return self.reply


def _env(sim_prompt="spec", backend=None, **kw):
  return ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
      sim_prompt=sim_prompt,
      sim_protocol="userlm",
      sim_chat_backend=backend if backend is not None else _Recorder(),
      **kw,
  )


# ── protocol validation ─────────────────────────────────────────────────────


def test_unknown_protocol_raises():
  with pytest.raises(ValueError, match="sim_protocol"):
    ColBenchSpecUserSimEnv(
        problem_description=PROBLEM,
        spec=SPEC,
        ground_truth=GT,
        test_cases=CALLS,
        sim_protocol="user_lm",  # typo
    )


def test_assistant_protocol_is_the_default():
  env = ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
  )
  assert env.sim_protocol == "assistant"
  # No chat backend is built for the assistant protocol -- the userlm one would
  # need an OpenAI client at construction time.
  assert env.sim_chat_backend is None


def test_assistant_protocol_still_flattens_the_dialogue():
  """The default path must be untouched: (system, flattened_user), no roles."""
  seen = {}

  def backend(system_content, user_content):
    seen["system"] = system_content
    seen["user"] = user_content
    return "ok"

  env = ColBenchSpecUserSimEnv(
      problem_description=PROBLEM,
      spec=SPEC,
      ground_truth=GT,
      test_cases=CALLS,
      sim_backend=backend,
  )
  env.generate_user_turn(DIALOGUE)
  assert seen["user"] == templates.str_dialogue_history(DIALOGUE)
  assert seen["user"].endswith(templates.DIALOGUE_CUE)


# ── the dialogue reaches the sim with real roles ────────────────────────────


def test_userlm_gets_role_tagged_dialogue_not_a_flattened_string():
  rec = _Recorder()
  env = _env(backend=rec)
  env.generate_user_turn(DIALOGUE)
  _, dialogue = rec.calls[0]
  assert dialogue == DIALOGUE
  # The flattened form must not appear anywhere in what was sent.
  assert templates.DIALOGUE_CUE not in "".join(m["content"] for m in dialogue)


def test_to_userlm_dialogue_coalesces_and_drops_empties():
  """Two assistant turns in a row is a shape WildChat never contained."""
  out = userlm.to_userlm_dialogue(
      [
          {"role": "user", "content": "a"},
          {"role": "assistant", "content": "b"},
          {"role": "assistant", "content": "c"},
          {"role": "user", "content": "   "},
          {"role": "user", "content": "d"},
      ]
  )
  assert out == [
      {"role": "user", "content": "a"},
      {"role": "assistant", "content": "b\n\nc"},
      {"role": "user", "content": "d"},
  ]
  assert [m["role"] for m in out] == ["user", "assistant", "user"]


def test_to_userlm_dialogue_folds_unknown_roles_into_user():
  out = userlm.to_userlm_dialogue(
      [
          {"role": "system", "content": "x"},
          {"role": "assistant", "content": "y"},
      ]
  )
  assert out == [
      {"role": "user", "content": "x"},
      {"role": "assistant", "content": "y"},
  ]


# ── the intent is the arm's own prompt ──────────────────────────────────────


@pytest.mark.parametrize(
    "arm", ["spec", "grounded", "grounded_v0", "grounded_v0_no_plot"]
)
def test_system_conditioned_arms_pass_their_system_prompt_as_the_intent(arm):
  rec = _Recorder()
  env = _env(sim_prompt=arm, backend=rec)
  env.generate_user_turn(DIALOGUE)
  intent, _ = rec.calls[0]
  # Byte-identical to what the assistant protocol would put in `system`.
  expected, _ = env._build_sim_prompt([])  # pylint: disable=protected-access
  assert intent == expected
  # And it is the arm's real text, not a constant.
  assert "role-playing a real person" in intent


@pytest.mark.parametrize("arm", ["codeonly", "plot"])
def test_user_conditioned_arms_pass_the_rendered_user_prompt_as_the_intent(arm):
  """codeonly/plot keep the conditioning in the USER message (sweet_rl shape)."""
  rec = _Recorder()
  env = _env(sim_prompt=arm, backend=rec)
  env.generate_user_turn(DIALOGUE)
  intent, _ = rec.calls[0]
  # The hidden GT conditions these arms, and it must survive into the intent --
  # otherwise the sim would be told nothing at all.
  assert GT.strip() in intent
  assert PROBLEM in intent
  # The "answer the agent" cue is cut -- the dialogue does not live in this
  # message, so a bare "agent:" line with nothing under it would be noise. In
  # HUMAN_SIMULATOR_PROMPT that cue sits in the MIDDLE of the text, so this also
  # pins that stripping it does not truncate the rest of the arm's prompt.
  assert templates.DIALOGUE_CUE not in intent
  assert "IN TWO SENTENCES" in intent
  assert "What should it do with negatives?" not in intent
  if arm == "plot":
    assert SPEC["plot"] in intent


def test_intent_does_not_depend_on_the_dialogue():
  """Resampling must not smuggle the transcript into the intent."""
  rec = _Recorder()
  env = _env(sim_prompt="codeonly", backend=rec)
  env.generate_user_turn(DIALOGUE)
  env.generate_user_turn(DIALOGUE + [{"role": "user", "content": "extra"}])
  assert rec.calls[0][0] == rec.calls[1][0]


def test_spec_arm_intent_never_carries_the_ground_truth():
  """The spec leak invariant is protocol-independent."""
  rec = _Recorder()
  env = _env(sim_prompt="spec", backend=rec)
  env.generate_user_turn(DIALOGUE)
  intent, dialogue = rec.calls[0]
  assert GT.strip() not in intent
  assert GT.strip() not in "".join(m["content"] for m in dialogue)


def test_strip_dialogue_cue_drops_only_the_bare_cue_line():
  assert templates.strip_dialogue_cue("no cue here") == "no cue here"
  # A cue line with content on it is content, not a cue.
  assert templates.strip_dialogue_cue("agent: hi") == "agent: hi"
  # The cue is removed wherever it sits, and the tail survives.
  assert (
      templates.strip_dialogue_cue("before\nagent:\n\nafter")
      == "before\n\nafter"
  )
  # This is what an empty-dialogue render actually looks like.
  assert (
      templates.strip_dialogue_cue(
          "history:\n" + templates.str_dialogue_history([]) + "\ntail"
      )
      == "history:\ntail"
  )


# ── termination bridges through to the existing machinery ───────────────────


def test_endconversation_token_becomes_the_terminate_sentinel():
  assert (
      userlm.translate_endconversation(userlm.ENDCONVERSATION_TOKEN)
      == templates.TERMINATE_MARKER
  )
  assert templates.sim_terminated(
      userlm.translate_endconversation(userlm.ENDCONVERSATION_TOKEN)
  )
  # A standalone token stays standalone, so `term_standalone` still fires.
  assert templates.sim_terminate_standalone(
      userlm.translate_endconversation(userlm.ENDCONVERSATION_TOKEN)
  )


def test_reply_without_the_token_is_untouched():
  assert userlm.translate_endconversation("looks good to me") == (
      "looks good to me"
  )


def test_premature_endconversation_is_rejected_by_the_guard():
  """allow_terminate=False must discard a terminating UserLM draw."""
  rec = _Recorder(reply=userlm.ENDCONVERSATION_TOKEN)
  # The backend is what translates the token in production; here the stub
  # returns the already-translated sentinel via the same helper.
  rec.reply = userlm.translate_endconversation(userlm.ENDCONVERSATION_TOKEN)
  env = _env(backend=rec, sim_max_tries=3)
  env.generate_user_turn(DIALOGUE, allow_terminate=False)
  assert env.last_sim_early_term_rejected == 3
  assert env.last_sim_early_term_exhausted


def test_is_userlm_model():
  assert userlm.is_userlm_model("microsoft/UserLM-8b")
  assert userlm.is_userlm_model("models/userlm-8b")
  assert not userlm.is_userlm_model("Qwen/Qwen3-32B")
  assert not userlm.is_userlm_model("")


# ── what the model actually sees ────────────────────────────────────────────

# The rendering of microsoft/UserLM-8b's chat_template.jinja for
# system+user+assistant. Two things it pins: turns keep their OWN role headers
# (so the simulated person's turns are `user` and the solver's are `assistant`),
# and the prompt ends on a `user` header -- the model speaks in the USER slot.
# There is no BOS: the template does not emit one and `apply_chat_template`
# tokenizes with add_special_tokens=False.
_EXPECTED_RENDER = (
    "<|start_header_id|>system<|end_header_id|>\nINTENT<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\nfirst<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\nreply<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
)
_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}\n"
    "{{ message['content'] }}<|eot_id|>{% endfor %}"
    "{{ '<|start_header_id|>user<|end_header_id|>' }}"
)


def test_userlm_chat_template_puts_the_model_in_the_user_slot():
  """Render UserLM-8b's own template over what the backend sends."""
  jinja2 = pytest.importorskip("jinja2")
  messages = [{"role": "system", "content": "INTENT"}] + (
      userlm.to_userlm_dialogue(
          [
              {"role": "user", "content": "first"},
              {"role": "assistant", "content": "reply"},
          ]
      )
  )
  rendered = jinja2.Template(_TEMPLATE).render(messages=messages)
  assert rendered == _EXPECTED_RENDER
  assert rendered.endswith("<|start_header_id|>user<|end_header_id|>")


# ── the request the backend actually sends ──────────────────────────────────


def _patched_backend(monkeypatch, reply, env=None):
  """Build the real backend with the OpenAI client swapped for a recorder."""
  captured = {}

  class _Msg:

    def __init__(self, content):
      self.content = content

  class _Choice:

    def __init__(self, content):
      self.message = _Msg(content)

  class _Completion:

    def __init__(self, content):
      self.choices = [_Choice(content)]

  class _Completions:

    def create(self, **params):
      captured.update(params)
      return _Completion(reply)

  class _Chat:
    completions = _Completions()

  class _Client:

    def __init__(self, **kw):
      captured["client_kwargs"] = kw
      self.chat = _Chat()

  import openai

  monkeypatch.setattr(openai, "OpenAI", _Client)
  for k, v in (env or {}).items():
    monkeypatch.setenv(k, v)
  return (
      userlm.make_userlm_sim_backend(
          base_url="http://sim/v1", model="colbench-sim", api_key="EMPTY"
      ),
      captured,
  )


def test_request_shape_defaults_to_the_model_cards_sampling(monkeypatch):
  backend, captured = _patched_backend(monkeypatch, "\nsounds good")
  # Make sure no ambient SIM_* from another test leaks in.
  for k in ("SIM_TEMPERATURE", "SIM_TOP_P", "SIM_TOP_K", "SIM_MAX_TOKENS"):
    monkeypatch.delenv(k, raising=False)
  reply = backend("INTENT", DIALOGUE)

  # The leading newline the chat template guarantees is stripped.
  assert reply == "sounds good"
  # The intent is the SYSTEM message; the dialogue keeps its roles after it.
  assert captured["messages"][0] == {"role": "system", "content": "INTENT"}
  assert [m["role"] for m in captured["messages"]] == [
      "system",
      "user",
      "assistant",
  ]
  # UserLM's sampling, not Qwen3's (0.7/0.8/20), and no top_k unless asked.
  assert captured["temperature"] == 1.0
  assert captured["top_p"] == 0.8
  assert "top_k" not in captured["extra_body"]
  # End-of-turn ids are sent explicitly: the checkpoint's config.json and
  # generation_config.json disagree about which token is EOS.
  assert captured["extra_body"]["stop_token_ids"] == list(userlm.EOT_TOKEN_IDS)
  assert captured["max_tokens"] == 256
  # No thinking kwarg is ever sent (UserLM has no such template hook).
  assert "chat_template_kwargs" not in captured["extra_body"]


def test_backend_translates_endconversation(monkeypatch):
  backend, _ = _patched_backend(
      monkeypatch, "\n" + userlm.ENDCONVERSATION_TOKEN
  )
  assert backend("INTENT", DIALOGUE) == templates.TERMINATE_MARKER


def test_backend_honors_the_shared_sim_envs(monkeypatch):
  backend, captured = _patched_backend(
      monkeypatch,
      "hi",
      env={
          "SIM_TEMPERATURE": "0.5",
          "SIM_TOP_P": "0.9",
          "SIM_TOP_K": "40",
          "SIM_MAX_TOKENS": "64",
      },
  )
  backend("INTENT", DIALOGUE)
  assert captured["temperature"] == 0.5
  assert captured["top_p"] == 0.9
  assert captured["extra_body"]["top_k"] == 40
  assert captured["max_tokens"] == 64

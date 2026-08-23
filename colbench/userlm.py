"""UserLM protocol for the ColBench user simulator (microsoft/UserLM-8b).

WHAT IS DIFFERENT ABOUT A USER LM
---------------------------------
Every simulator this tree has used so far is an ASSISTANT model asked, in prose,
to role-play the user: the whole dialogue is flattened into ONE user message
(``templates.str_dialogue_history`` -> ``"user:...\\n\\n\\n\\nassistant:...\\n\\n\\n\\nagent:"``)
and the model answers in the ASSISTANT slot. ``microsoft/UserLM-8b`` is the
opposite object: a Llama-3-8B *base* model post-trained on WildChat to predict
the **user** turn. It is not an assistant and does not follow instructions well;
what it does well is sound like a real user, shard information across turns, and
end conversations (see arXiv:2510.06552).

Consequences for how it must be driven -- all of them handled here:

1. REAL ROLES, NOT A FLATTENED TRANSCRIPT. Its chat template is::

     {% for message in messages %}{{ '<|start_header_id|>' + message['role'] +
     '<|end_header_id|>' }}\\n{{ message['content'] }}<|eot_id|>{% endfor %}{{
     '<|start_header_id|>user<|end_header_id|>' }}

   i.e. the turns are rendered with their OWN roles and the template
   UNCONDITIONALLY appends a ``user`` generation header (``add_generation_prompt``
   is ignored -- the model always speaks as the user). Our ``sim_dialogue``
   already carries exactly the right roles: ``user`` = the simulated person's own
   turns, ``assistant`` = the solver's. So the dialogue is passed through as
   MESSAGES and the flattened string is not used at all.

2. THE "TASK INTENT" IS THE SYSTEM MESSAGE. UserLM takes one input besides the
   dialogue: a high-level intent. We reuse the arm's EXISTING sim prompt as that
   intent verbatim (see ``env_spec.ColBenchSpecUserSimEnv._userlm_intent``), so
   switching protocol changes WHO answers and HOW the history is presented, and
   nothing about WHAT the simulator is told. NB the paper's intents are one
   under-specified sentence ("You are a user chatting with an assistant language
   model to ..."); our spec/grounded prompts are long instruction-heavy texts, so
   they are out of UserLM's conditioning distribution. That is a deliberate
   first-pass choice (a drop-in), not a claim that it is optimal -- an
   intent-shaped prompt is the obvious next arm.

3. TERMINATION IS A TOKEN, NOT A PHRASE. UserLM ends a conversation by emitting
   ``<|endconversation|>`` (id 128256). The spec loop's termination protocol is
   the ``[TERMINATE]`` sentinel in the reply TEXT, and UserLM will not follow a
   prose instruction to emit it. ``translate_endconversation`` rewrites the token
   into ``templates.TERMINATE_MARKER``, so the loop's entire termination
   machinery -- ``sim_terminated``, ``allow_terminate`` rejection sampling under
   ``early_term_guard``, ``term_standalone`` -- keeps working unmodified, and the
   simulator's single best-measured skill (dialogue termination) is the thing
   driving it. The token survives ``skip_special_tokens=True`` because it is an
   added token marked ``special: False``, so no server-side flag is needed.

4. SAMPLING. Defaults here are the model card's / paper's (temperature 1.0,
   top_p 0.8, no top_k), NOT the Qwen3 recommendation ``env._sim_sampling``
   returns. Same env var NAMES (SIM_TEMPERATURE / SIM_TOP_P / SIM_TOP_K /
   SIM_MAX_TOKENS / SIM_TIMEOUT / SIM_MAX_RETRIES) so nothing new has to be
   plumbed; only the defaults differ. ``stop_token_ids`` is sent explicitly
   because the checkpoint disagrees with itself about EOS (``config.json`` says
   128001 ``<|end_of_text|>``, ``generation_config.json`` and the tokenizer say
   128009 ``<|eot_id|>``); without it a server that trusts config.json would run
   past the end of the user turn and start hallucinating the assistant's reply.

NOT IMPLEMENTED (deliberately, and worth knowing before reading a run)
---------------------------------------------------------------------
Appendix C.1 of the paper lists four decoding guardrails the authors needed to
get usable simulations out of an 8B user model: (1) ban "I"/"You"/"Here" as the
first token, (2) ban ``<|endconversation|>``, (3) resample any turn over 25 words
or under 3, (4) resample verbatim repetitions of a previous turn or of the
intent. NONE of them are applied here. (2) is already covered better by the
loop's ``early_term_guard``. (1) needs a logit processor and is not reachable
over the OpenAI API. (3) and (4) are rejection-sampling rules and BELONG in
``env_spec.generate_user_turn``'s existing metered retry loop -- where their cost
shows up in ``sim_seconds`` / ``sim_raw_attempts`` -- rather than hidden inside a
backend. They also change the simulator's behavior distribution, so they are a
SEPARATE arm, not part of a drop-in swap. Until (3) lands, brevity is bounded
only by SIM_MAX_TOKENS (256 -> ~190 words, far looser than the paper's 25-word
cap), which on this task risks the sim volunteering the whole problem in one
turn -- the free-ride channel this project already fights. Watch
``sim_reply_chars``.

VERIFYING THE SERVER SIDE. SGLang applies the HF chat template it finds in the
model directory (``chat_template.jinja``; transformers >= 4.51 loads it into
``tokenizer.chat_template``). If it were to fall back to its built-in llama-3
conversation template instead, the generation header would say ``assistant`` and
the sim would silently answer in the wrong role. Two checks: the server log
should NOT report using a built-in/guessed template, and a
``COLBENCH_DEBUG_SIM=1`` dump should show short user-style utterances rather than
assistant prose. ``tests/test_userlm.py`` pins the expected rendering.
"""

# pylint: disable=g-importing-member
import logging
import os
from typing import Any
from typing import Callable
from typing import Optional

from colbench import templates

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# A STRUCTURED sim backend: (intent, dialogue) -> raw reply. The counterpart of
# ``env.SimBackend`` ((system, flattened_user) -> reply) for a simulator that
# needs the dialogue with its real roles. ``dialogue`` is the running
# conversation as ``[{role, content}, ...]`` with ``user`` = the simulated
# person's own turns and ``assistant`` = the solver's.
ChatSimBackend = Callable[[str, list[dict[str, str]]], str]

# The token UserLM emits to end a conversation. NOT a special token in the
# tokenizer (``special: False``), so it survives detokenization with
# ``skip_special_tokens=True`` and can be matched as text.
ENDCONVERSATION_TOKEN = "<|endconversation|>"

# End-of-turn ids for the UserLM (Llama-3) vocab, sent as SGLang
# ``stop_token_ids``. 128009 = <|eot_id|> (what the tokenizer and
# generation_config call EOS, and what the chat template ends every turn with);
# 128001 = <|end_of_text|> (what config.json calls EOS). Both are terminal for a
# user turn, so stopping on either is correct regardless of which one the server
# picked up.
EOT_TOKEN_IDS = (128009, 128001)

# Header markers of the Llama-3 chat format. Sent as TEXT stops as well, so a
# server configured with ``skip_special_tokens=False`` cannot leak the start of a
# hallucinated next turn into the reply.
_TEXT_STOPS = ("<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>")

# Model-name fragments that identify a UserLM checkpoint, used to auto-resolve
# the protocol from ``--sim_model`` so a launch cannot pair UserLM weights with
# the assistant protocol by omission. Matched case-insensitively on the model
# identity (e.g. "microsoft/UserLM-8b").
USERLM_MODEL_HINTS = ("userlm",)

# Legal values of the sim PROTOCOL axis (``+colbench.sim_protocol`` /
# SIM_PROTOCOL). Orthogonal to ``sim_prompt`` (WHAT the sim is told) and to
# ``sim_live`` (WHOSE weights answer): this is HOW the dialogue is presented and
# which role the model speaks in.
SIM_PROTOCOLS = frozenset({"assistant", "userlm"})


def is_userlm_model(model: str) -> bool:
  """True iff ``model`` names a UserLM checkpoint.

  Args:
    model: a model identity such as "microsoft/UserLM-8b", or "".

  Returns:
    True iff any entry of ``USERLM_MODEL_HINTS`` occurs in it.
  """
  low = (model or "").lower()
  return any(h in low for h in USERLM_MODEL_HINTS)


def to_userlm_dialogue(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
  """Normalize the running dialogue into UserLM's message list.

  The roles are already right (``user`` = the simulated person, ``assistant`` =
  the solver), so this only does what the chat template cannot: drop empty turns
  and COALESCE consecutive same-role turns. Coalescing matters because UserLM was
  trained on strictly alternating WildChat conversations; two ``assistant``
  headers in a row is a shape it never saw. The spec loop can produce that -- a
  solver turn that is rejected/timed out leaves no user reply between two
  assistant turns.

  Any role that is neither ``user`` nor ``assistant`` (there is none today, but a
  future ``system`` mid-dialogue would be one) is folded into ``user``: the
  template would otherwise emit an unseen header.

  Args:
    messages: the running dialogue as ``[{role, content}, ...]``.

  Returns:
    A new list of ``{role, content}`` dicts, alternating and non-empty.
  """
  out: list[dict[str, str]] = []
  for m in messages or []:
    content = str(m.get("content", "") or "").strip()
    if not content:
      continue
    role = str(m.get("role", "user") or "user")
    if role not in ("user", "assistant"):
      role = "user"
    if out and out[-1]["role"] == role:
      out[-1]["content"] = out[-1]["content"] + "\n\n" + content
      continue
    out.append({"role": role, "content": content})
  return out


def translate_endconversation(reply: str) -> str:
  """Rewrite UserLM's ``<|endconversation|>`` token as ``[TERMINATE]``.

  This is the whole termination bridge: downstream, ``templates.sim_terminated``
  and the ``allow_terminate`` rejection sampling in
  ``env_spec.generate_user_turn`` only ever look for
  ``templates.TERMINATE_MARKER`` in the reply text, so a one-line substitution
  here makes a UserLM sim terminate through the SAME code path as a prompted
  assistant sim -- including the premature-termination guard and the
  ``term_standalone`` metric (the token is normally the entire turn, so the
  translated reply is the bare marker).

  Args:
    reply: one raw sim reply, possibly containing the token.

  Returns:
    The reply with every occurrence of the token replaced by the sentinel.
  """
  if ENDCONVERSATION_TOKEN not in (reply or ""):
    return reply
  return reply.replace(ENDCONVERSATION_TOKEN, templates.TERMINATE_MARKER)


def _userlm_sampling() -> tuple[float, float, Optional[int]]:
  """Resolve UserLM's sampling params from the environment.

  Same env var NAMES as ``env._sim_sampling`` so nothing new has to be plumbed,
  but the DEFAULTS are the model card's / paper's (temperature 1.0, top_p 0.8),
  not Qwen3's (0.7 / 0.8 / top_k 20). ``top_k`` is left UNSET unless SIM_TOP_K is
  given: it is a Qwen recommendation, not a UserLM one, and the paper simulates
  at plain temperature 1.0.

  Returns:
    ``(temperature, top_p, top_k_or_None)``.
  """
  top_k = os.environ.get("SIM_TOP_K", "").strip()
  return (
      float(os.environ.get("SIM_TEMPERATURE", "1.0")),
      float(os.environ.get("SIM_TOP_P", "0.8")),
      int(top_k) if top_k else None,
  )


def make_userlm_sim_backend(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatSimBackend:
  """Build the UserLM sim backend: an OpenAI /chat/completions call, real roles.

  Reads the same env as ``env.openai_sim_backend`` when the arguments are
  omitted (OPENAI_BASE_URL / MULTITURN_MODEL_NAME / OPENAI_API_KEY, exported by
  ``entrypoint_colbench.sh``), so a UserLM sim is reached exactly like any other
  frozen sim server. The request differs from the assistant path in four ways,
  all of them forced by the model (see the module docstring): the intent goes in
  as ``system`` and the dialogue as REAL role-tagged messages; sampling defaults
  are UserLM's; end-of-turn ids are sent explicitly; and no thinking kwarg is
  ever sent (UserLM is a Llama-3 base derivative with no such template hook).

  Degrades to ``"No response."`` after 3 attempts, matching
  ``env.openai_sim_backend`` -- a poisoned turn beats a crashed rollout.

  Args:
    base_url: sim endpoint; default OPENAI_BASE_URL.
    model: served model name; default MULTITURN_MODEL_NAME.
    api_key: key for that endpoint; default OPENAI_API_KEY or "EMPTY".

  Returns:
    A ``ChatSimBackend`` mapping ``(intent, dialogue)`` to the raw reply, with
    ``<|endconversation|>`` already translated to ``[TERMINATE]``.
  """
  # pylint: disable=g-import-not-at-top
  from openai import OpenAI  # lazy: only the real sim path needs the SDK

  base_url = base_url or os.environ.get(
      "OPENAI_BASE_URL", "http://localhost:8000/v1"
  )
  model = model or os.environ.get("MULTITURN_MODEL_NAME", "")
  api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
  # max_retries=0 for the same reason as env.openai_sim_backend: keep the loop
  # below the single source of retry behavior.
  client = OpenAI(
      api_key=api_key,
      base_url=base_url,
      max_retries=int(os.environ.get("SIM_MAX_RETRIES", "0") or "0"),
  )

  def backend(intent: str, dialogue: list[dict[str, str]]) -> str:
    temperature, top_p, top_k = _userlm_sampling()
    messages = [{"role": "system", "content": intent}]
    messages += to_userlm_dialogue(dialogue)
    extra_body: dict[str, Any] = {"stop_token_ids": list(EOT_TOKEN_IDS)}
    if top_k is not None:
      extra_body["top_k"] = top_k
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": int(os.environ.get("SIM_MAX_TOKENS", "256") or "256"),
        "temperature": temperature,
        "top_p": top_p,
        "stop": list(_TEXT_STOPS),
        "extra_body": extra_body,
        "timeout": float(os.environ.get("SIM_TIMEOUT", "180") or "180"),
    }
    for _ in range(3):
      try:
        completion = client.chat.completions.create(**params)
        raw = completion.choices[0].message.content or ""
        # The chat template renders "<|end_header_id|>\n{content}", so the first
        # generated token of every turn is that newline. Strip it (and any
        # trailing whitespace) before the loop sees the reply.
        return translate_endconversation(raw.strip())
      except Exception as e:  # pylint: disable=broad-exception-caught  # degrade to a default reply, never crash rollout
        logger.warning("[colbench] userlm sim call failed: %r", e)
    return "No response."

  return backend

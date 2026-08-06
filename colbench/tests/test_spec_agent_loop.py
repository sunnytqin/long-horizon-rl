"""Agent-loop tests for the SPEC path.

Drive ``ColBenchSpecAgentLoop.run`` end-to-end with a scripted solver + scripted
sim, asserting the same ``terminated_by`` / reward / masking the pinned
``tests/test_env_spec.py::drive`` contract produces.

CONTAINER-ONLY: importing ``colbench.colbench_spec_agent`` pulls in
``verl.experimental.agent_loop`` (needs py>=3.11 + ray), which is NOT importable
in the light conda eval env. The whole module is skipped there via
``importorskip`` and runs inside the VERL/SGLang training container. It
complements the env-level tests (which run everywhere) by exercising the loop's
token/mask bookkeeping and extra_fields, not just the env seams.
"""

# These tests pin the behaviour of module-private helpers, so they reach for
# them directly.
# pylint: disable=protected-access

import asyncio
import os

import pytest

os.environ["CODECONTEST_ALLOW_INPROCESS"] = "1"
os.environ.pop("CODECONTEST_EXEC_URL", None)

# Skip the entire module unless the verl agent-loop stack is importable
# (container only).
pytest.importorskip("verl.experimental.agent_loop.agent_loop")

# The module-level setup above (env vars, sys.path) has to run
# before these imports resolve, so they cannot sit at the top.
# pylint: disable=g-import-not-at-top,wrong-import-position
from colbench.colbench_spec_agent import ColBenchSpecAgentLoop
from colbench.env_spec import ColBenchSpecUserSimEnv
from verl.workers.rollout.replica import TokenOutput

# Reuse the env-level fixtures' shape (kept local to avoid importing a test
# module).
GT = (
    "def f(x, y):\n    if x >= 10:\n        return x + y\n    else:\n "
    "       return x - y\n"
)
WRONG = "def f(x, y):\n    return x + y\n"  # ignores x<10 -> 0.5 pass-rate
CALLS = ["f(1, 2)", "f(20, 5)", "f(15, 15)", "f(3, 4)"]
PROBLEM = "Write a function f(x, y) with some personalized behavior."
SPEC = {
    "persona": {
        "who": "an analyst",
        "domain": "ops",
        "python_skill": "analyst",
        "communication_style": "brief",
    },
    "scenario": "Needs a small helper for a report.",
    "requirements": (
        "The user wants f(x,y): if x is at least 10 return x+y, otherwise "
        "x-y."
    ),
    "plot": (
        "The user reveals the threshold of 10 only if the assistant asks "
        "about the cutoff."
    ),
}


def _code_turn(src):
  return "Here's my function:\n```python\n" + src + "```"


def _scripted_backend(replies):
  seq = list(replies)
  state = {"i": 0}

  def backend(system_content, user_content):
    r = seq[min(state["i"], len(seq) - 1)]
    state["i"] += 1
    return r

  return backend


class _FakeTokenizer:
  """UTF-8 byte tokenizer.

  Exact encode/decode roundtrip + realistic token counts.
  """

  def encode(self, text, add_special_tokens=False):
    return list(text.encode("utf-8"))

  def decode(self, ids, skip_special_tokens=True):
    return bytes(ids).decode("utf-8", errors="ignore")


class _FakeServerManager:
  """Yields the scripted solver turns as TokenOutput.

  Encoded by the fake tokenizer.
  """

  def __init__(self, tokenizer, solver_turns):
    self._tok = tokenizer
    self._turns = list(solver_turns)
    self._i = 0

  async def generate(self, **kwargs):
    text = self._turns[min(self._i, len(self._turns) - 1)]
    self._i += 1
    ids = self._tok.encode(text)
    return TokenOutput(
        token_ids=ids,
        log_probs=[0.0] * len(ids),
        num_preempted=0,
        extra_fields={"min_global_steps": 0, "max_global_steps": 0},
    )


def _make_loop(
    solver_turns,
    sim_replies,
    *,
    max_assistant_turns=10,
    max_code_proposals=2,
    sim_max_tries=8,
    train_turns="all",
    terminate_on_allpass=False,
    binary_reward=False,
    grounded_sim=False,
    sim_prompt="",
    early_term_guard=True,
):
  """Construct a ColBenchSpecAgentLoop bypassing AgentLoopBase.__init__.

  Wired to fakes.
  """
  obj = object.__new__(ColBenchSpecAgentLoop)
  tok = _FakeTokenizer()
  obj.tokenizer = tok
  obj.server_manager = _FakeServerManager(tok, solver_turns)
  obj.loop = asyncio.new_event_loop()
  obj.prompt_length = 4096
  obj.response_length = 8192
  obj.max_assistant_turns = max_assistant_turns
  obj.max_new_tokens_per_turn = 1024
  obj.env_step_timeout = 60.0
  obj.reward_time_limit = 6.0
  obj.train_turns = train_turns
  obj.max_code_proposals = max_code_proposals
  obj.sim_max_tries = sim_max_tries
  # Reward-shaping / rollout-cleaning knobs run() reads (default off = baseline
  # behavior).
  obj.length_penalty_coef = 0.0
  obj.length_soft_cap = 2048.0
  obj.terminate_on_allpass = terminate_on_allpass
  obj.binary_reward = binary_reward
  # NB: run() reads every attribute set here -- this fake bypasses __init__, so
  #     a knob added to the loop and NOT mirrored here raises AttributeError
  #     mid-rollout.
  obj.grounded_sim = grounded_sim
  obj.sim_prompt = sim_prompt
  obj.early_term_guard = early_term_guard

  # apply_chat_template is normally an AgentLoopBase method; override on the
  # instance with a byte-encoding stub (only token COUNTS + mask placement
  # matter to the loop under test).
  async def _fake_act(messages, remove_system_prompt=False):
    text = "".join(m.get("content", "") for m in messages)
    return tok.encode(text)

  obj.apply_chat_template = _fake_act

  # Bind a spec env with the scripted sim backend (the loop builds its own env
  # in run(), but we inject the sim backend via extra_info-independent
  # monkeypatch on ColBenchSpecUserSimEnv is awkward; instead patch the default
  # backend by pre-seeding env creation through kwargs). The loop reads
  # extra_info.spec + ground_truth; we route the scripted backend via a
  # subclass.
  obj._test_sim_backend = _scripted_backend(sim_replies)
  return obj


def _run(obj, spec=None):
  spec = SPEC if spec is None else spec
  kwargs = {
      "raw_prompt": [
          {"role": "system", "content": "sys"},
          {"role": "user", "content": PROBLEM},
      ],
      "extra_info": {
          "spec": spec,
          "ground_truth": {
              "problem_description": PROBLEM,
              "ground_truth": GT,
              "test_cases": CALLS,
          },
      },
      "index": 0,
  }
  # Inject the scripted sim backend into every env the loop builds.
  orig_post_init = ColBenchSpecUserSimEnv.__post_init__

  def patched_post_init(self):
    # CHAIN, never replace. __post_init__ also RESOLVES THE ARM
    # (sim_prompt <- grounded) and validates it. Replacing it outright left
    # sim_prompt at "" so every mode silently fell through to the spec prompt --
    # a green suite hiding the one thing the mode tests exist to check.
    orig_post_init(self)
    self.sim_backend = obj._test_sim_backend

  ColBenchSpecUserSimEnv.__post_init__ = patched_post_init
  try:
    return obj.loop.run_until_complete(obj.run({"temperature": 0.7}, **kwargs))
  finally:
    ColBenchSpecUserSimEnv.__post_init__ = orig_post_init


def test_correct_code_then_user_terminate():
  # Solver asks, sim answers, solver shows GT, sim [TERMINATE] -> reward 1.0,
  # terminated_by user.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["It's 10.", "Perfect, thanks! [TERMINATE]"],
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_user"] == 1.0
  assert out.reward_score == 1.0
  assert rei["showed_code"] == 1.0
  assert rei["code_proposals"] == 1.0
  # Mask: solver turns are 1, the injected sim turn is 0, and at least one of
  #       each exists.
  assert any(m == 1 for m in out.response_mask)
  assert any(m == 0 for m in out.response_mask)


def test_code_cap_forces_grade():
  # Two WRONG proposals hit max_code_proposals=2 -> code_cap, grade last (0.5).
  obj = _make_loop(
      solver_turns=[_code_turn(WRONG), _code_turn(WRONG)],
      sim_replies=["Not quite, try again."],
      max_code_proposals=2,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_code_cap"] == 1.0
  assert rei["code_proposals"] == 2.0
  assert out.reward_score == 0.5


def test_user_terminate_without_code_is_no_code_zero():
  # The sim insists on ending across the whole try budget (the scripted backend
  # repeats its last reply), so the premature-termination guard is overruled and
  # the outcome is unchanged from before the guard existed -- it can only ever
  # remove premature terminations, never add a terminal state.
  obj = _make_loop(
      solver_turns=["Tell me more?"],
      sim_replies=["I think you've got it. [TERMINATE]"],
      sim_max_tries=3,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_no_code"] == 1.0
  assert rei["showed_code"] == 0.0
  assert out.reward_score == 0.0
  assert rei["sim_early_term_rejected"] == 3.0
  assert rei["term_early_term_exhausted"] == 1.0
  # A trailing marker in prose, not the bare sentinel the prompt now demands.
  assert rei["term_standalone"] == 0.0


def test_early_terminate_before_code_is_resampled_in_training():
  # Draw 1 wants out before any function exists -> rejected; draw 2 is a real
  # user turn, so the rollout goes on to a code proposal instead of being a
  # 0-reward stub. This is the training-side wiring of allow_terminate.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=[
          "Looks right to me. [TERMINATE]",
          "It's 10.",
          "[TERMINATE]",
      ],
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_user"] == 1.0
  assert rei["showed_code"] == 1.0
  assert out.reward_score == 1.0
  assert rei["sim_early_term_rejected"] == 1.0
  assert rei["term_early_term_exhausted"] == 0.0
  # The episode-ending reply WAS the bare sentinel this time.
  assert rei["term_standalone"] == 1.0


def test_sim_code_reject_exhaustion_aborts():
  # Sim always writes code -> exhaustion -> terminated_by sim_code_reject; grade
  # last shown code.
  obj = _make_loop(
      solver_turns=[_code_turn(GT), "anything"],
      sim_replies=["```python\ndef f(x, y): return x + y\n```"],
      sim_max_tries=3,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_sim_code_reject"] == 1.0
  assert rei["sim_code_rejected"] == 3.0
  assert out.reward_score == 1.0  # GT was shown before the abort


def test_all_turns_mask_keeps_every_solver_turn():
  # train_turns='all' (default) -> every solver span stays 1; only sim turns are
  # 0.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["It's 10.", "Great, thanks! [TERMINATE]"],
      train_turns="all",
  )
  out = _run(obj)
  # Two solver turns worth of 1s plus one sim turn of 0s.
  assert out.response_mask.count(1) == len(
      ("What's the cutoff?" + _code_turn(GT)).encode("utf-8")
  )
  assert out.response_mask.count(0) == len(b"It's 10.")


def test_upto_last_code_zeros_trailing_post_code_turn():
  # Solver: clarify -> code(GT) -> trailing ramble; sim keeps it going, then
  #         [TERMINATE]. 'upto_last_code' must KEEP the clarify + code turns
  #         (mask=1) and ZERO the trailing ramble.
  clarify, code, ramble = (
      "What's the cutoff?",
      _code_turn(GT),
      "You're absolutely right, thanks!",
  )
  obj = _make_loop(
      solver_turns=[clarify, code, ramble],
      sim_replies=[
          "It's 10.",
          "Looks good, anything else?",
          "Perfect. [TERMINATE]",
      ],
      train_turns="upto_last_code",
  )
  out = _run(obj)
  # Kept solver 1s == clarify + code bytes; the trailing ramble turn is fully
  # zeroed.
  assert out.response_mask.count(1) == len((clarify + code).encode("utf-8"))
  # Sanity: the graded reward is unaffected by masking (GT was the last code
  #         shown).
  assert out.reward_score == 1.0


def test_upto_last_code_no_code_keeps_all():
  # Never shows code -> last_code_idx is None -> fall back to 'all' (keep the
  # solver span), preserving the negative advantage on a no-code ramble.
  obj = _make_loop(
      solver_turns=["Tell me more?"],
      sim_replies=["I think you've got it. [TERMINATE]"],
      train_turns="upto_last_code",
  )
  out = _run(obj)
  assert out.extra_fields["reward_extra_info"]["term_no_code"] == 1.0
  assert out.response_mask.count(1) == len(b"Tell me more?")


def test_terminate_on_allpass_breaks_before_sim():
  # GT code on turn 0. With terminate_on_allpass, the loop grades mid-loop, sees
  # all_pass, and ends BEFORE the sim can press on -> terminated_by
  # oracle_solved, reward 1.0, and NO sim turn is ever injected (response_mask
  # has no zeros). The scripted sim reply is never consumed.
  obj = _make_loop(
      solver_turns=[_code_turn(GT)],
      sim_replies=["Are you sure? that looks off. [TERMINATE]"],
      terminate_on_allpass=True,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_oracle_solved"] == 1.0
  assert out.reward_score == 1.0
  assert rei["num_assistant_turns"] == 1.0
  assert (
      out.response_mask.count(0) == 0
  )  # no sim feedback injected -> all solver tokens


def test_terminate_on_allpass_partial_does_not_break():
  # WRONG (partial=0.5) code never all-passes, so terminate_on_allpass must NOT
  # fire; the loop runs to the code cap and grades normally, REUSING the cached
  # mid-loop grade (reward 0.5).
  obj = _make_loop(
      solver_turns=[_code_turn(WRONG), _code_turn(WRONG)],
      sim_replies=["Not quite, try again."],
      max_code_proposals=2,
      terminate_on_allpass=True,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_oracle_solved"] == 0.0
  assert rei["term_code_cap"] == 1.0
  assert out.reward_score == 0.5


def test_binary_reward_zeros_partial_but_keeps_fractional_metric():
  # binary_reward: a partial (0.5) pass becomes reward 0.0, but the raw
  #                pass_rate METRIC stays 0.5 -- the reward and the diagnostic
  #                are decoupled.
  obj = _make_loop(
      solver_turns=[_code_turn(WRONG), _code_turn(WRONG)],
      sim_replies=["Not quite, try again."],
      max_code_proposals=2,
      binary_reward=True,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert out.reward_score == 0.0  # binary: not all-pass -> 0
  assert rei["pass_rate"] == 0.5  # metric keeps the raw fractional rate
  assert rei["all_pass"] == 0.0


def test_binary_reward_all_pass_is_one():
  # binary_reward with GT -> all_pass -> reward 1.0 and pass_rate metric 1.0.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["It's 10.", "Perfect. [TERMINATE]"],
      binary_reward=True,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert out.reward_score == 1.0
  assert rei["pass_rate"] == 1.0
  assert rei["all_pass"] == 1.0


def _capturing_backend(replies):
  """Scripted backend that also records its (system, user) pairs."""
  inner = _scripted_backend(replies)
  seen = []

  def backend(system_content, user_content):
    seen.append((system_content, user_content))
    return inner(system_content, user_content)

  return backend, seen


def test_grounded_sim_flag_reaches_the_sim_prompt():
  # The loop's grounded_sim knob must land on the env it builds, i.e. the sim's
  # SYSTEM prompt carries the GT source + plot instead of the spec's
  # requirements/persona. The solver's own messages are unaffected (asserted by
  # the env-level leak test).
  backend, seen = _capturing_backend(
      ["It's 10.", "Perfect, thanks! [TERMINATE]"]
  )
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=[],
      grounded_sim=True,
  )
  obj._test_sim_backend = backend
  out = _run(obj)
  assert seen, "the sim was never called"
  sys_msg = seen[0][0]
  assert GT.strip() in sys_msg and SPEC["plot"] in sys_msg
  assert SPEC["requirements"] not in sys_msg and SPEC["scenario"] not in sys_msg
  assert out.extra_fields["reward_extra_info"]["term_user"] == 1.0


def test_grounded_sim_off_by_default_uses_spec_prompt():
  backend, seen = _capturing_backend(
      ["It's 10.", "Perfect, thanks! [TERMINATE]"]
  )
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)], sim_replies=[]
  )
  obj._test_sim_backend = backend
  _run(obj)
  sys_msg = seen[0][0]
  assert SPEC["requirements"] in sys_msg
  assert GT not in sys_msg


def test_new_reward_extra_info_scalars_present():
  # user_term_and_allpass / sim_reply_chars must be emitted on EVERY rollout:
  # verl reads the reward_extra_info key set from the first sample, so a key
  # missing on some path breaks logging.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["It's 10.", "Perfect, thanks! [TERMINATE]"],
  )
  rei = _run(obj).extra_fields["reward_extra_info"]
  assert (
      rei["user_term_and_allpass"] == 1.0
  )  # user-terminated AND all tests pass
  assert rei["sim_reply_chars"] == float(len("It's 10."))
  # A no-sim-turn episode still carries both keys (0.0), not a missing key.
  obj2 = _make_loop(
      solver_turns=[_code_turn(WRONG), _code_turn(WRONG)],
      sim_replies=["Not quite."],
      max_code_proposals=1,
  )
  rei2 = _run(obj2).extra_fields["reward_extra_info"]
  assert rei2["user_term_and_allpass"] == 0.0
  assert rei2["sim_reply_chars"] == 0.0


# ── A1/A2 ladder: the submit signal at max_code_proposals=1 ───────────────────
# templates.fenced_function documents why `contains_code` is safe at a 2-proposal
# budget and fatal at 1. These pin the cap-dependent switch.


def test_prose_def_mention_is_not_a_proposal_at_cap_one():
  # THE regression this fixes: a clarifying QUESTION that names a function must
  # not consume the only proposal and force-grade the episode at turn 1.
  obj = _make_loop(
      solver_turns=[
          "Should it be something like def parse(rows), is that right?",
          _code_turn(GT),
      ],
      sim_replies=["Yes, per row.", "Looks good."],
      max_code_proposals=1,
      sim_prompt="codeonly",
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["code_proposals"] == 1.0, "the prose `def` mention was counted"
  assert rei["num_assistant_turns"] == 2.0, "episode was force-graded at turn 1"
  assert out.reward_score == 1.0


def test_fenced_function_still_ends_the_episode_at_cap_one():
  obj = _make_loop(
      solver_turns=[_code_turn(GT), "unused"],
      sim_replies=["Looks good."],
      max_code_proposals=1,
      sim_prompt="codeonly",
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_code_cap"] == 1.0
  assert rei["num_assistant_turns"] == 1.0
  assert out.reward_score == 1.0


def test_cap_two_keeps_the_legacy_contains_code_detector():
  # Regression guard: spec/grounded (cap 2) stay byte-identical to before, where
  # a bare `def` in prose DOES count as a proposal.
  obj = _make_loop(
      solver_turns=["Maybe def parse(rows)?", _code_turn(GT)],
      sim_replies=["Sure.", "Looks good."],
      max_code_proposals=2,
  )
  out = _run(obj)
  assert out.extra_fields["reward_extra_info"]["code_proposals"] == 2.0


def test_final_turn_fallback_grades_unfenced_code_at_cap_one():
  # Naive parity: the naive arm's last turn submits whatever it said, so real
  # but UNFENCED code still gets graded instead of scoring 0 as `no_code`.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", GT],  # GT, no ```python fence
      sim_replies=["It's 10.", "ok"],
      max_assistant_turns=2,
      max_code_proposals=1,
      sim_prompt="codeonly",
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["showed_code"] == 1.0, "unfenced final-turn code was thrown away"
  assert rei["term_no_code"] == 0.0
  assert out.reward_score == 1.0


def test_final_turn_fallback_does_not_rescue_prose():
  # Prose is still submitted (as naive does) but cannot parse, so the reward is
  # the same 0 the `no_code` path gave -- the fallback changes WHICH case
  # differs, not the prose outcome.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", "Thanks, that helps a lot!"],
      sim_replies=["It's 10.", "ok"],
      max_assistant_turns=2,
      max_code_proposals=1,
      sim_prompt="codeonly",
  )
  out = _run(obj)
  assert out.reward_score == 0.0


def test_final_turn_fallback_is_off_above_cap_one():
  # spec/grounded must stay byte-identical: an episode that never shows code
  # still ends `no_code` at reward 0.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", "Thanks!"],
      sim_replies=["It's 10.", "ok"],
      max_assistant_turns=2,
      max_code_proposals=2,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["term_no_code"] == 1.0
  assert rei["showed_code"] == 0.0
  assert out.reward_score == 0.0


# ── Premature-[TERMINATE] guard toggle (+colbench.early_term_guard) ───────────


def test_guard_on_blocks_a_pre_code_terminate():
  # Default. The sim wants out before any code exists; the guard resamples it
  # and the conversation continues to a real submission.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["[TERMINATE]", "It's 10.", "Looks good."],
      early_term_guard=True,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["sim_early_term_rejected"] >= 1.0
  assert rei["showed_code"] == 1.0
  assert out.reward_score == 1.0


def test_guard_off_restores_pre_guard_termination():
  # The ONLY way to reproduce a run from before the guard landed: the same sim
  # reply now ends the episode at once, with no code and zero reward.
  obj = _make_loop(
      solver_turns=["What's the cutoff?", _code_turn(GT)],
      sim_replies=["[TERMINATE]", "It's 10.", "Looks good."],
      early_term_guard=False,
  )
  out = _run(obj)
  rei = out.extra_fields["reward_extra_info"]
  assert rei["sim_early_term_rejected"] == 0.0
  assert rei["showed_code"] == 0.0
  assert rei["term_no_code"] == 1.0
  assert out.reward_score == 0.0

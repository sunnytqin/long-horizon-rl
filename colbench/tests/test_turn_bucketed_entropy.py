"""Tests for the turn-bucketed entropy diagnostic (ray_trainer).

The metric recovers solver-turn boundaries by run-length decoding
`response_mask` instead of plumbing spans down from the agent loop, so the
decoding itself is the thing that can silently go wrong (an off-by-one would
misattribute every token by one turn and invert the read). These pin it.
"""

import pytest

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")

from verl.trainer.ppo.ray_trainer import compute_turn_bucketed_entropy


def _m(*rows):
  return torch.tensor(rows, dtype=torch.long)


def test_single_turn_all_tokens_in_bucket_zero():
  mask = _m([1, 1, 1, 0, 0])
  ent = torch.tensor([[2.0, 4.0, 6.0, 99.0, 99.0]])
  out = compute_turn_bucketed_entropy(ent, mask)
  assert out["actor/entropy_turns/mean"] == 1.0
  assert out["actor/entropy_turn0"] == pytest.approx(4.0)
  assert out["actor/entropy_turn0/tok_frac"] == pytest.approx(1.0)
  # Masked positions must not leak into any bucket.
  assert "actor/entropy_turn1" not in out
  assert out["actor/entropy_turn1/tok_frac"] == pytest.approx(0.0)


def test_three_runs_map_to_three_buckets_in_emission_order():
  # turn0 = 2 tok, sim reply, turn1 = 1 tok, sim reply, turn2 = 3 tok
  mask = _m([1, 1, 0, 0, 1, 0, 1, 1, 1])
  ent = torch.tensor([[1.0, 3.0, 9.0, 9.0, 5.0, 9.0, 10.0, 20.0, 30.0]])
  out = compute_turn_bucketed_entropy(ent, mask, n_buckets=3)
  assert out["actor/entropy_turns/mean"] == 3.0
  assert out["actor/entropy_turn0"] == pytest.approx(2.0)
  assert out["actor/entropy_turn1"] == pytest.approx(5.0)
  assert out["actor/entropy_turn2plus"] == pytest.approx(20.0)
  assert out["actor/entropy_turn0/tok_frac"] == pytest.approx(2 / 6)
  assert out["actor/entropy_turn1/tok_frac"] == pytest.approx(1 / 6)
  assert out["actor/entropy_turn2plus/tok_frac"] == pytest.approx(3 / 6)


def test_last_bucket_absorbs_turns_beyond_it():
  # 4 solver turns, 1 token each; bucket 2plus must pool turns 2 and 3.
  mask = _m([1, 0, 1, 0, 1, 0, 1])
  ent = torch.tensor([[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 7.0]])
  out = compute_turn_bucketed_entropy(ent, mask, n_buckets=3)
  assert out["actor/entropy_turns/mean"] == 4.0
  assert out["actor/entropy_turn2plus"] == pytest.approx(5.0)  # (3+7)/2
  assert out["actor/entropy_turn2plus/tok_frac"] == pytest.approx(2 / 4)


def test_turn_starting_at_position_zero_is_not_dropped():
  # The `prev` shift must treat column 0 as "no preceding token", or a turn
  # that opens the response is never counted as a run start.
  mask = _m([1, 1])
  ent = torch.tensor([[4.0, 6.0]])
  out = compute_turn_bucketed_entropy(ent, mask)
  assert out["actor/entropy_turns/mean"] == 1.0
  assert out["actor/entropy_turn0"] == pytest.approx(5.0)


def test_runs_are_per_row_not_across_the_batch():
  # Row 1's first run must be ITS bucket 0, not a continuation of row 0.
  mask = _m([1, 0, 1], [1, 1, 0])
  ent = torch.tensor([[1.0, 0.0, 3.0], [10.0, 20.0, 0.0]])
  out = compute_turn_bucketed_entropy(ent, mask, n_buckets=3)
  assert out["actor/entropy_turns/mean"] == pytest.approx(1.5)  # (2+1)/2
  assert out["actor/entropy_turn0"] == pytest.approx((1.0 + 10.0 + 20.0) / 3)
  assert out["actor/entropy_turn1"] == pytest.approx(3.0)


def test_fully_masked_row_contributes_no_tokens_and_does_not_divide_by_zero():
  mask = _m([0, 0, 0], [1, 1, 0])
  ent = torch.tensor([[9.0, 9.0, 9.0], [2.0, 4.0, 9.0]])
  out = compute_turn_bucketed_entropy(ent, mask)
  assert out["actor/entropy_turns/mean"] == pytest.approx(0.5)
  assert out["actor/entropy_turn0"] == pytest.approx(3.0)


def test_empty_mask_returns_only_the_turn_count():
  out = compute_turn_bucketed_entropy(torch.zeros(2, 3), _m([0, 0, 0], [0, 0, 0]))
  assert out == {"actor/entropy_turns/mean": 0.0}


def test_bucket_means_are_token_weighted_across_the_batch():
  # Row 0 turn0 has 3 tokens at 1.0; row 1 turn0 has 1 token at 9.0.
  # Token-mean = 12/4 = 3.0, NOT the mean of row means (5.0).
  mask = _m([1, 1, 1], [1, 0, 0])
  ent = torch.tensor([[1.0, 1.0, 1.0], [9.0, 0.0, 0.0]])
  out = compute_turn_bucketed_entropy(ent, mask)
  assert out["actor/entropy_turn0"] == pytest.approx(3.0)


def test_default_is_four_buckets_so_turn_growth_is_not_pooled_away():
  # The default must separate turn 2 from turn 3+, which is where the observed
  # turn growth sits; pooling at 3 would make the diagnostic unreadable.
  mask = _m([1, 0, 1, 0, 1, 0, 1, 0, 1])
  ent = torch.tensor([[1.0, 0, 2.0, 0, 3.0, 0, 4.0, 0, 8.0]])
  out = compute_turn_bucketed_entropy(ent, mask)
  assert out["actor/entropy_turn2"] == pytest.approx(3.0)
  assert out["actor/entropy_turn3plus"] == pytest.approx(6.0)  # (4+8)/2
  assert "actor/entropy_turn2plus" not in out


# ── Cov(advantage, log pi) ──────────────────────────────────────────────────────


def _cov_of(adv, logp, mask, **kw):
  from verl.trainer.ppo.ray_trainer import compute_advantage_logprob_cov

  return compute_advantage_logprob_cov(
      torch.tensor(adv, dtype=torch.float32),
      torch.tensor(logp, dtype=torch.float32),
      _m(*mask),
      **kw,
  )


def test_cov_sign_is_the_covariance_not_the_entropy_derivative():
  # Likely actions (logp near 0) earn HIGH advantage => Cov > 0 => sharpening.
  # The logged value must keep the covariance's sign, NOT -Cov.
  out = _cov_of(
      [[1.0, -1.0]], [[-0.1, -2.0]], [[1, 1]]
  )
  assert out["actor/adv_logp_cov"] > 0
  # Flip the pairing: unlikely actions earn the advantage => Cov < 0 => H rises.
  out = _cov_of([[-1.0, 1.0]], [[-0.1, -2.0]], [[1, 1]])
  assert out["actor/adv_logp_cov"] < 0


def test_cov_matches_the_closed_form_over_masked_tokens_only():
  # Masked-out positions carry values that would change the answer if included.
  adv = [[2.0, -2.0, 99.0]]
  logp = [[-1.0, -3.0, -99.0]]
  out = _cov_of(adv, logp, [[1, 1, 0]])
  # population cov over the two kept tokens
  a, l = [2.0, -2.0], [-1.0, -3.0]
  am, lm = sum(a) / 2, sum(l) / 2
  want = sum((x - am) * (y - lm) for x, y in zip(a, l)) / 2
  assert out["actor/adv_logp_cov"] == pytest.approx(want, rel=1e-6)


def test_cov_is_zero_when_advantage_is_constant():
  # No advantage variation => no drift, regardless of the logprob spread.
  out = _cov_of([[1.0, 1.0, 1.0]], [[-0.1, -1.0, -5.0]], [[1, 1, 1]])
  assert out["actor/adv_logp_cov"] == pytest.approx(0.0, abs=1e-9)


def test_cov_buckets_by_turn_and_can_disagree_with_the_overall_sign():
  # turn0: likely->high adv (Cov>0). turn1: likely->low adv (Cov<0).
  adv = [[1.0, -1.0, 0.0, -1.0, 1.0]]
  logp = [[-0.1, -2.0, 0.0, -0.1, -2.0]]
  out = _cov_of(adv, logp, [[1, 1, 0, 1, 1]], n_buckets=2)
  assert out["actor/adv_logp_cov_turn0"] > 0
  assert out["actor/adv_logp_cov_turn1plus"] < 0


def test_cov_bucket_with_one_token_is_omitted_not_reported_as_zero():
  # A single token has no covariance; reporting 0.0 would read as "no drift".
  out = _cov_of([[1.0, 0.0, 5.0]], [[-1.0, 0.0, -2.0]], [[1, 0, 1]], n_buckets=2)
  assert "actor/adv_logp_cov_turn0" not in out
  assert "actor/adv_logp_cov_turn1plus" not in out
  # kept tokens (1, -1.0) and (5, -2.0): mean adv 3, mean logp -1.5 => cov -1.0
  assert out["actor/adv_logp_cov"] == pytest.approx(-1.0, rel=1e-6)


def test_cov_is_token_weighted_not_trajectory_weighted():
  # 3 tokens at (adv 1, logp -0.1) and 1 at (adv -1, logp -3.0).
  # Token-weighted population cov = 4.35/4 = 1.0875.
  # Giving each ROW equal say instead would give 2.90/2 = 1.45 -- both positive,
  # so only the magnitude distinguishes them. Assert the weighting, not the sign.
  out = _cov_of(
      [[1.0, 1.0, 1.0], [-1.0, 0.0, 0.0]],
      [[-0.1, -0.1, -0.1], [-3.0, 0.0, 0.0]],
      [[1, 1, 1], [1, 0, 0]],
  )
  assert out["actor/adv_logp_cov"] == pytest.approx(1.0875, rel=1e-6)
  assert out["actor/adv_logp_cov"] != pytest.approx(1.45, rel=1e-3)


def test_cov_and_entropy_buckets_agree_on_turn_boundaries():
  # The two metrics must decode turns identically -- they share
  # _solver_turn_index, and this is what would catch a future divergence.
  # Every turn gets 2 tokens so BOTH metrics report it (see the next test for
  # the deliberate n<2 asymmetry).
  from verl.trainer.ppo.ray_trainer import compute_turn_bucketed_entropy

  mask = [[1, 1, 0, 1, 1, 0, 1, 1]]
  e = compute_turn_bucketed_entropy(torch.ones(1, 8), _m(*mask))
  c = _cov_of(
      [[1.0, 2.0, 0, 3.0, 4.0, 0, 5.0, 6.0]],
      [[-1.0, -2.0, 0, -3.0, -4.0, 0, -5.0, -6.0]],
      mask,
  )
  ent_buckets = {k.rsplit("_", 1)[-1] for k in e if k.endswith(("turn0", "turn1", "turn2"))}
  cov_buckets = {k.rsplit("_", 1)[-1] for k in c if k.endswith(("turn0", "turn1", "turn2"))}
  assert ent_buckets == cov_buckets == {"turn0", "turn1", "turn2"}
  assert e["actor/entropy_turns/mean"] == 3.0


def test_entropy_reports_a_one_token_bucket_but_cov_omits_it():
  # DELIBERATE asymmetry, not a bug: a mean needs 1 token, a covariance needs 2.
  # Reporting an undefined covariance as 0.0 would read as "no drift" on the
  # dashboard, which is a substantive false claim -- so it is omitted instead.
  from verl.trainer.ppo.ray_trainer import compute_turn_bucketed_entropy

  mask = [[1, 1, 0, 1]]  # turn0 = 2 tokens, turn1 = 1 token
  e = compute_turn_bucketed_entropy(torch.ones(1, 4), _m(*mask))
  c = _cov_of([[1.0, 2.0, 0, 3.0]], [[-1.0, -2.0, 0, -3.0]], mask)
  assert "actor/entropy_turn1" in e
  assert "actor/adv_logp_cov_turn1" not in c
  assert "actor/adv_logp_cov_turn0" in c


def test_solver_turn_len_is_masked_tokens_per_kept_turn():
  # 3 kept turns holding 2 + 1 + 3 = 6 masked tokens => 2.0 tokens/turn.
  # Must count MASKED tokens only: the 0s are simulator replies.
  mask = _m([1, 1, 0, 0, 1, 0, 1, 1, 1])
  out = compute_turn_bucketed_entropy(torch.ones(1, 9), mask)
  assert out["actor/entropy_turns/mean"] == 3.0
  assert out["actor/solver_turn_len/mean"] == pytest.approx(6 / 3)


def test_solver_turn_len_pools_over_the_batch_not_per_row():
  # Row 0: 1 turn of 3 tokens. Row 1: 2 turns of 1 token each.
  # Batch-pooled = 5 tokens / 3 turns. A per-row mean would give (3 + 1)/2 = 2.
  mask = _m([1, 1, 1, 0], [1, 0, 1, 0])
  out = compute_turn_bucketed_entropy(torch.ones(2, 4), mask)
  assert out["actor/entropy_turns/mean"] == pytest.approx(1.5)
  assert out["actor/solver_turn_len/mean"] == pytest.approx(5 / 3)


def test_solver_turn_len_absent_when_nothing_is_masked():
  out = compute_turn_bucketed_entropy(torch.zeros(1, 3), _m([0, 0, 0]))
  assert "actor/solver_turn_len/mean" not in out


# ── wiring ─────────────────────────────────────────────────────────────────────
# These exist because on 2026-09-15 both diagnostics were wired into
# verl/trainer/ppo/ray_trainer.py (the V0 RayPPOTrainer) while colbench actually
# runs TaskRunnerV1 -> trainer.v1.trainer_mode=sync -> PPOTrainerSync, whose loop
# lives in verl/trainer/ppo/v1/trainer_base.py. The functions imported fine and
# `actor/entropy` kept logging (it is computed in BOTH paths), so nothing looked
# wrong -- a 5-hour replay through the detonation produced zero new metrics.
# A unit test of the math cannot catch that; only a check of the call site can.


def _src(rel):
  import pathlib

  import verl

  return (pathlib.Path(verl.__file__).parent / rel).read_text()


@pytest.mark.parametrize(
    "path",
    ["trainer/ppo/v1/trainer_base.py", "trainer/ppo/ray_trainer.py"],
)
def test_both_trainer_paths_call_the_entropy_bucket_diagnostic(path):
  assert "compute_turn_bucketed_entropy(" in _src(path), (
      f"{path} never calls compute_turn_bucketed_entropy -- the metric will be"
      " silently absent for whichever trainer that file implements"
  )


@pytest.mark.parametrize(
    "path",
    ["trainer/ppo/v1/trainer_base.py", "trainer/ppo/ray_trainer.py"],
)
def test_both_trainer_paths_call_the_covariance_diagnostic(path):
  assert "compute_advantage_logprob_cov(" in _src(path), (
      f"{path} never calls compute_advantage_logprob_cov -- the metric will be"
      " silently absent for whichever trainer that file implements"
  )


def test_v1_trainer_is_the_one_colbench_actually_runs():
  # If this ever fails, the run scripts moved off trainer_mode=sync and the
  # wiring tests above may be guarding the wrong file.
  import re

  cfg = _src("trainer/config/ppo_trainer.yaml")
  assert re.search(r"^\s*trainer_mode:\s*sync\s*$", cfg, re.M), (
      "trainer.v1.trainer_mode default is no longer 'sync'; re-check which"
      " trainer implements the loop before trusting the diagnostics"
  )


def test_diagnostics_are_importable_from_the_v1_module():
  # Guards the import direction (v1 -> ray_trainer) against a future cycle.
  from verl.trainer.ppo.v1 import trainer_base

  assert callable(trainer_base.compute_turn_bucketed_entropy)
  assert callable(trainer_base.compute_advantage_logprob_cov)

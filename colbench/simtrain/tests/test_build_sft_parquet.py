# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Stage 4 tests: selection -> the three-message SFT row, and the refusals.

CPU only. The refusals matter more than the happy path here: every one of them
is a failure that produces a FILE rather than an error, and a bad SFT parquet
looks exactly like a good one until the trained simulator misbehaves.
"""

import json
import os
import tempfile

import pytest

from colbench import prompts
from colbench import templates
from colbench.simtrain import build_sft_parquet as B
from colbench.simtrain import judge_rubric


GT = "def f(x, y):\n    return x + y if x >= 10 else x - y\n"


def _prefix(pid, task_id=1, turn=0, system=None):
  return {
      "prefix_id": pid,
      "task_id": task_id,
      "task_index": int(task_id),
      "turn_idx": turn,
      "episode_idx": 0,
      "ground_truth": GT,
      "sim_system": system or prompts.SIM_ROLE_RESTRAINT_SYSTEM_PROMPT,
      "sim_user": f"the rendered dialogue for {pid}",
      "sim_user_sha16": "abc123",
      "sim_system_variant": "role_restraint",
      "collector_version": "c1",
  }


def _verdict(total, leak=1, cal=4, ic=4, vol=4):
  return {
      "code_leak": leak,
      "calibration": cal,
      "in_character": ic,
      "volunteering": vol,
      "total": total,
  }


def _judged(pid, candidates, scores, **extra):
  rec = {
      "prefix_id": pid,
      "ok": True,
      "candidates": candidates,
      "scores": scores,
      "margin": 1,
      "rubric_version": prompts.JUDGE_RUBRIC_VERSION,
      "harness_version": "h2",
      "judge_best_cand_idx": None,
  }
  rec.update(extra)
  return rec


def test_row_is_system_user_assistant_and_reuses_the_collected_prompt():
  """The row shape MultiTurnSFTDataset consumes, and where the system text comes from.

  `collected` (the default) takes the system message off the PREFIX record, so
  the SFT row is byte-identical to the prompt that drew the candidate and to
  what the sim is served with. Anything else is context distillation and has to
  be named explicitly.
  """
  pre = {"p1": _prefix("p1")}
  jd = [_judged("p1", ["good reply", "worse"], [_verdict(12), _verdict(9, cal=3)])]
  rows, stats = B.build_rows(pre, jd, judge_rubric.DEFAULT_MIN_DIMS)
  assert stats["KEPT"] == 1 and len(rows) == 1
  msgs = rows[0]["messages"]
  assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
  assert msgs[0]["content"] == prompts.SIM_ROLE_RESTRAINT_SYSTEM_PROMPT
  assert msgs[1]["content"] == pre["p1"]["sim_user"]
  assert msgs[2]["content"] == "good reply"  # the argmax, not candidate 0
  assert rows[0]["sim_system_variant"] == "role_restraint"
  assert rows[0]["rubric_version"] == prompts.JUDGE_RUBRIC_VERSION


def test_system_override_is_context_distillation_and_is_validated():
  """A named variant swaps ONLY the system message, and a typo must raise.

  The user message stays the materialized one -- that is the whole point of
  context distillation: draw under the teacher, train against the production
  prompt. A silently-defaulted typo would produce a run labelled as one arm and
  trained as another.
  """
  pre = {"p1": _prefix("p1")}
  jd = [_judged("p1", ["r"], [_verdict(12)])]
  rows, _ = B.build_rows(
      pre, jd, judge_rubric.DEFAULT_MIN_DIMS, system_variant="default"
  )
  assert rows[0]["messages"][0]["content"] == templates.resolve_sim_system("default")
  assert rows[0]["messages"][1]["content"] == pre["p1"]["sim_user"]
  assert rows[0]["sim_system_variant"] == "default"
  with pytest.raises(ValueError):
    B.build_rows(pre, jd, None, system_variant="role_restrant")


def test_drops_are_counted_by_reason_and_a_vetoed_group_yields_nothing():
  pre = {"a": _prefix("a"), "b": _prefix("b", task_id=2)}
  jd = [
      _judged("a", ["leaks"], [_verdict(0, leak=0)]),
      _judged("b", ["dumps"], [_verdict(9, cal=0)]),
  ]
  rows, stats = B.build_rows(pre, jd, judge_rubric.DEFAULT_MIN_DIMS)
  assert not rows
  assert stats[judge_rubric.DROP_ALL_VETOED] == 1
  assert stats[judge_rubric.DROP_BELOW_FLOOR] == 1


def test_a_judged_row_without_its_prefix_is_counted_not_crashed():
  """Judged and prefix files are separate artifacts and can be re-generated
  independently; a missing prefix is a bookkeeping fact, not a reason to lose
  the whole build."""
  jd = [_judged("ghost", ["r"], [_verdict(12)])]
  rows, stats = B.build_rows({}, jd, None)
  assert not rows and stats["missing_prefix"] == 1


def test_duplicate_prefix_ids_raise():
  """Two rows for one prefix would double-count that turn in the loss."""
  pre = {"p1": _prefix("p1")}
  jd = [_judged("p1", ["r"], [_verdict(12)]), _judged("p1", ["r2"], [_verdict(12)])]
  with pytest.raises(ValueError, match="duplicate prefix_id"):
    B.build_rows(pre, jd, None)


def test_an_empty_selected_target_raises():
  """An empty assistant turn trains the sim to say nothing at all."""
  pre = {"p1": _prefix("p1")}
  jd = [_judged("p1", ["   "], [_verdict(12)])]
  with pytest.raises(ValueError, match="empty target"):
    B.build_rows(pre, jd, None)


def test_task_split_is_stable_and_hash_based():
  """The split must NOT move when the collected slice grows.

  An index-based split reshuffles which tasks are held out every time more
  tasks are collected, which silently makes each run's val number
  incomparable with the last.
  """
  first = {t: B.task_split(t, 0.5) for t in range(200)}
  # ...simulate collecting 200 more tasks and re-deriving the split.
  second = {t: B.task_split(t, 0.5) for t in range(400)}
  assert all(first[t] == second[t] for t in first)
  n_val = sum(1 for t in range(400) if second[t] == "val")
  assert 150 < n_val < 250, n_val  # ~50%, loose enough not to be flaky
  assert all(B.task_split(t, 0.0) == "train" for t in range(50))


def test_min_dims_parsing_including_the_two_special_forms():
  assert B.parse_min_dims(None) == dict(judge_rubric.DEFAULT_MIN_DIMS)
  assert B.parse_min_dims("") is None  # ranking-only
  assert B.parse_min_dims("calibration=2,in_character=3") == {
      "calibration": 2,
      "in_character": 3,
  }
  for bad in ("calibration", "calibration=x"):
    with pytest.raises(ValueError):
      B.parse_min_dims(bad)


def test_cli_refuses_mixed_rubrics_and_an_undersized_val_split():
  """Two rubric versions in one dataset make it mean two things at once, and a
  val split below the SFT batch size reports val/loss = nan -- which reads as a
  broken run rather than as a too-small split. Both must fail loudly."""
  with tempfile.TemporaryDirectory() as d:
    pfile = os.path.join(d, "prefixes.jsonl")
    with open(pfile, "w") as f:
      for i in range(4):
        f.write(json.dumps(_prefix(f"p{i}", task_id=i)) + "\n")

    mixed = os.path.join(d, "judged_mixed.jsonl")
    with open(mixed, "w") as f:
      f.write(json.dumps(_judged("p0", ["r"], [_verdict(12)])) + "\n")
      f.write(
          json.dumps(
              _judged("p1", ["r"], [_verdict(12)], rubric_version="r1")
          )
          + "\n"
      )
    with pytest.raises(SystemExit, match="mix rubric versions"):
      B.main([
          "--prefixes", pfile, "--judged", mixed,
          "--out_dir", os.path.join(d, "out"),
      ])

    same = os.path.join(d, "judged.jsonl")
    with open(same, "w") as f:
      for i in range(4):
        f.write(json.dumps(_judged(f"p{i}", ["r"], [_verdict(12)])) + "\n")
    with pytest.raises(SystemExit, match="below --val_min"):
      B.main([
          "--prefixes", pfile, "--judged", same,
          "--out_dir", os.path.join(d, "out"),
          "--val_frac", "0.25", "--val_min", "64",
      ])


def test_cli_refuses_a_target_that_still_contains_code():
  """The one thing this pipeline exists to remove. If a target with code
  reaches here, stage 1 and the judge disagree, and writing the file would
  train the simulator to emit code."""
  with tempfile.TemporaryDirectory() as d:
    pfile = os.path.join(d, "prefixes.jsonl")
    with open(pfile, "w") as f:
      f.write(json.dumps(_prefix("p0")) + "\n")
    jfile = os.path.join(d, "judged.jsonl")
    with open(jfile, "w") as f:
      # The judge scored it clean; detect_code_leak does not agree.
      f.write(
          json.dumps(
              _judged("p0", ["sure: def f(x, y): return x + y"], [_verdict(12)])
          )
          + "\n"
      )
    with pytest.raises(SystemExit, match="still contain code"):
      B.main([
          "--prefixes", pfile, "--judged", jfile,
          "--out_dir", os.path.join(d, "out"),
          "--val_frac", "0", "--val_min", "0",
      ])


# ── --all_tied: every candidate at the group max, not just one ───────────────

def test_all_tied_emits_one_row_per_tied_candidate():
  """The tie is the judge saying it cannot separate these replies, so imitate
  all of them. Measured on the 85-prefix grounded pilot: 168 rows instead of
  76, with prefix/task/turn coverage and the [TERMINATE] count all unchanged.
  """
  pre = {"p1": _prefix("p1")}
  jd = [_judged(
      "p1",
      ["the cutoff is ten", "ten is the cutoff", "a clearly worse reply"],
      [_verdict(12), _verdict(12), _verdict(9, cal=3)],
  )]
  rows, stats = B.build_rows(pre, jd, judge_rubric.DEFAULT_MIN_DIMS)
  assert stats["KEPT"] == 2              # both 12s, not the 9
  assert stats["prefixes_kept"] == 1     # the keep RATE is still per prefix
  assert {r["messages"][2]["content"] for r in rows} == {
      "the cutoff is ten", "ten is the cutoff"}
  assert sorted(r["cand_idx"] for r in rows) == [0, 1]
  # ...and the old behaviour is still one row, the same one `select` returns.
  one, st1 = B.build_rows(pre, jd, judge_rubric.DEFAULT_MIN_DIMS, all_tied=False)
  assert st1["KEPT"] == 1
  assert one[0]["messages"][2]["content"] == rows[0]["messages"][2]["content"]


def test_all_tied_drops_near_duplicate_paraphrases():
  """Duplicate rows would silently reweight the loss toward whichever prefix
  produced the most paraphrases, which is a sampling artefact, not data."""
  pre = {"p1": _prefix("p1")}
  dup = "The cutoff is ten, and below that it subtracts instead."
  jd = [_judged("p1", [dup, dup + " ", "Ten is the cutoff."],
                [_verdict(12), _verdict(12), _verdict(12)])]
  rows, _ = B.build_rows(pre, jd, judge_rubric.DEFAULT_MIN_DIMS)
  assert len(rows) == 2                            # the near-dup is gone
  keep_all, _ = B.build_rows(
      pre, jd, judge_rubric.DEFAULT_MIN_DIMS, near_dup_ratio=1.0)
  assert len(keep_all) == 3                        # ...unless asked to keep it


def test_all_tied_never_promotes_a_vetoed_or_sub_floor_candidate():
  """A tie is taken only among candidates that already cleared the vetoes and
  the floors -- a vetoed draw has `scores[i] is None` and can never tie."""
  pre = {"p1": _prefix("p1")}
  jd = [_judged(
      "p1",
      ["clean reply", "vetoed draw", "floor failure"],
      [_verdict(12), None, _verdict(12, cal=0)],
  )]
  rows, stats = B.build_rows(pre, jd, {"calibration": 2})
  assert stats["KEPT"] == 1
  assert rows[0]["messages"][2]["content"] == "clean reply"


def test_select_all_returns_the_same_drop_reason_as_select():
  """A dropped prefix must be attributed identically either way, or the two
  builds' drop tallies stop being comparable."""
  jd = _judged("p1", ["x"], [None])
  idx, why1 = judge_rubric.select(jd)
  idxs, why2 = judge_rubric.select_all(jd)
  assert idx is None and idxs == []
  assert why1 == why2


def test_max_tokens_drops_rows_the_sft_dataset_would_reject():
  """verl's MultiTurnSFTDataset RAISES above `max_length` instead of
  truncating, so one over-long row kills the run. The filter is OPT-IN so the
  builder stays tokenizer-free by default."""

  class _Tok:
    """Stands in for a tokenizer: length = total content characters."""

    def apply_chat_template(self, messages, tokenize=True):
      del tokenize
      return list("".join(m["content"] for m in messages))

  pre = {"p1": _prefix("p1"), "p2": _prefix("p2", task_id=2)}
  jd = [
      _judged("p1", ["short"], [_verdict(12)]),
      _judged("p2", ["x" * 10_000], [_verdict(12)]),
  ]
  # Without the filter both survive, and the long one would blow up in verl.
  rows, _ = B.build_rows(pre, jd, None)
  assert len(rows) == 2
  # With it, only the short one is written, and the drop is ATTRIBUTED.
  rows, stats = B.build_rows(pre, jd, None, max_tokens=5_000, tokenizer=_Tok())
  assert len(rows) == 1
  assert rows[0]["prefix_id"] == "p1"
  assert stats["over_max_tokens"] == 1

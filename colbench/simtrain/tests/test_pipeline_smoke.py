"""End-to-end CPU smoke test of the Stage 1 -> 3 CLIs, on stubbed endpoints.

Runs the actual ``main()`` of every script against a three-row parquet with
``ChatEndpoint.chat`` monkeypatched, so the things unit tests cannot reach --
argument wiring, file naming, the prefix/candidate/judged join, the sha16
consistency checks, and resume -- are exercised without a server, a GPU or a
dollar.

Run: pytest colbench/simtrain/tests/test_pipeline_smoke.py
"""

import json
import os

import pandas as pd
import pytest

from colbench import prompts
from colbench.selfplay import llm_client
from colbench.selfplay.dataio import read_jsonl
from colbench.simtrain import collect_candidates
from colbench.simtrain import collect_prefixes
from colbench.simtrain import dump_judged
from colbench.simtrain import judge_candidates
from colbench.simtrain import judge_rubric

GT = "def f(x, y):\n    return x + y if x >= 10 else x - y\n"
SUBMISSION = "```python\ndef f(x, y):\n    return x + y\n```"


def _parquet(path, n=3):
  rows = []
  for i in range(n):
    rows.append({
        "data_source": "colbench",
        "prompt": [
            {"role": "system", "content": prompts.COLBENCH_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a python function f{i}(x, y)."},
        ],
        "ability": "code",
        "reward_model": {"style": "rule"},
        "extra_info": {
            "split": "train",
            "index": i,
            "ground_truth": {
                "problem_description": f"Create a python function f{i}(x, y).",
                "ground_truth": GT,
                "test_cases": ["f(1, 2)"],
            },
        },
    })
  pd.DataFrame(rows).to_parquet(path)


class _Router:
  """One stub for every role, dispatching on the message shape."""

  def __init__(self):
    self.solver_calls = 0
    self.sim_calls = 0
    self.judge_calls = 0
    self.truth_calls = 0

  def chat(self, endpoint, messages):
    del endpoint
    system = messages[0]["content"]
    if system == prompts.SIM_JUDGE_SYSTEM_PROMPT:
      self.judge_calls += 1
      return self._judge(messages[1]["content"])
    if system == prompts.SIM_TRUTH_SYSTEM_PROMPT:
      # r4 stage 2. Passes everything, so the smoke test still exercises the
      # ranking path; the veto itself is unit-tested in test_judge_rubric.
      self.truth_calls += 1
      return self._truth(messages[1]["content"])
    if system == prompts.SIM_SYSTEM_PROMPT:
      self.sim_calls += 1
      # Two distinct replies, so dedupe has something to collapse and the judge
      # has something to rank.
      return (
          "Ten is the cutoff, above it you add."
          if self.sim_calls % 2
          else "I would rather not say."
      )
    # The partner assistant: ask twice, then submit.
    self.solver_calls += 1
    return "What decides the branch?" if self.solver_calls % 3 else SUBMISSION

  def _labels(self, body):
    return [
        line.split()[-2]
        for line in body.split("\n")
        if line.startswith("--- Candidate ")
    ]

  def _truth(self, body):
    return json.dumps({
        "checks": [
            {"label": l, "verdict": "ok", "quote": "", "code": ""}
            for l in self._labels(body)
        ]
    })

  def _judge(self, body):
    labels = self._labels(body)
    verdicts = []
    for i, label in enumerate(labels):
      verdicts.append({
          "label": label,
          "code_leak": 1,
          "volunteering": 4 - i,
          "in_character": 4,
          "calibration": 4,
          "note": f"scored {label}",
      })
    return json.dumps({"verdicts": verdicts, "best_label": labels[0]})


@pytest.fixture
def wired(tmp_path, monkeypatch):
  monkeypatch.setenv("SIM_CHAR_LIMIT", "0")
  router = _Router()
  monkeypatch.setattr(
      llm_client.ChatEndpoint,
      "chat",
      lambda self, messages: router.chat(self, messages),
  )
  data = str(tmp_path / "train.fence.parquet")
  _parquet(data)
  return router, data, tmp_path


def _run_collect(data, out_dir, end=3):
  collect_prefixes.main([
      "--data_file", data, "--start", "0", "--end", str(end),
      "--out_dir", str(out_dir), "--tag", "smoke",
      "--max_assistant_turns", "4", "--concurrency", "2",
      "--solver_base_url", "http://x/v1", "--solver_model", "partner",
      "--sim_base_url", "http://y/v1", "--sim_model", "sim",
  ])


def test_full_pipeline_and_resume(wired):
  router, data, tmp_path = wired
  out_dir = tmp_path / "run"
  _run_collect(data, out_dir)

  prefix_file = str(out_dir / "prefixes.smoke.jsonl")
  episode_file = str(out_dir / "episodes.smoke.jsonl")
  prefixes = collect_prefixes.read_prefixes(prefix_file)
  assert prefixes, "the collector produced no prefixes"
  assert len(read_jsonl(episode_file)) == 3
  for rec in prefixes:
    assert GT in rec["sim_user"]
    assert rec["sim_user_sha16"] == collect_prefixes.sha16(rec["sim_user"])
    assert all(GT not in m["content"] for m in rec["sim_dialogue"])
    assert rec["episode_terminated_by"] in (
        "submit", "turn_cap", "sim_empty", "solver_empty"
    )

  # Resume is a no-op: every episode is already accounted for.
  before = router.solver_calls
  _run_collect(data, out_dir)
  assert router.solver_calls == before
  assert len(read_jsonl(episode_file)) == 3

  # ── Stage 3a ──
  cand_file = str(out_dir / "candidates.smoke.jsonl")
  collect_candidates.main([
      "--prefixes", prefix_file, "--out", cand_file, "--k", "4",
      "--concurrency", "2",
      "--sim_base_url", "http://y/v1", "--sim_model", "sim",
  ])
  cands = read_jsonl(cand_file)
  assert len(cands) == len(prefixes)
  assert all(c["n_draws"] == 4 for c in cands)
  assert all(1 <= c["n_unique"] <= 2 for c in cands)
  assert sum(c["dup_counts"][0] for c in cands) > len(cands), "dedupe did nothing"

  before = router.sim_calls
  collect_candidates.main([
      "--prefixes", prefix_file, "--out", cand_file, "--k", "4",
      "--sim_base_url", "http://y/v1", "--sim_model", "sim",
  ])
  assert router.sim_calls == before, "resume re-drew candidates"

  # ── Stage 3b ──
  judged_file = str(out_dir / "judged.smoke.jsonl")
  judge_candidates.main([
      "--prefixes", prefix_file, "--candidates", cand_file,
      "--out", judged_file, "--judge_model", "stub-judge",
      "--judge_api_key", "EMPTY", "--concurrency", "2",
  ])
  judged = read_jsonl(judged_file)
  assert len(judged) == len(prefixes)
  assert all(r["ok"] for r in judged), [
      r.get("parse_error") for r in judged if not r["ok"]
  ]
  assert all(r["rubric_version"] == prompts.JUDGE_RUBRIC_VERSION for r in judged)
  # The stub gives candidate-at-label-A the top score, so selection must land on
  # whichever candidate that label maps to -- the mis-attribution check, end to
  # end this time.
  for rec in judged:
    chosen, reason = judge_rubric.select(rec, min_total=12, min_dim=2)
    if chosen is not None:
      assert reason == ""
      assert rec["candidates"][chosen]

  before = router.judge_calls
  judge_candidates.main([
      "--prefixes", prefix_file, "--candidates", cand_file,
      "--out", judged_file, "--judge_model", "stub-judge",
      "--judge_api_key", "EMPTY",
  ])
  assert router.judge_calls == before, "resume re-paid for judged rows"

  # ── The dump ──
  dump_file = str(out_dir / "dump.txt")
  dump_judged.main([
      "--prefixes", prefix_file, "--judged", judged_file, "--out", dump_file,
  ])
  text = open(dump_file, encoding="utf-8").read()
  assert "SUMMARY over" in text
  assert "judge veto vs detect_code_leak" in text
  assert "Ten is the cutoff, above it you add." in text
  # Slicing partitions the pages without overlap.
  parts = []
  for i in (1, 2):
    p = str(out_dir / f"part{i}.txt")
    dump_judged.main([
        "--prefixes", prefix_file, "--judged", judged_file,
        "--slice", f"{i}/2", "--out", p,
    ])
    parts.append(open(p, encoding="utf-8").read())
  ids = [r["prefix_id"] for r in judged]
  for pid in ids:
    assert sum(f"prefix {pid} " in part for part in parts) == 1


def test_candidate_prefix_mismatch_is_fatal(wired, monkeypatch):
  _, data, tmp_path = wired
  out_dir = tmp_path / "run"
  _run_collect(data, out_dir)
  prefix_file = str(out_dir / "prefixes.smoke.jsonl")
  cand_file = str(out_dir / "candidates.smoke.jsonl")
  collect_candidates.main([
      "--prefixes", prefix_file, "--out", cand_file, "--k", "2",
      "--sim_base_url", "http://y/v1", "--sim_model", "sim",
  ])
  # Simulate a candidates file drawn from a different prefix rendering.
  rows = read_jsonl(cand_file)
  rows[0]["sim_user_sha16"] = "0" * 16
  with open(cand_file, "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r) + "\n")
  with pytest.raises(SystemExit):
    judge_candidates.main([
        "--prefixes", prefix_file, "--candidates", cand_file,
        "--out", str(out_dir / "j.jsonl"), "--judge_model", "stub",
        "--judge_api_key", "EMPTY",
    ])


def test_collector_refuses_to_run_without_sim_char_limit(wired, monkeypatch):
  _, data, tmp_path = wired
  monkeypatch.delenv("SIM_CHAR_LIMIT", raising=False)
  with pytest.raises(SystemExit):
    _run_collect(data, tmp_path / "run2")
  assert not os.path.exists(str(tmp_path / "run2" / "prefixes.smoke.jsonl"))

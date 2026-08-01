"""Shared row reading + JSONL cache helpers for the Phase-0 spec scripts.

``read_tasks`` normalizes a ColBench parquet row (either the raw InfoPO source
schema or our preprocessed schema) into ``{index, problem_description,
ground_truth, test_cases}`` -- the minimal task payload both ``generate_specs``
and ``diagnose_specs`` need. Test-case extraction for the raw schema mirrors
``colbench.preprocess_colbench._extract_test_cases`` (inlined below so this
Phase-0 subpackage stays free of that module's absl CLI dependency).
"""

import json
import logging
import os
from typing import Any
from typing import Iterator
from typing import Optional

import pandas as pd

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_test_cases(extra_info: dict[str, Any]) -> list[str]:
  """Non-None call-strings from the raw InfoPO nested tools_kwargs payload.

  Kept byte-identical to ``colbench.preprocess_colbench._extract_test_cases``;
  duplicated here only to avoid importing that module (its top-level absl import
  is CLI-only).

  Args:
    extra_info: a raw InfoPO row's ``extra_info``.

  Returns:
    The non-None call-strings, in payload order.
  """
  tools_kwargs = (extra_info or {}).get("tools_kwargs", {}) or {}
  create_kwargs = (tools_kwargs.get("interact_with_env", {}) or {}).get(
      "create_kwargs", {}
  ) or {}
  task = create_kwargs.get("task", {}) or {}
  test_cases = task.get("test_cases", {}) or {}
  return [str(v) for v in test_cases.values() if v is not None]


def _resolve_gt(row) -> dict[str, Any]:
  """Return the task ground_truth dict from either schema.

  Preprocessed rows carry a ready dict at ``extra_info.ground_truth`` /
  ``reward_model.ground_truth`` (with ``problem_description``, ``ground_truth``
  source, and ``test_cases``). Raw InfoPO rows carry
  ``reward_model.{problem_description, ground_truth}`` and nest test_cases under
  ``extra_info.tools_kwargs`` -- extracted here.

  Args:
    row: one parquet row, in either the raw or the preprocessed schema.

  Returns:
    ``{problem_description, ground_truth, test_cases}``, normalised to the
    preprocessed shape whichever schema the row came in.
  """
  extra_info = row.get("extra_info", {}) or {}
  rm = row.get("reward_model", {}) or {}
  gt = extra_info.get("ground_truth")
  if gt is None:
    gt = rm.get("ground_truth")
  # Preprocessed schema: gt is a dict with all three fields.
  if isinstance(gt, dict):
    _tc = gt.get("test_cases")
    return {
        "problem_description": gt.get(
            "problem_description", rm.get("problem_description", "")
        ),
        "ground_truth": gt.get("ground_truth", ""),
        "test_cases": list(_tc) if _tc is not None else [],
    }
  # Raw InfoPO schema: gt is the GT source string; test_cases nested in
  # extra_info.
  return {
      "problem_description": rm.get("problem_description", ""),
      "ground_truth": gt if isinstance(gt, str) else rm.get("ground_truth", ""),
      "test_cases": _extract_test_cases(extra_info),
  }


def read_tasks(
    data_file: str, max_rows: Optional[int] = None
) -> list[dict[str, Any]]:
  """Load tasks from a parquet, normalized to the minimal payload.

  Index = row position.

  Args:
    data_file: parquet to load.
    max_rows: keep only the first N rows; None loads all.

  Returns:
    One ``{index, problem_description, ground_truth, test_cases}`` per row, with
    ``index`` equal to the row position.
  """
  df = pd.read_parquet(os.path.expanduser(data_file))
  if max_rows is not None:
    df = df.iloc[:max_rows]
  tasks = []
  for i, (_, row) in enumerate(df.iterrows()):
    t = _resolve_gt(row)
    t["index"] = i
    tasks.append(t)
  return tasks


def read_jsonl(path: str) -> list[dict[str, Any]]:
  """Read a JSONL file into a list of dicts.

  An absent file yields an empty list.

  Tolerates a TRUNCATED FINAL LINE: a hard kill (walltime, OOM, ``scancel``) can
  interrupt an append mid-write, and for a metered generation run this file is
  the resume state -- refusing to read it would strand the run. A corrupt line
  anywhere EARLIER still raises, since that means real damage rather than an
  interrupted append.

  Args:
    path: the JSONL file; a missing path is not an error.

  Returns:
    One dict per line, dropping a truncated final line.
  """
  path = os.path.expanduser(path)
  if not os.path.exists(path):
    return []
  out = []
  lines = [ln.strip() for ln in open(path, encoding="utf-8")]
  for i, line in enumerate(lines):
    if not line:
      continue
    try:
      out.append(json.loads(line))
    except json.JSONDecodeError:
      if i == len(lines) - 1:
        logger.warning(
            "[selfplay] %s: dropping truncated final line (interrupted "
            "append); it will be re-authored on resume.",
            path,
        )
        continue
      raise
  return out


def existing_indices(path: str) -> set[int]:
  """Indices already present in a JSONL cache (for resumable generation).

  Args:
    path: the JSONL cache to scan; a missing file reads as empty.

  Returns:
    The ``index`` of every record that has one, so a resumed run can skip
    them.
  """
  return {r["index"] for r in read_jsonl(path) if "index" in r}


def append_jsonl(path: str, records: Iterator[dict[str, Any]]) -> None:
  """Append records to a JSONL file, creating parent dirs as needed.

  Args:
    path: destination JSONL; ``~`` is expanded and parent dirs are created.
    records: records to append, one JSON object per line.
  """
  path = os.path.expanduser(path)
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  with open(path, "a", encoding="utf-8") as f:
    for r in records:
      f.write(json.dumps(r) + "\n")

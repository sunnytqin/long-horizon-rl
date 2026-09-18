#!/usr/bin/env python3
"""Parse the per-step metric lines out of colbench GRPO slurm logs.

Offline, stdlib-only. Reads the `.out` files a run produced and writes one tidy
JSON series per run under `series/`, so the figures stay reproducible after the
logs age out of netscratch (~90d atime retention).

WHY THIS IS NOT A ONE-LINE grep
-------------------------------
1. Chained jobs OVERLAP, and the overlap is a RE-ROLL, not a continuation. A
   `--chain` resume starts from the last checkpoint, so if job A wrote steps
   0-223 and resumed at step 200, job B's steps 201-223 are a *fresh rollout*
   of those steps on a different RNG stream. They are close but not equal
   (measured on spec_simsft_s2: H 0.310 vs 0.332 at step 220). Mixing the two
   splices two trajectories together. Rule here: the LATER job wins, because
   its branch is the one that continues into the rest of the trace.
2. A step's metrics can arrive on MORE THAN ONE line (train metrics and
   `val-core/*` are emitted separately, and val only every `test_freq` steps),
   so per-step dicts must be merged, not replaced.
3. Lines are wrapped in Ray's ANSI colour codes and prefixed with
   `(TaskRunnerV1 pid=NNNNN)`, so the record does not start at column 0.
4. Metric NAMES contain `:` (`val-core/colbench_spec_local/reward/mean@1`), so
   the key/value split has to be on the LAST colon, not the first.

Usage:
    python3 colbench/collapse/parse_runs.py                    # all known runs
    python3 colbench/collapse/parse_runs.py spec_simsft_s2     # just one
"""

from __future__ import annotations

import json
import os
import re
import sys

# exp_name -> glob for its slurm stdout logs. exp_name is the `--exp_name` the
# launcher was given; the log basename is `cb_<exp_name>_<jobid>.out`.
SLURM_LOGS = os.environ.get(
    "SLURM_LOGS", "/n/netscratch/barak_lab/Lab/sqin/slurm_logs"
)
# Runs, in the order they should be read. The key is the `--exp_name`; logs are
# located by the EXACT basename `cb_<exp_name>_<jobid>.out`. Do NOT glob on
# `cb_<exp_name>_*` -- exp names are prefixes of each other
# (`spec_simsft_s2` vs `spec_simsft_s2_ent003`) and a prefix glob silently
# splices a different experiment's steps into this one's series.
RUNS = (
    "spec_simsft_s2",
    "spec_simsft_s2_ent003",
    "spec_ent003_replay900_diag2",
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SERIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "series")


def _parse_one_log(path: str) -> dict[int, dict[str, float]]:
    """Every `step:N - k:v - k:v ...` record in one log, keyed by step."""
    out: dict[int, dict[str, float]] = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if "step:" not in line:
                continue
            line = _ANSI.sub("", line)
            body = line[line.index("step:") :]
            rec: dict[str, float] = {}
            for field in body.split(" - "):
                field = field.strip()
                if ":" not in field:
                    continue
                key, _, val = field.rpartition(":")  # names contain ':'
                try:
                    rec[key] = float(val)
                except ValueError:
                    pass  # non-numeric (timing strings etc.)
            if "step" in rec:
                out.setdefault(int(rec["step"]), {}).update(rec)
    return out


def _logs_for(exp_name: str) -> list[str]:
    """This run's logs, oldest job first. Exact match on the job-id suffix."""
    pat = re.compile(rf"^cb_{re.escape(exp_name)}_(\d+)\.out$")
    hits = []
    for name in os.listdir(SLURM_LOGS):
        m = pat.match(name)
        if m:
            hits.append((int(m.group(1)), os.path.join(SLURM_LOGS, name)))
    return [path for _, path in sorted(hits)]


def parse_run(exp_name: str) -> list[dict[str, float]]:
    """Stitch a run's chained logs into one step-ordered series."""
    logs = _logs_for(exp_name)
    if not logs:
        raise SystemExit(f"{exp_name}: no cb_{exp_name}_<jobid>.out in {SLURM_LOGS}")
    merged: dict[int, dict[str, float]] = {}
    for path in logs:  # oldest job first => later job wins the overlap
        per_job = _parse_one_log(path)
        if not per_job:
            print(f"    {os.path.basename(path)}: no step lines (skipped)")
            continue
        overlap = sorted(set(merged) & set(per_job))
        for step, rec in per_job.items():
            merged.setdefault(step, {}).update(rec)
        note = f", re-rolled {overlap[0]}-{overlap[-1]}" if overlap else ""
        print(
            f"    {os.path.basename(path)}: steps "
            f"{min(per_job)}-{max(per_job)}{note}"
        )
    return [merged[s] for s in sorted(merged)]


def main(argv: list[str]) -> None:
    wanted = argv[1:] or list(RUNS)
    for exp_name in wanted:
        if exp_name not in RUNS:
            raise SystemExit(f"unknown run {exp_name!r}; known: {list(RUNS)}")
        print(f"{exp_name}:")
        rows = parse_run(exp_name)
        steps = [int(r["step"]) for r in rows]
        gaps = sorted(set(range(min(steps), max(steps) + 1)) - set(steps))
        # Which steps carry the turn-bucket diagnostics -- they only exist from
        # whenever the run was (re)started with the instrumented trainer, so a
        # figure must not interpolate across the boundary.
        diag = [s for s, r in zip(steps, rows) if "actor/entropy_turn0" in r]
        dest = os.path.join(_SERIES_DIR, f"{exp_name}.json")
        with open(dest, "w") as fh:
            json.dump(rows, fh)
        print(
            f"    -> {len(rows)} steps {min(steps)}-{max(steps)}, "
            f"{len(gaps)} gaps, turn-buckets from step "
            f"{min(diag) if diag else None}\n"
            f"    -> {dest} ({os.path.getsize(dest) / 1e3:.0f} kB)"
        )


if __name__ == "__main__":
    main(sys.argv)

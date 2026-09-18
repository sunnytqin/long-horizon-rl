#!/bin/bash
# Judge a candidate file against an ALREADY-RUNNING sim server. CPU only.
#
# The rubric-iteration loop. A `--mode judge` job pays ~15 minutes of 235B load
# for ~70 seconds of judging, which is the wrong shape when the thing being
# iterated is the RUBRIC TEXT. So: start one long-lived server
#
#   bash slurm_setup/launch_eval_slurm.sh --mode serve --exp_name g1_iterate \
#     --partition kempner_h100 --time 02:00:00
#
# and then run THIS after every rubric edit. Each pass is ~1 minute.
#
# THE VERSION IS THE FILENAME, AND THAT IS LOAD-BEARING. The output name is
# derived from the rubric version, and `judge_candidates` resumes by prefix_id,
# so re-judging after a reword WITHOUT bumping the version finds every row
# already present and silently does nothing -- you would read the OLD scores and
# conclude the reword changed nothing. Bump GROUNDED_JUDGE_RUBRIC_VERSION in the
# same edit that rewords the rubric. This script refuses a no-op pass rather
# than letting it look like a result.
#
# Usage:
#   PREFIXES=... CANDIDATES=... bash colbench/simtrain/judge_against_server.sh
# Options: ARM (gt|grounded, default grounded), MAX_PREFIXES (default 50),
#          RUN_ROOT (default: the newest serve run), CONCURRENCY (default 16).
set -euo pipefail

REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
CONDA_SH=${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf}
# shellcheck source=../../slurm_setup/paths.sh
source "$REPO/slurm_setup/paths.sh"

ARM=${ARM:-grounded}
MAX_PREFIXES=${MAX_PREFIXES:-50}
CONCURRENCY=${CONCURRENCY:-16}
: "${PREFIXES:?set PREFIXES=/path/to/prefixes.*.jsonl}"
: "${CANDIDATES:?set CANDIDATES=/path/to/candidates.*.jsonl}"
[ -s "$PREFIXES" ]   || { echo "[judge] ERROR: no prefixes at $PREFIXES"; exit 1; }
[ -s "$CANDIDATES" ] || { echo "[judge] ERROR: no candidates at $CANDIDATES"; exit 1; }

# The newest serve run, unless told otherwise. Its sentinel carries the URL.
if [ -z "${RUN_ROOT:-}" ]; then
  RUN_ROOT=$(ls -dt "$RUN_ROOT_BASE"/colbench_spec_eval/*/*/ 2>/dev/null \
             | while read -r d; do [ -n "$(ls "$d"sim_url.*.txt 2>/dev/null)" ] && echo "$d" && break; done)
fi
[ -n "${RUN_ROOT:-}" ] || { echo "[judge] ERROR: no serve run with a published endpoint. Start one with --mode serve."; exit 1; }
SENTINEL=$(ls -t "$RUN_ROOT"/sim_url.*.txt 2>/dev/null | head -1)
[ -n "$SENTINEL" ] || { echo "[judge] ERROR: no sentinel under $RUN_ROOT (server still loading?)"; exit 1; }
BASE_URL=$(tr -d '[:space:]' < "$SENTINEL")
[ -n "$BASE_URL" ] || { echo "[judge] ERROR: empty sentinel $SENTINEL"; exit 1; }

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1

# The served alias IS the judge identity recorded on every row, so read it off
# the server rather than assuming: a mismatch would be rejected by the resume
# guard anyway, but with a confusing message.
MODEL=$(curl -sf --max-time 10 "$BASE_URL/models" | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])') \
  || { echo "[judge] ERROR: $BASE_URL is not reachable from this host."; exit 1; }

RUBRIC=$(python -c "from colbench.simtrain.judge_candidates import rubric_version_for; print(rubric_version_for('$ARM'))")
TAG=$(basename "$CANDIDATES" .jsonl); TAG=${TAG#candidates.}
SUFFIX=$([ "$MAX_PREFIXES" -gt 0 ] && echo ".pilot${MAX_PREFIXES}" || echo "")
OUT=${OUT:-$(dirname "$CANDIDATES")/judged.${TAG}.${RUBRIC}.${MODEL}${SUFFIX}.jsonl}
DUMP=${DUMP:-${OUT%.jsonl}.dump.txt}

echo "[judge] endpoint  $BASE_URL  (model $MODEL)"
echo "[judge] arm=$ARM rubric=$RUBRIC max_prefixes=$MAX_PREFIXES"
echo "[judge] -> $OUT"

# Refuse the silent no-op described at the top.
if [ -s "$OUT" ]; then
  _have=$(wc -l < "$OUT")
  _want=$([ "$MAX_PREFIXES" -gt 0 ] && echo "$MAX_PREFIXES" || wc -l < "$CANDIDATES")
  if [ "$_have" -ge "$_want" ]; then
    echo "[judge] ERROR: $OUT already holds $_have rows -- this pass would judge NOTHING."
    echo "[judge]   Reworded the rubric? BUMP prompts.GROUNDED_JUDGE_RUBRIC_VERSION (now $RUBRIC)"
    echo "[judge]   so the reword writes its own file and stays comparable with this one."
    echo "[judge]   Re-reading the same scores? They are already in $DUMP."
    exit 1
  fi
fi

python -m colbench.simtrain.judge_candidates \
  --prefixes "$PREFIXES" --candidates "$CANDIDATES" --out "$OUT" \
  --arm "$ARM" \
  ${MAX_PREFIXES:+$([ "$MAX_PREFIXES" -gt 0 ] && echo --max_prefixes "$MAX_PREFIXES" || true)} \
  --judge_model "$MODEL" --judge_base_url "$BASE_URL" \
  --judge_vendor vllm --judge_api_key EMPTY \
  --temperature 0.0 --concurrency "$CONCURRENCY" --timeout 900

python -m colbench.simtrain.dump_judged \
  --prefixes "$PREFIXES" --judged "$OUT" --out "$DUMP"

echo "[judge] dump -> $DUMP"
echo "[judge] READ IT, then reword the rubric AND bump the version together."

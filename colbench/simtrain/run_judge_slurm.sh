#!/bin/bash
# Judge the collected candidates against the rubric. CPU ONLY -- no GPU, no local server.
#
# Two modes, and the DEFAULT IS THE PILOT because that is the decision point:
#   MAX_PREFIXES=50 (default)  ~50 calls, ~$0.60. Judge a small batch, dump it readably, READ it,
#                              reword the rubric, re-judge the SAME batch. Three or four rounds
#                              in an afternoon before anything scales.
#   MAX_PREFIXES=0             the bulk pass (~1,800 calls, ~$16). Only after the pilot reads
#                              like it means something.
#
# Judged output is keyed by JUDGE_RUBRIC_VERSION, so a reworded rubric writes a NEW file instead
# of mixing two scoring standards in one dataset. Bump prompts.JUDGE_RUBRIC_VERSION in the same
# edit that rewords the rubric -- the filename is derived from it.
#
# PAID-RUN SAFETY is inherited wholesale from selfplay/run_specs_openai_slurm.sh: resumable, a
# failed CALL is never persisted (so a resume re-pays only for gaps), bounded retries with
# jittered backoff honouring Retry-After, fatal errors abort instead of repeating a doomed call,
# a hard per-pass MAX_COST, and mop-up passes for rows deferred by transient errors.
#
# Submit:  RUN_TAG=gs200 sbatch verl/colbench/simtrain/run_judge_slurm.sh
# Bulk:    RUN_TAG=gs200 MAX_PREFIXES=0 MAX_COST=40 sbatch ...
#
#SBATCH -p shared
#SBATCH --account=dam_lab
#SBATCH -c 4
#SBATCH -t 0-06:00
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH --job-name=simtrain_judge
#SBATCH -o /n/home05/sqin/long-horizon-RL/verl/colbench/simtrain/slurm_out/judge-%j.out
#SBATCH -e /n/home05/sqin/long-horizon-RL/verl/colbench/simtrain/slurm_out/judge-%j.out

set -euo pipefail

REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
CONDA_SH=${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf}

# shellcheck source=../../slurm_setup/paths.sh
source "$REPO/slurm_setup/paths.sh"

RUN_TAG=${RUN_TAG:-gs200}
OUT_DIR=${OUT_DIR:-$SIMTRAIN_ROOT/$RUN_TAG}
TAG=${TAG:-train.fence.c1}
PREFIX_FILE=${PREFIX_FILE:-$OUT_DIR/prefixes.${TAG}.jsonl}
CAND_FILE=${CAND_FILE:-$OUT_DIR/candidates.${TAG}.jsonl}

JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.4-mini}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-https://api.openai.com/v1}
GEN_API_KEY_FILE=${GEN_API_KEY_FILE:-$HOME/.openai_key}
# Temperature 0: the rubric asks for absolute anchored scores, so run-to-run sampling noise in
# the SCORES is pure loss. Candidate ORDER is still permuted per prefix (judge_rubric.permute).
TEMPERATURE=${TEMPERATURE:-0.0}
CONCURRENCY=${CONCURRENCY:-16}
PERM_SEED=${PERM_SEED:-0}
MAX_PREFIXES=${MAX_PREFIXES:-50}           # 0 = the full bulk pass
RETRIES=${RETRIES:-8}
SERVICE_TIER=${SERVICE_TIER:-}
if [ "$SERVICE_TIER" = "flex" ]; then TIMEOUT=${TIMEOUT:-900}; else TIMEOUT=${TIMEOUT:-600}; fi
MAX_PASSES=${MAX_PASSES:-3}

# USD per 1M tokens. VERIFY against the pricing page before a bulk run: MAX_COST is only as
# accurate as these two numbers, and HALVE them for SERVICE_TIER=flex.
PRICE_IN=${PRICE_IN:-0.25}
PRICE_OUT=${PRICE_OUT:-2.00}
MAX_COST=${MAX_COST:-5}                    # per pass; the pilot should cost ~$0.60

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1

# The rubric version names the output file, so it comes from the source of truth rather than
# being retyped here -- retyping it is how two rubrics end up in one file.
RUBRIC=$(python -c 'from colbench import prompts; print(prompts.JUDGE_RUBRIC_VERSION)')
SUFFIX=$([ "$MAX_PREFIXES" -gt 0 ] && echo ".pilot${MAX_PREFIXES}" || echo "")
JUDGED_FILE=${JUDGED_FILE:-$OUT_DIR/judged.${TAG}.${RUBRIC}${SUFFIX}.jsonl}
DUMP_FILE=${DUMP_FILE:-$OUT_DIR/dump.${TAG}.${RUBRIC}${SUFFIX}.txt}

echo "[harness] node=$(hostname)  rubric=$RUBRIC  judge=$JUDGE_MODEL"
echo "[harness] prefixes=$PREFIX_FILE"
echo "[harness] candidates=$CAND_FILE"
echo "[harness] -> $JUDGED_FILE"
echo "[harness] max_prefixes=$MAX_PREFIXES (0 = all)  cap=\$$MAX_COST per pass  tier=${SERVICE_TIER:-default}"
[ -s "$PREFIX_FILE" ] || { echo "[harness] ERROR: no prefixes at $PREFIX_FILE"; exit 1; }
[ -s "$CAND_FILE" ]   || { echo "[harness] ERROR: no candidates at $CAND_FILE"; exit 1; }
[ -s "$GEN_API_KEY_FILE" ] || { echo "[harness] ERROR: key file $GEN_API_KEY_FILE missing/empty"; exit 1; }

# An empty array rather than an inline conditional: MAX_PREFIXES=0 must pass NO flag at all,
# and word-splitting a conditionally-empty string under `set -u` is how that breaks silently.
LIMIT=()
[ "$MAX_PREFIXES" -gt 0 ] && LIMIT=(--max_prefixes "$MAX_PREFIXES")

for PASS in $(seq 1 "$MAX_PASSES"); do
  BEFORE=$( [ -f "$JUDGED_FILE" ] && wc -l < "$JUDGED_FILE" || echo 0 )
  echo "[harness] === pass $PASS/$MAX_PASSES (rows on disk: $BEFORE) ==="
  RC=0
  python -m colbench.simtrain.judge_candidates \
    --prefixes "$PREFIX_FILE" --candidates "$CAND_FILE" --out "$JUDGED_FILE" \
    "${LIMIT[@]}" \
    --judge_model "$JUDGE_MODEL" --judge_base_url "$JUDGE_BASE_URL" \
    --judge_vendor openai --judge_api_key_file "$GEN_API_KEY_FILE" \
    --temperature "$TEMPERATURE" --perm_seed "$PERM_SEED" \
    --concurrency "$CONCURRENCY" --retries "$RETRIES" --timeout "$TIMEOUT" \
    ${SERVICE_TIER:+--service_tier "$SERVICE_TIER"} \
    --price_in "$PRICE_IN" --price_out "$PRICE_OUT" --max_cost "$MAX_COST" || RC=$?
  AFTER=$( [ -f "$JUDGED_FILE" ] && wc -l < "$JUDGED_FILE" || echo 0 )
  # Exit 4 = spend cap. Each pass has its OWN counter, so continuing would re-spend the cap.
  [ "$RC" = "4" ] && { echo "[harness] pass $PASS: $BEFORE -> $AFTER rows, then the SPEND CAP -> stopping."; break; }
  # Exit 3 = non-retryable (bad key, no credits, unknown model). Further passes are pointless.
  [ "$RC" = "3" ] && { echo "[harness] pass $PASS: $BEFORE -> $AFTER rows, then a FATAL API error -> stopping."; break; }
  echo "[harness] pass $PASS: $BEFORE -> $AFTER rows"
  [ "$AFTER" -le "$BEFORE" ] && { echo "[harness] no progress this pass -- stopping early"; break; }
done

# ── The readable artifact. THIS is the deliverable of a pilot pass. ───────────
python -m colbench.simtrain.dump_judged \
  --prefixes "$PREFIX_FILE" --judged "$JUDGED_FILE" --out "$DUMP_FILE"

echo "[harness] DONE. judged -> $JUDGED_FILE"
echo "[harness]       dump   -> $DUMP_FILE"
echo "[harness] READ THE DUMP before spending the bulk budget. Fan it out with"
echo "[harness]   python -m colbench.simtrain.dump_judged --prefixes $PREFIX_FILE \\"
echo "[harness]     --judged $JUDGED_FILE --slice i/4 --out part_i.txt"
echo "[harness] and start with the cases the judge and the regex disagree about:"
echo "[harness]   ... --only leak_disagree"

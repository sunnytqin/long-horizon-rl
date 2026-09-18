#!/bin/bash
# Merge one FSDP actor checkpoint into a servable HF directory.
#
#   SRC=<.../global_step_N/actor> DST=<hf dir> sbatch merge_partner.sbatch
#
# NOT serial_requeue: a 90 s preemption mid-merge leaves *.safetensors on disk with no
# model.safetensors.index.json and no tokenizer, and vLLM starts on that and serves nonsense.
#SBATCH -p shared,sapphire
#SBATCH -A dam_lab
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-01:30
#SBATCH --job-name=merge_partner
#SBATCH -o /n/home05/sqin/long-horizon-RL/verl/colbench/simtrain/slurm_out/merge-%j.out
#SBATCH -e /n/home05/sqin/long-horizon-RL/verl/colbench/simtrain/slurm_out/merge-%j.out
set -euo pipefail

REPO=${REPO:-/n/home05/sqin/long-horizon-RL/verl}
CONDA_SH=${CONDA_SH:-/n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/n/holylabs/LABS/dam_lab/Lab/sqin/envs/openrlhf}
: "${SRC:?set SRC to the .../global_step_N/actor dir}"
: "${DST:?set DST to the target HF dir}"

echo "[merge] node=$(hostname -s)"
echo "[merge] src=$SRC"
echo "[merge] dst=$DST"

# conda's MKL activate.d is not `set -u` clean (MKL_INTERFACE_LAYER: unbound variable).
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"; conda activate "$CONDA_ENV"
set -u
export PYTHONPATH="$REPO"

python -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DST"

# A merge that died partway still leaves shards behind, so check the index and the tokenizer,
# not just "some weights exist".
n=$(find "$DST" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) | wc -l)
[ "$n" -gt 0 ]                            || { echo "[merge] ERROR: no weight files"; exit 1; }
[ -f "$DST/config.json" ]                 || { echo "[merge] ERROR: no config.json"; exit 1; }
[ -f "$DST/tokenizer_config.json" ]       || { echo "[merge] ERROR: no tokenizer"; exit 1; }
if [ "$n" -gt 1 ]; then
  [ -f "$DST/model.safetensors.index.json" ] || { echo "[merge] ERROR: sharded but no index.json"; exit 1; }
fi
echo "[merge] OK  $n weight file(s)  $(du -sh "$DST" | cut -f1)"
ls -la "$DST"

#!/bin/bash
#SBATCH --job-name=x20_smkR256
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=340G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk_res256_%j.out
#SBATCH --error=logs/smk_res256_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6 X20_ANOMALY=1
echo "start $(date)"
python -u scripts/run_phase1.py --config configs/ablations/lb7/smk_frozenRES256.yaml --device cpu 2>&1 \
  | grep -E "frozen_go|latent_dim|Parameters|Phase|Epoch|epoch|pert|recon|nan|NaN|Error|Traceback|anomaly|Function|done"
echo "done $(date)"

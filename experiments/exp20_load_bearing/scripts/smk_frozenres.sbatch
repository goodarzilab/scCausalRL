#!/bin/bash
#SBATCH --job-name=x20_smkRES
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=340G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk_res_%j.out
#SBATCH --error=logs/smk_res_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "start $(date)"
python -u scripts/run_phase1.py --config configs/ablations/lb7/smk_frozenRES64.yaml --device cpu 2>&1 \
  | grep -E "frozen_go|latent_dim|Epoch|epoch|pert=|recon=|nan|NaN|Error|Traceback|Parameters|DAG|Phase"
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_smkfz
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smkfz_%j.out
#SBATCH --error=logs/smkfz_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
python -u scripts/run_phase1.py --config configs/ablations/lb2smk/frozenD256_smk.yaml --device cpu 2>&1 \
  | grep -iE "frozen_go|Model:|latent_dim|Epoch |dag_ramp|nan|error|traceback|Phase 1 complete" | head -20
echo "done $(date)"

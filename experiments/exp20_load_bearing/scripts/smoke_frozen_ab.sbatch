#!/bin/bash
#SBATCH --job-name=x20_smkab
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=3:00:00
#SBATCH --output=logs/smkab_%j.out
#SBATCH --error=logs/smkab_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for CFG in frozenD256_ridge_smk frozenD64_smk; do
  echo "############## SMOKE $CFG ##############"
  python -u scripts/run_phase1.py --config configs/ablations/lb2smk/$CFG.yaml --device cpu 2>&1 \
    | grep -iE "frozen_go|latent_dim|Epoch |dag_ramp|pert=|nan|error|traceback|Phase 1 complete" | head -14 || echo "FAILED $CFG"
done
echo "done $(date)"

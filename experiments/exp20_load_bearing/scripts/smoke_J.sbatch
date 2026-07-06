#!/bin/bash
#SBATCH --job-name=x20_smkJ
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smkJ_%j.out
#SBATCH --error=logs/smkJ_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
python -u scripts/run_phase1.py --config configs/ablations/lb5smk/J_hardself_k4.yaml --device cpu 2>&1 \
  | grep -iE "hard-self-gate|Model:|Epoch |pert=|oracle|nan|error|traceback|Phase 1 complete" | head -14
echo "done $(date)"

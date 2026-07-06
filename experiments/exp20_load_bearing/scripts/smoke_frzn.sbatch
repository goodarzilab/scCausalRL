#!/bin/bash
#SBATCH --job-name=x20_smkfn
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smkfn_%j.out
#SBATCH --error=logs/smkfn_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
python -u scripts/run_phase1.py --config configs/ablations/lb2smk/frozenD64_nonlin_smk.yaml --device cpu 2>&1 \
  | grep -iE "frozen_go|scm_nonlinear|Model:|Epoch |pert=|nan|error|traceback|Phase 1 complete" | head -14
echo "done $(date)"

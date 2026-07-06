#!/bin/bash
#SBATCH --job-name=x20_synR4
#SBATCH --partition=cpu_batch,cpu,cpu_high_mem,cpu_batch_high_mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/synR4_%j.out
#SBATCH --error=logs/synR4_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=4
echo "start $(date) on $(hostname)"
echo "### R4 (learnable decoder, NO frozen) = the arch that engaged A on Joung; sweep L0 to force gate sparse ###"
for SCM in linear nonlinear; do
  for L0 in 0.02 0.1 0.3 1.0; do
    python -u evaluation/synthetic_scm_poc.py --scm $SCM --decoder learn --l0 $L0 --epochs 800 2>&1 | grep -E "RESULT" || echo "FAILED $SCM l0=$L0"
  done
done
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_synth
#SBATCH --partition=cpu_batch,cpu,cpu_high_mem,cpu_batch_high_mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/synth_%j.out
#SBATCH --error=logs/synth_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=4
echo "start $(date) on $(hostname)"
for SCM in linear nonlinear; do
  for DEC in frozen learn; do
    python -u evaluation/synthetic_scm_poc.py --scm $SCM --decoder $DEC --epochs 600 2>&1 | grep -E "===|RESULT" || echo "FAILED $SCM $DEC"
  done
done
echo "done $(date)"

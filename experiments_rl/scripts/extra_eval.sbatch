#!/bin/bash
#SBATCH --job-name=x20_extra
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/extra_eval_%j.out
#SBATCH --error=logs/extra_eval_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for GRID in x20po x20arch; do
  echo "############## $GRID ##############"
  python -u analysis/extra_eval.py --grid $GRID --device cpu 2>&1 | tail -25 || echo "FAILED $GRID"
done
echo "done $(date)"

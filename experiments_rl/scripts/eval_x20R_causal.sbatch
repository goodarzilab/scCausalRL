#!/bin/bash
#SBATCH --job-name=x20R_causal
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_x20R_causal_%j.out
#SBATCH --error=logs/eval_x20R_causal_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
python -u analysis/grn_dag_sweep.py --grid x20R --device cpu 2>&1 | tail -18 || echo "GRN FAILED"
python -u analysis/extra_eval.py --grid x20R --device cpu 2>&1 | tail -14 || echo "EXTRA FAILED"
echo "done $(date)"

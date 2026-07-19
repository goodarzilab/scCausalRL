#!/bin/bash
#SBATCH --job-name=x20fn_causal
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_frzn_causal_%j.out
#SBATCH --error=logs/eval_frzn_causal_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
python -u analysis/grn_dag_sweep.py --grid x20frz --device cpu 2>&1 | tail -12 || echo GRN_FAIL
python -u analysis/extra_eval.py --grid x20frz --device cpu 2>&1 | tail -10 || echo EXTRA_FAIL
echo "done $(date)"

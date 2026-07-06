#!/bin/bash
#SBATCH --job-name=x20arch_grn
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_grn_x20arch_%j.out
#SBATCH --error=logs/eval_grn_x20arch_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
echo "###### grn_dag_sweep x20arch (singles+combos drift-removed + A real/zero/shuf) ######"
python -u analysis/grn_dag_sweep.py --grid x20arch --device cpu 2>&1 | tail -40 || echo "GRN FAILED"
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20po_grn
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_grn_x20po_%j.out
#SBATCH --error=logs/eval_grn_x20po_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
echo "###### grn_dag_sweep x20po (GRN-bio + A real/zero/shuf, PRIOR OFF) ######"
python -u analysis/grn_dag_sweep.py --grid x20po --device cpu 2>&1 | tail -40 || echo "GRN FAILED"
echo "###### component_ablation x20po ######"
python -u analysis/component_ablation.py --grid x20po --device cpu 2>&1 | tail -30 || echo "COMPABL FAILED"
echo "done $(date)"

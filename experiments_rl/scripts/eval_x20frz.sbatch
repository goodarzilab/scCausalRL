#!/bin/bash
#SBATCH --job-name=x20frz_cb
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_x20frz_%j.out
#SBATCH --error=logs/eval_x20frz_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
echo "###### CAUSAL: grn_dag_sweep x20frz (S_Agap drift-removed + CollecTRI) ######"
python -u analysis/grn_dag_sweep.py --grid x20frz --device cpu 2>&1 | tail -25 || echo "GRN FAILED"
echo "###### RAW A_gap + BIOLOGY (fate-hierarchy) x20frz ######"
python -u analysis/extra_eval.py --grid x20frz --device cpu 2>&1 | tail -20 || echo "EXTRA FAILED"
echo "done $(date)"

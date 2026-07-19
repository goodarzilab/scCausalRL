#!/bin/bash
#SBATCH --job-name=x20_bigtable
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --output=logs/bigtable_%j.out
#SBATCH --error=logs/bigtable_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/eval_bigtable.py --device cpu 2>&1 | grep -vE "load_joung|Loaded|Warning|warn|UserWarning"
echo "done $(date)"

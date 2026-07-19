#!/bin/bash
#SBATCH --job-name=x20_lceval
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=logs/lceval_%j.out
#SBATCH --error=logs/lceval_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/eval_learning_curve.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "fin $(date)"

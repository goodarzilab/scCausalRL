#!/bin/bash
#SBATCH --job-name=x20_seedstab
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G
#SBATCH --time=1:30:00
#SBATCH --output=logs/seedstab_%j.out
#SBATCH --error=logs/seedstab_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
python -u evaluation/diag_seed_stability.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "fin $(date)"

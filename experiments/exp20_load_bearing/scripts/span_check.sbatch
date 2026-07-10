#!/bin/bash
#SBATCH --job-name=x20_span
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=logs/span_%j.out
#SBATCH --error=logs/span_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
python -u evaluation/frozen_span_check.py --device cpu 2>&1 | grep -vE "^\[|Warning|warn"

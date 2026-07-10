#!/bin/bash
#SBATCH --job-name=x20_fulltable
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=340G
#SBATCH --time=6:00:00
#SBATCH --output=logs/fulltable_%j.out
#SBATCH --error=logs/fulltable_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date)"
python -u evaluation/full_table.py --device cpu 2>&1 | grep -vE "Warning|warn|Downcasting|ImplicitModification"
echo "done $(date)"

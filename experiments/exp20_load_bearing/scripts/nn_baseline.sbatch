#!/bin/bash
#SBATCH --job-name=x20_nnbase
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=1:30:00
#SBATCH --output=logs/nnbase_%j.out
#SBATCH --error=logs/nnbase_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
python -u evaluation/nn_baseline.py 2>&1 | grep -vE "Warning|warn|Downcasting|ImplicitMod"

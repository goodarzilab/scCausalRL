#!/bin/bash
#SBATCH --job-name=x20_procv2
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=4:00:00
#SBATCH --output=logs/procv2_%j.out
#SBATCH --error=logs/procv2_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/build_proc_v2.py 2>&1 | grep -vE "Warning|warn|Downcasting|ImplicitMod"

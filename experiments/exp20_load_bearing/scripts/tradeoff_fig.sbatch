#!/bin/bash
#SBATCH --job-name=x20_tradefig
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=logs/tradefig_%j.out
#SBATCH --error=logs/tradefig_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
python -u evaluation/make_tradeoff_figure.py --device cpu 2>&1 | grep -vE "Warning|warn|Downcasting|ImplicitMod"

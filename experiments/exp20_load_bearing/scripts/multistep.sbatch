#!/bin/bash
#SBATCH --job-name=x20_multistep
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=logs/multistep_%j.out
#SBATCH --error=logs/multistep_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/briggs/multistep_probe.py --T 5 --n_ctrl 64 --n_combos 1500 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"

#!/bin/bash
#SBATCH --job-name=joung_featprobe
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=2:00:00
#SBATCH --output=logs/featprobe_%j.out
#SBATCH --error=logs/featprobe_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u evaluation/feature_probe.py --tag epoch80
echo "done $(date)"

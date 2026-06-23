#!/bin/bash
#SBATCH --job-name=joung_celltypist
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=logs/celltypist_%j.out
#SBATCH --error=logs/celltypist_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
mkdir -p logs
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u analysis/human_dev_annotation/try_celltypist.py
echo "done $(date)"

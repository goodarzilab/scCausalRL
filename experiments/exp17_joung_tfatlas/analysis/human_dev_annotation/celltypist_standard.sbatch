#!/bin/bash
#SBATCH --job-name=joung_ct_std
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=logs/ct_std_%j.out
#SBATCH --error=logs/ct_std_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
mkdir -p logs
OMP_NUM_THREADS=8 python -u analysis/human_dev_annotation/celltypist_standard.py

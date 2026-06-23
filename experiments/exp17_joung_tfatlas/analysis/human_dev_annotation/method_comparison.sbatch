#!/bin/bash
#SBATCH --job-name=joung_method_cmp
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=logs/method_cmp_%j.out
#SBATCH --error=logs/method_cmp_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
OMP_NUM_THREADS=8 python -u analysis/human_dev_annotation/method_comparison.py

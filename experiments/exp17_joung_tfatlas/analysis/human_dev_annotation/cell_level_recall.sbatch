#!/bin/bash
#SBATCH --job-name=joung_recall_cmp
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=1:30:00
#SBATCH --output=logs/recall_cmp_%j.out
#SBATCH --error=logs/recall_cmp_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
OMP_NUM_THREADS=8 python -u analysis/human_dev_annotation/cell_level_recall.py

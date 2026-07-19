#!/bin/bash
#SBATCH --job-name=x20_bestboth
#SBATCH --partition=gpu,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/bestboth_%j.out
#SBATCH --error=logs/bestboth_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO"
echo "start $(date) on $(hostname)"
python -u scripts/run_phase1.py \
  --config configs/ablations/lb25bestboth/v2b_deg3_t25_frozen_r32_all40.yaml \
  --device cuda 2>&1 | grep -ivE "warn|deprecat"
echo "done $(date)"

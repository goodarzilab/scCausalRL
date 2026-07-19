#!/bin/bash
#SBATCH --job-name=x20_briggs_pred
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/briggs_pred_%j.out
#SBATCH --error=logs/briggs_pred_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/briggs/predict_combo.py \
  --ckpt results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt \
  --answer results/briggs/observed_combo_targets.csv \
  --device cpu --n_ctrl 256 2>&1 | grep -vE "Warning|warn|verbose"
echo "fin $(date)"

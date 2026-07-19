#!/bin/bash
#SBATCH --job-name=x20_briggs_seed
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=logs/briggs_seed_%j.out
#SBATCH --error=logs/briggs_seed_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/briggs/sweep_fidelity.py \
  --runs_file configs/briggs_seedcheck_tags.txt \
  --answer results/briggs/observed_combo_targets.csv \
  --max_k 4 --n_ctrl 128 --device cpu 2>&1 | grep -vE "Warning|warn|verbose"
echo "fin $(date)"

#!/bin/bash
#SBATCH --job-name=exp16_crosscell
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=3:00:00
#SBATCH --output=logs/crosscell_%j.out
#SBATCH --error=logs/crosscell_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp16_multicellline
mkdir -p logs
CKPT="${CKPT:-results/grid/exp16_multicellline/checkpoints/phase1_epoch80.pt}"
echo "start $(date) on $(hostname)  ckpt=$CKPT"
python -u evaluation/run_crosscell_eval.py --checkpoint "$CKPT" \
    --output_dir results/grid/exp16_multicellline/crosscell --device cuda
echo "done $(date)"

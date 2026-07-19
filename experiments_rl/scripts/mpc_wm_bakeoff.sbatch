#!/bin/bash
#SBATCH --job-name=wm_bakeoff
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:30:00
#SBATCH --output=logs/wm_bakeoff_%j.out
#SBATCH --error=logs/wm_bakeoff_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
echo "start $(date) on $(hostname)"
python -u analysis/mpc_wm_bakeoff.py
echo "done $(date)"

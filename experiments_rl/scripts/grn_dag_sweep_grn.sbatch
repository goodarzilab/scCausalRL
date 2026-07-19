#!/bin/bash
#SBATCH --job-name=grn_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=04:00:00
#SBATCH --output=logs/grn_eval_%j.out
#SBATCH --error=logs/grn_eval_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
echo "start $(date) on $(hostname)"
python -u analysis/grn_dag_sweep.py --grid grn
echo "done $(date)"

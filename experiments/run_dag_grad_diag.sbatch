#!/bin/bash
#SBATCH --job-name=dag_grad_diag
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=00:30:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp2_shift_linear
python -u evaluation/dag_grad_diagnostic.py --max_cells 20000

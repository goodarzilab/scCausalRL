#!/bin/bash
#SBATCH --job-name=joung_esm_embed
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/esm_embed_%j.out
#SBATCH --error=logs/esm_embed_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u utils/build_esm_embeddings.py
echo "done $(date)"; ls -la data/joung_esm2_emb.npz

#!/bin/bash
#SBATCH --job-name=x20_briggs_tgt
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=340G
#SBATCH --time=3:00:00
#SBATCH --output=logs/briggs_tgt_%j.out
#SBATCH --error=logs/briggs_tgt_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/briggs/reconstruct_targets.py --h5ad /large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad --min_cells 3 2>&1 | grep -vE "Warning|warn|verbose"
echo "fin $(date)"

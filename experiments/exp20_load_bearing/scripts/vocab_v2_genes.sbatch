#!/bin/bash
#SBATCH --job-name=x20_vocabgenes
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=4:00:00
#SBATCH --output=logs/vocabgenes_%j.out
#SBATCH --error=logs/vocabgenes_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/build_vocab_v2_genes.py --n_hvg 4000 --subsample 200000 2>&1 | grep -vE "Warning|warn|Downcasting|ImplicitMod"

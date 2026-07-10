#!/bin/bash
#SBATCH --job-name=x20_vocabgen
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=2:00:00
#SBATCH --output=logs/vocabgen_%j.out
#SBATCH --error=logs/vocabgen_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
python -u evaluation/vocab_generalizability.py 2>&1 | grep -vE "Warning|warn|Downcasting|ImplicitMod"

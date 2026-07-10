#!/bin/bash
#SBATCH --job-name=x20_smksw
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=6
#SBATCH --mem=220G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk_sw_%x_%j.out
#SBATCH --error=logs/smk_sw_%x_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
python -u scripts/run_phase1.py --config "$CFG" --device cpu 2>&1 \
  | grep -E "frozen_go|latent_dim|Model|Parameters|Phase|Epoch|Best DAG|nan|NaN|Error|Traceback|done"

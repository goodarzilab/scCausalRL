#!/bin/bash
#SBATCH --job-name=x20_smkft
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=6
#SBATCH --mem=220G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk_ft_%x_%j.out
#SBATCH --error=logs/smk_ft_%x_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
# Keep warm-start / freeze / interaction-head diagnostics (they gate the fine-tune arms).
python -u scripts/run_phase1.py --config "$CFG" --device cpu 2>&1 \
  | grep -iE "warm|init_from|load_state|frozen|freeze|requires_grad|interaction|trainable|descriptor|keeping|pert_label|shift encoder|latent_dim|Model|Phase|Epoch|Best DAG|nan|NaN|Error|Traceback|done"

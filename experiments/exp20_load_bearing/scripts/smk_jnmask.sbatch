#!/bin/bash
#SBATCH --job-name=x20_smkJNm
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=6
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk_jnmask_%j.out
#SBATCH --error=logs/smk_jnmask_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "start $(date)"
python -u scripts/run_phase1.py --config configs/ablations/lb7v2b/smk_JN_masked_go_l3.yaml --device cpu 2>&1 \
  | grep -E "jn|masked_go|effective_tfs|latent_dim|Model|Parameters|Phase|Epoch|train_disc|Best DAG|nan|NaN|Error|Traceback|done"
echo "done $(date)"

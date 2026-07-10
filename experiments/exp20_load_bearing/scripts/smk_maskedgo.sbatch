#!/bin/bash
#SBATCH --job-name=x20_smkmask
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk_mask_%j.out
#SBATCH --error=logs/smk_mask_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "start $(date)"
python -u scripts/run_phase1.py --config configs/ablations/lb7v2/smk_v2_masked_go.yaml --device cpu 2>&1 \
  | grep -E "masked_go|load_joung|effective_tfs|latent_dim|Model|Parameters|Phase|Epoch|Best DAG|pert|recon|nan|NaN|Error|Traceback|done"
echo "done $(date)"

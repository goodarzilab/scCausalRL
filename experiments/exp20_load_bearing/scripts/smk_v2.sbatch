#!/bin/bash
#SBATCH --job-name=x20_smkv2
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk_v2_%j.out
#SBATCH --error=logs/smk_v2_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "start $(date)"
python -u scripts/run_phase1.py --config configs/ablations/lb7/smk_v2_sparse_con.yaml --device cpu 2>&1 \
  | grep -E "load_joung|gene_vocab|effective_tfs|Model|latent_dim|Parameters|Phase|Epoch|pert|recon|nan|NaN|Error|Traceback|done"
echo "done $(date)"

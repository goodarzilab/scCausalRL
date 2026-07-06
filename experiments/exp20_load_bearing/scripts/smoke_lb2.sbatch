#!/bin/bash
#SBATCH --job-name=x20_smk2
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=380G
#SBATCH --time=3:00:00
#SBATCH --output=logs/smk2_%j.out
#SBATCH --error=logs/smk2_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for CFG in struct_frozenD_whiten struct_whiten; do
  echo "############## SMOKE $CFG ##############"
  python -u scripts/run_phase1.py --config configs/ablations/lb2smk/$CFG.yaml --device cpu 2>&1 \
    | grep -iE "frozen_go|whiten target|Model:|Parameters|Δ-drift|Epoch |pert=|pert_mse|error|traceback|nan|decoder=|latent_dim" | head -30 \
    || echo "SMOKE FAILED $CFG"
done
echo "done $(date)"

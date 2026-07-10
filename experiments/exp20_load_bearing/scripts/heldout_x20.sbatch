#!/bin/bash
#SBATCH --job-name=x20_heldout
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=logs/heldout_x20_%j.out
#SBATCH --error=logs/heldout_x20_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for TAG in struct_min_esm struct_sparse_esm struct_sparse_con struct_sparse_resid struct_sparse_pertonly struct_sparse_residrecon struct_sparse_all; do
  echo "===== $TAG ====="
  python -u evaluation/heldout_zeroshot_eval.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --tag hz_x20_$TAG --device cpu 2>&1 | tail -14 || echo "FAILED $TAG"
done
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_tvh
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/tvh_%j.out
#SBATCH --error=logs/tvh_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for TAG in R2_pseudobulk R4_pseudobulk_projectout R6_R4_energy struct_sparse_con_po; do
  python -u evaluation/train_vs_heldout.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --tag $TAG 2>&1 | grep -E "===|split|train|test|GAP|-->" || echo "FAILED $TAG"
done
echo "done $(date)"

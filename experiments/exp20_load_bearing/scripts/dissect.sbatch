#!/bin/bash
#SBATCH --job-name=x20_dissect
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=6:00:00
#SBATCH --output=logs/dissect_%j.out
#SBATCH --error=logs/dissect_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date)"
echo "########## DATA: is the residual signal or noise? ##########"
python -u evaluation/objective_dissection.py --device cpu 2>&1 | grep -vE "^\[|Warning" || echo "DISSECT FAILED"
echo "########## which objective component kills train-fit? (train vs heldout disc) ##########"
for TAG in struct_sparse_con_po struct_sparse_resid_po struct_sparse_esm_po struct_min_esm_po; do
  python -u evaluation/train_vs_heldout.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --tag $TAG 2>&1 | grep -E "===|split|train |test |-->" || echo "FAIL $TAG"
done
echo "done $(date)"

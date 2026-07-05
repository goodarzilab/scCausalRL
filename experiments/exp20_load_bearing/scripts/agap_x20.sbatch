#!/bin/bash
#SBATCH --job-name=x20_agap
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/agap_x20_%j.out
#SBATCH --error=logs/agap_x20_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date) on $(hostname)"
for TAG in struct_min_esm struct_sparse_esm struct_sparse_con struct_sparse_resid struct_sparse_pertonly struct_sparse_residrecon struct_sparse_all; do
  echo "############ $TAG ############"
  python -u evaluation/agap_probe.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --n_tf 30 2>&1 | grep -E "===|gate_open|A_gap" || echo "FAILED $TAG"
done
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_probepo
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/probepo_%j.out
#SBATCH --error=logs/probepo_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date)"
for TAG in struct_min_esm_po struct_sparse_esm_po struct_sparse_con_po struct_sparse_pertonly_po; do
  echo "#### $TAG ####"
  python -u evaluation/component_probe.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --n_tf 24 2>&1 | grep -E "===|probe_|crossTF|mean"
done
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_mech
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=5:00:00
#SBATCH --output=logs/mech_x20_%j.out
#SBATCH --error=logs/mech_x20_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date) on $(hostname)"
for TAG in struct_sparse_con struct_sparse_esm struct_sparse_pertonly struct_min_esm; do
  echo "################## $TAG ##################"
  echo "----- COMPONENT PROBE -----"
  python -u evaluation/component_probe.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --n_tf 24 2>&1 | grep -E "===|probe_|crossTF|mean" || echo "PROBE FAILED $TAG"
  echo "----- GRADIENT FLOW -----"
  python -u evaluation/gradient_flow.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --n_tf 60 2>&1 | grep -E "===|encoder|shift|dag|decoder|shift_scale|share|%" || echo "GRAD FAILED $TAG"
done
echo "done $(date)"

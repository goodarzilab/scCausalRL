#!/bin/bash
#SBATCH --job-name=fate_sep
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs/fate_sep_%j.out
#SBATCH --error=logs/fate_sep_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
echo "start $(date) on $(hostname)"
for TAG in v2_esm_con_ugate v2_combo_resid; do
  CKPT=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  OMP_NUM_THREADS=8 python -u evaluation/fate_separability.py --checkpoint "$CKPT" --device cpu --tag "$TAG"
done
echo "done $(date)"

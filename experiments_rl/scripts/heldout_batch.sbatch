#!/bin/bash
#SBATCH --job-name=heldout_batch
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/heldout_batch_%j.out
#SBATCH --error=logs/heldout_batch_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
cd "$REPO"; export PYTHONPATH="$REPO:${PYTHONPATH:-}" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for ck in results/grid/grnfix_*/checkpoints/phase1_epoch60.pt; do
  tag=$(basename $(dirname $(dirname "$ck")))          # grnfix_<name>
  echo "===== $tag ====="
  python -u evaluation/heldout_zeroshot_eval.py --checkpoint "$ck" --tag "hz_$tag" --device cpu 2>&1 | tail -12 || echo "FAILED $tag"
done
echo "done $(date)"

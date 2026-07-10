#!/bin/bash
#SBATCH --job-name=of_eval
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem
#SBATCH --mem=72G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/ofeval_%j.out
#SBATCH --error=logs/ofeval_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "eval start $(date) on $(hostname)"
for TAG in of_frozenRES256_base of_frozenRES256_noL0 of_frozenRES256_pert30 of_J_hardself_k8_base of_J_hardself_k8_pert30; do
  CK="$REPO/results/grid/$TAG/checkpoints/phase1_epoch200.pt"
  echo "==================== EVAL $TAG ===================="
  if [ -f "$CK" ]; then
    python -u "$REPO/evaluation/train_vs_heldout.py" --checkpoint "$CK" --tag "$TAG" --device cpu --match_train_split || echo "EVAL_FAILED $TAG"
  else
    echo "MISSING checkpoint: $CK"; ls -la "$REPO/results/grid/$TAG/checkpoints/" 2>/dev/null || true
  fi
done
echo "eval done $(date)"

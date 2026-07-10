#!/bin/bash
#SBATCH --job-name=x20_evalv2
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=4:00:00
#SBATCH --output=logs/evalv2_%j.out
#SBATCH --error=logs/evalv2_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date)"
for TAG in v2_frozenRES64_s26 v2_frozenRES128_s26 v2_frozenRES256_s26 v2_J_hardself_k4 v2_J_hardself_k8; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  [ -f "$CK" ] || { echo "SKIP $TAG (no ckpt)"; continue; }
  echo "################## $TAG ##################"
  echo "--- train vs test disc_resid (effective-TF split; ceiling ~0.80) ---"
  python -u evaluation/train_vs_heldout.py --checkpoint "$CK" --tag "$TAG" 2>&1 | grep -E "===|split|train |test |-->|Error|Traceback" || echo "FAIL tvh $TAG"
  echo "--- A_gap ---"
  python -u evaluation/agap_probe.py --checkpoint "$CK" --n_tf 40 2>&1 | grep -E "===|A_gap|gate_open|Error|Traceback" || echo "FAIL agap $TAG"
done
echo "done $(date)"

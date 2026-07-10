#!/bin/bash
#SBATCH --job-name=x20_evalv2b
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evalv2b_%j.out
#SBATCH --error=logs/evalv2b_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date)"
for TAG in v2_sparse_con v2_masked_go v2_masked_go_l3 v2_JN0_multictx v2_JN_R4_multictx v2_JN1_multictx_hardself; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  [ -f "$CK" ] || { echo "SKIP $TAG (no epoch60 ckpt)"; continue; }
  echo "################## $TAG ##################"
  echo "--- train vs test disc_resid (effective-TF split; ceiling ~0.80) ---"
  python -u evaluation/train_vs_heldout.py --checkpoint "$CK" --tag "$TAG" 2>&1 | grep -E "===|split|train |test |-->|Error|Traceback" || echo "FAIL tvh $TAG"
  echo "--- A_gap ---"
  python -u evaluation/agap_probe.py --checkpoint "$CK" --n_tf 40 2>&1 | grep -E "===|A_gap|gate_open|Error|Traceback" || echo "FAIL agap $TAG"
done
echo "done $(date)"

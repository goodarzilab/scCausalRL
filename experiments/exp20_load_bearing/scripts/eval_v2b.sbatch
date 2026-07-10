#!/bin/bash
#SBATCH --job-name=x20_evalv2b
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/evalV2B_%j.out
#SBATCH --error=logs/evalV2B_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date)"
for TAG in v2b_sparse_con v2b_frozenRES64_s26 v2b_frozenRES128_s26 v2b_frozenRES256_s26 v2b_J_hardself_k4 v2b_J_hardself_k8 v2b_masked_go v2b_masked_go_l3 v2b_JN0_multictx v2b_JN_R4_multictx v2b_JN1_multictx_hardself; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  [ -f "$CK" ] || { echo "SKIP $TAG (no epoch60)"; continue; }
  echo "################## $TAG ##################"
  python -u evaluation/train_vs_heldout.py --checkpoint "$CK" --tag "$TAG" 2>&1 | grep -E "===|split|train |test |-->|Traceback" || echo "FAIL tvh $TAG"
  python -u evaluation/agap_probe.py --checkpoint "$CK" --n_tf 49 2>&1 | grep -E "===|A_gap|gate_open|Traceback" || echo "FAIL agap $TAG"
done
echo "done $(date)"

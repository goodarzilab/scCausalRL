#!/bin/bash
#SBATCH --job-name=x20_evalretr
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evalretr_%j.out
#SBATCH --error=logs/evalretr_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date)"
for TAG in v2b_retr_masked_k5 v2b_retr_masked_k10 v2b_retr_masked_k20 v2b_retr_masked_k50 v2b_retr_loc_k5 v2b_retr_loc_k10 v2b_retr_loc_k20 v2b_retr_loc_k50; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  [ -f "$CK" ] || { echo "SKIP $TAG"; continue; }
  echo "################## $TAG ##################"
  python -u evaluation/train_vs_heldout.py --checkpoint "$CK" --tag "$TAG" 2>&1 | grep -E "===|train |test |-->|Traceback" || echo "FAIL $TAG"
  python -u evaluation/agap_probe.py --checkpoint "$CK" --n_tf 49 2>&1 | grep -E "A_gap|gate_open" || true
done
echo "done $(date)"

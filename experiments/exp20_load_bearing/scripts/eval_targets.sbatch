#!/bin/bash
#SBATCH --job-name=x20_evaltgt
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evaltgt_%j.out
#SBATCH --error=logs/evaltgt_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date)"
for TAG in v2b_masked_tgt_delta_plus_cellrecon v2b_masked_tgt_delta_residual v2b_masked_tgt_statematch_only v2b_masked_tgt_delta_plus_mmd v2b_JN_masked_go_l3; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  [ -f "$CK" ] || { echo "SKIP $TAG"; continue; }
  echo "################## $TAG ##################"
  python -u evaluation/train_vs_heldout.py --checkpoint "$CK" --tag "$TAG" 2>&1 | grep -E "===|train |test |-->|Traceback" || echo "FAIL $TAG"
  python -u evaluation/agap_probe.py --checkpoint "$CK" --n_tf 49 2>&1 | grep -E "A_gap|gate_open" || true
done
echo "done $(date)"

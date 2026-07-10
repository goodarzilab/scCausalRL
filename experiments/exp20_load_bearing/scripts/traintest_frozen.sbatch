#!/bin/bash
#SBATCH --job-name=x20_tt_frz
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/tt_frz_%j.out
#SBATCH --error=logs/tt_frz_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date) on $(hostname)"
# can these OVERFIT train? frozen (chance on test) vs learned sparse_con (near-ceiling on test)
for TAG in struct_sparse_con struct_frozenD64 staged_frozenD64_s26 frozenD64_nonlin; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  echo "############ $TAG ############"
  [ -f "$CK" ] || { echo "no epoch60 for $TAG"; continue; }
  python -u evaluation/train_vs_heldout.py --checkpoint "$CK" --tag "$TAG" 2>&1 | grep -E "===|split|train |test |-->|Error|Traceback" || echo "FAILED $TAG"
done
echo "done $(date)"

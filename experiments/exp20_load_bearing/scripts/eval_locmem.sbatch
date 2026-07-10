#!/bin/bash
#SBATCH --job-name=x20_locmem_eval
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=logs/locmemeval_%j.out
#SBATCH --error=logs/locmemeval_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
for TAG in of_Jhs_L128_k16 of_Jhs_L256_k16 of_Jhs_L256_k32; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch200.pt
  [ -f "$CK" ] || CK=$(ls -t results/grid/$TAG/checkpoints/phase1_epoch*.pt 2>/dev/null | head -1)
  [ -f "$CK" ] || { echo "SKIP $TAG"; continue; }
  echo "### $TAG ###"
  python -u evaluation/train_vs_heldout.py --checkpoint "$CK" --tag "$TAG" --match_train_split 2>&1 | grep -E "train |test |-->" || echo "FAIL $TAG"
done

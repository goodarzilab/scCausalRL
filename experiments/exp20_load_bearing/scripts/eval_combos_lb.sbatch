#!/bin/bash
#SBATCH --job-name=x20_evalcombo
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=220G
#SBATCH --time=4:00:00
#SBATCH --output=logs/evalcombo_%j.out
#SBATCH --error=logs/evalcombo_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
for spec in "v2b_mmd1_deg2:deg2_top" "v2b_mmd1_deg2_Jcombo_ho50:deg2_Jcombo" "v2b_mmd1_deg2_JNcombo_ho50:deg2_JNcombo"; do
  TAG=${spec%%:*}; OUT=${spec##*:}; CK="results/grid/$TAG/checkpoints/phase1_epoch60.pt"
  if [ -f "$CK" ]; then
    echo "#### COMBO EVAL $TAG ####"
    python -u evaluation/combo_eval_full.py --checkpoint "$CK" --tag "$OUT" --device cpu 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
  else echo "SKIP $TAG (no ckpt)"; fi
done
echo "done $(date)"

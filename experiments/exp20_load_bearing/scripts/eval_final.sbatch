#!/bin/bash
#SBATCH --job-name=x20_evalfinal
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=220G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evalfinal_%j.out
#SBATCH --error=logs/evalfinal_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "==== FINAL RANKING (all arms incl. the last 3) $(date) ===="
python -u evaluation/agap_correctness.py 2>&1 | grep -vE "load_joung|Loaded shift|Warning|warn"
for T in v2b_mmd1_Jcombo_ho50 v2b_mmd1_JNcombo_ho50; do
  CK="results/grid/$T/checkpoints/phase1_epoch60.pt"
  if [ -f "$CK" ]; then
    echo "==== COMBO EVAL $T ===="
    python -u evaluation/combo_eval_full.py --checkpoint "$CK" --tag "${T#v2b_mmd1_}" --device cpu 2>&1 | grep -vE "load_joung|Loaded shift|Warning|warn"
  else echo "SKIP $T (no ckpt)"; fi
done
echo "==== done $(date) ===="

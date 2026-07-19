#!/bin/bash
#SBATCH --job-name=x20_evaltoday
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=240G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evaltoday_%j.out
#SBATCH --error=logs/evaltoday_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "########## SINGLES: all arms (held-out seed-42, drift-removed) $(date) ##########"
python -u evaluation/agap_correctness.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "########## SINGLES: split-seed robustness arms (own splits) ##########"
python -u evaluation/eval_seedarms.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "########## COMBOS: all combo-trained arms + top singles model ##########"
for spec in "v2b_mmd1_deg2:deg2_top(singles)" "v2b_mmd1_Jcombo_ho50:mmd1_Jcombo" "v2b_mmd1_JNcombo_ho50:mmd1_JNcombo" "v2b_mmd1_deg2_Jcombo_ho50:deg2_Jcombo" "v2b_mmd1_deg2_JNcombo_ho50:deg2_JNcombo"; do
  TAG=${spec%%:*}; OUT=${spec##*:}; CK="results/grid/$TAG/checkpoints/phase1_epoch60.pt"
  [ -f "$CK" ] && { echo "#### $TAG ####"; python -u evaluation/combo_eval_full.py --checkpoint "$CK" --tag "$OUT" --device cpu 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"; } || echo "SKIP $TAG"
done
echo "########## done $(date) ##########"

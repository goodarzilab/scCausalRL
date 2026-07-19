#!/bin/bash
#SBATCH --job-name=x20_evalmaster
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=240G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evalmaster_%j.out
#SBATCH --error=logs/evalmaster_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "##### SINGLES (all arms, @0.15) #####"; python -u evaluation/agap_correctness.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "##### SPLIT-SEED (own splits, @0.15) #####"; python -u evaluation/eval_seedarms.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "##### COMBOS (@0.15) #####"
for spec in "v2b_mmd1_Jcombo_ho50:mmd1_Jcombo" "v2b_mmd1_deg2_Jcombo_ho50:deg2_Jcombo" "v2b_mmd1_JNcombo_ho50:mmd1_JNcombo" "v2b_mmd1_deg2_JNcombo_ho50:deg2_JNcombo" "v2b_mmd1_deg2_t15_Jcombo_ho50:deg2t15_Jcombo"; do
  T=${spec%%:*}; O=${spec##*:}; CK="results/grid/$T/checkpoints/phase1_epoch60.pt"
  [ -f "$CK" ] && { echo "#### $O ####"; python -u evaluation/combo_eval_full.py --checkpoint "$CK" --tag "$O" --device cpu 2>&1 | grep -E "combo\]|do_mean|modeladd|obsadd|epistasis\]"; } || echo "SKIP $T"
done
echo "done $(date)"

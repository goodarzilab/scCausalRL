#!/bin/bash
#SBATCH --job-name=x20_evalft
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=240G
#SBATCH --time=4:00:00
#SBATCH --output=logs/evalft_%j.out
#SBATCH --error=logs/evalft_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6

echo "##### SINGLES (fine-tune arms + refs, real-A, @0.15) #####"
python -u evaluation/eval_ft.py --device cpu 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"

echo; echo "##### COMBOS (held-out, do_mean + epistasis r) #####"
for tag in v2b_ft_ixhead_frozen_fx v2b_ft_replay_fx v2b_ft_fullcombo_fx v2b_ft_combofrac0p1_fx \
           v2b_mmd1_deg2 v2b_mmd1_deg2_Jcombo_ho50; do
  CK=$(ls -1 results/grid/$tag/checkpoints/phase1_epoch*.pt 2>/dev/null \
        | sed -E 's/.*epoch([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  if [ -n "${CK:-}" ]; then
    echo "#### $tag  ($(basename "$CK")) ####"
    python -u evaluation/combo_eval_full.py --checkpoint "$CK" --tag "ft_${tag}" --device cpu 2>&1 \
      | grep -E "combo\]|held-out\]|do_sum1|do_mean|modeladd|obsadd|epistasis\]"
  else
    echo "#### $tag  (no checkpoint yet) ####"
  fi
done
echo "done $(date)"

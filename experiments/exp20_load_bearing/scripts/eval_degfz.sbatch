#!/bin/bash
#SBATCH --job-name=x20_evaldegfz
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evaldegfz_%j.out
#SBATCH --error=logs/evaldegfz_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
ARMS="v2b_mmd1_deg1_fzhead,v2b_mmd1_deg1_t25_fzhead,v2b_fthead_r32,v2b_mmd1_deg2_t15_fzhead,v2b_mmd1_deg3_fzhead,v2b_mmd1_deg3_t25_fzhead,v2b_deg5_frozen_r32,v2b_mmd1_deg5_t25_fzhead"
echo "##### SINGLES (DEG-weight frozen-head sweep, seed-42) #####"
python -u evaluation/eval_ft.py --device cpu --arms "$ARMS" 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo; echo "##### COMBOS (do_mean + r_pair, seed-42, same 11-pair split) #####"
for tag in $(echo "$ARMS" | tr ',' ' '); do
  CK=$(ls -1 results/grid/$tag/checkpoints/phase1_epoch*.pt 2>/dev/null | sed -E 's/.*epoch([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  [ -n "${CK:-}" ] && { echo "#### $tag ####"; python -u evaluation/combo_eval_full.py --checkpoint "$CK" --tag "degfz_$tag" --device cpu 2>&1 | grep -E "do_mean|obsadd|epistasis\]"; } || echo "#### $tag (no ckpt) ####"
done
echo "done $(date)"

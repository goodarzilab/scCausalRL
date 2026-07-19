#!/bin/bash
#SBATCH --job-name=x20_evallb1415
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evallb1415_%j.out
#SBATCH --error=logs/evallb1415_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
LB14="v2b_epi_l2_lo,v2b_epi_l2_hi,v2b_deg5_frozen_r32,v2b_distill_l2sp_lo,v2b_distill_l2sp_hi,v2b_dag_only_nohead,v2b_dag_head_r16"
DISC="v2b_disc_c2_t2,v2b_disc_c2_t1,v2b_disc_c4_t1,v2b_disc_esmreg_c2_t1"
DISCFT="v2b_disc_c2_t2_ftHead,v2b_disc_c2_t1_ftHead,v2b_disc_c4_t1_ftHead,v2b_disc_esmreg_c2_t1_ftHead"

echo "##### SINGLES (lb14 + lb15 + refs, @0.15, seed-42 held-out) #####"
python -u evaluation/eval_ft.py --device cpu --arms "$LB14,$DISC,$DISCFT,v2b_fthead_r32" 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"

echo; echo "##### COMBOS (do_mean + epistasis r, seed-42 held-out) #####"
for tag in $(echo "$LB14,$DISCFT,v2b_fthead_r32,v2b_mmd1_deg2_Jcombo_ho50" | tr ',' ' '); do
  CK=$(ls -1 results/grid/$tag/checkpoints/phase1_epoch*.pt 2>/dev/null | sed -E 's/.*epoch([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
  [ -n "${CK:-}" ] && { echo "#### $tag ($(basename $CK)) ####"; python -u evaluation/combo_eval_full.py --checkpoint "$CK" --tag "lb_$tag" --device cpu 2>&1 | grep -E "combo\]|held-out\]|do_mean|modeladd|obsadd|epistasis\]"; } || echo "#### $tag (no ckpt) ####"
done
echo "done $(date)"

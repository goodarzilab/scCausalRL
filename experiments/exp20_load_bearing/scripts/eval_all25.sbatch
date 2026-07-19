#!/bin/bash
#SBATCH --job-name=x20_eval25
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=240G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval25_%j.out
#SBATCH --error=logs/eval25_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
gv(){ grep -vE "load_joung|Loaded|Warning|warn"; }
echo "##### QUARTILE EPISTASIS (traditional score, allDEG@0.25) #####"; python -u evaluation/quartile_epistasis.py 2>&1 | gv
echo "##### CRACKS EPISTASIS #####"; python -u evaluation/cracks_epistasis.py 2>&1 | gv
echo "##### COMBOS @0.25 (5 arms) #####"
for spec in "v2b_mmd1_deg2:deg2_top" "v2b_mmd1_Jcombo_ho50:mmd1_Jcombo" "v2b_mmd1_JNcombo_ho50:mmd1_JNcombo" "v2b_mmd1_deg2_Jcombo_ho50:deg2_Jcombo" "v2b_mmd1_deg2_JNcombo_ho50:deg2_JNcombo"; do
  T=${spec%%:*}; O=${spec##*:}; python -u evaluation/combo_eval_full.py --checkpoint results/grid/$T/checkpoints/phase1_epoch60.pt --tag "$O" --device cpu 2>&1 | grep -E "combo\]|do_sum1|do_mean|modeladd|obsadd|epistasis\]"
done
echo "##### SINGLES table @0.25 (agap) #####"; python -u evaluation/agap_correctness.py 2>&1 | gv | tail -40
echo "##### n=27 eval @0.25 #####"; python -u evaluation/eval_lfc025.py 2>&1 | gv
echo "##### best_model + pertf + fate @0.25 #####"
python -u evaluation/best_model_figure.py 2>&1 | gv; python -u evaluation/pertf_plot.py 2>&1 | gv
python -u evaluation/tf_fate_cluster.py 2>&1 | gv; python -u evaluation/tf_fate_cluster_fine.py 2>&1 | gv
echo "done $(date)"

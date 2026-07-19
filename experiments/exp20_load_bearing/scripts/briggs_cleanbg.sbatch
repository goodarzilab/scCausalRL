#!/bin/bash
#SBATCH --job-name=x20_cleanbg
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --output=logs/cleanbg_%j.out
#SBATCH --error=logs/cleanbg_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
H=/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad
CK=results/grid/v2b_deg3_t25_frozen_r32_all40/checkpoints/phase1_epoch40.pt
echo "########## rebuild joung_ctrl answer key (pure Joung controls, matched norm) ##########"
python -u evaluation/briggs/reconstruct_targets.py --h5ad "$H" --background joung_ctrl \
  --out results/briggs/observed_combo_targets_joungctrl.csv 2>&1 | grep -vE "Warning|warn|verbose"
echo "########## rebuild hesc_joung with FIXED matched normalization (re-check poolability) ##########"
python -u evaluation/briggs/reconstruct_targets.py --h5ad "$H" --background hesc_joung \
  --out results/briggs/observed_combo_targets_hescjoung.csv 2>&1 | grep -E "poolability|axes|agreement|P1 "
echo "########## readout (target=Briggs-native rho_briggs) across backgrounds ##########"
for ANS in observed_combo_targets observed_combo_targets_hesc observed_combo_targets_joungctrl; do
  echo "===== background: $ANS ====="
  python -u evaluation/briggs/readout_head.py --ckpt "$CK" --answer results/briggs/${ANS}.csv \
    --target rho_briggs --holdout random 2>&1 | grep -E "overall|k>=7|baseline|wrote|feats"
done
echo "fin $(date)"

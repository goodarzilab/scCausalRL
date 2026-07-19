#!/bin/bash
#SBATCH --job-name=x20_tgt_hesc
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=380G
#SBATCH --time=4:00:00
#SBATCH --output=logs/tgt_hesc_%j.out
#SBATCH --error=logs/tgt_hesc_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
H=/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad
echo "########## hesc (Briggs pluripotent subset) ##########"
python -u evaluation/briggs/reconstruct_targets.py --h5ad "$H" --background hesc \
  --out results/briggs/observed_combo_targets_hesc.csv 2>&1 | grep -vE "Warning|warn|verbose"
echo "########## hesc_joung (pooled, shared genes) ##########"
python -u evaluation/briggs/reconstruct_targets.py --h5ad "$H" --background hesc_joung \
  --out results/briggs/observed_combo_targets_hescjoung.csv 2>&1 | grep -vE "Warning|warn|verbose"
echo "fin $(date)"

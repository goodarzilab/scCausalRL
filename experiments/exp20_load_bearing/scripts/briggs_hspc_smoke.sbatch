#!/bin/bash
#SBATCH --job-name=x20_briggs_hspc
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=340G
#SBATCH --time=3:00:00
#SBATCH --output=logs/briggs_hspc_%j.out
#SBATCH --error=logs/briggs_hspc_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
BDIR=/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026
DEC=$BDIR/_decompressed; mkdir -p "$DEC"
H5=$DEC/all_cells_final_v2.h5ad
if [ ! -f "$H5" ]; then echo "decompressing all_cells ($(date))..."; gunzip -c "$BDIR/GSE328457_all_cells_final_v2.h5ad.gz" > "$H5" && echo "decompressed -> $H5 ($(du -h "$H5"|cut -f1))"; fi
python -u evaluation/briggs/hspc_sim.py --h5ad "$H5" 2>&1 | grep -vE "Warning|warn|verbose"
echo "fin $(date)"

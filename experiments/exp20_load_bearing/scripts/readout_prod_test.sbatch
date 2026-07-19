#!/bin/bash
#SBATCH --job-name=x20_prodreadout
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=4:00:00
#SBATCH --output=logs/prodreadout_%j.out
#SBATCH --error=logs/prodreadout_%j.err
set -uo pipefail
E=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$E"; export PYTHONPATH="$E" OMP_NUM_THREADS=8
# Does the JOUNG-CHAMPION (prod deg8) latent also decode Briggs fate via a learned readout?
for CKTAG in v2b_prod_deg8_all40 v2b_deg3_t25_frozen_r32_all40; do
  echo "########## $CKTAG (readout on joung_ctrl clean background) ##########"
  python -u evaluation/briggs/readout_head.py \
    --ckpt results/grid/$CKTAG/checkpoints/phase1_epoch40.pt \
    --answer results/briggs/observed_combo_targets_joungctrl.csv \
    --target rho_briggs --holdout random 2>&1 | grep -E "READOUT|overall|k>=7|baseline|wrote"
done
echo "fin $(date)"

#!/bin/bash
#SBATCH --job-name=x20_briggs_readout
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/briggs_readout_%j.out
#SBATCH --error=logs/briggs_readout_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
CK=results/grid/v2b_deg3_t25_frozen_r32_s1/checkpoints/phase1_epoch40.pt
echo "=== random split ==="
python -u evaluation/briggs/readout_head.py --ckpt "$CK" --n_ctrl 128 --holdout random 2>&1 | grep -vE "Warning|warn|verbose|Convergence"
echo "=== leave-TF-out split (strict generalization) ==="
python -u evaluation/briggs/readout_head.py --ckpt "$CK" --n_ctrl 128 --holdout loto 2>&1 | grep -vE "Warning|warn|verbose|Convergence"
echo "fin $(date)"

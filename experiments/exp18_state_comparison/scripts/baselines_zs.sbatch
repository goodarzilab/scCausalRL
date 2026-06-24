#!/bin/bash
#SBATCH --job-name=zs_baselines
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs/baselines_zs_%j.out
#SBATCH --error=logs/baselines_zs_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp18_state_comparison
mkdir -p logs results/zs
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u compute_baselines.py --exp_dir ../exp16_multicellline \
  --state_hvg results/state_hvg.json --perts results/perts_zs_union.json --out_dir results/zs
echo "done $(date)"; ls -la results/zs/

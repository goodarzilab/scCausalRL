#!/bin/bash
#SBATCH --job-name=comp_abl
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/comp_abl_%x_%j.out
#SBATCH --error=logs/comp_abl_%x_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
export OMP_NUM_THREADS=16 PYTHONPATH="$PWD:/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose"
echo "start $(date) on $(hostname) | grid=${GRID:-sena}"
python -u analysis/component_ablation.py --grid "${GRID:-sena}" --device cpu
echo "done $(date)"

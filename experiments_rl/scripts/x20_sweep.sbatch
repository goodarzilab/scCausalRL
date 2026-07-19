#!/bin/bash
#SBATCH --job-name=x20_sweep
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=10:00:00
#SBATCH --output=logs/x20_sweep_%j.out
#SBATCH --error=logs/x20_sweep_%j.err
set -uo pipefail
cd /home/sgoudarzi/scCausalRL/experiments_rl
export OMP_NUM_THREADS=16
export PYTHONPATH="/home/sgoudarzi/scCausalRL/experiments_rl:/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"
echo "start $(date) on $(hostname)"
echo "===== grn_dag_sweep --grid x20 (A real/zero/shuffled + GRN biological-ness) ====="
python -u analysis/grn_dag_sweep.py --grid x20 --device cpu
echo "===== component_ablation --grid x20 (shift vs decoder vs A) ====="
python -u analysis/component_ablation.py --grid x20 --device cpu
echo "done $(date)"

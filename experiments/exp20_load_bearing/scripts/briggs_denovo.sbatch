#!/bin/bash
#SBATCH --job-name=x20_denovo
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=logs/denovo_%j.out
#SBATCH --error=logs/denovo_%j.err
set -uo pipefail
RL=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
python -u analysis/briggs_fate_mpc.py --ensemble 1 --lam 1 --support_gamma 2 \
  --kmax 7 --beam 24 --n_ctrl 256 --n_dag 8 --topN 40 --div_jaccard 0.6 \
  --out /home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/denovo_candidates.csv 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"

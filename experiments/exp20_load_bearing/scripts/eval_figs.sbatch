#!/bin/bash
#SBATCH --job-name=x20_figs
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=logs/figs_%j.out
#SBATCH --error=logs/figs_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "#### combo_vs_singles ####"; python -u evaluation/combo_singles_fig.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "#### cracks_epistasis ####"; python -u evaluation/cracks_epistasis.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "done $(date)"

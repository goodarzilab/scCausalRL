#!/bin/bash
#SBATCH --job-name=x20_deckfigs
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=logs/deckfigs_%j.out
#SBATCH --error=logs/deckfigs_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
echo "### lb9_figures ###"; python -u evaluation/lb9_figures.py 2>&1 | grep -vE "load_joung|Loaded shift|Warning|warn|RuntimeWarning|Downcasting|ImplicitMod"
echo "### epistasis_sidebyside ###"; python -u evaluation/epistasis_sidebyside.py 2>&1 | grep -vE "load_joung|Loaded shift|Warning|warn|RuntimeWarning|Downcasting|ImplicitMod"
echo "### DONE ###"; ls -la results/figures/lb9_ranking.png results/figures/disc_vs_correctness.png results/figures/epistasis_singles_vs_combo.png

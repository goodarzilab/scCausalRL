#!/bin/bash
#SBATCH --job-name=joung_preprocess
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=8:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
mkdir -p logs data
echo "start $(date) on $(hostname)"
# full preprocessing: chunk-read 172GB dense atlas → compact axis-aligned singles +
# combinatorial h5ads. ALL cells (no per-TF cap, all GFP/mCherry control).
python -u data/preprocess_joung.py --out_dir data \
    --ctrl_cells 0 --max_per_tf 0 --chunk 20000
echo "done $(date)"
ls -la data/joung_*_proc.h5ad

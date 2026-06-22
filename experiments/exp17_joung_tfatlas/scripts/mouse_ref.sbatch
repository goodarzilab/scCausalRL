#!/bin/bash
#SBATCH --job-name=mouse_ref
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mouse_ref_%j.out
#SBATCH --error=logs/mouse_ref_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
mkdir -p logs
RS=~/miniconda/envs/renv/bin/Rscript
echo "start $(date) on $(hostname)"
# Build Cao 2019 reference pseudobulks at both named tiers (38 main cell types + ~56
# named sub-trajectories). Reads the 1.9GB sparse RDS + 827MB cell_annotate once each.
$RS analysis/mouse_dev_annotation/build_mouse_reference.R Main_cell_type
$RS analysis/mouse_dev_annotation/build_mouse_reference.R Sub_trajectory_name
echo "--- NNLS deconvolution of Joung TFs ---"
python -u analysis/mouse_dev_annotation/nnls_annotate.py --ref mouse_ref_Main_cell_type
python -u analysis/mouse_dev_annotation/nnls_annotate.py --ref mouse_ref_Sub_trajectory_name
echo "done $(date)"
ls -la analysis/mouse_dev_annotation/

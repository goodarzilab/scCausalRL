#!/bin/bash
#SBATCH --job-name=cao_ct_train
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ct_train_%j.out
#SBATCH --error=logs/ct_train_%j.err
set -euo pipefail
cd /home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas
mkdir -p logs
D=/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal
[ -f $D/GSE156793_S3_gene_count.loom ] || { echo "decompressing loom..."; gunzip -kf $D/GSE156793_S3_gene_count.loom.gz; }
echo "start $(date) on $(hostname)"
OMP_NUM_THREADS=8 python -u analysis/human_dev_annotation/celltypist_custom_train.py
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_procv2b
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=4:00:00
#SBATCH --output=logs/procv2b_%j.out
#SBATCH --error=logs/procv2b_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
python -u evaluation/build_proc_v2.py --vocab /large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2/vocab_v2b_genes.txt --out /large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2/joung_singles_proc_v2b.h5ad 2>&1 | grep -vE "Warning|warn|Downcasting|ImplicitMod"

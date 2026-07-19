#!/bin/bash
#SBATCH --job-name=x20_joungfine
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/joungfine_%j.out
#SBATCH --error=logs/joungfine_%j.err
set -uo pipefail
E=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
RL=/home/sgoudarzi/scCausalRL/experiments_rl
V=/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
python -u analysis/joung_rl_ladder.py --dataset joung \
  --checkpoint $E/results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt --wm_root $E \
  --singles_path $V/joung_singles_proc_v2b.h5ad --combo_path $V/joung_combo_proc_v2b.h5ad \
  --lineage_csv $E/results/briggs/joung_tf_celltype.csv \
  --min_drivers 2 --qmax 0.15 --pool curated --device cpu --iters 5000 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"

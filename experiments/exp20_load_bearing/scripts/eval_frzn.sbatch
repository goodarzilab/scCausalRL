#!/bin/bash
#SBATCH --job-name=x20fn_eval
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_frzn_%j.out
#SBATCH --error=logs/eval_frzn_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
echo "########## HELDOUT frozenD64_nonlin ##########"
python -u evaluation/heldout_zeroshot_eval.py --checkpoint results/grid/frozenD64_nonlin/checkpoints/phase1_epoch60.pt --tag hz_frozenD64_nonlin --device cpu 2>&1 | tail -6
echo "########## PROBE frozenD64_nonlin ##########"
python -u evaluation/component_probe.py --checkpoint results/grid/frozenD64_nonlin/checkpoints/phase1_epoch60.pt --n_tf 24 2>&1 | grep -E "===|probe_|crossTF|mean"
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20R_eval
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_x20R_%j.out
#SBATCH --error=logs/eval_x20R_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for TAG in R1_perex_driftrem R2_pseudobulk R3_projectout R4_pseudobulk_projectout R5_R4_contrastive10 R6_R4_energy; do
  echo "########## HELDOUT $TAG ##########"
  python -u evaluation/heldout_zeroshot_eval.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --tag hz_$TAG --device cpu 2>&1 | tail -6 || echo "HZ FAILED $TAG"
  echo "########## PROBE $TAG (crossTF_div = THE gate metric) ##########"
  python -u evaluation/component_probe.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --n_tf 24 2>&1 | grep -E "===|probe_|crossTF|mean" || echo "PROBE FAILED $TAG"
done
echo "done $(date)"

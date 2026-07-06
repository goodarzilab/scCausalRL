#!/bin/bash
#SBATCH --job-name=x20arch_hz
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_hz_x20arch_%j.out
#SBATCH --error=logs/eval_hz_x20arch_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for TAG in struct_frozenD struct_whiten struct_frozenD_whiten staged_esm_s26 staged_esm_s36 staged_con_s26 staged_frozenD_s26 staged_nonlinSCM_s26; do
  echo "########## HELDOUT $TAG ##########"
  python -u evaluation/heldout_zeroshot_eval.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --tag hz_$TAG --device cpu 2>&1 | tail -10 || echo "HZ FAILED $TAG"
  echo "########## PROBE $TAG ##########"
  python -u evaluation/component_probe.py --checkpoint results/grid/$TAG/checkpoints/phase1_epoch60.pt --n_tf 24 2>&1 | grep -E "===|probe_|crossTF|mean" || echo "PROBE FAILED $TAG"
done
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_agapfrz
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/agap_frz_%j.out
#SBATCH --error=logs/agap_frz_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=4
echo "start $(date) on $(hostname)"
# raw-target + expressive-A arms whose A_gap was never measured:
# frozen decoder (breaks D(I-A)^-1 absorption) +/- nonlinear A +/- A-learns-first schedule + localization
for TAG in struct_frozenD64 struct_frozenD frozenD64_nonlin staged_frozenD64_s26 staged_frozenD_s26 J_hardself_k4 J_hardself_k8; do
  CK=results/grid/$TAG/checkpoints/phase1_epoch60.pt
  echo "############ $TAG ############"
  [ -f "$CK" ] || { echo "no epoch60 for $TAG"; continue; }
  python -u evaluation/agap_probe.py --checkpoint "$CK" --n_tf 40 2>&1 | grep -E "===|gate_open|A_gap|Error|Traceback" || echo "FAILED $TAG"
done
echo "done $(date)"

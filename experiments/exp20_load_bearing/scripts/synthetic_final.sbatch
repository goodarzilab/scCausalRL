#!/bin/bash
#SBATCH --job-name=x20_synf
#SBATCH --partition=cpu_batch,cpu,cpu_high_mem,cpu_batch_high_mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=logs/synf_%j.out
#SBATCH --error=logs/synf_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=4
echo "start $(date) on $(hostname)"
echo "###### (1) FINE L0 SCAN [0.3-1.0] cyclic linear (soft gate): sparsity<->prediction frontier ######"
for L0 in 0.3 0.4 0.5 0.6 0.7 0.85 1.0; do
  python -u evaluation/synthetic_scm_poc.py --scm linear --decoder learn --graph cyclic --l0 $L0 --epochs 800 2>&1 | grep -E "RESULT" || echo "FAIL L0=$L0"
done
echo "###### (2) PERTURBATION-BLIND (hard oracle target) — does A become load-bearing / recovered? ######"
for SCM in linear nonlinear; do
  python -u evaluation/synthetic_scm_poc.py --scm $SCM --decoder learn --graph cyclic --hard_target --l0 0.0 --epochs 800 2>&1 | grep -E "RESULT" || echo "FAIL hard $SCM"
done
echo "  (baseline soft-gate for contrast:)"
python -u evaluation/synthetic_scm_poc.py --scm linear --decoder learn --graph cyclic --l0 0.02 --epochs 800 2>&1 | grep -E "RESULT" || true
echo "done $(date)"

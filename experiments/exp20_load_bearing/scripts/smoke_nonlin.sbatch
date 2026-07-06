#!/bin/bash
#SBATCH --job-name=x20_smknl
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=380G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smknl_%j.out
#SBATCH --error=logs/smknl_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
python -u scripts/run_phase1.py --config configs/ablations/lb3smk/staged_nonlinSCM.yaml --device cpu 2>&1 \
  | grep -iE "Model:|latent_dim|Δ-drift|whiten|Epoch |dag_ramp|l0_ramp|pert=|error|traceback|nan|Phase 1 complete" | head -30
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=x20_smk4
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=2:00:00
#SBATCH --output=logs/smk4_%j.out
#SBATCH --error=logs/smk4_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" WANDB_MODE=disabled OMP_NUM_THREADS=8
echo "start $(date) on $(hostname)"
for CFG in R4_pseudobulk_projectout R6_R4_energy; do
  echo "############## SMOKE $CFG ##############"
  python -u scripts/run_phase1.py --config configs/ablations/lb4smk/$CFG.yaml --device cpu 2>&1 \
    | grep -iE "pseudobulk|project|Model:|Δ-drift|Epoch |pert=|energy|nan|error|traceback|Phase 1 complete" | head -16 || echo "FAILED $CFG"
done
echo "done $(date)"

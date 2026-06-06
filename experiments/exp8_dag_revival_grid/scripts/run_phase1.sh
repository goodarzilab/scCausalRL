#!/bin/bash
#SBATCH --job-name=CausalRLBioPhase1
#SBATCH --output=/home/sgoudarzi/scCausalRL/logs/phase1_train_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/logs/phase1_train_%j.err
#SBATCH --partition=goodarzilab_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ── paths ─────────────────────────────────────────────────────────────────────
REPO=/home/sgoudarzi/scCausalRL
CONFIG=${REPO}/configs/default.yaml

cd "${REPO}"
mkdir -p logs checkpoints

echo "=================================================="
echo "  scCausalRL  —  Phase 1: Causal VAE Training"
echo "  SLURM job ID : ${SLURM_JOB_ID}"
echo "  Node         : $(hostname)"
echo "  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Start        : $(date)"
echo "=================================================="

python scripts/run_phase1.py \
    --config "${CONFIG}" \
    --n_epochs 100

echo ""
echo "Phase 1 finished at $(date)"
echo "Best checkpoint: ${REPO}/checkpoints/phase1_best.pt"

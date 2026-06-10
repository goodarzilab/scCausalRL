#!/bin/bash
#SBATCH --job-name=CausalRLBioPhase2
#SBATCH --output=/home/sgoudarzi/scCausalRL/logs/phase2_train_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/logs/phase2_train_%j.err
#SBATCH --partition=goodarzilab_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

# ── environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate causalbiorld        # change to your conda env name

# ── paths ─────────────────────────────────────────────────────────────────────
REPO=/home/sgoudarzi/scCausalRL
CONFIG=${REPO}/configs/default.yaml
VAE_CKPT=${REPO}/checkpoints/phase1_best.pt

cd "${REPO}"
mkdir -p logs checkpoints

# Guard: Phase 1 must have run first
if [[ ! -f "${VAE_CKPT}" ]]; then
    echo "ERROR: Phase 1 checkpoint not found at ${VAE_CKPT}"
    echo "       Run phase 1 first:  sbatch scripts/run_phase1.sh"
    exit 1
fi

echo "=================================================="
echo "  scCausalRL  —  Phase 2: Joint VAE + PPO"
echo "  SLURM job ID : ${SLURM_JOB_ID}"
echo "  Node         : $(hostname)"
echo "  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  VAE ckpt     : ${VAE_CKPT}"
echo "  Start        : $(date)"
echo "=================================================="

python scripts/run_phase2.py \
    --config "${CONFIG}" \
    --vae_checkpoint "${VAE_CKPT}"

echo ""
echo "Phase 2 finished at $(date)"
echo "Policy checkpoint : ${REPO}/checkpoints/phase2_policy_final.pt"
echo "VAE checkpoint    : ${REPO}/checkpoints/phase2_vae_final.pt"

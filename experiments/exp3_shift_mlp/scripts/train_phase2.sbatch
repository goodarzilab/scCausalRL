#!/bin/bash
#SBATCH --job-name=cbrl_phase2
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=48:00:00
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err

set -euo pipefail
conda init
# ── Setup ────────────────────────────────────────────────────────────────────
REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs checkpoints

# Phase 1 checkpoint must exist
CKPT="checkpoints/phase1_best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $CKPT"
    echo "Run Phase 1 first: sbatch scripts/train_phase1.sbatch"
    exit 1
fi

echo "=== Phase 2: Joint VAE + PPO Co-Training ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "VAE ckpt:  $CKPT"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────
python scripts/run_phase2.py \
    --config configs/default.yaml \
    --vae_checkpoint "$CKPT" \
    --device cuda

echo ""
echo "=== Phase 2 Complete ==="
echo "End:       $(date)"
echo "Policy:    checkpoints/phase2_policy_final.pt"
echo "VAE:       checkpoints/phase2_vae_final.pt"
echo "Log:       logs/phase2_training.csv"

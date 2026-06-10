#!/bin/bash
#SBATCH --job-name=cbrl_phase1
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err

set -euo pipefail

# ── Setup ────────────────────────────────────────────────────────────────────
REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs checkpoints

echo "=== Phase 1: Causal VAE Training ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Working:   $(pwd)"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────
python scripts/run_phase1.py \
    --config configs/default.yaml \
    --device cuda

echo ""
echo "=== Phase 1 Complete ==="
echo "End:       $(date)"
echo "Checkpoint: checkpoints/phase1_best.pt"
echo "Log:        logs/phase1_training.csv"

#!/bin/bash
#SBATCH --job-name=cbrl_replogle_val
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=4:00:00
#SBATCH --output=logs/replogle_val_%j.out
#SBATCH --error=logs/replogle_val_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs results/replogle_validation

CKPT="checkpoints/phase1_best.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $CKPT"
    exit 1
fi

echo "=== Replogle External Validation ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:      $(date)"
echo "Checkpoint: $CKPT"
echo ""

python evaluation/run_replogle_validation.py \
    --checkpoint "$CKPT" \
    --replogle_path /large_storage/goodarzilab/causal_state/pretrain_data/replogle \
    --max_cells 50000 \
    --output_dir results/replogle_validation \
    --device cuda

echo ""
echo "=== Done ==="
echo "End:     $(date)"
echo "Results: results/replogle_validation/"

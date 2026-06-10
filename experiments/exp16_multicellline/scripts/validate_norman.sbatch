#!/bin/bash
#SBATCH --job-name=cbrl_norman_val
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=4:00:00
#SBATCH --output=logs/norman_val_%j.out
#SBATCH --error=logs/norman_val_%j.err

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Configurable via env vars (set before sbatch, or pass with --export=ALL,VAR=...)
#   CKPT        — path to Phase 1 checkpoint to validate
#   OUTPUT_DIR  — where to dump validation results
#
# Examples:
#   sbatch scripts/validate_norman.sbatch
#       (uses defaults: checkpoints/phase1_best.pt → results/norman_validation/)
#
#   CKPT=checkpoints/phase1_best_no_dag.pt \
#   OUTPUT_DIR=results/norman_validation_no_dag \
#       sbatch scripts/validate_norman.sbatch
# ──────────────────────────────────────────────────────────────────────────────

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"

CKPT="${CKPT:-checkpoints/phase1_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/norman_validation}"
NORMAN_PATH="${NORMAN_PATH:-/large_storage/goodarzilab/shervin/CausalBioRL/Norman/}"

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

echo "=== Norman External Validation (Replogle-trained → Norman) ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:      $(date)"
echo "Checkpoint: $CKPT"
echo "Output:     $OUTPUT_DIR"
echo "Norman:     $NORMAN_PATH"
echo ""

python evaluation/run_norman_validation.py \
    --checkpoint "$CKPT" \
    --norman_path "$NORMAN_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "=== Done ==="
echo "End:        $(date)"
echo "Results:    $OUTPUT_DIR/"

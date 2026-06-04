#!/bin/bash
#SBATCH --job-name=cbrl_p1_norman_us
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_norman_us_%j.out
#SBATCH --error=logs/phase1_norman_us_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"

OUT_DIR="results/phase1_norman_u_score"
mkdir -p "$OUT_DIR/checkpoints" "$OUT_DIR/logs" logs

echo "=== Phase 1 — Norman-only with u-score ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Output:    $OUT_DIR/"
echo "Config:    configs/norman_u_score.yaml"
echo ""
echo "Expectation: ~93k Norman cells (single + control + combo, post good_coverage),"
echo "             105 single-pert genes (84 train / 10 val / 11 test by default split),"
echo "             131 combo perts (all held out — never used in training)."
echo ""

cd "$OUT_DIR"
python -u "$REPO_DIR/scripts/run_phase1.py" \
    --config "$REPO_DIR/configs/norman_u_score.yaml" \
    --device cuda

echo ""
echo "=== Phase 1 Norman u-score run complete ==="
echo "End:        $(date)"
echo "Checkpoint: $REPO_DIR/$OUT_DIR/checkpoints/phase1_best.pt"
echo "Log:        $REPO_DIR/$OUT_DIR/logs/phase1_training.csv"

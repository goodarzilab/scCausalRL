#!/bin/bash
#SBATCH --job-name=cbrl_p1_union_hvg
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_union_hvg_%j.out
#SBATCH --error=logs/phase1_union_hvg_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"

# Same layout as the promotion run — separate output dir.
OUT_DIR="results/phase1_union_hvg"
mkdir -p "$OUT_DIR/checkpoints" "$OUT_DIR/logs" logs

echo "=== Phase 1 — Replogle with union HVG (Replogle HVGs ∪ Norman pert genes) ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Output:    $OUT_DIR/"
echo "Config:    configs/replogle_u_score_union_hvg.yaml"
echo ""
echo "Expectation: ~2086 HVGs (Replogle's 2000 + ~86 added from Norman pert set"
echo "             not already in Replogle's variance-ranked top 2000)."
echo "Downstream: Norman external eval becomes ~105/105 singles and ~131/131 combos"
echo "             (vs 19/105 and 4/131 with the original HVG axis)."
echo ""

cd "$OUT_DIR"
python -u "$REPO_DIR/scripts/run_phase1.py" \
    --config "$REPO_DIR/configs/replogle_u_score_union_hvg.yaml" \
    --device cuda

echo ""
echo "=== Phase 1 union-HVG run complete ==="
echo "End:        $(date)"
echo "Checkpoint: $REPO_DIR/$OUT_DIR/checkpoints/phase1_best.pt"
echo "Log:        $REPO_DIR/$OUT_DIR/logs/phase1_training.csv"

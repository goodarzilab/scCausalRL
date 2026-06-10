#!/bin/bash
#SBATCH --job-name=cbrl_p1_essential
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=36:00:00
#SBATCH --output=logs/phase1_essential_%j.out
#SBATCH --error=logs/phase1_essential_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"

OUT_DIR="results/phase1_essential"
mkdir -p "$OUT_DIR/checkpoints" "$OUT_DIR/logs" logs

echo "=== Phase 1 — K562 essential (raw counts, full transcriptome 8,563 genes) ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Output:    $OUT_DIR/"
echo "Config:    configs/replogle_essential.yaml"
echo ""
echo "Data:  /large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/K562_essential_raw_singlecell_01.h5ad"
echo "  - 310,385 cells (K562-only, no cell_line filter needed)"
echo "  - 8,563 genes (var_names auto-translated ENSG -> symbol by load_replogle.py)"
echo "  - 2,058 perturbations (vs train_v5's 1,371)"
echo "  - RAW counts (preprocess will normalize_total + log1p correctly; no double-log)"
echo "  - max_cells = 200,000 (same scale as promotion run for fair comparison)"
echo ""
echo "Expected vs train_v5 promotion:"
echo "  - 4× larger gene axis (8,563 vs 2,000 HVG) -> Decoder D scales 2,000x64 -> 8,563x64"
echo "  - Recon loss scales ~4x in raw magnitude (no rescaling done; lambda weights are unchanged)"
echo "  - Norman external coverage at eval: 57/105 single perts (vs 19/105 with train_v5)"
echo ""

cd "$OUT_DIR"
python -u "$REPO_DIR/scripts/run_phase1.py" \
    --config "$REPO_DIR/configs/replogle_essential.yaml" \
    --device cuda

echo ""
echo "=== Phase 1 essential run complete ==="
echo "End:        $(date)"
echo "Checkpoint: $REPO_DIR/$OUT_DIR/checkpoints/phase1_epoch150.pt  (final epoch — use this, not phase1_best.pt)"
echo "Log:        $REPO_DIR/$OUT_DIR/logs/phase1_training.csv"

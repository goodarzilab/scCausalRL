#!/bin/bash
#SBATCH --job-name=cbrl_p1_lsparse_sweep
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/p1_lsparse_sweep_%A_%a.out
#SBATCH --error=logs/p1_lsparse_sweep_%A_%a.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs results/grid_search

CELL=$SLURM_ARRAY_TASK_ID

# 8-cell sweep parallelised across slurm array indices 0..7
#   idx 0..3 — mode='product', lambda_sparse ∈ {0.1, 1.0, 10.0, 100.0}
#   idx 4..7 — mode='gates',   lambda_sparse ∈ {0.1, 1.0, 10.0, 100.0}
# Diagnostics for each cell land in results/grid_search/<run_id>/ and one row
# is appended to results/grid_search/summary.csv per cell.

echo "=== Phase 1 sweep cell $CELL — lambda_sparse mode × weight ==="
echo "Job ID:    $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo ""

python -u scripts/run_phase1_sweep_lsparse_mode.py \
    --base_config configs/replogle_u_score.yaml \
    --output_dir results/grid_search \
    --n_epochs 80 \
    --max_cells 50000 \
    --cell_index "$CELL" \
    --device cuda

echo ""
echo "=== Cell $CELL complete ==="
echo "End:    $(date)"

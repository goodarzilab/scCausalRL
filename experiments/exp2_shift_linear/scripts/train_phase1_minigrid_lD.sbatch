#!/bin/bash
#SBATCH --job-name=cbrl_p1_minigrid
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase1_minigrid_%j.out
#SBATCH --error=logs/phase1_minigrid_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs results/grid_search

echo "=== Phase J mini-grid: lambda_D_sparse ∈ {30, 100} ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo ""
echo "Pinned hyperparameters (Phase J grid 2369390 winner):"
echo "  lambda_sparse      = 10.0"
echo "  lambda_prior       = 10.0,   prior_margin = 0.5"
echo "  intervention.scheme = u_score (factored A = gate × weight × mask)"
echo ""
echo "Sweep axis:"
echo "  lambda_D_sparse ∈ {30, 100}   (extending beyond 10 to drive programs/gene below 15)"
echo ""
echo "Per-run output: results/grid_search/<run_id>/"
echo "Aggregate:      results/grid_search/summary.csv  (appended)"
echo ""

python -u scripts/run_phase1_minigrid_lD.py \
    --base_config configs/replogle_u_score.yaml \
    --output_dir results/grid_search \
    --n_epochs 80 \
    --max_cells 50000 \
    --device cuda

echo ""
echo "=== Mini-grid complete ==="
echo "End:       $(date)"

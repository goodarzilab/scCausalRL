#!/bin/bash
#SBATCH --job-name=cbrl_p1_grid
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_grid_%j.out
#SBATCH --error=logs/phase1_grid_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs results/grid_search

echo "=== Phase 1 grid search (u-score / cell-sentence pipeline, Phase F config) ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Working:   $(pwd)"
echo "Base:      configs/replogle_u_score.yaml (intervention.scheme = u_score)"
echo "Data:      50k K562 cells (cell-sentence loader handles per-block sampling)"
echo ""
echo "Sweep (Phase J — scale-up of all axes after Phase I grid showed they were too weak):"
echo "  lambda_sparse    ∈ {1.0, 10.0}                   (was {0.01, 0.1} — 100× scale up)"
echo "  lambda_prior + m ∈ {(10.0, 0.5), (50.0, 0.5)}    (was {1, 10} — extend the working direction)"
echo "  lambda_D_sparse  ∈ {1.0, 10.0}                   (was {0.01, 0.1} — 100× scale up; MultiOmics used 1000)"
echo "Total: 2 × 2 × 2 = 8 runs, 80 epochs each."
echo ""
echo "Fixed (Phase F):"
echo "  pert_loss_kind       = mse  (was energy; cell-sentence pairing handles batch-effects)"
echo "  pert_n_dag_samples   = 1    (was 8; bootstrap was for RL uncertainty, not training)"
echo "  DAGMA acyclicity     = OFF  (dagma_rho_init=0; spectral_stability keeps SCM solvable;"
echo "                                real GRNs are cyclic so this is biologically correct)"
echo "  min_block_cells      = 3    (was 8 → silent fallback bug; 50k×1371÷48 ≈ 0.76 cells/block)"
echo "  N-DAG inner loop in perturbation_set_loss_u is VECTORISED (one batched torch.linalg.solve)"
echo "Per-run output: results/grid_search/<run_id>/"
echo "Aggregate:      results/grid_search/summary.csv"
echo ""

# Note: -u for unbuffered stdout so the per-cell progress shows up in real time.
# --max_cells 50000 pins the grid to the 50k subset (the config has full K562 for the promotion run).
python -u scripts/run_phase1_grid.py \
    --base_config configs/replogle_u_score.yaml \
    --output_dir results/grid_search \
    --n_epochs 80 \
    --max_cells 50000 \
    --device cuda

echo ""
echo "=== Grid search complete ==="
echo "End:       $(date)"
echo "Summary:   results/grid_search/summary.csv"
echo "Next:      pick a winning row, update configs/replogle.yaml or"
echo "           configs/replogle_u_score.yaml, and submit the full Phase 1 job."

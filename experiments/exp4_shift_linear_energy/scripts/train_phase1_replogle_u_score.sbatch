#!/bin/bash
#SBATCH --job-name=cbrl_p1_promote
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_promote_%j.out
#SBATCH --error=logs/phase1_promote_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"

# Dedicated output dir so the promotion run doesn't clobber the grid driver's
# top-level checkpoints/logs and stays separable in results/.
PROMOTE_DIR="results/phase1_promote"
mkdir -p "$PROMOTE_DIR/checkpoints" "$PROMOTE_DIR/logs" logs

echo "=== Phase 1 promotion run — full K562 with grid 2365673 winner ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Working:   $(pwd)"
echo "Output:    $PROMOTE_DIR/"
echo "Config:    configs/replogle_u_score.yaml  (Phase J promoted values)"
echo ""
echo "Promoted hyperparameters (from grid 2365673 winning cell):"
echo "  lambda_sparse      = 0.01"
echo "  lambda_prior       = 10.0,   prior_margin = 0.5"
echo "  lambda_D_sparse    = 0.01"
echo "  lambda_spectral    = 0.1"
echo "  intervention.scheme = u_score (factored A = sigmoid_gate × weight × mask)"
echo "  pert_loss_kind     = mse,    pert_match_delta = true,    pert_n_dag_samples = 1"
echo "  replogle_max_cells = 200000  (full K562, ~186k cells)"
echo "  n_epochs           = 150"
echo ""

# chdir into PROMOTE_DIR so run_phase1.py's relative output paths (checkpoints/, logs/) go there.
cd "$PROMOTE_DIR"
python -u "$REPO_DIR/scripts/run_phase1.py" \
    --config "$REPO_DIR/configs/replogle_u_score.yaml" \
    --device cuda

echo ""
echo "=== Phase 1 promotion complete ==="
echo "End:        $(date)"
echo "Checkpoint: $REPO_DIR/$PROMOTE_DIR/checkpoints/phase1_best.pt"
echo "Log:        $REPO_DIR/$PROMOTE_DIR/logs/phase1_training.csv"
echo ""
echo "Diagnostic command:"
echo "  python $REPO_DIR/evaluation/dag_diagnostics.py \\"
echo "      --checkpoint $REPO_DIR/$PROMOTE_DIR/checkpoints/phase1_best.pt \\"
echo "      --output_dir $REPO_DIR/results/dag_diagnostics_promote"

#!/bin/bash
#SBATCH --job-name=cbrl_p1_verify
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase1_verify_%j.out
#SBATCH --error=logs/phase1_verify_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs results/verify

echo "=== Phase 1 single-cell verification — Phase H config ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo ""
echo "Phase H stack:"
echo "  theta_init             = randn(d, d) * 0.05    (centred at 0)"
echo "  L normalization        = row-max (per-gene max(L)=1)"
echo "  u_clip_max             = 0.5"
echo "  pert_match_delta       = true                  (decode(z_post)-decode(z_prev) vs x_pert-x_ctrl)"
echo "  pert_loss_kind         = mse"
echo "  pert_n_dag_samples     = 1"
echo "  lambda_spectral        = 0.1"
echo "  DAGMA                  = OFF (rho_init=0)"
echo ""
echo "Hyperparameters: (lambda_sparse=0.01, lambda_prior=1.0, margin=0.5, lambda_D=0.001)"
echo "Data:            50000 K562 cells, 80 epochs"
echo ""
echo "Acceptance:"
echo "  |theta_mean − 0.0| ≥ 1.0   (θ actually moved)"
echo "  edges_gt_0.1 ≥ 50"
echo "  recon_blowup_ratio ≤ 2.0"
echo "  mean_programs_per_gene ≤ 15"
echo "  ortho_off_diag_mean_abs ≤ 0.05"
echo ""

python -u scripts/run_phase1_verify.py

echo ""
echo "=== Verification complete ==="
echo "End:       $(date)"
echo "Artifacts: results/verify/cell_ls0p01_lp1p0_m0p5_ld0p001/"
echo "  - checkpoints/phase1_best.pt"
echo "  - logs/phase1_training.csv"
echo "  - verify_metrics.json"

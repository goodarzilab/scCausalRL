#!/bin/bash
#SBATCH --job-name=cbrl_p1_replogle_noDAG
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_replogle_no_dag_constraint_%j.out
#SBATCH --error=logs/phase1_replogle_no_dag_constraint_%j.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs checkpoints

echo "=== Phase 1: Replogle K562 — NO Acyclicity Constraint (ablation) ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Working:   $(pwd)"
echo "Config:    configs/replogle_no_dag_constraint.yaml"
echo ""
echo "Note: DAGMA augmented Lagrangian zeroed (lambda=0, rho=0, rho_max=0)."
echo "      h(A) is still computed + logged each epoch but does not enter the loss."
echo "      Sparsity + spectral stability + perturbation consistency still active."
echo ""

python scripts/run_phase1.py \
    --config configs/replogle_no_dag_constraint.yaml \
    --device cuda

echo ""
echo "=== Phase 1 Complete ==="
echo "End:       $(date)"
echo "Checkpoint: checkpoints/phase1_best.pt"
echo "Log:        logs/phase1_training.csv"
echo "Wandb run:  phase1_replogle_no_dag_constraint"

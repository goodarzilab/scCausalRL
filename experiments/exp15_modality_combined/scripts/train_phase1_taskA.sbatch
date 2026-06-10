#!/bin/bash
#SBATCH --job-name=cbrl_p1_taskA
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=36:00:00
#SBATCH --output=logs/phase1_taskA_%j.out
#SBATCH --error=logs/phase1_taskA_%j.err

set -euo pipefail

# Generic Task-A Phase-1 retrain. Each of the 6 retrains points at one config:
#   CFG=configs/replogle_essential_taskA_V1.yaml   OUTDIR=results/phase1_replogle_essential_taskA_V1
#   CFG=configs/replogle_essential_taskA_V2.yaml   OUTDIR=results/phase1_replogle_essential_taskA_V2
#   CFG=configs/norman_taskA_V1.yaml               OUTDIR=results/phase1_norman_taskA_V1
#   CFG=configs/norman_taskA_V2.yaml               OUTDIR=results/phase1_norman_taskA_V2
#   CFG=configs/combined_taskA_V1.yaml             OUTDIR=results/phase1_combined_taskA_V1
#   CFG=configs/combined_taskA_V2.yaml             OUTDIR=results/phase1_combined_taskA_V2
#
# Usage:
#   CFG=configs/replogle_essential_taskA_V1.yaml OUTDIR=results/phase1_replogle_essential_taskA_V1 \
#       sbatch scripts/train_phase1_taskA.sbatch

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"

CFG="${CFG:?must set CFG env var to a config path}"
OUTDIR="${OUTDIR:?must set OUTDIR env var to an output directory}"

mkdir -p "$OUTDIR/checkpoints" "$OUTDIR/logs" logs

echo "=== Phase 1 Task-A retrain ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo "Config:    $CFG"
echo "Output:    $OUTDIR"
echo ""

cd "$OUTDIR"
python -u "$REPO_DIR/scripts/run_phase1.py" \
    --config "$REPO_DIR/$CFG" \
    --device cuda

echo ""
echo "=== Phase 1 Task-A run complete ==="
echo "End:        $(date)"
echo "Checkpoint: $REPO_DIR/$OUTDIR/checkpoints/phase1_epoch150.pt  (final epoch — eval this, not phase1_best.pt)"
echo "Log:        $REPO_DIR/$OUTDIR/logs/phase1_training.csv"

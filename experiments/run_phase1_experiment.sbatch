#!/bin/bash
#SBATCH --job-name=cbrl_exp_p1
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=36:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err

set -euo pipefail

# Generic Phase-1 launcher for a self-contained experiment folder.
# Each experiment dir has its own copy of models/ training/ data/ scripts/ etc.,
# so run_phase1.py resolves imports from EXP_DIR (its sys.path.insert adds the
# experiment root). Outputs (checkpoints/logs/wandb) are written under OUTDIR so
# each experiment's artifacts stay with its code.
#
# Usage:
#   EXP_DIR=/home/sgoudarzi/scCausalRL/experiments/exp2_shift_linear \
#   CFG=configs/exp2_shift_linear.yaml \
#   OUTDIR=/home/sgoudarzi/scCausalRL/experiments/exp2_shift_linear/results/phase1 \
#       sbatch scripts/run_phase1_experiment.sbatch
#   (override --time / --job-name on the sbatch command line for smoke runs)

EXP_DIR="${EXP_DIR:?must set EXP_DIR to an experiment folder}"
CFG="${CFG:?must set CFG (path to a config, relative to EXP_DIR or absolute)}"
OUTDIR="${OUTDIR:?must set OUTDIR (output dir for checkpoints/logs)}"

# Resolve CFG to an absolute path if it was given relative to EXP_DIR.
case "$CFG" in
  /*) CFG_ABS="$CFG" ;;
  *)  CFG_ABS="$EXP_DIR/$CFG" ;;
esac

mkdir -p "$OUTDIR/checkpoints" "$OUTDIR/logs"

echo "=== Phase-1 experiment run ==="
echo "Job ID:    ${SLURM_JOB_ID:-<interactive>}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "Start:     $(date)"
echo "ExpDir:    $EXP_DIR"
echo "Config:    $CFG_ABS"
echo "Output:    $OUTDIR"
echo ""

cd "$OUTDIR"
python -u "$EXP_DIR/scripts/run_phase1.py" --config "$CFG_ABS" --device cuda

echo ""
echo "=== run complete ==="
echo "End:        $(date)"
echo "Checkpoint: $OUTDIR/checkpoints/phase1_epoch<N>.pt  (use the final epoch, not phase1_best.pt)"
echo "Log:        $OUTDIR/logs/phase1_training.csv"

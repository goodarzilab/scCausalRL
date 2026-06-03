#!/bin/bash
#SBATCH --job-name=cbrl_shift_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err

set -euo pipefail

# Shift-pathway eval for one experiment checkpoint.
# Usage:
#   EXP_DIR=/home/sgoudarzi/scCausalRL/experiments/exp2_shift_linear \
#   CKPT=$EXP_DIR/results/phase1/checkpoints/phase1_epoch150.pt \
#   CFG=configs/exp2_shift_linear.yaml \
#   OUTDIR=$EXP_DIR/results/phase1/shift_eval \
#   CMODE=both \
#       sbatch scripts/run_shift_eval.sbatch

EXP_DIR="${EXP_DIR:?must set EXP_DIR}"
CKPT="${CKPT:?must set CKPT (checkpoint path)}"
CFG="${CFG:?must set CFG (config path, relative to EXP_DIR or absolute)}"
OUTDIR="${OUTDIR:?must set OUTDIR}"
CMODE="${CMODE:-both}"
WANDB_FLAG="${WANDB_FLAG:-}"   # set to "--wandb" to log summary to W&B

case "$CFG" in /*) CFG_ABS="$CFG" ;; *) CFG_ABS="$EXP_DIR/$CFG" ;; esac
mkdir -p "$OUTDIR"

echo "=== shift eval ===  job ${SLURM_JOB_ID:-<i>}  node $(hostname)  $(date)"
echo "ExpDir: $EXP_DIR | Ckpt: $CKPT | Cfg: $CFG_ABS | Out: $OUTDIR | c_mode: $CMODE"

cd "$EXP_DIR"
python -u "$EXP_DIR/evaluation/run_shift_eval.py" \
    --checkpoint "$CKPT" --config "$CFG_ABS" \
    --output_dir "$OUTDIR" --c_mode "$CMODE" --device cuda $WANDB_FLAG

echo "=== eval complete $(date) ===  summary: $OUTDIR/summary.csv"

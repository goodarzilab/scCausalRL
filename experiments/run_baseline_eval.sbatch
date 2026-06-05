#!/bin/bash
#SBATCH --job-name=cbrl_baseline_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=04:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err

set -euo pipefail

# Baseline (legacy D⁺·dx pathway) eval via run_perturbation_eval.py.
# Uses exp1_baseline's code (a faithful copy of the baseline) against the Task-A
# checkpoints in the CausalBioRL repo. Same held-out split + AUROC metric as the
# shift eval, so numbers are directly comparable.
#
# Usage:
#   EXP_DIR=/home/sgoudarzi/scCausalRL/experiments/exp1_baseline \
#   CKPT=/home/sgoudarzi/CausalBioRL/results/phase1_replogle_essential_taskA_V1/checkpoints/phase1_epoch150.pt \
#   CFG=configs/replogle_essential_taskA_V1.yaml \
#   INPUT_MODE=canonical OUTDIR=$EXP_DIR/results/eval_V1_canonical \
#       sbatch scripts/run_baseline_eval.sbatch

EXP_DIR="${EXP_DIR:?set EXP_DIR}"
CKPT="${CKPT:?set CKPT}"
CFG="${CFG:?set CFG (rel to EXP_DIR or absolute)}"
INPUT_MODE="${INPUT_MODE:?set INPUT_MODE: target_lfc|canonical|observed_full}"
OUTDIR="${OUTDIR:?set OUTDIR}"

case "$CFG" in /*) CFG_ABS="$CFG" ;; *) CFG_ABS="$EXP_DIR/$CFG" ;; esac
mkdir -p "$OUTDIR"

echo "=== baseline eval ===  job ${SLURM_JOB_ID:-<i>}  node $(hostname)  $(date)"
echo "Ckpt: $CKPT | Cfg: $CFG_ABS | input_mode: $INPUT_MODE | Out: $OUTDIR"

cd "$EXP_DIR"
python -u "$EXP_DIR/evaluation/run_perturbation_eval.py" \
    --checkpoint "$CKPT" --config "$CFG_ABS" \
    --input_mode "$INPUT_MODE" --n_ctrl_cells 500 \
    --output_dir "$OUTDIR" --device cuda

echo "=== baseline eval complete $(date) ===  summary: $OUTDIR/summary.csv"

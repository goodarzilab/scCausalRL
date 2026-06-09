#!/bin/bash
#SBATCH --job-name=exp12_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err
set -euo pipefail
# Coherent eval for a mutilation-trained cell: held-out SINGLE AUROC via the SAME
# mutilation operator (auto-detected from ckpt config) + Norman combo/epistasis eval
# (shift_uscore = coherent mutilation predictor). Works for Replogle (cross-dataset
# Norman combos) and Norman (native combos) checkpoints.
EXP_DIR="${EXP_DIR:?}"; CKPT="${CKPT:?}"; CONFIG="${CONFIG:?}"; OUTROOT="${OUTROOT:?}"
cd "$EXP_DIR"
echo "########## single held-out eval (mutilation-coherent) ##########"
python -u evaluation/run_shift_eval.py --checkpoint "$CKPT" --config "$CONFIG" \
    --output_dir "$OUTROOT/shift_eval" --c_mode fixed --device cuda
echo "########## Norman combo + epistasis eval (mutilation-coherent) ##########"
python -u evaluation/run_norman_combo_eval.py --checkpoint "$CKPT" --config "$CONFIG" \
    --output_dir "$OUTROOT/norman_combo"
echo "########## done ##########"

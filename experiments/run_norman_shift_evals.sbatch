#!/bin/bash
#SBATCH --job-name=nrm_shift_ev
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err
set -euo pipefail
# Evals for the Norman-native shift model: held-out SINGLE-pert AUROC (in-modality,
# Norman's own pert_split) + combo/epistasis eval (combos auto-held-out from the
# single-gene shift training). Both are coherent with additive-shift training.
EXP_DIR="${EXP_DIR:?}"; CKPT="${CKPT:?}"; CONFIG="${CONFIG:?}"; OUTROOT="${OUTROOT:?}"
cd "$EXP_DIR"
echo "########## Norman held-out SINGLE eval (in-modality) ##########"
python -u evaluation/run_shift_eval.py --checkpoint "$CKPT" --config "$CONFIG" \
    --output_dir "$OUTROOT/shift_eval" --c_mode fixed --device cuda
echo "########## Norman COMBO + epistasis eval ##########"
python -u evaluation/run_norman_combo_eval.py --checkpoint "$CKPT" --config "$CONFIG" \
    --output_dir "$OUTROOT/norman_combo"
echo "########## done ##########"

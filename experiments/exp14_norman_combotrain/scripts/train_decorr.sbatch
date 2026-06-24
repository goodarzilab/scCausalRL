#!/bin/bash
#SBATCH --job-name=exp14_decorr
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/decorr_%j.out
#SBATCH --error=logs/decorr_%j.err
set -euo pipefail
# usage: CFG=exp14_decorr100.yaml RUN=exp14_decorr100 sbatch scripts/train_decorr.sbatch
REPO=/home/sgoudarzi/scCausalRL/experiments/exp14_norman_combotrain
OUT="$REPO/results/grid/${RUN}"
mkdir -p "$OUT" "$REPO/logs"
cd "$OUT"; export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "start $(date) on $(hostname)  CFG=$CFG RUN=$RUN"
python -u "$REPO/scripts/run_phase1.py" --config "$REPO/configs/grid/$CFG" --device cuda
CKPT="$OUT/checkpoints/phase1_epoch150.pt"; [ -f "$CKPT" ] || CKPT="$OUT/checkpoints/phase1_best.pt"
echo "=== combo eval on $CKPT ==="
python -u "$REPO/evaluation/run_norman_combo_eval.py" --checkpoint "$CKPT" \
    --config "$REPO/configs/grid/$CFG" --output_dir "$OUT/norman_combo" --device cuda
echo "=== singles shift eval ==="
python -u "$REPO/evaluation/run_shift_eval.py" --config "$REPO/configs/grid/$CFG" \
    --checkpoint "$CKPT" --output_dir "$OUT/shift_eval" --device cuda || echo "shift_eval skipped"
echo "done $(date)"

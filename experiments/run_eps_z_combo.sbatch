#!/bin/bash
#SBATCH --job-name=epsz_combo
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=00:40:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err
set -euo pipefail
EXP_DIR="${EXP_DIR:?}"; CKPT="${CKPT:?}"; CONFIG="${CONFIG:?}"
OUTROOT="${OUTROOT:?}"; TAG="${TAG:?}"; RUN_COMBO="${RUN_COMBO:-0}"
cd "$EXP_DIR"
echo "=== eps/z diagnostic ($TAG) ==="
python -u evaluation/eps_z_diagnostic.py --checkpoint "$CKPT" --config "$CONFIG" \
    --out "$OUTROOT/eps_z" --tag "$TAG"
if [ "$RUN_COMBO" = "1" ]; then
  echo "=== Norman combo eval ($TAG) ==="
  python -u evaluation/run_norman_combo_eval.py --checkpoint "$CKPT" --config "$CONFIG" \
      --output_dir "$OUTROOT/norman_combo"
fi
echo "=== done $TAG ==="

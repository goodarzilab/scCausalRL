#!/bin/bash
#SBATCH --job-name=exp10_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err
set -euo pipefail
# Combined post-training eval for one exp10 cell: held-out Replogle AUROC (pred
# survived?) + DAG grad/bypass diagnostic (A still active?) + ε/z table
# (disentanglement: did off-diag corr ε drop?).
EXP_DIR="${EXP_DIR:?}"; CKPT="${CKPT:?}"; CONFIG="${CONFIG:?}"; OUTROOT="${OUTROOT:?}"; TAG="${TAG:?}"
cd "$EXP_DIR"
echo "########## [$TAG] shift eval (Replogle held-out) ##########"
python -u evaluation/run_shift_eval.py --checkpoint "$CKPT" --config "$CONFIG" \
    --output_dir "$OUTROOT/shift_eval" --c_mode fixed --device cuda
echo "########## [$TAG] DAG grad / bypass diagnostic ##########"
python -u evaluation/dag_grad_diagnostic.py --ckpt "$CKPT" --max_cells 20000
echo "########## [$TAG] eps/z entanglement table ##########"
python -u evaluation/eps_z_diagnostic.py --checkpoint "$CKPT" --config "$CONFIG" \
    --out "$OUTROOT/eps_z" --tag "$TAG"
echo "########## [$TAG] done ##########"

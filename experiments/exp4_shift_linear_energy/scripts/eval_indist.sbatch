#!/bin/bash
#SBATCH --job-name=cbrl_eval_indist
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_indist_%j.out
#SBATCH --error=logs/eval_indist_%j.err

set -euo pipefail

# Minimal in-distribution eval — just run_perturbation_eval.py against one
# checkpoint + one config. The script dispatches on config.data.dataset.
# Useful for one-off evals of a single checkpoint (e.g. a sweep cell or
# a retrain final-epoch snapshot) without launching the full eval_u_sweep array.
#
# Usage:
#   CKPT=path/to/ckpt.pt CFG=configs/foo.yaml OUTDIR=results/foo \
#       sbatch scripts/eval_indist.sbatch

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs

CKPT="${CKPT:?must set CKPT env var to a checkpoint path}"
CFG="${CFG:?must set CFG env var to a config path}"
OUTDIR="${OUTDIR:?must set OUTDIR env var to an output directory}"
U_MAX="${U_MAX:-0.5}"
COMBO_MODE="${COMBO_MODE:-parallel}"
INPUT_MODE="${INPUT_MODE:-target_lfc}"
PERT_DIRECTION="${PERT_DIRECTION:--1}"
CANONICAL_MAGNITUDE="${CANONICAL_MAGNITUDE:-2.0}"
A_MODE="${A_MODE:-learned}"
GENE_IDX_MODE="${GENE_IDX_MODE:-correct}"
N_CTRL_CELLS="${N_CTRL_CELLS:-500}"
N_CTRL_COMBO="${N_CTRL_COMBO:-200}"
SEED="${SEED:-42}"

mkdir -p "$OUTDIR"

echo "=== In-distribution eval ==="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"
echo "Ckpt:   $CKPT"
echo "Config: $CFG"
echo "Output: $OUTDIR"
echo "input_mode:    $INPUT_MODE   pert_direction: $PERT_DIRECTION   magnitude: $CANONICAL_MAGNITUDE"
echo "a_mode:        $A_MODE       gene_idx_mode:  $GENE_IDX_MODE"
echo "n_ctrl_cells:  $N_CTRL_CELLS (singles) / $N_CTRL_COMBO (combos)   seed: $SEED"
echo "u_max:         $U_MAX  combo_mode: $COMBO_MODE"
echo ""

python -u evaluation/run_perturbation_eval.py \
    --checkpoint "$CKPT" \
    --config "$CFG" \
    --output_dir "$OUTDIR" \
    --u_clip_max "$U_MAX" \
    --combo_mode "$COMBO_MODE" \
    --input_mode "$INPUT_MODE" \
    --pert_direction "$PERT_DIRECTION" \
    --canonical_magnitude "$CANONICAL_MAGNITUDE" \
    --a_mode "$A_MODE" \
    --gene_idx_mode "$GENE_IDX_MODE" \
    --n_ctrl_cells "$N_CTRL_CELLS" \
    --n_ctrl_combo "$N_CTRL_COMBO" \
    --seed "$SEED" \
    --n_dag_samples 8 \
    --device cuda

echo ""
echo "=== Done ==="
echo "End:    $(date)"

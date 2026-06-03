#!/bin/bash
#SBATCH --job-name=cbrl_eval_u_sweep
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/eval_u_sweep_%A_%a.out
#SBATCH --error=logs/eval_u_sweep_%A_%a.err

set -euo pipefail

REPO_DIR="/home/sgoudarzi/CausalBioRL"
cd "$REPO_DIR"
mkdir -p logs

CKPT="${CKPT:-results/phase1_promote/checkpoints/phase1_best.pt}"
CFG="${CFG:-configs/replogle_u_score.yaml}"
NORMAN_PATH="${NORMAN_PATH:-/large_storage/goodarzilab/shervin/CausalBioRL/Norman/}"
# Prefix for all output dirs so different checkpoints don't clobber each other.
# Default keeps the legacy `results/eval_sweep/...` layout.
OUTDIR_PREFIX="${OUTDIR_PREFIX:-results/eval_sweep}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

# 9-cell sweep: 3 eval slices × 3 u_max values.
#   idx 0..2 — Replogle held-out single-pert (in-distribution test split)
#   idx 3..5 — Norman external (single + combo PARALLEL)
#   idx 6..8 — Norman external (single + combo SEQUENTIAL — ablation)
U_MAX_VALUES=(0.5 0.8 0.9)
IDX=$SLURM_ARRAY_TASK_ID

GROUP=$(( IDX / 3 ))
SLOT=$(( IDX % 3 ))
U_MAX="${U_MAX_VALUES[$SLOT]}"
U_TAG=$(printf "u%s" "${U_MAX/./p}")  # e.g. 0.5 → u0p5

case "$GROUP" in
    0)
        SLICE="replogle_single"
        OUTDIR="${OUTDIR_PREFIX}/${SLICE}_${U_TAG}"
        mkdir -p "$OUTDIR"
        echo "=== Eval slice: Replogle single-pert (in-distribution) | u_max=$U_MAX ==="
        echo "Job:       $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
        echo "Start:     $(date)"
        echo "Output:    $OUTDIR"
        python evaluation/run_perturbation_eval.py \
            --checkpoint "$CKPT" \
            --config "$CFG" \
            --output_dir "$OUTDIR" \
            --u_clip_max "$U_MAX" \
            --combo_mode parallel \
            --n_dag_samples 8 \
            --device cuda
        ;;
    1)
        SLICE="norman_parallel"
        OUTDIR="${OUTDIR_PREFIX}/${SLICE}_${U_TAG}"
        mkdir -p "$OUTDIR"
        echo "=== Eval slice: Norman external (single + combo PARALLEL) | u_max=$U_MAX ==="
        echo "Job:       $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
        echo "Start:     $(date)"
        echo "Output:    $OUTDIR"
        python evaluation/run_norman_validation.py \
            --checkpoint "$CKPT" \
            --config "$CFG" \
            --norman_path "$NORMAN_PATH" \
            --output_dir "$OUTDIR" \
            --u_clip_max "$U_MAX" \
            --combo_mode parallel \
            --device cuda
        ;;
    2)
        SLICE="norman_sequential"
        OUTDIR="${OUTDIR_PREFIX}/${SLICE}_${U_TAG}"
        mkdir -p "$OUTDIR"
        echo "=== Eval slice: Norman external (single + combo SEQUENTIAL ablation) | u_max=$U_MAX ==="
        echo "Job:       $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
        echo "Start:     $(date)"
        echo "Output:    $OUTDIR"
        python evaluation/run_norman_validation.py \
            --checkpoint "$CKPT" \
            --config "$CFG" \
            --norman_path "$NORMAN_PATH" \
            --output_dir "$OUTDIR" \
            --u_clip_max "$U_MAX" \
            --combo_mode sequential \
            --device cuda
        ;;
esac

echo ""
echo "=== Done ==="
echo "End:    $(date)"
echo "Slice:  $SLICE  u_max=$U_MAX"

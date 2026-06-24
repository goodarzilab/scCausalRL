#!/bin/bash
#SBATCH --job-name=exp15_modality_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=2:00:00
#SBATCH --output=logs/modality_eval_%j.out
#SBATCH --error=logs/modality_eval_%j.err

set -euo pipefail

# Modality-AWARE eval of the exp15 combined (Replogle CRISPRi + Norman CRISPRa) model.
# Norman perturbations are scored with the CRISPRa modality scale (the stock combo
# eval omits modality and silently uses the global scale). Runs singles + combos.

REPO_DIR="/home/sgoudarzi/scCausalRL/experiments/exp15_modality_combined"
cd "$REPO_DIR"
mkdir -p logs

CKPT="${CKPT:-results/grid/exp15_modality_combined/checkpoints/phase1_epoch150.pt}"
CFG="${CFG:-configs/combined.yaml}"
OUTDIR="${OUTDIR:-results/grid/exp15_modality_combined/modality_eval_norman}"
# combined.yaml's norman_path is stale (scCausalRL/Norman doesn't exist); use the
# real Norman location (same as exp14/the validated smoke).
NORMAN_PATH="${NORMAN_PATH:-/large_storage/goodarzilab/shervin/CausalBioRL/Norman/}"

echo "=== exp15 modality-aware eval (Norman = CRISPRa) ==="
echo "Job:    ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "Ckpt:   $CKPT"
echo "Config: $CFG"
echo "Output: $OUTDIR"
echo ""

python -u evaluation/run_modality_eval.py \
    --checkpoint "$CKPT" \
    --config "$CFG" \
    --norman_path "$NORMAN_PATH" \
    --pert_modality crispra \
    --output_dir "$OUTDIR" \
    --n_ctrl_cells 500 \
    --device cuda

echo ""
echo "=== Done ==="
echo "End:     $(date)"
echo "Results: $OUTDIR/"

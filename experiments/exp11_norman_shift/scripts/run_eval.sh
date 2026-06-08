#!/bin/bash
#SBATCH --job-name=CausalRLBioEval
#SBATCH --output=/home/sgoudarzi/scCausalRL/logs/eval_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/logs/eval_%j.err
#SBATCH --partition=goodarzilab_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

# ── environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate causalbiorld        # change to your conda env name

# ── paths ─────────────────────────────────────────────────────────────────────
REPO=/home/sgoudarzi/scCausalRL
CONFIG=${REPO}/configs/default.yaml
VAE_CKPT=${REPO}/checkpoints/phase1_best.pt
POLICY_CKPT=${REPO}/checkpoints/phase2_policy_final.pt

cd "${REPO}"
mkdir -p logs/dag_eval

echo "=================================================="
echo "  scCausalRL  —  Evaluation"
echo "  SLURM job ID : ${SLURM_JOB_ID}"
echo "  Node         : $(hostname)"
echo "  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  VAE ckpt     : ${VAE_CKPT}"
echo "  Policy ckpt  : ${POLICY_CKPT}"
echo "  Start        : $(date)"
echo "=================================================="

# Run all three evaluations; policy checkpoint is optional (planning eval skips RL
# comparison gracefully if the file doesn't exist yet)
POLICY_ARG=""
if [[ -f "${POLICY_CKPT}" ]]; then
    POLICY_ARG="--policy_checkpoint ${POLICY_CKPT}"
fi

python scripts/run_eval.py \
    --config "${CONFIG}" \
    --vae_checkpoint "${VAE_CKPT}" \
    ${POLICY_ARG} \
    --evals combinatorial dag planning

echo ""
echo "Evaluation finished at $(date)"
echo "Results: ${REPO}/logs/eval_results.json"
echo "DAG plots: ${REPO}/logs/dag_eval/"

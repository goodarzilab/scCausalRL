#!/bin/bash
#SBATCH --job-name=dag_analysis
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=02:00:00
#SBATCH --output=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.out
#SBATCH --error=/home/sgoudarzi/scCausalRL/experiments/logs/%x_%j.err
set -euo pipefail
NB_DIR=/home/sgoudarzi/scCausalRL/experiments/notebooks
EXP_DIR="${DAG_EXP_DIR:?set DAG_EXP_DIR}"
TAG="${TAG:-exp}"
cd "$NB_DIR"
echo "=== executing dag_analysis.ipynb on $EXP_DIR ($(date)) on $(hostname) ==="
jupyter nbconvert --to notebook --execute --allow-errors \
  --ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=python3 \
  --output "dag_analysis_${TAG}_executed.ipynb" dag_analysis.ipynb
echo "=== done $(date) ==="
echo "executed notebook: $NB_DIR/dag_analysis_${TAG}_executed.ipynb"
echo "figures + scorecard: $EXP_DIR/results/phase1/dag_analysis/"

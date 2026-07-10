#!/bin/bash
#SBATCH --job-name=exp20_smk
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_preemptible,cpu_batch,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=340G
#SBATCH --time=04:00:00
#SBATCH --output=logs/smk_%x_%j.out
#SBATCH --error=logs/smk_%x_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
CFG="$REPO/configs/ablations/lbsmk/${TAG}.yaml"
OUT="$REPO/results/grid/smk_${TAG}"; mkdir -p "$OUT"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"; export OMP_NUM_THREADS=16
echo "start $(date) on $(hostname) | smoke $TAG"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --n_epochs 3 --device cpu
echo "=== Phase-B diag (gate/target_l0) ==="; grep -iE "gate_open|target_l0|pert=|Phase B" logs/phase1_training.csv 2>/dev/null | tail -3 || true
echo "=== A_gap probe ==="
CK=$(ls checkpoints/phase1_best.pt checkpoints/phase1_epoch*.pt 2>/dev/null | head -1)
python -u "$REPO/evaluation/agap_probe.py" --checkpoint "$CK" --n_tf 20 --n_ctrl 500 || echo "probe failed"
echo "done $(date)"

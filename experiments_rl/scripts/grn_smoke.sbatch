#!/bin/bash
#SBATCH --job-name=grn_smoke
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=340G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:30:00
#SBATCH --output=logs/grn_smoke_%x_%j.out
#SBATCH --error=logs/grn_smoke_%x_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CFG="$REPO/configs/ablations/grn/${TAG}.yaml"
OUT="$REPO/results/grid/smoke_grn_${TAG}"; mkdir -p "$OUT"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "start $(date) on $(hostname) | smoke TAG=$TAG (2 epochs)"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --n_epochs 2 --device cuda
echo "--- A stats at epoch 2 ---"
python - <<PY
import torch,glob,numpy as np,numpy.linalg as la,sys
sys.path.insert(0,"$REPO")
from evaluation.run_shift_eval_norman import load_shift_model
ck=sorted(glob.glob("checkpoints/phase1_epoch*.pt"))[-1]
m,gn,_=load_shift_model(ck,'cpu')
Ar=m.dag.get_edge_probs().detach().numpy(); Ac=m.dag.sample_A(n_samples=1).squeeze(0).detach().numpy()
print(f"SMOKE_RESULT $TAG: ckpt={ck} dens_raw={(np.abs(Ar)>1e-3).mean():.3f} spec_raw={np.abs(la.eigvals(Ar)).max():.2f} dens_solve={(np.abs(Ac)>1e-3).mean():.3f} spec_solve={np.abs(la.eigvals(Ac)).max():.3f} |A_raw|mean={np.abs(Ar).mean():.4f}")
PY
echo "done $(date)"

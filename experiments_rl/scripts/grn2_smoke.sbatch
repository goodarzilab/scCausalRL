#!/bin/bash
#SBATCH --job-name=grn2_smoke
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=340G
#SBATCH --cpus-per-task=12
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --time=01:30:00
#SBATCH --output=logs/grn2_smoke_%x_%j.out
#SBATCH --error=logs/grn2_smoke_%x_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CFG="$REPO/configs/ablations/grn2/${TAG}.yaml"
OUT="$REPO/results/grid/smoke_grn2_${TAG}"; mkdir -p "$OUT"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "start $(date) on $(hostname) | smoke grn2 TAG=$TAG (2 epochs)"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --n_epochs 2 --device cuda
python - <<PY
import torch,glob,numpy as np,numpy.linalg as la,sys
sys.path.insert(0,"$REPO"); from evaluation.run_shift_eval_norman import load_shift_model
ck=sorted(glob.glob("checkpoints/phase1_epoch*.pt"))[-1]; m,_,_=load_shift_model(ck,'cpu')
th=m.dag.theta.detach().numpy(); w=m.dag.weights.detach().numpy(); mk=m.dag.mask.detach().numpy(); tau=float(m.dag.tau)
g=1/(1+np.exp(-th/tau))*mk; Ac=m.dag.sample_A(n_samples=1).squeeze(0).detach().numpy(); off=~np.eye(th.shape[0],bool)
print(f"SMOKE2 $TAG: gate_mean={g[off].mean():.3f} frac_on>0.9={np.mean(g[off]>0.9):.3f} frac_off<0.1={np.mean(g[off]<0.1):.3f} "
      f"|w|mean={np.abs(w[off]).mean():.3f} solveA_dens={(np.abs(Ac)>1e-3).mean():.3f} spec_solve={np.abs(la.eigvals(Ac)).max():.3f}")
PY
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=grnsmk
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_preemptible,cpu_batch,cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=340G
#SBATCH --time=04:00:00
#SBATCH --output=logs/grnsmk_%x_%j.out
#SBATCH --error=logs/grnsmk_%x_%j.err
set -euo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose
CFG="$REPO/configs/ablations/grnsmk/${TAG}.yaml"
OUT="$REPO/results/grid/grnsmk_${TAG}"; mkdir -p "$OUT"; cd "$OUT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"; export OMP_NUM_THREADS=16
echo "start $(date) on $(hostname) | recon1 smoke $TAG (3 epochs, reaches DAG+pert phase)"
python -u "$REPO/scripts/run_phase1.py" --config "$CFG" --n_epochs 3 --device cpu
echo "=== dag_util_term / pert diag from training (phase B) ==="
grep -iE "dag_util|pert_diag|pert=" logs/phase1_training.csv 2>/dev/null | tail -3 || true
python - <<PY
import torch,glob,numpy as np,numpy.linalg as la,sys
sys.path.insert(0,"$REPO"); from evaluation.run_shift_eval_norman import load_shift_model
cks=glob.glob("checkpoints/phase1_best.pt")+glob.glob("checkpoints/phase1_best_recon.pt")+sorted(glob.glob("checkpoints/phase1_epoch*.pt"))
m,_,_=load_shift_model(cks[0],'cpu'); th=m.dag.theta.detach().numpy(); w=m.dag.weights.detach().numpy(); mk=m.dag.mask.detach().numpy(); tau=float(m.dag.tau)
g=1/(1+np.exp(-th/tau))*mk; Ac=m.dag.sample_A(1).squeeze(0).detach().numpy(); off=~np.eye(th.shape[0],dtype=bool)
print(f"SMOKE3 $TAG (ckpt={cks[0].split('/')[-1]}): gate_mean={g[off].mean():.3f} frac_off<0.1={np.mean(g[off]<0.1):.3f} "
      f"|w|mean={np.abs(w[off]).mean():.3f} solveA_dens={(np.abs(Ac)>1e-3).mean():.3f} spec_solve={np.abs(la.eigvals(Ac)).max():.3f}")
PY
echo "done $(date)"

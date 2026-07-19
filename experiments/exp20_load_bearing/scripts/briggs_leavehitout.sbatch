#!/bin/bash
#SBATCH --job-name=x20_leavehit
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=logs/leavehit_%j.out
#SBATCH --error=logs/leavehit_%j.err
set -uo pipefail
E=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
RL=/home/sgoudarzi/scCausalRL/experiments_rl
export PYTHONPATH="$E"
CK=$E/results/grid/v2b_deg3_t25_frozen_r32_all40/checkpoints/phase1_epoch40.pt
TAG=v2b_deg3_t25_frozen_r32_all40
HITS="TAL1,RPA3,ZFP90,RNF138,LMO2,KLF2,DDIT3,MZF1,HOXB3,MIER1,SIRT6,ZNF85"
cd "$E"
echo "########## leave-HIT-out readouts (fit on non-P1/P2 combos, test on hit combos) ##########"
for T in rho_cao rho_briggs; do
  python -u evaluation/briggs/readout_head.py --ckpt "$CK" --answer results/briggs/observed_combo_targets.csv \
    --target $T --holdout holdhits --exclude_tfs "$HITS" 2>&1 | grep -E "READOUT|overall|k>=7|wrote|leave-HIT"
done
echo "########## fate-MPC with leak-free readout + hit-excluded support (does it STILL recover P1/P2?) ##########"
cd "$RL"; export PYTHONPATH="$RL"
python -u analysis/briggs_fate_mpc.py --ensemble 1 --lam 1 --support_gamma 2 --kmax 7 --beam 12 --n_ctrl 256 --n_dag 8 \
  --ckpt "$CK" \
  --readout_cao $E/results/briggs/readout_ridge_${TAG}_rho_cao_holdhits.npz \
  --readout_briggs $E/results/briggs/readout_ridge_${TAG}_rho_briggs_holdhits.npz \
  --support_feats $E/results/briggs/readout_feats_${TAG}_ans.npz \
  --exclude_tfs "$HITS" 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"

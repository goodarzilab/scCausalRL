#!/bin/bash
#SBATCH --job-name=x20_cleanRL
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=logs/cleanRL_%j.out
#SBATCH --error=logs/cleanRL_%j.err
set -uo pipefail
E=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
RL=/home/sgoudarzi/scCausalRL/experiments_rl
CK=$E/results/grid/v2b_deg3_t25_frozen_r32_all40/checkpoints/phase1_epoch40.pt
TAG=v2b_deg3_t25_frozen_r32_all40
declare -A ANS=( [screenbulk]=observed_combo_targets [joung_ctrl]=observed_combo_targets_joungctrl [joung_briggs]=observed_combo_targets_hescjoung )
declare -A ANSB=( [screenbulk]=ans [joung_ctrl]=ans_joungctrl [joung_briggs]=ans_hescjoung )
for BG in screenbulk joung_ctrl joung_briggs; do
  echo "############################## BACKGROUND: $BG ##############################"
  cd "$E"; export PYTHONPATH="$E"
  echo "----- WM fidelity (readout Spearman, Briggs-native axis) -----"
  python -u evaluation/briggs/readout_head.py --ckpt "$CK" --answer results/briggs/${ANS[$BG]}.csv \
    --target rho_briggs --holdout random 2>&1 | grep -E "READOUT|overall|k>=7|wrote"
  echo "----- RL recovery (fate-MPC, clean-bg readout + clean-bg support) -----"
  cd "$RL"; export PYTHONPATH="$RL"
  python -u analysis/briggs_fate_mpc.py --ensemble 0 --lam 1 --support_gamma 2 --kmax 7 --beam 12 --n_ctrl 256 --n_dag 8 \
    --ckpt "$CK" \
    --readout_cao $E/results/briggs/readout_ridge_${TAG}_${ANSB[$BG]}_rho_briggs.npz \
    --support_feats $E/results/briggs/readout_feats_${TAG}_${ANSB[$BG]}.npz 2>&1 | grep -E "recovery|driver rate|P1 mem|P2 mem|drivers rec|pool="
done
echo "fin $(date)"

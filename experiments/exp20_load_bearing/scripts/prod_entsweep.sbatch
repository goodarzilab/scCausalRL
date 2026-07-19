#!/bin/bash
#SBATCH --job-name=x20_entsweep
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/entsweep_%j.out
#SBATCH --error=logs/entsweep_%j.err
set -uo pipefail
E=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
RL=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
CK=$E/results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt
RO=$E/results/briggs/readout_ridge_v2b_prod_deg8_all40_ans_joungctrl_rho_briggs.npz
SF=$E/results/briggs/readout_feats_v2b_prod_deg8_all40_ans_joungctrl.npz
for EC in 0.02 0.05 0.1 0.2; do
  echo "############### ent_coeff=$EC (prod WM, clean joung_ctrl readout) ###############"
  python -u analysis/briggs_setppo_bcfinetune.py --ckpt "$CK" --readout "$RO" --support_feats "$SF" \
    --ent_coeff $EC --kmax 7 --bc_epochs 500 --ft_iters 3000 --n_ctrl 256 --n_dag 6 2>&1 | grep -E "bc\+ft|PARETO|VERDICT|BC-init|POST-FINETUNE|ft it"
done
echo "fin $(date)"

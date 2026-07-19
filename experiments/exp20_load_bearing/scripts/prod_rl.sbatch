#!/bin/bash
#SBATCH --job-name=x20_prodRL
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/prodRL_%j.out
#SBATCH --error=logs/prodRL_%j.err
set -uo pipefail
E=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
RL=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
CK=$E/results/grid/v2b_prod_deg8_all40/checkpoints/phase1_epoch40.pt
RO=$E/results/briggs/readout_ridge_v2b_prod_deg8_all40_ans_joungctrl_rho_briggs.npz
SF=$E/results/briggs/readout_feats_v2b_prod_deg8_all40_ans_joungctrl.npz
echo "########## CANONICAL RL on PRODUCTION model (prod_deg8_all40) + clean joung_ctrl readout ##########"
python -u analysis/briggs_fate_mpc.py --ensemble 0 --lam 1 --support_gamma 2 --kmax 7 --beam 24 \
  --topN 40 --div_jaccard 0.6 --n_ctrl 256 --n_dag 8 \
  --ckpt "$CK" --readout_cao "$RO" --support_feats "$SF" \
  --out $E/results/briggs/denovo_candidates_prod.csv 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"
echo "fin $(date)"

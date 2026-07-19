#!/bin/bash
#SBATCH --job-name=x20_fate_mpc
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/fate_mpc_%j.out
#SBATCH --error=logs/fate_mpc_%j.err
set -uo pipefail
RL=/home/sgoudarzi/scCausalRL/experiments_rl
cd "$RL"; export PYTHONPATH="$RL" OMP_NUM_THREADS=8
run () { echo "########## $* ##########"; python -u analysis/briggs_fate_mpc.py --kmax 7 --beam 12 --n_ctrl 256 --n_dag 8 "$@" 2>&1 | grep -vE "Warning|warn|verbose|FutureWarning"; echo; }
run --ensemble 1 --lam 0  --support_gamma 0     # no pessimism (baseline exploitation)
run --ensemble 1 --lam 3  --support_gamma 0     # DAG pessimism only (does epistemic-std alone fix it?)
run --ensemble 1 --lam 1  --support_gamma 2     # light off-manifold support (MOReL/MOPO)
run --ensemble 1 --lam 1  --support_gamma 5     # medium support (expected sweet spot)
run --ensemble 1 --lam 1  --support_gamma 10    # strong support
echo "fin $(date)"

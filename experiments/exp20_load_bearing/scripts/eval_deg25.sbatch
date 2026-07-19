#!/bin/bash
#SBATCH --job-name=x20_evaldeg25
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=logs/evaldeg25_%j.out
#SBATCH --error=logs/evaldeg25_%j.err
set -uo pipefail
REPO=/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing
cd "$REPO"; export PYTHONPATH="$REPO" OMP_NUM_THREADS=6
python -u evaluation/agap_correctness.py 2>&1 | grep -vE "load_joung|Loaded|Warning|warn"
echo "=== DEG-0.25 (t25) vs thr=0.1 winner (allDEG) ==="
python - <<'PY'
import csv
r={x['arm']:x for x in csv.DictReader(open('results/figures/agap_correctness.csv'))}
for a in ['v2b_mmd1_deg1_t25','v2b_mmd1_deg2_t25','v2b_mmd1_deg3_t25','v2b_mmd1_deg5_t25','v2b_mmd1_deg2']:
    if a in r: print(f"  {a:24} allDEG {float(r[a]['aDEG_real']):+.3f}  Pnk {float(r[a]['Pnk_real']):+.3f}")
PY
echo "done $(date)"

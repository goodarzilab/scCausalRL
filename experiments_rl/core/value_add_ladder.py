"""The value-add ladder: random → greedy-1 → additive → MPC (→ PPO policies later).

Runs every planner through the same data-grounded validation (gene recovery, latent
distance, decoded-state DE fidelity) on the held-out splits and emits one CSV row per
rung, so the monotone improvement (or lack of it) is auditable. PPO/set-PPO/distilled
rungs are appended by passing extra planners via `extra_planners`.

Usage:
  python core/value_add_ladder.py --split combo --max_set 2 --beam 8 --alpha 1.0 \
      --n_dag 16 --output rl0_mpc/results/ladder_combo.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.wm_data import WMData
from core.baselines import Planners
from core.validate import evaluate_planner
from rl0_mpc.mpc_planner import MPCPlanner


def run_ladder(op, data, split='combo', max_set=2, beam=8, alpha=1.0, n_dag=16,
               random_seeds=(0, 1, 2), extra_planners=None):
    keys = data.combo_split['test'] if split == 'combo' else data.single_split['test']
    planners = Planners(op, data)
    mpc = MPCPlanner(op, data, max_set=max_set, beam_width=beam,
                     n_dag_samples=n_dag, pessimism_alpha=alpha)

    rungs = []
    # random: average over seeds (it's noisy)
    rand_summaries = []
    for s in random_seeds:
        rng = np.random.default_rng(s)
        _, summ = evaluate_planner(lambda k: planners.random(k, rng=rng), keys, planners, data,
                                   label=f'random_s{s}')
        rand_summaries.append(summ)
    rand_mean = {k: (np.mean([d[k] for d in rand_summaries]) if isinstance(rand_summaries[0][k], float)
                     else rand_summaries[0][k]) for k in rand_summaries[0]}
    rand_mean['label'] = 'random'
    rungs.append(rand_mean)

    for label, fn in [('greedy1', planners.greedy1), ('additive', planners.additive),
                      ('MPC', mpc)]:
        _, summ = evaluate_planner(fn, keys, planners, data, label=label)
        rungs.append(summ)

    for label, fn in (extra_planners or []):
        _, summ = evaluate_planner(fn, keys, planners, data, label=label)
        rungs.append(summ)

    return pd.DataFrame(rungs)[['label', 'n', 'match', 'recall', 'full_recovery',
                                'dist_ratio', 'auroc', 'spearman', 'pearson_deg']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='combo', choices=['combo', 'single'])
    ap.add_argument('--max_set', type=int, default=2)
    ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--n_dag', type=int, default=16)
    ap.add_argument('--n_ctrl', type=int, default=300)
    ap.add_argument('--output', default=None)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, meta = load_frozen_wm(device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=args.n_dag)
    data = WMData(model, meta, op, n_ctrl_cells=args.n_ctrl)

    df = run_ladder(op, data, split=args.split, max_set=args.max_set, beam=args.beam,
                    alpha=args.alpha, n_dag=args.n_dag)
    pd.set_option('display.width', 160)
    print(f"\n===== VALUE-ADD LADDER ({args.split}, max_set={args.max_set}, beam={args.beam}, "
          f"alpha={args.alpha}, N={args.n_dag}) =====")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"saved → {args.output}")


if __name__ == "__main__":
    main()

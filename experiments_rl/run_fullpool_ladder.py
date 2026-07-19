"""Large-pool (zero-shot, ~8000-gene) value-add ladder: baselines + MPC on held-out
Norman combos. The candidate pool is the FULL axis (≈99% never screened), so this asks:
given an observed combo's target, can a planner find a 2-gene set that reaches its DE
when almost all candidates are unseen? MPC (beam search at scale) vs simple baselines.

Run (GPU): python run_fullpool_ladder.py --beam 16 --n_dag 8 --output rl0_mpc/results/fullpool
"""
import os
import sys
import argparse
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.setup_stack import add_stack_args, build_stack
from core.baselines import Planners
from core.validate import evaluate_planner
from core.effect_clusters import EffectClusters
from rl0_mpc.mpc_planner import MPCPlanner


def main():
    ap = argparse.ArgumentParser()
    add_stack_args(ap)
    ap.add_argument('--beam', type=int, default=16)
    ap.add_argument('--n_dag', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--n_ctrl', type=int, default=300)
    ap.add_argument('--pool', default='full', choices=['full', 'screened', 'curated'])
    ap.add_argument('--output', default='rl0_mpc/results/fullpool')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model, meta, op, data = build_stack(args, dev, args.n_dag)
    data.set_pool(args.pool)
    planners = Planners(op, data)                          # batched precompute over the pool
    ec = EffectClusters(op, data, planners)                # posthoc effect modules (soft recall)
    mpc = MPCPlanner(op, data, max_set=2, beam_width=args.beam, n_dag_samples=args.n_dag,
                     pessimism_alpha=args.alpha)
    greedy1 = MPCPlanner(op, data, max_set=1, beam_width=1, n_dag_samples=args.n_dag,
                         pessimism_alpha=args.alpha)
    greedy2 = MPCPlanner(op, data, max_set=2, beam_width=1, n_dag_samples=args.n_dag,
                         pessimism_alpha=args.alpha)  # sequential greedy-2 (joint, beam-1)

    keys = data.combo_split['test']
    rng = np.random.default_rng(0)
    rungs = []
    methods = [
        ('random', lambda k: [int(x) for x in rng.choice(data.candidate_idx, 2, replace=False)]),
        ('greedy1', greedy1),
        ('top2_marginal', lambda k: mpc.top_singles(k, 2)),   # 2 best independent singles
        ('greedy2', greedy2),                                  # best gene, then best 2nd | first
        ('MPC', mpc),
    ]
    for label, fn in methods:
        df, summ = evaluate_planner(fn, keys, planners, data, label=label, effect_clusters=ec)
        df.to_csv(os.path.join(args.output, f'{label}.csv'), index=False)
        rungs.append(summ)
        print(f"  [{label}] " + "  ".join(f"{k}={summ[k]:.4f}" for k in
              ['match', 'recall', 'soft_recall_corr', 'soft_recall_clust', 'pearson_deg']
              if isinstance(summ.get(k), float)), flush=True)

    tab = pd.DataFrame(rungs)[['label', 'n', 'match', 'recall', 'soft_recall_corr',
                               'soft_recall_clust', 'full_recovery', 'dist_ratio',
                               'auroc', 'spearman', 'pearson_deg']]
    pd.set_option('display.width', 170)
    print(f"\n===== FULL-POOL LADDER ({args.pool}, |cand|={len(data.candidate_idx)}, "
          f"held-out combos n={len(keys)}) =====")
    print(tab.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    tab.to_csv(os.path.join(args.output, 'ladder.csv'), index=False)
    print(f"saved → {args.output}/")


if __name__ == "__main__":
    main()

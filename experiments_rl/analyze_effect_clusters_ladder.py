"""Effect-cluster soft-recall across the Norman RL ladder.

The DE-match ladder asks "did the planner reach the target expression state". This asks a
complementary question: **did each method pick genes with the SAME perturbation effect as
the held-out combo's true genes?** We cluster genes by their perturbation-effect profile
(EffectClusters: observed-DE correlation modules), then for every held-out combo and every
method score whether the chosen genes land in the true genes' effect module (cluster-hit)
and how effect-similar they are (graded correlation). No re-planning — we read each
method's already-saved (target, chosen) and only recompute the soft-recall.

Outputs (results/effect_clusters/):
  - per_method.csv               (method × combo soft-recall)
  - summary.csv                  (method means ± SEM)
  - effect_cluster_softrecall.png   (the headline: cluster-hit + graded, per method)
  - effect_cluster_map.png          (MDS of screened genes coloured by effect module)
Run (GPU or CPU):  python analyze_effect_clusters_ladder.py
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator
from core.wm_data import WMData
from core.baselines import Planners
from core.effect_clusters import EffectClusters

R = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(R, 'results', 'effect_clusters')
HUE = '#3C5488'

# (label, csv path, is_non_learned_baseline) — same order/role as the DE-match ladder figure
METHODS = [
    ('random',       'rl0_mpc/results/fullpool/random.csv',        True),
    ('seq-PPO',      'rl1_seq_ppo/results/seqppo_test.csv',        False),
    ('set-PPO',      'rl2_set_ppo/results/setppo_test.csv',        False),
    ('MPC→distill',  'rl3_mpc_distill/results/distill_test.csv',   False),
    ('top2-marginal','rl0_mpc/results/fullpool/top2_marginal.csv', True),
    ('greedy-1',     'rl0_mpc/results/fullpool/greedy1.csv',       True),
    ('MPC',          'rl0_mpc/results/fullpool/MPC.csv',           False),
]


def sem(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (float(v.mean()) if len(v) else np.nan, 0.0)


def main(n_clusters=12, device=None):
    os.makedirs(OUT, exist_ok=True)
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, meta = load_frozen_wm(device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=8)
    data = WMData(model, meta, op, n_ctrl_cells=300)
    data.set_pool('full')
    planners = Planners(op, data)
    ec = EffectClusters(op, data, planners, n_clusters=n_clusters)

    def to_idx(names):
        return [data.gene_to_idx[g] for g in names if g in data.gene_to_idx]

    rows = []
    for label, rel, isb in METHODS:
        p = os.path.join(R, rel)
        if not os.path.exists(p):
            print(f'  [skip] {label}: {rel} missing'); continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            true_gi = to_idx(str(r['target']).split('+'))
            chosen_gi = to_idx(str(r['chosen']).split('+'))
            sr = ec.soft_recall(chosen_gi, true_gi)
            rows.append({'method': label, 'baseline': isb, 'target': r['target'],
                         'chosen': r['chosen'], **sr})
        print(f'  [{label}] scored {len(df)} combos', flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, 'per_method.csv'), index=False)

    # summary ± SEM, preserve METHODS order
    labels = [m[0] for m in METHODS if m[0] in per['method'].unique()]
    isb_map = {m[0]: m[2] for m in METHODS}
    summ = []
    for lab in labels:
        s = per[per['method'] == lab]
        cm, ce = sem(s['soft_recall_clust']); gm, ge = sem(s['soft_recall_corr'])
        summ.append({'method': lab, 'baseline': isb_map[lab], 'n': len(s),
                     'clust_hit': cm, 'clust_sem': ce, 'corr': gm, 'corr_sem': ge})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    print('\n', summ.to_string(index=False))

    _fig_softrecall(summ)
    _fig_clustermap(ec)
    print(f'\nsaved → {OUT}/')


def _fig_softrecall(summ):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, col, se, title, ylab in [
        (ax[0], 'clust_hit', 'clust_sem', 'Effect-cluster HIT rate',
         'fraction of true genes whose effect\nmodule a chosen gene hits'),
        (ax[1], 'corr', 'corr_sem', 'Graded effect similarity',
         'max effect-profile correlation\n(chosen vs true gene)')]:
        bars = a.bar(range(len(summ)), summ[col], yerr=summ[se], capsize=3,
                     color=HUE, edgecolor='k', linewidth=0.6)
        for b, isb in zip(bars, summ['baseline']):
            if isb: b.set_hatch('//'); b.set_alpha(0.55)
        for i, (m, e) in enumerate(zip(summ[col], summ[se])):
            a.text(i, m + e + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        a.set_xticks(range(len(summ))); a.set_xticklabels(summ['method'], fontsize=8.5, rotation=20)
        a.set_ylabel(ylab, fontsize=9); a.set_title(title, fontsize=11)
        a.spines[['top', 'right']].set_visible(False); a.grid(axis='y', alpha=0.25)
        a.set_ylim(0, min(1.0, max(summ[col] + summ[se]) * 1.25 + 0.02))
    fig.suptitle('Did each method pick genes with the TRUE combo genes\' perturbation effect?\n'
                 f'(Norman held-out combos, n={int(summ["n"].iloc[0])}; hatched = non-learned baselines)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(R, 'figures', 'effect_cluster_softrecall.png') if os.path.isdir(os.path.join(R, 'figures'))
                else os.path.join(OUT, 'effect_cluster_softrecall.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'effect_cluster_softrecall.png'), dpi=200, bbox_inches='tight')
    plt.close(fig); print('saved effect_cluster_softrecall.png')


def _fig_clustermap(ec):
    # 2D MDS of the screened genes' observed-DE profiles, coloured by effect module
    from sklearn.manifold import MDS
    D = 1.0 - np.corrcoef(ec.E_obs); np.fill_diagonal(D, 0.0); D = np.clip(D, 0, 2)
    xy = MDS(n_components=2, dissimilarity='precomputed', random_state=0,
             normalized_stress='auto').fit_transform(D)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=ec.labels, cmap='tab20', s=40, edgecolor='k', linewidth=0.3)
    for i, g in enumerate(ec.ref_genes):
        ax.annotate(g, xy[i], fontsize=6, alpha=0.7)
    ax.set_title(f'Perturbation-effect modules of the {len(ec.ref_genes)} screened genes\n'
                 f'(MDS of observed-DE correlation; {ec.K} clusters)', fontsize=11)
    ax.set_xlabel('MDS-1'); ax.set_ylabel('MDS-2')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'effect_cluster_map.png'), dpi=200, bbox_inches='tight')
    plt.close(fig); print('saved effect_cluster_map.png')


if __name__ == '__main__':
    main()

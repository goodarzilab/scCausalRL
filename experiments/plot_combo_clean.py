"""Combo-prediction figure (paper quality): every model evaluated on the SAME held-out
test combos (the exp14 30%-held-out split, n≈31), each shown with ONLY its trained
predictor, baselines once, SEM error bars, single colour. Metrics: DEG-AUROC + pearson-DEG.
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(R, 'figures')
PP = 'norman_combo/combo_per_pert.csv'
CK = f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/checkpoints/phase1_epoch150.pt'

# held-out test combos (the only combos exp14 did NOT train on; all other models are
# singles-trained so they are zero-shot on these too → fully fair, same set for all).
test_fs = {frozenset(k) for k in torch.load(CK, map_location='cpu', weights_only=False)['config']['phase1']['combo_test_keys']}

def filt(d):
    x = pd.read_csv(f'{R}/{d}/{PP}')
    x['fs'] = x['combo'].map(lambda s: frozenset(s.split('+')))
    return x[x['fs'].isin(test_fs)]

# (label, results_dir, predictor, is_baseline)
spec = [
    ('corr',                       'exp14_norman_combotrain/results/grid/exp14_norman_combotrain', 'corr', True),
    ('additive_obs\n(obs oracle)', 'exp14_norman_combotrain/results/grid/exp14_norman_combotrain', 'additive_obs', True),
    ('exp11\nshift_add',           'exp11_norman_shift/results/phase1', 'shift_add', False),
    ('exp12-load\nu-score (loading)', 'exp12_shift_mutilate/results/grid/exp12_norman_loadclamp', 'shift_uscore', False),
    ('exp12-shift\nu-score (shift)',  'exp12_shift_mutilate/results/grid/exp12_norman_shiftclamp', 'shift_uscore', False),
    ('exp14\nu-score\n(combo-trained)', 'exp14_norman_combotrain/results/grid/exp14_norman_combotrain', 'shift_uscore', False),
]
cache = {}
def stats(d, m, col):
    if d not in cache: cache[d] = filt(d)
    v = cache[d][f'{m}_{col}'].dropna().values
    return v.mean(), (v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.0), len(v)

n_combos = len(filt(spec[0][1]))
COLOR = '#3C5488'  # single professional colour
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, col, title, ylo in [(axes[0], 'auroc', 'DEG-AUROC', 0.70),
                            (axes[1], 'pearson_deg', 'pearson-DEG', 0.0)]:
    means, sems, labs, hatches = [], [], [], []
    for lab, d, m, isb in spec:
        mu, se, _ = stats(d, m, col); means.append(mu); sems.append(se); labs.append(lab)
        hatches.append('//' if isb else '')
    xs = np.arange(len(labs))
    bars = ax.bar(xs, means, yerr=sems, capsize=3, color=COLOR,
                  edgecolor='black', linewidth=0.6, error_kw=dict(lw=1.0))
    for b, h in zip(bars, hatches):
        if h: b.set_hatch(h); b.set_alpha(0.55)
    for i, (mu, se) in enumerate(zip(means, sems)):
        ax.text(i, mu + se + 0.008, f'{mu:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel(col.replace('_', '-')); ax.set_title(title, fontsize=11)
    ax.set_ylim(ylo, 1.0); ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    if ylo > 0: ax.axhline(0.5, ls='--', c='gray', lw=0.7)
fig.suptitle(f'Norman combo prediction on held-out combos (n={n_combos}; SEM error bars; hatched = baselines)',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f'{FIG}/combo_prediction_clean.png'
fig.savefig(out, dpi=200, bbox_inches='tight'); print('saved', out, '| n_test_combos =', n_combos)

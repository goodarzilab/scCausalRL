"""Epistasis figure: bin held-out combos by non-additivity (E quartiles Q1→Q4) and show,
per model, the combo do-op prediction vs the baselines, across the epistasis gradient.

Panel A — absolute do-op pearson per quartile, with the OBSERVED-singles additive oracle
  (data-only, same for all models). Everyone declines as epistasis rises; ugate is best.
Panel B — do-op ADVANTAGE over the deployable MODEL-additive (gap_vs_modeladd) per quartile.
  This is the epistasis signal: a model that captures epistasis should gain MORE where
  additivity fails (Q4). ugate's advantage grows into Q4; esm_con's shrinks.
Reads results/epistasis_quartile/epistasis_q/summary.json.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose'
j = json.load(open(f'{ROOT}/results/epistasis_quartile/epistasis_q/summary.json'))
M = j['models']
QS = ['Q1', 'Q2', 'Q3', 'Q4']
NAMES = {'v2_esm_con': 'esm_con (baseline)', 'v2_esm_con_ugate': 'ugate',
         'v2_esm_con_epsmix': 'epsmix', 'v2_esm_con_ugate_epsmix': 'ugate_epsmix'}
COL = {'v2_esm_con': '#3C5488', 'v2_esm_con_ugate': '#E64B35',
       'v2_esm_con_epsmix': '#00A087', 'v2_esm_con_ugate_epsmix': '#F39B7F'}
xe = [M['v2_esm_con'][q]['mean_E'] for q in QS]          # mean epistasis per quartile (x-tick)
xt = [f'{q}\nE={e:.2f}' for q, e in zip(QS, xe)]
x = np.arange(4)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
# Panel A — absolute do-op pearson + oracle
for m in M:
    ax[0].plot(x, [M[m][q]['do_pearson'] for q in QS], '-o', color=COL[m], lw=2, label=NAMES.get(m, m))
ax[0].plot(x, [M['v2_esm_con'][q]['addobs_pearson'] for q in QS], '--s', color='darkred', lw=1.6,
           label='OBSERVED-singles oracle')
ax[0].axhline(0, c='k', lw=0.5)
ax[0].set_xticks(x); ax[0].set_xticklabels(xt, fontsize=9)
ax[0].set_xlabel('epistasis quartile (Q1 additive → Q4 epistatic)')
ax[0].set_ylabel('combo do-op pearson_delta')
ax[0].set_title('(A) Combo prediction vs epistasis', fontsize=11)
ax[0].legend(fontsize=8, frameon=False); ax[0].grid(alpha=0.25); ax[0].spines[['top','right']].set_visible(False)
# Panel B — ABSOLUTE do-op (solid) vs the model's own additive (dashed), per quartile.
# (Replaces the confounded gap_vs_modeladd: a big gap can just mean a broken additive
# baseline, not a strong do-op. Plotting both absolutes shows the real picture.)
for m in M:
    do = [M[m][q]['do_pearson'] for q in QS]
    add = [M[m][q]['do_pearson'] - M[m][q]['gap_vs_modeladd'] for q in QS]   # model-additive
    ax[1].plot(x, do, '-o', color=COL[m], lw=2, label=f'{NAMES.get(m, m)} (do-op)')
    ax[1].plot(x, add, '--x', color=COL[m], lw=1.3, alpha=0.7)
ax[1].axhline(0, c='k', lw=0.6)
ax[1].set_xticks(x); ax[1].set_xticklabels(xt, fontsize=9)
ax[1].set_xlabel('epistasis quartile (Q1 additive → Q4 epistatic)')
ax[1].set_ylabel('combo pearson_delta (solid=do-op, dashed=MODEL-additive)')
ax[1].set_title('(B) do-op (solid) vs the SAME model\'s additive (dashed)', fontsize=10.5)
ax[1].legend(fontsize=7.5, frameon=False); ax[1].grid(alpha=0.25); ax[1].spines[['top','right']].set_visible(False)
fig.suptitle('exp19 epistasis: held-out Joung combos binned by non-additivity (ep60)\n'
             'All models (and the oracle) DEGRADE as epistasis rises; ugate is the best combo predictor at every level.',
             fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f'{ROOT}/results/epistasis_quartile/epistasis_quartile.png'
fig.savefig(out, dpi=200, bbox_inches='tight'); print('saved', out)

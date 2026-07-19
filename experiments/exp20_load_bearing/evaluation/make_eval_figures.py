"""Comprehensive eval figures (prior-off): SINGLES (all arms, test_disc + A_gap, SEM error bars across
the 49 held-out TFs) and DOUBLES (top model: do-op vs additive baselines + epistasis quartiles, SEM
across the 40 held-out pairs). Reads full_eval_table.csv + combo_eval/<top>/summary.json.

Run: python evaluation/make_eval_figures.py
"""
import os, csv, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FAM = {'learned':'#2a78d6', 'masked':'#1baf7a', 'frozen':'#eda100', 'localized':'#4a3aa7', 'loc+masked':'#c0392b'}
CHANCE, NN, CEIL = 0.500, 0.602, 0.699
MUT = '#6b6b6b'
TOP = 'v2b_retrfix_refresh_k20'
os.makedirs('results/figures', exist_ok=True)

rows = list(csv.DictReader(open('results/figures/full_eval_table.csv')))
g = lambda r, k: float(r.get(k, 0) or 0)

# ============ FIGURE 1 — SINGLES ============
fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 8))

# Panel A: test_disc bars, sorted, family color, SEM error bars, reference lines
rs = sorted(rows, key=lambda r: g(r, 'test_disc'))
y = np.arange(len(rs))
axA.barh(y, [g(r, 'test_disc') for r in rs], xerr=[g(r, 'test_disc_sem') for r in rs],
         color=[FAM.get(r['family'], '#888') for r in rs], alpha=0.9,
         error_kw=dict(ecolor='#333', elinewidth=0.8, capsize=2))
axA.set_yticks(y); axA.set_yticklabels([r['tag'].replace('v2b_', '') for r in rs], fontsize=6.5)
for x, c, t in [(CHANCE, MUT, 'chance'), (NN, '#d03b3b', 'ESM-NN 0.60'), (CEIL, '#0ca30c', 'ceiling 0.70')]:
    axA.axvline(x, ls='--', lw=1.2, color=c); axA.text(x, len(rs)+0.4, t, color=c, fontsize=8, ha='center')
axA.set_xlim(0.46, 0.72); axA.set_xlabel('held-out disc_resid  (±1 SEM over 49 test TFs)', fontsize=11)
axA.set_title('SINGLES: zero-shot discrimination, all arms (prior-off)', fontsize=12)
axA.legend(handles=[Line2D([0],[0],marker='s',color='w',mfc=c,ms=9,label=f) for f,c in FAM.items()],
           title='decoder', loc='lower right', fontsize=8)

# Panel B: A_gap vs test_disc scatter, SEM both axes
for r in rows:
    c = FAM.get(r['family'], '#888'); best = r['tag'] == TOP
    axB.errorbar(g(r,'test_disc'), g(r,'A_gap'), xerr=g(r,'test_disc_sem'), yerr=g(r,'A_gap_sem'),
                 fmt='*' if best else 'o', ms=20 if best else 8, mfc=c, mec='k' if best else 'white',
                 mew=1.6 if best else 0.6, ecolor=c, elinewidth=1, capsize=2, alpha=0.9, zorder=5 if best else 3)
axB.axvline(CHANCE, ls='--', lw=1, color=MUT, alpha=0.6); axB.axvline(NN, ls='--', lw=1.1, color='#d03b3b')
axB.text(NN, axB.get_ylim()[1], ' ESM-NN', color='#d03b3b', fontsize=8, va='top')
axB.annotate('top model\n(refresh_k20)', (g([r for r in rows if r['tag']==TOP][0],'test_disc'),
             g([r for r in rows if r['tag']==TOP][0],'A_gap')), textcoords='offset points', xytext=(10,-24),
             fontsize=8.5, fontweight='bold')
axB.set_xlabel('held-out disc_resid  →', fontsize=11); axB.set_ylabel('A_gap  (↑ latent DAG A load-bearing)', fontsize=11)
axB.set_title('SINGLES: the A-load-bearing ↔ prediction tradeoff', fontsize=12); axB.grid(alpha=0.15)
axB.legend(handles=[Line2D([0],[0],marker='o',color='w',mfc=c,ms=8,label=f) for f,c in FAM.items()],
           title='decoder', loc='upper right', fontsize=8)
fig.tight_layout(); fig.savefig('results/figures/eval_singles.png', dpi=150, bbox_inches='tight')
print('wrote results/figures/eval_singles.png')

# ============ FIGURE 2 — DOUBLES (top model) ============
S = json.load(open(f'results/combo_eval/{TOP}/summary.json'))
fig2, (axC, axD) = plt.subplots(1, 2, figsize=(15, 6.5))
MCOL = {'do': '#c0392b', 'add': '#1baf7a', 'addobs': '#888'}
MLAB = {'do': 'do-op (structural)', 'add': 'model-additive', 'addobs': 'obs-additive (null)'}

# Panel C: do vs additive baselines — disc + pearson, SEM over the 40 pairs
ps = S['pairs_per_set']; methods = ['do', 'add', 'addobs']
xc = np.arange(2); w = 0.25
for i, m in enumerate(methods):
    dvals = np.array([v for v in ps[m]['disc'] if v == v]); pvals = np.array([v for v in ps[m]['pear'] if v == v])
    means = [dvals.mean(), pvals.mean()]; sems = [dvals.std(ddof=1)/np.sqrt(len(dvals)), pvals.std(ddof=1)/np.sqrt(len(pvals))]
    axC.bar(xc + (i-1)*w, means, w, yerr=sems, color=MCOL[m], label=MLAB[m], alpha=0.9,
            error_kw=dict(ecolor='#333', elinewidth=0.9, capsize=3))
axC.axhline(0.5, ls='--', lw=1, color=MUT); axC.text(0.02, 0.5, 'chance', color=MUT, fontsize=8, va='bottom')
axC.set_xticks(xc); axC.set_xticklabels(['disc_resid', 'pearson (drift-removed)'], fontsize=10)
axC.set_title('DOUBLES: do-op vs additive baselines (top model, 40 pairs, ±SEM)', fontsize=11.5)
axC.legend(fontsize=9); axC.set_ylim(0, 0.95)

# Panel D: epistasis quartiles — do-op vs additive pearson
Q = S['epistasis_quartiles']; qs = ['Q1', 'Q2', 'Q3', 'Q4']; xq = np.arange(4)
for m, c, lab in [('do', MCOL['do'], 'do-op'), ('add', MCOL['add'], 'model-add'), ('addobs', MCOL['addobs'], 'obs-add')]:
    axD.plot(xq, [Q[q][m][1] for q in qs], 'o-', color=c, label=lab, lw=2, ms=8)
axD.set_xticks(xq); axD.set_xticklabels([f"{q}\nE={Q[q]['meanE']}" for q in qs], fontsize=9)
axD.set_xlabel('epistasis quartile (Q4 = most non-additive) →', fontsize=10.5)
axD.set_ylabel('pearson (drift-removed)', fontsize=10.5)
axD.set_title('DOUBLES: do-op never beats additive, even at high epistasis', fontsize=11.5)
axD.legend(fontsize=9); axD.grid(alpha=0.15)
fig2.tight_layout(); fig2.savefig('results/figures/eval_doubles.png', dpi=150, bbox_inches='tight')
print('wrote results/figures/eval_doubles.png')

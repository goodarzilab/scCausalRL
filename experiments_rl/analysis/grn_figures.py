"""Clean figures for the GRN load-bearing / interpretability diagnostics (pre-fix models).
Fig 1  CollecTRI AUROC is a decoder/prior artifact (real vs shuffled-A vs random-D vs corr vs prior-off)
Fig 2  The latent DAG is NOT load-bearing for prediction (real-A vs A=0 vs shuffled-A)
Fig 3  Product-sparsity is degenerate: collapses edge WEIGHTS, not topology (gates stay on)
Run: python analysis/grn_figures.py
"""
import os, sys
import numpy as np, pandas as pd, torch, numpy.linalg as la
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
EXP19 = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose'
FIG = f'{EXP19}/results/figures/grn'; os.makedirs(FIG, exist_ok=True)
CSV = f'{HERE}/grn_dag_sweep.csv'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 140})

df = pd.read_csv(CSV).set_index('WM')
B = 'v2_esm_con_ugate'                                            # baseline singles WM

# ───────────────────────── Fig 1 — CollecTRI causality ─────────────────────────
fig, ax = plt.subplots(figsize=(6.6, 4.2))
bars = {'real GRN\nD·A·Dᵀ': df.loc[B, 'collectri_auroc'], 'shuffle\nlatent A': df.loc[B, 'shuffA'],
        'randomize\ndecoder D': df.loc[B, 'randomD'], 'gene–gene\ncorr': df.loc[B, 'corrGRN'],
        'prior OFF\n(retrain)': df.loc['grn_prior0_spA', 'collectri_auroc']}
cols = ['#2c7fb8', '#41b6c4', '#bdbdbd', '#bdbdbd', '#d95f0e']
x = np.arange(len(bars)); ax.bar(x, list(bars.values()), color=cols, width=0.62)
for i, v in enumerate(bars.values()): ax.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=10)
ax.axhline(0.5, ls='--', c='k', lw=1, alpha=0.6); ax.text(len(bars) - 0.4, 0.515, 'chance', fontsize=9, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(bars.keys(), fontsize=9.5); ax.set_ylim(0.4, 1.02)
ax.set_ylabel('CollecTRI recovery AUROC')
ax.set_title('GRN–CollecTRI agreement is a decoder/prior artifact, not causal A\n'
             'shuffling A barely changes it · randomizing D → chance · no prior → corr baseline', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_collectri_causality.png'); plt.close(fig)

# ───────────────────────── Fig 2 — DAG load-bearing ─────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 4.2))
groups = ['singles\npΔ_resid', 'singles\neffect-spearman', 'combos\npΔ_resid']
real = [df.loc[B, 's_pear_real'], df.loc[B, 's_spear_real'], df.loc[B, 'c_pear_real']]
zero = [df.loc[B, 's_pear_zero'], df.loc[B, 's_spear_zero'], df.loc[B, 'c_pear_zero']]
shuf = [np.nan, df.loc[B, 's_spear_shuf'], df.loc[B, 'c_pear_shuf']]
x = np.arange(len(groups)); w = 0.26
ax.bar(x - w, real, w, label='real A', color='#2c7fb8')
ax.bar(x, zero, w, label='A = 0 (no DAG)', color='#d95f0e')
ax.bar(x + w, shuf, w, label='shuffled A', color='#bdbdbd')
ax.axhline(0, c='k', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel('drift-removed match'); ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_title('The latent DAG is NOT load-bearing: removing A (A=0) matches or beats real A\n'
             '(baseline ugate WM; pattern universal across all 14 ablations)', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_dag_load_bearing.png'); plt.close(fig)

# ───────────────────────── Fig 3 — product-sparsity degeneracy ─────────────────────────
sys.path.insert(0, EXP19); torch.set_grad_enabled(False)
from evaluation.run_shift_eval_norman import load_shift_model
sweep = [('λ=0\n(ugate)', 'v2_esm_con_ugate'), ('λ=1', 'grn_spA1'), ('λ=5', 'grn_spA5'),
         ('λ=10', 'grn_spA10'), ('λ=30', 'grn_spA30')]
gate_on, wmean, dens = [], [], []
for _, tag in sweep:
    m, _, _ = load_shift_model(f'{EXP19}/results/grid/{tag}/checkpoints/phase1_epoch60.pt', 'cpu')
    th = m.dag.theta.numpy(); w = m.dag.weights.numpy(); mk = m.dag.mask.numpy(); tau = float(m.dag.tau)
    g = 1 / (1 + np.exp(-th / tau)) * mk; off = ~np.eye(th.shape[0], dtype=bool)
    gate_on.append(np.mean(g[off] > 0.9)); wmean.append(np.abs(w[off]).mean())
    dens.append(float(df.loc[tag, 'dens_solve']))
x = np.arange(len(sweep))
fig, ax = plt.subplots(figsize=(6.8, 4.2)); ax2 = ax.twinx()
l1, = ax.plot(x, wmean, 'o-', color='#d95f0e', label='|edge weights| (mean)')
l2, = ax.plot(x, gate_on, 's-', color='#2c7fb8', label='frac gates ON (>0.9)')
l3, = ax2.plot(x, dens, '^--', color='#7a7a7a', label='solve-A density (>1e-3)')
ax.set_xticks(x); ax.set_xticklabels([s for s, _ in sweep])
ax.set_xlabel('product-mode sparsity weight  λ_sparse'); ax.set_ylabel('weight mag / gate fraction'); ax2.set_ylabel('solve-A density')
ax.set_ylim(0, max(max(wmean), max(gate_on)) * 1.25); ax2.set_ylim(0, 1)
ax.legend(handles=[l1, l2, l3], frameon=False, fontsize=9, loc='center right')
ax.set_title("Product-L1 sparsity is degenerate: it collapses edge WEIGHTS (0.11→0.02),\n"
             "while gates stay ON — density drops only via weight-shrink, not edge pruning", fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_product_sparsity_degeneracy.png'); plt.close(fig)

print("saved 3 figures ->", FIG)
for f in ['fig1_collectri_causality', 'fig2_dag_load_bearing', 'fig3_product_sparsity_degeneracy']:
    print(" ", f + '.png')
print(f"\nFig1 AUROC: real {df.loc[B,'collectri_auroc']:.3f} | shufA {df.loc[B,'shuffA']:.3f} | randD {df.loc[B,'randomD']:.3f} | corr {df.loc[B,'corrGRN']:.3f} | priorOFF {df.loc['grn_prior0_spA','collectri_auroc']:.3f}")
print(f"Fig2 real/zero: s_pear {df.loc[B,'s_pear_real']:.3f}/{df.loc[B,'s_pear_zero']:.3f} | s_spear {df.loc[B,'s_spear_real']:.3f}/{df.loc[B,'s_spear_zero']:.3f} | c_pear {df.loc[B,'c_pear_real']:.3f}/{df.loc[B,'c_pear_zero']:.3f}")
print(f"Fig3 |w|: {[round(v,3) for v in wmean]} | gate_on: {[round(v,2) for v in gate_on]} | dens: {[round(v,2) for v in dens]}")

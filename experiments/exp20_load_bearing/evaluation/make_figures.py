"""Step-by-step figure story for the exp20 load-bearing-A investigation.
Reads the sweep CSVs + heldout summaries + synthetic logs; emits PNGs to docs/figures/.
Palette = dataviz reference (validated); every bar gets a direct value label (contrast relief rule);
one axis per panel; chance/ceiling reference lines; recessive chrome.
"""
import os, sys, json, glob, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RL = '/home/sgoudarzi/scCausalRL/experiments_rl/analysis'
EXP = '/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing'
OUT = f'{EXP}/docs/figures'; os.makedirs(OUT, exist_ok=True)

# --- palette (dataviz reference, light) ---
SURF, INK, SEC, MUT = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, BASE = '#e1e0d9', '#c3c2b7'
BLUE, AQUA, YEL, VIO, ORA = '#2a78d6', '#1baf7a', '#eda100', '#4a3aa7', '#eb6834'
GOOD, CRIT, WARN = '#0ca30c', '#d03b3b', '#fab219'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def style(ax, xlabel='', ylabel='', title=''):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=SEC, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=SEC, fontsize=10)
    if title: ax.set_title(title, color=INK, fontsize=11, fontweight='bold', loc='left', pad=8)


def grn(grid):
    return pd.read_csv(f'{RL}/grn_dag_sweep_{grid}.csv')


def disc(tag):
    for name in (f'hz_{tag}', f'hz_x20_{tag}', f'hz_x20po_{tag}'):
        p = f'{EXP}/results/heldout_eval/{name}/summary.json'
        if os.path.exists(p):
            return json.load(open(p))
    return {}


# ============ FIG 1 — perturbation prediction ladder (nothing beats the ceiling) ============
def fig1():
    rows = []  # (label, disc, regime)
    po = {'struct_sparse_con_po': 'sparse_con', 'struct_sparse_pertonly_po': 'sparse_pertonly',
          'struct_min_esm_po': 'min_esm', 'struct_sparse_esm_po': 'sparse_esm',
          'struct_sparse_resid_po': 'sparse_resid', 'struct_sparse_all_po': 'sparse_all',
          'struct_sparse_residrecon_po': 'sparse_residrecon'}
    for t, lab in po.items():
        d = disc(t)
        if d: rows.append((lab, d['discrimination_cos_resid'], 'prior-off'))
    for t, lab in [('staged_con_s26', 'staged_con'), ('staged_esm_s26', 'staged_esm26'),
                   ('staged_esm_s36', 'staged_esm36'), ('staged_nonlinSCM_s26', 'nonlinSCM'),
                   ('R2_pseudobulk', 'R2·pseudobulk'), ('R4_pseudobulk_projectout', 'R4·pbk+proj'),
                   ('R6_R4_energy', 'R6·+energy'), ('whiten', 'whiten')]:
        d = disc(t)
        if d: rows.append((lab, d['discrimination_cos_resid'], 'objective sweep'))
    for t, lab in [('struct_frozenD64', 'frozenD'), ('frozenD64_nonlin', 'frozenD+nonlin'),
                   ('staged_frozenD64_s26', 'frozenD+staged')]:
        d = disc(t)
        if d: rows.append((lab, d['discrimination_cos_resid'], 'frozen decoder'))
    rows.sort(key=lambda r: r[1])
    labs = [r[0] for r in rows]; vals = [r[1] for r in rows]
    cmap = {'prior-off': BLUE, 'objective sweep': ORA, 'frozen decoder': VIO}
    cols = [cmap[r[2]] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.66, edgecolor=SURF, lw=1.2, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.004, yi, f'{v:.3f}', va='center', ha='left', color=SEC, fontsize=8, fontweight='bold')
    ax.axvline(0.5, color=CRIT, ls='--', lw=1.6, zorder=2)
    ax.text(0.5, len(rows) - 0.2, ' chance 0.50', color=CRIT, fontsize=8.5, va='top')
    ax.axvline(0.647, color=GOOD, ls='--', lw=1.6, zorder=2)
    ax.text(0.647, len(rows) - 1.4, ' exp19 ceiling 0.647', color='#006300', fontsize=8.5, va='top')
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8.5, color=INK)
    ax.set_xlim(0.48, 0.68); ax.grid(axis='y', visible=False)
    style(ax, xlabel='held-out drift-removed discrimination (disc_resid, chance 0.5)',
          title='1 · Perturbation prediction — no fix beats the prior-off ceiling')
    ax.legend(handles=[Patch(fc=cmap[k], label=k) for k in cmap], loc='lower right',
              frameon=False, fontsize=8.5)
    fig.text(0.01, 0.005, 'Zero-shot held-out TFs. Best = prior-off sparse_con 0.639 ≈ exp19 epsmix 0.647; '
             'whiten/resid/frozen collapse to chance.', color=MUT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f'{OUT}/fig1_prediction_ladder.png', dpi=150)
    plt.close(fig); print('fig1')


# ============ FIG 2 — causality: A is not load-bearing (real vs zero vs shuffled A) ============
def fig2():
    g = grn('x20po'); g['lab'] = g.WM.str.replace('struct_', '').str.replace('_po', '')
    key = ['sparse_con', 'sparse_pertonly', 'min_esm', 'sparse_esm']
    g = g[g.lab.isin(key)].set_index('lab').loc[key].reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={'width_ratios': [1.35, 1]})
    x = np.arange(len(g)); w = 0.26
    a1.bar(x - w, g.s_spear_real, w, color=BLUE, edgecolor=SURF, lw=1, label='real A', zorder=3)
    a1.bar(x, g.s_spear_zero, w, color=MUT, edgecolor=SURF, lw=1, label='A = 0', zorder=3)
    a1.bar(x + w, g.s_spear_shuf, w, color=BASE, edgecolor=SURF, lw=1, label='A shuffled', zorder=3)
    for xi, (r, z, s) in enumerate(zip(g.s_spear_real, g.s_spear_zero, g.s_spear_shuf)):
        for dx, v in [(-w, r), (0, z), (w, s)]:
            a1.text(xi + dx, v + 0.006, f'{v:.2f}', ha='center', color=SEC, fontsize=7)
    a1.set_xticks(x); a1.set_xticklabels(g.lab, fontsize=8.5, rotation=15, color=INK)
    a1.set_ylim(0, 0.55); a1.grid(axis='x', visible=False); a1.grid(axis='y', color=GRID, lw=0.6)
    style(a1, ylabel='drift-removed spearman (singles)',
          title='2 · Causality: real ≈ zero ≈ shuffled A')
    a1.legend(frameon=False, fontsize=8.5, loc='upper right')
    # A_gap across all prior-off arms
    gall = grn('x20po'); gall['lab'] = gall.WM.str.replace('struct_', '').str.replace('_po', '')
    gall['agap'] = gall.s_spear_real - np.maximum(gall.s_spear_zero, gall.s_spear_shuf)
    gall = gall.sort_values('agap')
    yy = np.arange(len(gall))
    cols = [GOOD if v > 0.05 else (CRIT if v < -0.02 else MUT) for v in gall.agap]
    a2.barh(yy, gall.agap, color=cols, height=0.6, edgecolor=SURF, lw=1, zorder=3)
    for yi, v in zip(yy, gall.agap):
        a2.text(v + (0.002 if v >= 0 else -0.002), yi, f'{v:+.3f}', va='center',
                ha='left' if v >= 0 else 'right', color=SEC, fontsize=7.5)
    a2.axvline(0, color=BASE, lw=1)
    a2.set_yticks(yy); a2.set_yticklabels(gall.lab, fontsize=8, color=INK)
    a2.set_xlim(-0.09, 0.09); a2.grid(axis='y', visible=False)
    style(a2, xlabel='A_gap = real − max(zero, shuf)', title='A_gap ≈ 0 (all arms)')
    fig.text(0.01, 0.005, 'A neither improves prediction (A_gap≈0) nor is needed — removing/shuffling it barely changes the drift-removed match.',
             color=MUT, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=3); fig.savefig(f'{OUT}/fig2_causality_Agap.png', dpi=150)
    plt.close(fig); print('fig2')


# ============ FIG 3 — mechanism: decoder absorbs the operator; shift = magnitude not identity ==
def fig3():
    arms = ['min_esm', 'sparse_esm', 'sparse_con', 'sparse_pertonly']
    probe_A = [0.014, 0.007, 0.156, 0.459]
    probe_D = [1.007, 1.007, 0.984, 0.985]
    crossTF = [0.001, 0.000, 0.018, 0.018]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(arms)); w = 0.26
    b1 = ax.bar(x - w, probe_D, w, color=BLUE, edgecolor=SURF, lw=1, zorder=3, label='probe_Dperm (shuffle decoder)')
    b2 = ax.bar(x, probe_A, w, color=ORA, edgecolor=SURF, lw=1, zorder=3, label='probe_A (remove A)')
    b3 = ax.bar(x + w, crossTF, w, color=VIO, edgecolor=SURF, lw=1, zorder=3, label='crossTF_div (pred diversity)')
    for xi, (d, a, c) in enumerate(zip(probe_D, probe_A, crossTF)):
        for dx, v in [(-w, d), (0, a), (w, c)]:
            ax.text(xi + dx, v + 0.02, f'{v:.2f}', ha='center', color=SEC, fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=9, color=INK)
    ax.set_ylim(0, 1.15); ax.grid(axis='x', visible=False); ax.grid(axis='y', color=GRID, lw=0.6)
    style(ax, ylabel='1 − cos(Δ, Δ_ablated)   /   diversity',
          title='3 · Mechanism — the decoder carries prediction; A is a minor add-on; predictions collapse')
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', ncol=1)
    fig.text(0.01, 0.005, 'probe_Dperm≈1.0 (shuffling D destroys the output) ≫ probe_A (0.0–0.5). crossTF_div≈0.02 ⇒ ~99% identical predictions across TFs (mode collapse → chance discrimination).',
             color=MUT, fontsize=7.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f'{OUT}/fig3_mechanism.png', dpi=150)
    plt.close(fig); print('fig3')


# ============ FIG 4 — biology: the GRN is prior memorization, not causal recovery ============
def fig4():
    cats = ['CollecTRI\nprior-ON', 'CollecTRI\nprior-off', 'shuffled-A\nprior-off',
            'corr-GRN\nbaseline', 'fate-hierarchy\nprior-off', 'random-D']
    vals = [0.976, 0.569, 0.569, 0.568, 0.508, 0.49]
    cols = [WARN, BLUE, BASE, MUT, AQUA, BASE]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(cats))
    ax.bar(x, vals, color=cols, width=0.62, edgecolor=SURF, lw=1.2, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.008, f'{v:.2f}', ha='center', color=SEC, fontsize=9, fontweight='bold')
    ax.axhline(0.5, color=CRIT, ls='--', lw=1.5, zorder=2); ax.text(len(cats) - 0.5, 0.51, 'chance', color=CRIT, fontsize=8, ha='right')
    ax.axhline(0.568, color=MUT, ls=':', lw=1.3, zorder=2); ax.text(0.0, 0.578, 'co-expression floor', color=MUT, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.3, color=INK)
    ax.set_ylim(0.45, 1.02); ax.grid(axis='x', visible=False); ax.grid(axis='y', color=GRID, lw=0.6)
    style(ax, ylabel='GRN vs prior AUROC', title='4 · Biology — the projected GRN is prior memorization, not discovery')
    fig.text(0.01, 0.005, 'Prior-ON 0.98 collapses to the 0.57 co-expression floor when the prior is off; shuffling A doesn’t change it (lives in the decoder). '
             'The fetal-atlas fate hierarchy is at chance (0.51) — no developmental structure recovered.', color=MUT, fontsize=7.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f'{OUT}/fig4_biology_grn.png', dpi=150)
    plt.close(fig); print('fig4')


# ============ FIG 5 — R-sweep: objective breaks mode collapse + engages A, but not discrimination
def fig5():
    R = ['R1\nper-ex', 'R2\npseudobulk', 'R3\nproject', 'R4\npbk+proj', 'R5\n+contr10', 'R6\n+energy']
    crossTF = [0.031, 0.193, 0.029, 0.204, 0.226, 0.166]
    probe_A = [0.16, 0.68, 0.15, 0.87, 1.04, 0.57]
    disc = [0.502, 0.503, 0.502, 0.502, 0.504, 0.505]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1, 1]})
    x = np.arange(len(R)); w = 0.38
    a1.bar(x - w / 2, crossTF, w, color=VIO, edgecolor=SURF, lw=1, zorder=3, label='crossTF_div')
    a1.bar(x + w / 2, [p / 1.1 for p in probe_A], w, color=ORA, edgecolor=SURF, lw=1, zorder=3, label='probe_A (÷1.1 scale)')
    for xi, (c, p) in enumerate(zip(crossTF, probe_A)):
        a1.text(xi - w / 2, c + 0.02, f'{c:.2f}', ha='center', color=SEC, fontsize=7.5)
        a1.text(xi + w / 2, p / 1.1 + 0.02, f'{p:.2f}', ha='center', color=SEC, fontsize=7.5)
    a1.axhline(0.10, color=WARN, ls='--', lw=1.4); a1.text(len(R) - .5, 0.11, 'gate 0.10', color='#9a6b00', fontsize=8, ha='right')
    a1.set_xticks(x); a1.set_xticklabels(R, fontsize=8, color=INK); a1.set_ylim(0, 1.05)
    a1.grid(axis='x', visible=False); a1.grid(axis='y', color=GRID, lw=0.6)
    style(a1, title='5 · Pseudobulk breaks collapse, engages A')
    a1.legend(frameon=False, fontsize=8.5, loc='upper left')
    a2.bar(x, disc, 0.6, color=[AQUA if 'pseudobulk' in r or 'pbk' in r or 'contr' in r or 'energy' in r else BASE for r in R],
           edgecolor=SURF, lw=1.2, zorder=3)
    for xi, v in zip(x, disc): a2.text(xi, v + 0.002, f'{v:.3f}', ha='center', color=SEC, fontsize=8.5, fontweight='bold')
    a2.axhline(0.5, color=CRIT, ls='--', lw=1.5); a2.text(len(R) - .5, 0.5008, 'chance', color=CRIT, fontsize=8, ha='right', va='bottom')
    a2.set_xticks(x); a2.set_xticklabels(R, fontsize=8, color=INK); a2.set_ylim(0.49, 0.56)
    a2.grid(axis='x', visible=False); a2.grid(axis='y', color=GRID, lw=0.6)
    style(a2, ylabel='disc_resid', title='…but disc stays at CHANCE')
    fig.text(0.01, 0.005, 'Pseudobulk (R2/R4/R5/R6) lifts crossTF 0.03→0.2 and probe_A→0.9 (A now engaged) — yet held-out disc stays 0.50: diverse-but-WRONG. '
             'The wall is zero-shot generalization, not the objective.', color=MUT, fontsize=7.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=3); fig.savefig(f'{OUT}/fig5_Rsweep.png', dpi=150)
    plt.close(fig); print('fig5')


# ============ FIG 6 — generalization is the wall: train vs held-out ============
def fig6():
    arms = ['sparse_con\n(best)', 'R2\npseudobulk', 'R4\npbk+proj', 'R6\n+energy']
    tr = [0.926, 0.573, 0.558, 0.601]; te = [0.618, 0.503, 0.502, 0.505]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(arms)); w = 0.36
    ax.bar(x - w / 2, tr, w, color=BLUE, edgecolor=SURF, lw=1, zorder=3, label='TRAIN genes (seen)')
    ax.bar(x + w / 2, te, w, color=AQUA, edgecolor=SURF, lw=1, zorder=3, label='held-out genes (zero-shot)')
    for xi, (a, b) in enumerate(zip(tr, te)):
        ax.text(xi - w / 2, a + 0.008, f'{a:.2f}', ha='center', color=SEC, fontsize=8.5, fontweight='bold')
        ax.text(xi + w / 2, b + 0.008, f'{b:.2f}', ha='center', color=SEC, fontsize=8.5, fontweight='bold')
        ax.annotate('', (xi + w / 2, b), (xi - w / 2, a), arrowprops=dict(arrowstyle='->', color=CRIT, lw=1.3))
    ax.axhline(0.5, color=CRIT, ls='--', lw=1.4); ax.text(len(arms) - .5, 0.505, 'chance', color=CRIT, fontsize=8, ha='right')
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5, color=INK); ax.set_ylim(0.48, 1.0)
    ax.grid(axis='x', visible=False); ax.grid(axis='y', color=GRID, lw=0.6)
    style(ax, ylabel='discrimination (drift-removed)', title='6 · Generalization is the wall — models memorize seen TFs, don’t transfer')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    fig.text(0.01, 0.005, 'sparse_con nails TRAIN TFs (0.93) but drops to 0.62 held-out (+0.31 gap). Shift ‖S‖ stays full-magnitude on unseen TFs — confidently WRONG direction, not mean-regression.',
             color=MUT, fontsize=7.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f'{OUT}/fig6_train_vs_heldout.png', dpi=150)
    plt.close(fig); print('fig6')


# ============ FIG 7 — synthetic positive control: method sound, A non-identifiable ============
def fig7():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # panel A: disc_resid vs graph_AUROC across synthetic settings
    sets = ['soft gate\n(cyclic)', 'L0=0.5\n(best)', 'perturbation-\nblind (linear)', 'perturbation-\nblind (nonlin)']
    disc = [0.659, 0.661, 0.895, 0.875]; graph = [0.477, 0.475, 0.537, 0.495]
    x = np.arange(len(sets)); w = 0.38
    a1.bar(x - w / 2, disc, w, color=AQUA, edgecolor=SURF, lw=1, zorder=3, label='held-out disc_resid')
    a1.bar(x + w / 2, graph, w, color=VIO, edgecolor=SURF, lw=1, zorder=3, label='graph-recovery AUROC')
    for xi, (d, gg) in enumerate(zip(disc, graph)):
        a1.text(xi - w / 2, d + 0.01, f'{d:.2f}', ha='center', color=SEC, fontsize=8, fontweight='bold')
        a1.text(xi + w / 2, gg + 0.01, f'{gg:.2f}', ha='center', color=SEC, fontsize=8)
    a1.axhline(0.5, color=CRIT, ls='--', lw=1.4); a1.text(len(sets) - .5, 0.51, 'chance', color=CRIT, fontsize=8, ha='right')
    a1.set_xticks(x); a1.set_xticklabels(sets, fontsize=8, color=INK); a1.set_ylim(0.42, 1.0)
    a1.grid(axis='x', visible=False); a1.grid(axis='y', color=GRID, lw=0.6)
    style(a1, ylabel='synthetic (planted A*)', title='7 · Synthetic: generalizes, A not recoverable')
    a1.legend(frameon=False, fontsize=8.5, loc='upper left')
    # panel B: L0 frontier — disc vs gate_open
    l0 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    dl = [0.645, 0.647, 0.661, 0.609, 0.587, 0.576, 0.582]
    gate = [0.84, 0.61, 0.45, 0.36, 0.30, 0.25, 0.21]
    a2.plot(gate, dl, '-o', color=BLUE, lw=2, ms=7, mec=SURF, zorder=3)
    for gx, dy, lx in zip(gate, dl, l0):
        a2.text(gx, dy + 0.004, f'L0={lx}', ha='center', color=MUT, fontsize=7)
    a2.axhline(0.5, color=CRIT, ls='--', lw=1.4); a2.text(0.85, 0.503, 'chance', color=CRIT, fontsize=8)
    a2.set_xlim(0.9, 0.15); a2.set_ylim(0.49, 0.69)
    a2.grid(color=GRID, lw=0.6)
    style(a2, xlabel='gate open fraction  (← sparser)', ylabel='held-out disc_resid',
          title='L0 frontier: sparsity ≠ recovery')
    fig.text(0.01, 0.005, 'On ideal data the shift-encoder generalizes (disc 0.66; perturbation-blind localization → 0.90) but graph-recovery stays at chance (~0.5): '
             'A is non-identifiable — the shift+decoder are a sufficient statistic. Forcing the gate sparse never lifts recovery.', color=MUT, fontsize=7.1)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=3); fig.savefig(f'{OUT}/fig7_synthetic.png', dpi=150)
    plt.close(fig); print('fig7')


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
    try:
        f()
    except Exception as e:
        print('FAIL', f.__name__, e)
print('saved ->', OUT)

"""Thorough summary figure panels for the exp17 Joung TF-atlas ablation.

Loads the model + baseline eval summaries and builds 4 multi-panel PNGs in
experiments/figures/joung/:
  exp17_singles_panels.png   held-out TF zero-shot: model (mutilate vs additive) + baseline BARS
  exp17_combos_panels.png    singles->combo composition: do-op vs additive + baseline BARS
  exp17_pairs_vs_triples.png combo breakdown by set size
  exp17_per_tf_dists.png     per-TF distributions + example pred-vs-obs scatter

Baselines (context_mean, perturb_mean, corr, coexpression, additive) are drawn as their own
bars next to the models. Run (base env, after the evals):  python evaluation/plot_summary.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.abspath(os.path.join(ROOT, '..', 'figures', 'joung')); os.makedirs(FIG, exist_ok=True)

MODELS = ['v2_orig', 'v2_esm', 'v2_esm_resid', 'v2_esm_resid_con', 'v2_esm_con']
LABEL = {'v2_orig': 'orig', 'v2_esm': 'esm', 'v2_esm_resid': 'esm_resid*',
         'v2_esm_resid_con': 'esm_resid_con*', 'v2_esm_con': 'esm_con'}
COL = dict(zip(MODELS, ['#7f7f7f', '#1f77b4', '#d62728', '#9467bd', '#2ca02c']))
BCOL = {'context_mean': '#c7c7c7', 'perturb_mean': '#525252', 'additive': '#111111',
        'corr': '#8c564b', 'coexpression': '#e377c2'}
BLAB = {'context_mean': 'context_mean', 'perturb_mean': 'perturb_mean', 'additive': 'additive_obs',
        'corr': 'corr', 'coexpression': 'coexp'}
NAN = float('nan')


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def g(d, k):
    v = (d or {}).get(k, NAN)
    return NAN if v is None else v


def panel(ax, title, ylabel, model_a, model_b, lab_a, lab_b, base_vals,
          chance=None, zero=False, ylim=None):
    """model_a/model_b: per-model values (grouped bars). base_vals: dict name->value (single bars)."""
    nM = len(MODELS); x = np.arange(nM); w = 0.38
    ax.bar(x - w/2, model_a, w, color=[COL[m] for m in MODELS], label=lab_a)
    ax.bar(x + w/2, model_b, w, color=[COL[m] for m in MODELS], alpha=0.45, hatch='//',
           edgecolor='white', label=lab_b)
    # baseline bars, offset to the right with a divider
    bx = nM + 0.6
    xticks, xlabs = list(x), [LABEL[m] for m in MODELS]
    for name, val in base_vals.items():
        ax.bar(bx, val, 0.7, color=BCOL.get(name, '#999999'), edgecolor='k', linewidth=0.4)
        xticks.append(bx); xlabs.append(BLAB.get(name, name)); bx += 1
    ax.axvline(nM - 0.2, ls='-', lw=0.8, color='k', alpha=0.25)
    if chance is not None:
        ax.axhline(chance, ls=':', lw=1.1, color='r', alpha=0.7); ax.text(0, chance, ' chance', fontsize=7, color='r', va='bottom')
    if zero:
        ax.axhline(0.0, ls=':', lw=1.0, color='k', alpha=0.5)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabs, rotation=40, ha='right', fontsize=7.5)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', ls=':', alpha=0.4)
    if ylim: ax.set_ylim(*ylim)


def fig_singles(S, B):
    bs = B.get('singles', {})
    bnames = ['context_mean', 'perturb_mean', 'corr', 'coexpression']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    specs = [
        ('pearson_delta',            'Pearson Δ (raw)',           dict()),
        ('effect_spearman',          'Effect-size Spearman',      dict()),
        ('deg_auroc',                'DEG-detection AUROC',       dict(chance=0.5, ylim=(0.4, 1.0))),
        ('pearson_delta_resid',      'Pearson Δ (drift-removed)', dict(zero=True)),
        ('discrimination_cos_resid', 'Discrimination (resid)',    dict(chance=0.5)),
        ('fate_top1',                'Fate top-1 (20-way)',       dict(chance=0.05)),
    ]
    for ax, (key, ylab, kw) in zip(axes.ravel(), specs):
        ma = [g(S.get(m, {}), key) for m in MODELS]
        mb = [g(S.get(m, {}), f'add_{key}') for m in MODELS]
        bv = {n: g(bs.get(n, {}), key) for n in bnames}
        panel(ax, ylab, ylab, ma, mb, 'mutilate', 'additive', bv, **kw)
    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle('exp17 Joung — held-out TF zero-shot (epoch 60): models (mutilate vs additive) + non-learned baselines (bars)\n'
                 '* = residual-target run (collapsed). Raw pearson/Spearman/AUROC are saturated by perturb_mean (drift); '
                 'the model only wins on drift-removed + discrimination.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{FIG}/exp17_singles_panels.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_combos(C, B):
    bc = B.get('combos', {}).get('all', {})
    bnames = ['perturb_mean', 'additive', 'corr', 'coexpression']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [
        ('pearson_delta',      'Pearson Δ',            dict()),
        ('deg_auroc',          'DEG-detection AUROC',  dict(chance=0.5, ylim=(0.4, 1.0))),
        ('effect_spearman',    'Effect-size Spearman', dict(zero=True)),
        ('discrimination_cos', 'Discrimination',       dict(chance=0.5)),
    ]
    for ax, (key, ylab, kw) in zip(axes.ravel(), specs):
        ma = [g(C.get(m, {}).get('all', {}), f'do_{key}') for m in MODELS]
        mb = [g(C.get(m, {}).get('all', {}), f'add_{key}') for m in MODELS]
        bv = {n: g(bc.get(n, {}), key) for n in bnames}
        panel(ax, ylab, ylab, ma, mb, 'do-op (mutilate)', 'additive (model)', bv, **kw)
    axes[0, 0].legend(fontsize=8, loc='upper right')
    fig.suptitle('exp17 Joung — singles→combo composition (all multi-TF sets, n=42): do-op vs model-additive + baselines (bars)\n'
                 'additive_obs (Σ OBSERVED singles) is the strong null — it beats every learned do-op.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"{FIG}/exp17_combos_panels.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_pairs_triples(C, B):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ylab in [(axes[0], 'pearson_delta', 'Combo Pearson Δ'),
                          (axes[1], 'deg_auroc', 'Combo DEG-AUROC')]:
        x = np.arange(len(MODELS)); w = 0.38
        pair = [g(C.get(m, {}).get('pairs', {}), f'do_{key}') for m in MODELS]
        tri = [g(C.get(m, {}).get('triples', {}), f'do_{key}') for m in MODELS]
        ax.bar(x - w/2, pair, w, color=[COL[m] for m in MODELS], label='pairs (n≈40)')
        ax.bar(x + w/2, tri, w, color=[COL[m] for m in MODELS], alpha=0.4, hatch='xx',
               edgecolor='white', label='triples (n=2!)')
        rv = g(B.get('combos', {}).get('pairs', {}).get('additive', {}), key)
        if np.isfinite(rv):
            ax.axhline(rv, ls='-.', color='k', alpha=0.6, lw=1.2)
            ax.text(len(MODELS) - 1, rv, f' additive_obs(pairs)={rv:.2f}', va='bottom', ha='right', fontsize=7)
        if 'auroc' in key: ax.axhline(0.5, ls=':', color='r', alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=30, ha='right', fontsize=8)
        ax.set_title(ylab, fontsize=10); ax.set_ylabel(ylab, fontsize=8); ax.grid(axis='y', ls=':', alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle('exp17 Joung — combo composition by set size (do-op). Triples n=2 → not statistically meaningful.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = f"{FIG}/exp17_pairs_vs_triples.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_per_tf():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    dists_p, dists_d, labels, cols = [], [], [], []
    for m in MODELS:
        f = f"{ROOT}/results/heldout_eval/{m}_ep60/per_tf.csv"
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        dists_p.append(df['pearson_delta'].dropna().values)
        dists_d.append(df['discrimination_beaten'].dropna().values)
        labels.append(LABEL[m]); cols.append(COL[m])
    for ax, dists, ttl, chance in [(axes[0, 0], dists_p, 'Per-TF Pearson Δ (raw)', None),
                                   (axes[0, 1], dists_d, 'Per-TF discrimination (beaten frac)', 0.5)]:
        bp = ax.boxplot(dists, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp['boxes'], cols): patch.set_facecolor(c); patch.set_alpha(0.55)
        if chance is not None: ax.axhline(chance, ls=':', color='r', alpha=0.7)
        ax.set_title(ttl, fontsize=10); ax.grid(axis='y', ls=':', alpha=0.4)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    pf = f"{ROOT}/results/heldout_eval/v2_esm_con_ep60/profiles.npz"
    if os.path.exists(pf):
        z = np.load(pf, allow_pickle=True); pred, obs, tests = z['pred'], z['obs'], z['test']
        pears = [np.corrcoef(pred[i], obs[i])[0, 1] if pred[i].std() > 1e-8 else -1 for i in range(len(tests))]
        bi = int(np.nanargmax(pears))
        ax = axes[1, 0]; ax.scatter(obs[bi], pred[bi], s=4, alpha=0.3, color=COL['v2_esm_con'])
        lim = max(np.abs(obs[bi]).max(), np.abs(pred[bi]).max())
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.6)
        ax.set_xlabel('observed Δ'); ax.set_ylabel('predicted Δ')
        ax.set_title(f'esm_con best TF = {tests[bi]}  (r={pears[bi]:.2f})', fontsize=10); ax.grid(ls=':', alpha=0.4)
    dfc = f"{ROOT}/results/heldout_eval/v2_esm_con_ep60/per_tf.csv"
    if os.path.exists(dfc):
        df = pd.read_csv(dfc); ax = axes[1, 1]
        ax.scatter(df['add_pearson_delta'], df['pearson_delta'], s=8, alpha=0.4, color=COL['v2_esm_con'])
        ax.plot([-0.4, 1.0], [-0.4, 1.0], 'k--', lw=0.8, alpha=0.6)
        ax.set_xlabel('additive per-TF Pearson Δ'); ax.set_ylabel('mutilate per-TF Pearson Δ')
        ax.set_title('esm_con: mutilate vs additive per TF (above line = mutilate better)', fontsize=9); ax.grid(ls=':', alpha=0.4)
    fig.suptitle('exp17 Joung — per-TF distributions (held-out singles, epoch 60)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{FIG}/exp17_per_tf_dists.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def main():
    S = {m: _load(f"{ROOT}/results/heldout_eval/{m}_ep60/summary.json") for m in MODELS}
    C = {m: _load(f"{ROOT}/results/combo_eval/{m}_ep60/summary.json") for m in MODELS}
    B = _load(f"{ROOT}/results/baseline_eval/baselines/summary.json")
    for p in [fig_singles(S, B), fig_combos(C, B), fig_pairs_triples(C, B), fig_per_tf()]:
        print("saved", p)


if __name__ == '__main__':
    main()

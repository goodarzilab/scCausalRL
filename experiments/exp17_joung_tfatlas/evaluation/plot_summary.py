"""Thorough summary figure panels for the exp17 Joung TF-atlas ablation.

Loads the model + baseline eval summaries and builds 4 multi-panel PNGs in
experiments/figures/joung/:
  exp17_singles_panels.png   held-out TF zero-shot: model (mutilate vs additive) vs baselines
  exp17_combos_panels.png    singles->combo composition: do-op vs additive vs null baselines
  exp17_pairs_vs_triples.png combo breakdown by set size
  exp17_per_tf_dists.png     per-TF distributions + example pred-vs-obs scatter

Run (base env, after the evals):  python evaluation/plot_summary.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.abspath(os.path.join(ROOT, '..', 'figures', 'joung')); os.makedirs(FIG, exist_ok=True)

# model order + display labels (* = residual-target run, collapsed)
MODELS = ['v2_orig', 'v2_esm', 'v2_esm_resid', 'v2_esm_resid_con', 'v2_esm_con']
LABEL = {'v2_orig': 'orig', 'v2_esm': 'esm', 'v2_esm_resid': 'esm_resid*',
         'v2_esm_resid_con': 'esm_resid_con*', 'v2_esm_con': 'esm_con'}
COL = dict(zip(MODELS, ['#7f7f7f', '#1f77b4', '#d62728', '#9467bd', '#2ca02c']))
NAN = float('nan')


def load_singles():
    out = {}
    for m in MODELS:
        f = f"{ROOT}/results/heldout_eval/{m}_ep60/summary.json"
        out[m] = json.load(open(f)) if os.path.exists(f) else {}
    return out


def load_combos():
    out = {}
    for m in MODELS:
        f = f"{ROOT}/results/combo_eval/{m}_ep60/summary.json"
        out[m] = json.load(open(f)) if os.path.exists(f) else {}
    return out


def load_baselines():
    f = f"{ROOT}/results/baseline_eval/baselines/summary.json"
    return json.load(open(f)) if os.path.exists(f) else {}


def g(d, k):
    v = d.get(k, NAN)
    return NAN if v is None else v


def grouped_bars(ax, vals_a, vals_b, label_a, label_b, title, ylabel,
                 reflines=None, ylim=None):
    """Two bars (mutilate vs additive) per model + optional reference lines."""
    x = np.arange(len(MODELS)); w = 0.38
    ax.bar(x - w/2, vals_a, w, color=[COL[m] for m in MODELS], label=label_a)
    ax.bar(x + w/2, vals_b, w, color=[COL[m] for m in MODELS], alpha=0.45,
           hatch='//', edgecolor='white', label=label_b)
    for refname, refval, style in (reflines or []):
        if refval is not None and np.isfinite(refval):
            ax.axhline(refval, ls=style, lw=1.2, color='k', alpha=0.6)
            ax.text(len(MODELS)-0.5, refval, f' {refname}={refval:.2f}', va='bottom',
                    ha='right', fontsize=7, color='k', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=30, ha='right', fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', ls=':', alpha=0.4)
    if ylim: ax.set_ylim(*ylim)


# ───────────────────────── Figure 1: SINGLES ─────────────────────────
def fig_singles(S, B):
    bs = B.get('singles', {}); pm = bs.get('perturb_mean', {}); cm = bs.get('context_mean', {})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    panels = [
        ('pearson_delta',            'Pearson Δ (raw)',           [('perturb_mean', g(pm,'pearson_delta'), '--')],     None),
        ('effect_spearman',          'Effect-size Spearman',      [('perturb_mean', g(pm,'effect_spearman'), '--')],   None),
        ('deg_auroc',                'DEG-detection AUROC',       [('perturb_mean', g(pm,'deg_auroc'), '--'), ('chance', 0.5, ':')], (0.4, 1.0)),
        ('pearson_delta_resid',      'Pearson Δ (drift-removed)', [('perturb_mean', g(pm,'pearson_delta_resid'), '--'), ('0', 0.0, ':')], None),
        ('discrimination_cos_resid', 'Discrimination (resid)',    [('chance', 0.5, ':')],                              None),
        ('fate_top1',                'Fate top-1',                [('perturb_mean', g(pm,'fate_top1'), '--')],         None),
    ]
    for ax, (key, ylab, refs, ylim) in zip(axes.ravel(), panels):
        va = [g(S[m], key) for m in MODELS]
        vb = [g(S[m], f'add_{key}') for m in MODELS]
        grouped_bars(ax, va, vb, 'mutilate', 'additive', ylab, ylab, reflines=refs, ylim=ylim)
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('exp17 Joung — held-out TF zero-shot (epoch 60): model (mutilate vs additive) vs baselines\n'
                 '* = residual-target run (collapsed). Drift-removed + discrimination are the load-bearing metrics; '
                 'perturb_mean (drift) is the floor.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{FIG}/exp17_singles_panels.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


# ───────────────────────── Figure 2: COMBOS ─────────────────────────
def fig_combos(C, B):
    bc = B.get('combos', {}).get('all', {})
    add_obs = bc.get('additive', {}); pm = bc.get('perturb_mean', {})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ('pearson_delta',      'Pearson Δ', [('additive-obs null', g(add_obs,'pearson_delta'), '-.'), ('perturb_mean', g(pm,'pearson_delta'), '--')], None),
        ('deg_auroc',          'DEG-detection AUROC', [('additive-obs null', g(add_obs,'deg_auroc'), '-.'), ('chance', 0.5, ':')], (0.4, 1.0)),
        ('effect_spearman',    'Effect-size Spearman', [('additive-obs null', g(add_obs,'effect_spearman'), '-.')], None),
        ('discrimination_cos', 'Discrimination', [('chance', 0.5, ':')], None),
    ]
    for ax, (key, ylab, refs, ylim) in zip(axes.ravel(), panels):
        va = [g(C[m].get('all', {}), f'do_{key}') for m in MODELS]
        vb = [g(C[m].get('all', {}), f'add_{key}') for m in MODELS]
        grouped_bars(ax, va, vb, 'do-op (mutilate)', 'additive (model)', ylab, ylab, reflines=refs, ylim=ylim)
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('exp17 Joung — singles→combo composition (all multi-TF sets, n=42)\n'
                 'do-op (coherent mutilation) vs model-additive; dash-dot = additive-of-OBSERVED-singles null '
                 '(strong, uses ground-truth singles).', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"{FIG}/exp17_combos_panels.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


# ───────────────────── Figure 3: pairs vs triples ─────────────────────
def fig_pairs_triples(C, B):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, ylab in [(axes[0], 'pearson_delta', 'Combo Pearson Δ'),
                          (axes[1], 'deg_auroc', 'Combo DEG-AUROC')]:
        x = np.arange(len(MODELS)); w = 0.38
        pair = [g(C[m].get('pairs', {}), f'do_{key}') for m in MODELS]
        tri = [g(C[m].get('triples', {}), f'do_{key}') for m in MODELS]
        ax.bar(x - w/2, pair, w, color=[COL[m] for m in MODELS], label='pairs (n≈40)')
        ax.bar(x + w/2, tri, w, color=[COL[m] for m in MODELS], alpha=0.4, hatch='xx',
               edgecolor='white', label='triples (n=2!)')
        bc = B.get('combos', {})
        for split, style in (('pairs', '-.'), ):
            rv = g(bc.get(split, {}).get('additive', {}), key)
            if np.isfinite(rv):
                ax.axhline(rv, ls=style, color='k', alpha=0.6, lw=1.2)
                ax.text(len(MODELS)-0.5, rv, f' additive-obs(pairs)={rv:.2f}', va='bottom', ha='right', fontsize=7)
        if 'auroc' in key: ax.axhline(0.5, ls=':', color='r', alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=30, ha='right', fontsize=8)
        ax.set_title(ylab, fontsize=10); ax.set_ylabel(ylab, fontsize=8); ax.grid(axis='y', ls=':', alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle('exp17 Joung — combo composition by set size (do-op). Triples n=2 → not statistically meaningful.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = f"{FIG}/exp17_pairs_vs_triples.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


# ───────────────────── Figure 4: per-TF distributions ─────────────────
def fig_per_tf():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # (a) per-TF pearson_delta distribution across models
    # (b) per-TF discrimination_beaten distribution
    dists_p, dists_d, labels, cols = [], [], [], []
    for m in MODELS:
        f = f"{ROOT}/results/heldout_eval/{m}_ep60/per_tf.csv"
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        dists_p.append(df['pearson_delta'].dropna().values)
        dists_d.append(df['discrimination_beaten'].dropna().values)
        labels.append(LABEL[m]); cols.append(COL[m])
    for ax, dists, ttl, chance in [(axes[0,0], dists_p, 'Per-TF Pearson Δ (raw)', None),
                                   (axes[0,1], dists_d, 'Per-TF discrimination (beaten frac)', 0.5)]:
        bp = ax.boxplot(dists, labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp['boxes'], cols): patch.set_facecolor(c); patch.set_alpha(0.55)
        if chance is not None: ax.axhline(chance, ls=':', color='r', alpha=0.7, label='chance')
        ax.set_title(ttl, fontsize=10); ax.grid(axis='y', ls=':', alpha=0.4)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    # (c) example pred-vs-obs Δ scatter for esm_con's best-retrieved TF
    # (d) per-TF mutilate vs additive pearson for esm_con
    pf = f"{ROOT}/results/heldout_eval/v2_esm_con_ep60/profiles.npz"
    if os.path.exists(pf):
        z = np.load(pf, allow_pickle=True); pred, obs, tests = z['pred'], z['obs'], z['test']
        # best TF by per-gene pearson
        pears = [np.corrcoef(pred[i], obs[i])[0,1] if pred[i].std()>1e-8 else -1 for i in range(len(tests))]
        bi = int(np.nanargmax(pears))
        ax = axes[1,0]; ax.scatter(obs[bi], pred[bi], s=4, alpha=0.3, color=COL['v2_esm_con'])
        lim = max(np.abs(obs[bi]).max(), np.abs(pred[bi]).max())
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.6)
        ax.set_xlabel('observed Δ'); ax.set_ylabel('predicted Δ')
        ax.set_title(f'esm_con best TF = {tests[bi]}  (r={pears[bi]:.2f})', fontsize=10)
        ax.grid(ls=':', alpha=0.4)
    dfc = f"{ROOT}/results/heldout_eval/v2_esm_con_ep60/per_tf.csv"
    if os.path.exists(dfc):
        df = pd.read_csv(dfc); ax = axes[1,1]
        ax.scatter(df['add_pearson_delta'], df['pearson_delta'], s=8, alpha=0.4, color=COL['v2_esm_con'])
        lo, hi = -0.4, 1.0; ax.plot([lo,hi],[lo,hi],'k--',lw=0.8,alpha=0.6)
        ax.set_xlabel('additive per-TF Pearson Δ'); ax.set_ylabel('mutilate per-TF Pearson Δ')
        ax.set_title('esm_con: mutilate vs additive (per TF; above line = mutilate better)', fontsize=9)
        ax.grid(ls=':', alpha=0.4)
    fig.suptitle('exp17 Joung — per-TF distributions (held-out singles, epoch 60)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{FIG}/exp17_per_tf_dists.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def main():
    S, C, B = load_singles(), load_combos(), load_baselines()
    paths = [fig_singles(S, B), fig_combos(C, B), fig_pairs_triples(C, B), fig_per_tf()]
    print("saved figures:")
    for p in paths: print("  ", p)


if __name__ == '__main__':
    main()

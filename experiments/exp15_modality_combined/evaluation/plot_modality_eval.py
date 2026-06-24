"""Plot the exp15 combined-model modality-aware eval (Norman = CRISPRa).

Reads the per-pert CSVs written by run_modality_eval.py and makes a 2x2 figure:
  combos (AUROC, Spearman) over methods + singles (AUROC, Spearman) over methods.
Highlights the zero-shot mutilation (shift_uscore) vs additive-shift (shift_add)
gap, with the observed-singles oracle (additive_obs) and corr baseline for context.

Usage:
  python evaluation/plot_modality_eval.py \
      --eval_dir results/grid/exp15_modality_combined/modality_eval_norman \
      --out ../figures/exp15_modality_eval.png
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMBO_METHODS = [('corr', 'corr\n(baseline)'), ('shift_add', 'shift_add\n(additive, zero-shot)'),
                 ('shift_uscore', 'shift_uscore\n(mutilation, zero-shot)'),
                 ('additive_obs', 'additive_obs\n(observed oracle)')]
SINGLE_METHODS = [('shift_add', 'shift_add'), ('shift_uscore', 'shift_uscore\n(mutilation)')]
COLORS = {'corr': '#999999', 'shift_add': '#4C72B0', 'shift_uscore': '#C44E52',
          'additive_obs': '#55A868'}


def _box(ax, df, methods, metric, title):
    data, labels, cols = [], [], []
    for key, lab in methods:
        c = f'{key}_{metric}'
        if c in df.columns:
            v = df[c].dropna().values
            if len(v):
                data.append(v); labels.append(lab); cols.append(COLORS.get(key, '#777'))
    if not data:
        ax.set_visible(False); return
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i + 1) + np.random.uniform(-0.08, 0.08, len(v)), v,
                   s=8, c='k', alpha=0.3, zorder=3)
        ax.text(i + 1, np.mean(v), f'{np.mean(v):.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric.upper()); ax.set_title(title, fontsize=10)
    if metric == 'auroc':
        ax.axhline(0.5, ls='--', c='gray', lw=0.8)
    ax.grid(axis='y', alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sdf = pd.read_csv(os.path.join(args.eval_dir, 'singles_per_pert.csv'))
    cdf = pd.read_csv(os.path.join(args.eval_dir, 'combo_per_pert.csv'))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _box(axes[0, 0], cdf, COMBO_METHODS, 'auroc', f'Norman COMBOS — DEG AUROC (n={len(cdf)})')
    _box(axes[0, 1], cdf, COMBO_METHODS, 'spearman', f'Norman COMBOS — Spearman(|effect|) (n={len(cdf)})')
    _box(axes[1, 0], sdf, SINGLE_METHODS, 'auroc', f'Norman SINGLES — DEG AUROC (n={len(sdf)})')
    _box(axes[1, 1], sdf, SINGLE_METHODS, 'spearman', f'Norman SINGLES — Spearman(|effect|) (n={len(sdf)})')
    fig.suptitle('exp15 combined (Replogle CRISPRi + Norman CRISPRa) — modality-aware eval on Norman (CRISPRa)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out or os.path.join(args.eval_dir, 'modality_eval.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    # also drop a copy next to the CSVs
    fig.savefig(os.path.join(args.eval_dir, 'modality_eval.png'), dpi=150, bbox_inches='tight')
    print(f"saved → {out}")
    # quick textual summary
    for name, df, methods in [('COMBOS', cdf, COMBO_METHODS), ('SINGLES', sdf, SINGLE_METHODS)]:
        line = "  ".join(f"{k}={df[f'{k}_auroc'].mean():.3f}" for k, _ in methods if f'{k}_auroc' in df.columns)
        print(f"{name} AUROC: {line}")


if __name__ == '__main__':
    main()

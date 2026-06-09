"""Cleaner combo eval plots + finer epistasis-trend analysis.

Fig 1 (per-model bars): AUROC and Spearman, ALL methods as bars (shift_add,
  additive_obs, shift_uscore, uscore_obs, correlation), grouped by model. SEM bars.
Fig 2 (epistasis trend, shift-native): instead of 3 tertiles, the PER-COMBO
  mutilation advantage (shift_uscore − shift_add) vs observed epistasis as a
  scatter (heat = epistasis) with a linear fit + Spearman r, plus decile-binned
  means — to see whether the advantage actually trends with epistasis or is flat.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, linregress
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
try: plt.style.use('seaborn-v0_8-whitegrid')
except Exception: pass
ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ('exp11 additive', 'exp11_norman_shift/results/phase1/norman_combo/combo_per_pert.csv'),
    ('exp12 mut loading', 'exp12_shift_mutilate/results/grid/exp12_norman_loadclamp/norman_combo/combo_per_pert.csv'),
    ('exp12 mut shift-native ★', 'exp12_shift_mutilate/results/grid/exp12_norman_shiftclamp/norman_combo/combo_per_pert.csv'),
]
METHODS = [('shift_add', '#2171b5'), ('additive_obs', '#9e9ac8'), ('shift_uscore', '#41ab5d'),
           ('uscore_obs', '#238b45'), ('corr', '#bdbdbd')]


def msem(a):
    a = np.asarray([x for x in a if x == x])
    return (float(a.mean()), float(a.std() / np.sqrt(len(a)))) if len(a) else (np.nan, 0.0)


def fig_bars(dfs, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, metric, ylab, ylo in [(axes[0], 'auroc', 'AUROC', 0.7), (axes[1], 'spearman', 'Spearman ρ', 0.0)]:
        x = np.arange(len(MODELS)); w = 0.16
        for k, (m, c) in enumerate(METHODS):
            means, sems = [], []
            for _, df in dfs:
                mu, se = msem(df[f'{m}_{metric}']); means.append(mu); sems.append(se)
            ax.bar(x + (k - 2) * w, means, w, yerr=sems, capsize=3, color=c, edgecolor='black', lw=0.7,
                   label=m if ax is axes[0] else None)
            for xi, (mu, se) in enumerate(zip(means, sems)):
                if mu == mu: ax.text(x[xi] + (k - 2) * w, mu + se + 0.004, f'{mu:.2f}', ha='center', fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([n for n, _ in MODELS], fontsize=9)
        ax.set_ylabel(ylab, fontsize=12); ax.set_ylim(ylo, None)
        if metric == 'auroc': ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5)
        ax.set_title(f'Norman combo {ylab} (n={len(dfs[0][1])}; SEM across combos)', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[0].legend(fontsize=8, loc='lower right', ncol=2)
    fig.suptitle('Combo prediction by method — all baselines as bars (AUROC + Spearman)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out, 'combo_methods_auroc_spearman.png'); plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print('wrote', p)


def fig_epistasis_scatter(df, label, out):
    df = df[df['epistasis'].notna() & df['shift_uscore_auroc'].notna() & df['shift_add_auroc'].notna()].copy()
    df['adv'] = df['shift_uscore_auroc'] - df['shift_add_auroc']
    rho, pval = spearmanr(df['epistasis'], df['adv'])
    lr = linregress(df['epistasis'], df['adv'])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # left: per-combo advantage vs epistasis (heat = epistasis), fit
    sccat = axes[0].scatter(df['epistasis'], df['adv'], c=df['epistasis'], cmap='viridis', s=45, edgecolor='k', lw=0.4)
    xs = np.linspace(df['epistasis'].min(), df['epistasis'].max(), 50)
    axes[0].plot(xs, lr.intercept + lr.slope * xs, 'r-', lw=2, label=f'fit slope={lr.slope:+.3f} (p={lr.pvalue:.2f})')
    axes[0].axhline(0, color='gray', ls='--', lw=1)
    axes[0].set_xlabel('observed epistasis  ‖obs−additive‖/‖obs‖ (DEGs)', fontsize=11)
    axes[0].set_ylabel('mutilation advantage  (shift_uscore − shift_add) AUROC', fontsize=11)
    axes[0].set_title(f'{label}: per-combo advantage vs epistasis\nSpearman ρ={rho:+.3f} (p={pval:.2f}), n={len(df)}', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9); plt.colorbar(sccat, ax=axes[0], label='epistasis')
    # right: decile-binned method AUROC vs epistasis (finer than tertiles) + extreme-tail callout
    df['dec'] = pd.qcut(df['epistasis'], 10, labels=False, duplicates='drop')
    for m, c in [('shift_uscore', '#41ab5d'), ('shift_add', '#2171b5'), ('additive_obs', '#9e9ac8'), ('corr', '#bdbdbd')]:
        g = df.groupby('dec').agg(mu=(f'{m}_auroc', 'mean'), ep=('epistasis', 'mean'))
        axes[1].plot(g['ep'], g['mu'], 'o-', color=c, lw=1.5, ms=5, label=m)
    axes[1].set_xlabel('observed epistasis (decile mean)', fontsize=11); axes[1].set_ylabel('combo AUROC', fontsize=11)
    axes[1].set_title('AUROC vs epistasis decile (10 bins) — finer than tertiles', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    for a in axes: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    plt.tight_layout(); p = os.path.join(out, f'combo_epistasis_scatter_{label}.png'); plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print('wrote', p)
    # extreme-tail table (top vs bottom decile)
    lo = df[df['dec'] == df['dec'].min()]; hi = df[df['dec'] == df['dec'].max()]
    print(f"\n  extreme deciles ({label}): bottom n={len(lo)} (epi {lo['epistasis'].min():.2f}-{lo['epistasis'].max():.2f}), top n={len(hi)} (epi {hi['epistasis'].min():.2f}-{hi['epistasis'].max():.2f})")
    for m, _ in METHODS:
        print(f"    {m:14s} bottom={lo[f'{m}_auroc'].mean():.3f}  top={hi[f'{m}_auroc'].mean():.3f}  Δ(top-bot)={hi[f'{m}_auroc'].mean()-lo[f'{m}_auroc'].mean():+.3f}")
    print(f"    advantage(uscore-add): bottom={lo['adv'].mean():+.3f}  top={hi['adv'].mean():+.3f}")


def main():
    out = os.path.join(ROOT, 'figures'); os.makedirs(out, exist_ok=True)
    dfs = [(n, pd.read_csv(os.path.join(ROOT, p))) for n, p in MODELS]
    fig_bars(dfs, out)
    sn = dict(dfs)['exp12 mut shift-native ★']
    fig_epistasis_scatter(sn, 'shift_native', out)


if __name__ == '__main__':
    main()

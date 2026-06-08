"""Norman combo AUROC stratified by observed-epistasis tertile.

Hypothesis: the additive baselines should dominate on near-additive combos, but on
the most epistatic (non-linear) combos the causal mutilation model should hold up
relatively better. We split combos into LOW / MID / HIGH tertiles of observed
epistasis (normalized deviation of the observed combo from observed-additive on its
DEGs — a model-independent genetic-interaction strength) and compare mean AUROC
(±SEM across combos) of: shift_add (zero-shot additive), additive_obs (observed
additive null), shift_uscore (zero-shot mutilation), uscore_obs (mutilation fed
observed singles), correlation.

Reads results/grid/<cell>/norman_combo/combo_per_pert.csv (needs the `epistasis`
column added by run_norman_combo_eval.py).
Usage: python plot_combo_epistasis.py --cell exp8_sl0p0_ls0p0 [--label sl0_ls0]
"""
import os, argparse, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
try: plt.style.use('seaborn-v0_8-whitegrid')
except Exception: pass

ROOT = os.path.dirname(os.path.abspath(__file__))
METHODS = [('shift_add', '#2171b5', 'shift_add (zero-shot additive)'),
           ('additive_obs', '#9e9ac8', 'additive_obs (observed null)'),
           ('shift_uscore', '#41ab5d', 'shift_uscore (zero-shot mutilation)'),
           ('uscore_obs', '#238b45', 'uscore_obs (obs-fed mutilation)'),
           ('corr', '#bdbdbd', 'correlation')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='exp8_sl0p0_ls0p0')
    ap.add_argument('--label', default=None)
    ap.add_argument('--exp', default='exp8_dag_revival_grid')
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    args = ap.parse_args()
    label = args.label or args.cell
    os.makedirs(args.out, exist_ok=True)
    csv = os.path.join(ROOT, args.exp, 'results/grid', args.cell, 'norman_combo', 'combo_per_pert.csv')
    df = pd.read_csv(csv)
    df = df[df['epistasis'].notna() & df['shift_add_auroc'].notna()].copy()
    # tertiles of observed epistasis (low/mid/high genetic interaction)
    df['tertile'] = pd.qcut(df['epistasis'], 3, labels=['low', 'mid', 'high'])
    terts = ['low', 'mid', 'high']

    # table: mean AUROC per method per tertile
    rows = []
    for t in terts:
        sub = df[df['tertile'] == t]
        rec = {'tertile': t, 'n': len(sub), 'epistasis_range': f"{sub['epistasis'].min():.2f}-{sub['epistasis'].max():.2f}"}
        for m, _, _ in METHODS:
            a = sub[f'{m}_auroc'].dropna().values
            rec[f'{m}_auroc'] = float(np.mean(a)) if len(a) else np.nan
            rec[f'{m}_sem'] = float(np.std(a) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, f'combo_epistasis_tertiles_{label}.csv'), index=False)
    print(f"\n=== {label}: Norman combo AUROC by observed-epistasis tertile (n={len(df)}) ===")
    cols = ['tertile', 'n', 'epistasis_range'] + [f'{m}_auroc' for m, _, _ in METHODS]
    print(tab[cols].to_string(index=False))

    # grouped bar plot
    x = np.arange(len(terts)); w = 0.16
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (m, c, lab) in enumerate(METHODS):
        means = [tab[tab.tertile == t][f'{m}_auroc'].values[0] for t in terts]
        sems = [tab[tab.tertile == t][f'{m}_sem'].values[0] for t in terts]
        ax.bar(x + (i - 2) * w, means, w, yerr=sems, capsize=3, color=c, edgecolor='black', lw=0.8, label=lab)
    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.6, label='random')
    ns = [int(tab[tab.tertile == t]['n'].values[0]) for t in terts]
    rngs = [tab[tab.tertile == t]['epistasis_range'].values[0] for t in terts]
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t} epistasis\n(n={n}; {r})' for t, n, r in zip(terts, ns, rngs)], fontsize=9)
    ax.set_ylabel('Norman combo AUROC (direction-agnostic)', fontsize=12)
    ax.set_ylim(0.45, max(0.92, float(np.nanmax([tab[f'{m}_auroc'].max() for m, _, _ in METHODS])) + 0.05))
    ax.set_title(f'Combo prediction vs observed-epistasis tertile — {label}\n'
                 f'(error = SEM across combos; epistasis = ‖obs_combo − additive‖/‖obs_combo‖ on DEGs)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(args.out, f'combo_epistasis_tertiles_{label}.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"\nwrote {p}")
    print(f"wrote {os.path.join(args.out, f'combo_epistasis_tertiles_{label}.csv')}")


if __name__ == '__main__':
    main()

"""exp8 winners (sl0_ls0 & sl0.1_ls0) — Replogle + Norman, AUROC + Spearman vs correlation.

The two A-active exp8 cells: `sl0_ls0` (no shift penalty) and `sl0.1_ls0` (small
shift penalty, cleanest causal cell). Renders a 2x2 panel:
    rows  = Replogle K562 held-out test   /   Norman external (CRISPRa)
    cols  = DEG detection (AUROC)          /   effect-size ranking (Spearman rho)
Each bar = mean over held-out perturbations; error = SEM across perturbations;
x-labels annotated with the perturbation count. Correlation baseline (gray) on
every panel for reference.

Re-runnable; reads the per-perturbation eval CSVs written by run_shift_eval.py /
run_shift_eval_norman.py under results/grid/<cell>/{shift_eval,norman_eval}/.
Usage: python plot_exp8_winners.py [--out figures]
"""
import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
try: plt.style.use('seaborn-v0_8-whitegrid')
except Exception: pass

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(ROOT, 'exp8_dag_revival_grid', 'results', 'grid')

# (display label, color, cell dir)
MODELS = [
    ('sl0_ls0',    '#2171b5', 'exp8_sl0p0_ls0p0'),
    ('sl0.1_ls0 ★', '#41ab5d', 'exp8_sl0p1_ls0p0'),
]
CORR_COL = '#bdbdbd'


def _arr(path, col):
    try:
        df = pd.read_csv(path)
        return df[col].dropna().values if col in df else np.array([])
    except Exception:
        return np.array([])

def _mean(a): return float(np.mean(a)) if len(a) else np.nan
def _sem(a):  return float(np.std(a) / np.sqrt(len(a))) if len(a) > 1 else 0.0
def _lbl(name, a): return f"{name}\n(n={len(a)})"


def collect():
    """Returns dict: split -> metric -> list of (label, color, value_array)."""
    out = {'replogle': {'auroc': [], 'spearman': []},
           'norman':   {'auroc': [], 'spearman': []}}
    corr_added = {'replogle': False, 'norman': False}
    for label, color, cell in MODELS:
        sd = os.path.join(GRID, cell, 'shift_eval', 'test_model_fixed.csv')
        nd = os.path.join(GRID, cell, 'norman_eval', 'norman_model_fixed.csv')
        # correlation baseline (once per split; identical across cells/same split)
        if not corr_added['replogle']:
            cp = os.path.join(GRID, cell, 'shift_eval', 'test_corr.csv')
            out['replogle']['auroc'].append(('correlation', CORR_COL, _arr(cp, 'auroc')))
            out['replogle']['spearman'].append(('correlation', CORR_COL, _arr(cp, 'spearman')))
            corr_added['replogle'] = True
        if not corr_added['norman']:
            cp = os.path.join(GRID, cell, 'norman_eval', 'norman_corr.csv')
            out['norman']['auroc'].append(('correlation', CORR_COL, _arr(cp, 'auroc')))
            out['norman']['spearman'].append(('correlation', CORR_COL, _arr(cp, 'spearman')))
            corr_added['norman'] = True
        out['replogle']['auroc'].append((label, color, _arr(sd, 'auroc')))
        out['replogle']['spearman'].append((label, color, _arr(sd, 'spearman')))
        out['norman']['auroc'].append((label, color, _arr(nd, 'auroc')))
        out['norman']['spearman'].append((label, color, _arr(nd, 'spearman')))
    return out


def _panel(ax, items, ylab, rnd, title):
    names = [_lbl(l, a) for l, _, a in items]
    colors = [c for _, c, _ in items]
    means = [_mean(a) for _, _, a in items]
    sems = [_sem(a) for _, _, a in items]
    x = np.arange(len(items))
    bars = ax.bar(x, means, yerr=sems, color=colors, edgecolor='black', capsize=4, width=0.6, lw=1.1)
    for b, m, s in zip(bars, means, sems):
        if not np.isnan(m):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+(s or 0.004)+0.004,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(rnd, color='red', ls='--', lw=1.3, alpha=0.6, label='random')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(ylab, fontsize=11); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    d = collect()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    _panel(axes[0, 0], d['replogle']['auroc'],    'AUROC',      0.5, 'Replogle test — DEG detection (AUROC)')
    _panel(axes[0, 1], d['replogle']['spearman'], 'Spearman ρ', 0.0, 'Replogle test — effect-size ranking (Spearman)')
    _panel(axes[1, 0], d['norman']['auroc'],       'AUROC',      0.5, 'Norman external — DEG detection (AUROC)')
    _panel(axes[1, 1], d['norman']['spearman'],    'Spearman ρ', 0.0, 'Norman external — effect-size ranking (Spearman)')
    fig.suptitle('exp8 A-active winners vs correlation baseline\n'
                 '(held-out perturbations; error = SEM across perts)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(args.out, 'exp8_winners_repl_norman.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()

    # also dump the numbers
    rows = []
    for split in ('replogle', 'norman'):
        for metric in ('auroc', 'spearman'):
            for label, _, a in d[split][metric]:
                rows.append(dict(split=split, metric=metric, model=label,
                                 n=len(a), mean=_mean(a), sem=_sem(a)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, 'exp8_winners_summary.csv'), index=False)
    print(tab.to_string(index=False))
    print('\nwrote', p)
    print('wrote', os.path.join(args.out, 'exp8_winners_summary.csv'))


if __name__ == '__main__':
    main()

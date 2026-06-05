"""Cross-experiment eval comparison figures (with error bars).

Reads every experiment's saved eval CSVs and renders comparison figures on the
held-out TEST split, plus an in-distribution-vs-external(Norman) panel.

Error bars are SEM across perturbations (std/sqrt(n) over the per-perturbation
metric values) — the same convention as the legacy run_perturbation_eval plots.
NOT across seeds.

Handles both CSV formats:
  - shift experiments (exp2-7): results/phase1/shift_eval/{summary.csv,
    test_model_fixed.csv, test_corr.csv, test_coexp.csv}; external:
    results/phase1/norman_eval/{summary.csv, norman_model_fixed.csv, norman_corr.csv}
  - baseline (exp1, legacy D⁺·dx): results/eval_<V1|V2>_<mode>/single_pert_model_test.csv

Re-runnable: silently skips experiments whose eval hasn't finished.
Usage: python plot_eval_comparison.py [--out figures]
"""
import os, glob, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
COL = {'shift': '#2171b5', 'baseline_old': '#9e9ac8', 'corr': '#bdbdbd',
       'coexp': '#fd8d3c', 'norman': '#41ab5d'}


def _csv(p):
    try: return pd.read_csv(p)
    except Exception: return None

def _arr(df, col):
    if df is None or col not in df: return np.array([])
    return df[col].dropna().values

def _mean(a): return float(np.mean(a)) if len(a) else np.nan
def _sem(a):  return float(np.std(a) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def collect():
    entries = []   # model, kind, indist{auroc,spearman,pearson}, norman{auroc,spearman}, mmd, mmd_floor
    refs_done = False
    for exp in sorted(glob.glob(os.path.join(ROOT, 'exp*'))):
        sdir = os.path.join(exp, 'results/phase1/shift_eval')
        summ = _csv(os.path.join(sdir, 'summary.csv'))
        perp = _csv(os.path.join(sdir, 'test_model_fixed.csv'))
        if summ is None or perp is None:
            continue
        s = summ.iloc[0]
        short = os.path.basename(exp).replace('_shift', '').replace('shift_', '')
        nperp = _csv(os.path.join(exp, 'results/phase1/norman_eval', 'norman_model_fixed.csv'))
        entries.append({
            'model': short, 'kind': 'shift',
            'indist': {'auroc': _arr(perp, 'auroc'), 'spearman': _arr(perp, 'spearman'),
                       'pearson': _arr(perp, 'pearson_deg')},
            'norman': ({'auroc': _arr(nperp, 'auroc'), 'spearman': _arr(nperp, 'spearman')}
                       if nperp is not None else None),
            'mmd': float(s.get('test_model_fixed_mmd', np.nan)),
            'mmd_floor': float(s.get('test_model_fixed_mmd_ctrl_floor', np.nan)),
        })
        if not refs_done:
            cdf = _csv(os.path.join(sdir, 'test_corr.csv'))
            xdf = _csv(os.path.join(sdir, 'test_coexp.csv'))
            if cdf is not None:
                entries.append({'model': 'correlation', 'kind': 'corr',
                                'indist': {'auroc': _arr(cdf, 'auroc'), 'spearman': _arr(cdf, 'spearman'),
                                           'pearson': np.array([])}, 'norman': None,
                                'mmd': np.nan, 'mmd_floor': np.nan})
            if xdf is not None:
                entries.append({'model': 'co-exp nbr', 'kind': 'coexp',
                                'indist': {'auroc': _arr(xdf, 'auroc'), 'spearman': _arr(xdf, 'spearman'),
                                           'pearson': _arr(xdf, 'pearson_deg')}, 'norman': None,
                                'mmd': np.nan, 'mmd_floor': np.nan})
            refs_done = True

    base = os.path.join(ROOT, 'exp1_baseline', 'results')
    for v in ('V1', 'V2'):
        for mode in ('target_lfc', 'canonical'):
            perp = _csv(os.path.join(base, f'eval_{v}_{mode}', 'single_pert_model_test.csv'))
            if perp is None: continue
            entries.append({'model': f'base_{v}_{mode}', 'kind': 'baseline_old',
                            'indist': {'auroc': _arr(perp, 'auroc'), 'spearman': _arr(perp, 'spearman'),
                                       'pearson': _arr(perp, 'pearson_deg')}, 'norman': None,
                            'mmd': np.nan, 'mmd_floor': np.nan})
    return entries


def _order(entries):
    rank = {'baseline_old': 0, 'corr': 1, 'coexp': 2, 'shift': 3}
    return sorted(entries, key=lambda e: (rank.get(e['kind'], 9),
                                          -_mean(e['indist']['auroc'])))


def fig_panels(entries, out):
    """Two-panel AUROC + Spearman bar chart with SEM error bars (held-out test)."""
    order = _order(entries)
    names = [e['model'] for e in order]
    colors = [COL[e['kind']] for e in order]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.3 * len(names)), 5.5))
    for ax, key, ylab, rnd, title in [
        (axes[0], 'auroc', 'AUROC', 0.5, 'DEG Detection (AUROC)'),
        (axes[1], 'spearman', 'Spearman ρ', 0.0, 'Effect-size Ranking (Spearman)')]:
        means = [_mean(e['indist'][key]) for e in order]
        sems = [_sem(e['indist'][key]) for e in order]
        bars = ax.bar(x, means, yerr=sems, color=colors, edgecolor='black',
                      capsize=4, width=0.62, linewidth=1.1)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + (sems[0] or 0.005) + 0.004,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(rnd, color='red', ls='--', lw=1.3, alpha=0.6, label='random')
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(ylab, fontsize=12); ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Held-out single-perturbation prediction (Replogle K562, test split; '
                 'error = SEM across perts)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, 'headline_indist_panels.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close(); return p


def fig_indist_vs_norman(entries, out):
    """Grouped bars: in-dist (Replogle) vs external (Norman) AUROC per model."""
    sh = [e for e in _order(entries) if e['kind'] == 'shift' and e['norman'] is not None]
    if not sh:
        return None
    names = [e['model'] for e in sh]
    x = np.arange(len(names)); w = 0.38
    id_m = [_mean(e['indist']['auroc']) for e in sh]; id_s = [_sem(e['indist']['auroc']) for e in sh]
    no_m = [_mean(e['norman']['auroc']) for e in sh]; no_s = [_sem(e['norman']['auroc']) for e in sh]
    # baseline references
    corr_id = next((_mean(e['indist']['auroc']) for e in entries if e['kind'] == 'corr'), None)
    norman_corr = None
    for exp in glob.glob(os.path.join(ROOT, 'exp*')):
        c = _csv(os.path.join(exp, 'results/phase1/norman_eval', 'norman_corr.csv'))
        if c is not None: norman_corr = _mean(_arr(c, 'auroc')); break

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names)), 5.5))
    ax.bar(x - w/2, id_m, w, yerr=id_s, capsize=4, color=COL['shift'], edgecolor='black',
           label='Replogle held-out')
    ax.bar(x + w/2, no_m, w, yerr=no_s, capsize=4, color=COL['norman'], edgecolor='black',
           label='Norman external (CRISPRa)')
    for xi, (a, b) in enumerate(zip(id_m, no_m)):
        ax.text(xi - w/2, a + 0.008, f'{a:.3f}', ha='center', fontsize=8, fontweight='bold')
        ax.text(xi + w/2, b + 0.008, f'{b:.3f}', ha='center', fontsize=8, fontweight='bold')
    if corr_id: ax.axhline(corr_id, color=COL['corr'], ls='--', lw=1.5, label=f'corr (in-dist) {corr_id:.3f}')
    if norman_corr: ax.axhline(norman_corr, color=COL['norman'], ls=':', lw=1.5, label=f'corr (Norman) {norman_corr:.3f}')
    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('In-distribution vs external generalization (held-out AUROC; SEM across perts)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, 'indist_vs_norman_auroc.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close(); return p


def fig_box(entries, out):
    order = [e for e in _order(entries) if len(e['indist']['auroc'])]
    data = [e['indist']['auroc'] for e in order]; names = [e['model'] for e in order]
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(names)), 5))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, e in zip(bp['boxes'], order):
        patch.set_facecolor(COL[e['kind']]); patch.set_alpha(0.75)
    for med in bp['medians']: med.set_color('black')
    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(np.arange(1, len(names)+1)); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Per-perturbation AUROC', fontsize=12)
    ax.set_title('Per-perturbation held-out AUROC distribution', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, 'per_pert_auroc_box.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close(); return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    entries = collect()
    if not entries:
        print("No eval CSVs found yet."); return

    tab = pd.DataFrame([{
        'model': e['model'], 'kind': e['kind'],
        'indist_auroc': _mean(e['indist']['auroc']), 'indist_auroc_sem': _sem(e['indist']['auroc']),
        'indist_spearman': _mean(e['indist']['spearman']),
        'norman_auroc': (_mean(e['norman']['auroc']) if e['norman'] else np.nan),
        'norman_auroc_sem': (_sem(e['norman']['auroc']) if e['norman'] else np.nan),
        'mmd': e['mmd'], 'mmd_floor': e['mmd_floor'],
    } for e in entries]).sort_values(['kind', 'indist_auroc'], ascending=[True, False])
    tab.to_csv(os.path.join(args.out, 'comparison_summary.csv'), index=False)
    print(tab.to_string(index=False))
    for fn in (fig_panels, fig_indist_vs_norman, fig_box):
        p = fn(entries, args.out)
        if p: print("wrote", p)
    print(f"\nFigures + comparison_summary.csv in {args.out}/")


if __name__ == '__main__':
    main()

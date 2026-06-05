"""Cross-experiment eval comparison figures (with error bars + perturbation counts).

Reads every experiment's saved eval CSVs and renders comparison figures on the
held-out TEST split, an external-Norman figure, and an in-dist-vs-Norman panel.

Error bars are SEM across perturbations (std/sqrt(n) over per-perturbation metric
values) — the legacy convention, NOT across seeds. Each bar is annotated with the
number of perturbations (genes) it averages over.

Formats handled:
  - shift (exp2-7): results/phase1/shift_eval/{summary,test_model_fixed,test_corr,test_coexp}.csv
    external: results/phase1/norman_eval/{summary,norman_model_fixed,norman_corr}.csv
  - baseline (exp1, legacy D⁺·dx): results/eval_<V1|V2>_<mode>/single_pert_model_test.csv

Re-runnable: skips experiments whose eval hasn't finished.
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
       'coexp': '#fd8d3c', 'norman': '#41ab5d', 'norman_corr': '#bdbdbd'}


def _csv(p):
    try: return pd.read_csv(p)
    except Exception: return None

def _arr(df, col):
    if df is None or col not in df: return np.array([])
    return df[col].dropna().values

def _mean(a): return float(np.mean(a)) if len(a) else np.nan
def _sem(a):  return float(np.std(a) / np.sqrt(len(a))) if len(a) > 1 else 0.0
def _xl(name, arr): return f"{name}\n(n={len(arr)})"   # x-label with perturbation count


def collect():
    entries, refs_done = [], False
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
            for label, fn, kind in [('correlation', 'test_corr.csv', 'corr'),
                                    ('co-exp nbr', 'test_coexp.csv', 'coexp')]:
                d = _csv(os.path.join(sdir, fn))
                if d is not None:
                    entries.append({'model': label, 'kind': kind, 'norman': None,
                                    'indist': {'auroc': _arr(d, 'auroc'), 'spearman': _arr(d, 'spearman'),
                                               'pearson': _arr(d, 'pearson_deg')},
                                    'mmd': np.nan, 'mmd_floor': np.nan})
            refs_done = True

    base = os.path.join(ROOT, 'exp1_baseline', 'results')
    for v in ('V1', 'V2'):
        for mode in ('target_lfc', 'canonical'):
            perp = _csv(os.path.join(base, f'eval_{v}_{mode}', 'single_pert_model_test.csv'))
            if perp is None: continue
            entries.append({'model': f'base_{v}_{mode}', 'kind': 'baseline_old', 'norman': None,
                            'indist': {'auroc': _arr(perp, 'auroc'), 'spearman': _arr(perp, 'spearman'),
                                       'pearson': _arr(perp, 'pearson_deg')},
                            'mmd': np.nan, 'mmd_floor': np.nan})
    return entries


def _norman_corr():
    """Per-pert Norman correlation-baseline arrays (from any experiment's norman_corr.csv)."""
    for exp in sorted(glob.glob(os.path.join(ROOT, 'exp*'))):
        c = _csv(os.path.join(exp, 'results/phase1/norman_eval', 'norman_corr.csv'))
        if c is not None:
            return {'auroc': _arr(c, 'auroc'), 'spearman': _arr(c, 'spearman')}
    return None


def _order(entries):
    rank = {'baseline_old': 0, 'corr': 1, 'coexp': 2, 'shift': 3}
    return sorted(entries, key=lambda e: (rank.get(e['kind'], 9), -_mean(e['indist']['auroc'])))


def _bar_panels(items, get, title, out, fname, ref_lines=()):
    """items: list of (label, color, auroc_arr, spearman_arr). get unused (kept simple)."""
    names = [_xl(lbl, a) for lbl, _, a, _ in items]
    colors = [c for _, c, _, _ in items]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.4 * len(names)), 5.6))
    for ax, idx, ylab, rnd, sub in [(axes[0], 2, 'AUROC', 0.5, 'DEG Detection (AUROC)'),
                                     (axes[1], 3, 'Spearman ρ', 0.0, 'Effect-size Ranking (Spearman)')]:
        means = [_mean(it[idx]) for it in items]; sems = [_sem(it[idx]) for it in items]
        bars = ax.bar(x, means, yerr=sems, color=colors, edgecolor='black', capsize=4, width=0.62, lw=1.1)
        for b, m, sm in zip(bars, means, sems):
            if not np.isnan(m):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+(sm or 0.004)+0.004,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.axhline(rnd, color='red', ls='--', lw=1.3, alpha=0.6, label='random')
        for yv, lab, col in ref_lines:
            ax.axhline(yv, ls=':', lw=1.4, color=col, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(ylab, fontsize=12); ax.set_title(sub, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out, fname); plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    return p


def fig_indist(entries, out):
    order = _order(entries)
    items = [(e['model'], COL[e['kind']], e['indist']['auroc'], e['indist']['spearman']) for e in order]
    n_typ = int(np.median([len(e['indist']['auroc']) for e in order if len(e['indist']['auroc'])]))
    return _bar_panels(items, None,
                       f'Held-out single-perturbation prediction — Replogle K562 test split '
                       f'(~{n_typ} perts; error = SEM across perts)', out, 'headline_indist_panels.png')


def fig_norman(entries, out):
    sh = [e for e in _order(entries) if e['kind'] == 'shift' and e['norman'] is not None]
    if not sh: return None
    nc = _norman_corr()
    items = []
    if nc is not None:
        items.append(('correlation', COL['norman_corr'], nc['auroc'], nc['spearman']))
    for e in sh:
        items.append((e['model'], COL['norman'], e['norman']['auroc'], e['norman']['spearman']))
    n_ext = len(sh[0]['norman']['auroc'])
    return _bar_panels(items, None,
                       f'External validation on Norman (CRISPRa) — model trained on K562 CRISPRi '
                       f'({n_ext} perts; AUROC is direction-agnostic)', out, 'norman_external_panels.png')


def fig_indist_vs_norman(entries, out):
    sh = [e for e in _order(entries) if e['kind'] == 'shift' and e['norman'] is not None]
    if not sh: return None
    names = sh and [e['model'] for e in sh]
    id_n = len(sh[0]['indist']['auroc']); nm_n = len(sh[0]['norman']['auroc'])
    x = np.arange(len(names)); w = 0.38
    id_m = [_mean(e['indist']['auroc']) for e in sh]; id_s = [_sem(e['indist']['auroc']) for e in sh]
    no_m = [_mean(e['norman']['auroc']) for e in sh]; no_s = [_sem(e['norman']['auroc']) for e in sh]
    corr_id = next((_mean(e['indist']['auroc']) for e in entries if e['kind'] == 'corr'), None)
    nc = _norman_corr(); norman_corr = _mean(nc['auroc']) if nc else None
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5.6))
    ax.bar(x - w/2, id_m, w, yerr=id_s, capsize=4, color=COL['shift'], edgecolor='black',
           label=f'Replogle held-out (n={id_n})')
    ax.bar(x + w/2, no_m, w, yerr=no_s, capsize=4, color=COL['norman'], edgecolor='black',
           label=f'Norman external (n={nm_n})')
    for xi, (a, b) in enumerate(zip(id_m, no_m)):
        ax.text(xi - w/2, a + 0.008, f'{a:.3f}', ha='center', fontsize=8, fontweight='bold')
        ax.text(xi + w/2, b + 0.008, f'{b:.3f}', ha='center', fontsize=8, fontweight='bold')
    if corr_id: ax.axhline(corr_id, color=COL['corr'], ls='--', lw=1.5, label=f'corr in-dist {corr_id:.3f}')
    if norman_corr: ax.axhline(norman_corr, color=COL['norman'], ls=':', lw=1.6, label=f'corr Norman {norman_corr:.3f}')
    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('In-distribution vs external generalization (held-out AUROC; SEM across perts)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, 'indist_vs_norman_auroc.png'); plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    return p


def fig_box(entries, out):
    order = [e for e in _order(entries) if len(e['indist']['auroc'])]
    data = [e['indist']['auroc'] for e in order]
    names = [_xl(e['model'], e['indist']['auroc']) for e in order]
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * len(names)), 5))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, e in zip(bp['boxes'], order):
        patch.set_facecolor(COL[e['kind']]); patch.set_alpha(0.75)
    for med in bp['medians']: med.set_color('black')
    ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(np.arange(1, len(names)+1)); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Per-perturbation AUROC', fontsize=12)
    ax.set_title('Per-perturbation held-out AUROC distribution (Replogle test)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, 'per_pert_auroc_box.png'); plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    entries = collect()
    if not entries:
        print("No eval CSVs found yet."); return
    tab = pd.DataFrame([{
        'model': e['model'], 'kind': e['kind'],
        'indist_n': len(e['indist']['auroc']),
        'indist_auroc': _mean(e['indist']['auroc']), 'indist_auroc_sem': _sem(e['indist']['auroc']),
        'indist_spearman': _mean(e['indist']['spearman']),
        'norman_n': (len(e['norman']['auroc']) if e['norman'] else 0),
        'norman_auroc': (_mean(e['norman']['auroc']) if e['norman'] else np.nan),
        'norman_auroc_sem': (_sem(e['norman']['auroc']) if e['norman'] else np.nan),
        'mmd': e['mmd'], 'mmd_floor': e['mmd_floor'],
    } for e in entries]).sort_values(['kind', 'indist_auroc'], ascending=[True, False])
    tab.to_csv(os.path.join(args.out, 'comparison_summary.csv'), index=False)
    print(tab.to_string(index=False))
    for fn in (fig_indist, fig_norman, fig_indist_vs_norman, fig_box):
        p = fn(entries, args.out)
        if p: print("wrote", p)
    print(f"\nFigures + comparison_summary.csv in {args.out}/")


if __name__ == '__main__':
    main()

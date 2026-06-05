"""Cross-experiment eval comparison figures.

Reads the saved eval CSVs from every experiment and produces side-by-side
comparison figures on the held-out TEST split. Handles both CSV formats:
  - shift experiments (exp2-7): results/phase1/shift_eval/{summary.csv,
    test_model_fixed.csv, test_corr.csv, test_coexp.csv}
  - baseline experiment (exp1, legacy D⁺·dx): results/eval_<V1|V2>_<mode>/
    {summary.csv, single_pert_model_test.csv}

Re-runnable: silently skips experiments whose eval hasn't finished yet, so you
can run it now (exp2-5) and again once exp6/7 + the baselines land.

Usage: python plot_eval_comparison.py [--out figures]
"""
import os, glob, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
C_SHIFT, C_BASE, C_CORR, C_COEXP = '#2171b5', '#9e9ac8', '#bdbdbd', '#fd8d3c'


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def collect():
    """Return (entries, refs). entries: list of dicts per model. refs: corr/coexp AUROC."""
    entries, refs = [], {}

    # ── Shift experiments ──────────────────────────────────────────────────
    for exp in sorted(glob.glob(os.path.join(ROOT, 'exp*'))):
        name = os.path.basename(exp)
        sdir = os.path.join(exp, 'results/phase1/shift_eval')
        summ = _read_csv(os.path.join(sdir, 'summary.csv'))
        perp = _read_csv(os.path.join(sdir, 'test_model_fixed.csv'))
        if summ is None or perp is None:
            continue
        s = summ.iloc[0]
        short = name.replace('_shift', '').replace('shift_', '')
        entries.append({
            'model': short, 'kind': 'shift',
            'auroc': float(s.get('test_model_fixed_auroc', np.nan)),
            'pearson': float(s.get('test_model_fixed_pearson_deg', np.nan)),
            'mmd': float(s.get('test_model_fixed_mmd', np.nan)),
            'mmd_floor': float(s.get('test_model_fixed_mmd_ctrl_floor', np.nan)),
            'per_pert_auroc': perp['auroc'].dropna().values if 'auroc' in perp else np.array([]),
        })
        # capture the fair baselines once (identical data across experiments)
        if 'corr' not in refs and not np.isnan(s.get('test_corr_auroc', np.nan)):
            refs['corr'] = float(s['test_corr_auroc'])
        if 'coexp' not in refs and not np.isnan(s.get('test_coexp_auroc', np.nan)):
            refs['coexp'] = float(s['test_coexp_auroc'])

    # ── Baseline (exp1) legacy-pathway evals ───────────────────────────────
    base = os.path.join(ROOT, 'exp1_baseline', 'results')
    for v in ('V1', 'V2'):
        for mode in ('target_lfc', 'canonical'):
            d = os.path.join(base, f'eval_{v}_{mode}')
            summ = _read_csv(os.path.join(d, 'summary.csv'))
            perp = _read_csv(os.path.join(d, 'single_pert_model_test.csv'))
            if summ is None:
                continue
            s = summ.iloc[0]
            entries.append({
                'model': f'baseline_{v}_{mode}', 'kind': 'baseline',
                'auroc': float(s.get('single_test_model_auroc', np.nan)),
                'pearson': float(s.get('single_test_model_pearson', np.nan)),
                'mmd': np.nan, 'mmd_floor': np.nan,
                'per_pert_auroc': (perp['auroc'].dropna().values
                                   if perp is not None and 'auroc' in perp else np.array([])),
            })
    return entries, refs


def fig_headline(entries, refs, out):
    """Bar chart: held-out test AUROC per model + corr/co-exp reference lines."""
    order = sorted(entries, key=lambda e: (e['kind'] != 'baseline', e['model']))
    names = [e['model'] for e in order]
    aurocs = [e['auroc'] for e in order]
    colors = [C_BASE if e['kind'] == 'baseline' else C_SHIFT for e in order]

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(names)), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, aurocs, color=colors, edgecolor='black', width=0.6, linewidth=1.1)
    for b, a in zip(bars, aurocs):
        if not np.isnan(a):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f'{a:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if 'corr' in refs:
        ax.axhline(refs['corr'], color=C_CORR, ls='--', lw=2, label=f"correlation ({refs['corr']:.3f})")
    if 'coexp' in refs:
        ax.axhline(refs['coexp'], color=C_COEXP, ls='--', lw=2, label=f"co-exp neighbour ({refs['coexp']:.3f})")
    ax.axhline(0.5, color='red', ls=':', lw=1, alpha=0.6, label='random (0.5)')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Held-out test AUROC (downstream-DEG)', fontsize=12)
    ax.set_title('Perturbation prediction — held-out test AUROC (shift models c=1 vs baselines)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0.45, max(0.75, max([a for a in aurocs if not np.isnan(a)] + [0.7]) + 0.05))
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, 'headline_test_auroc.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    return p


def fig_box(entries, refs, out):
    """Per-perturbation AUROC distribution across models."""
    order = sorted(entries, key=lambda e: (e['kind'] != 'baseline', e['model']))
    data = [e['per_pert_auroc'] for e in order if len(e['per_pert_auroc'])]
    names = [e['model'] for e in order if len(e['per_pert_auroc'])]
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(names)), 5))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, e in zip(bp['boxes'], [e for e in order if len(e['per_pert_auroc'])]):
        patch.set_facecolor(C_BASE if e['kind'] == 'baseline' else C_SHIFT); patch.set_alpha(0.7)
    for med in bp['medians']:
        med.set_color('black')
    if 'coexp' in refs:
        ax.axhline(refs['coexp'], color=C_COEXP, ls='--', lw=2, label=f"co-exp ({refs['coexp']:.3f})")
    if 'corr' in refs:
        ax.axhline(refs['corr'], color=C_CORR, ls='--', lw=2, label=f"corr ({refs['corr']:.3f})")
    ax.axhline(0.5, color='red', ls=':', lw=1, alpha=0.6)
    ax.set_xticks(np.arange(1, len(names) + 1)); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Per-perturbation AUROC', fontsize=12)
    ax.set_title('Per-perturbation held-out AUROC distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, 'per_pert_auroc_box.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    return p


def fig_metrics_panel(entries, out):
    """AUROC | Pearson-on-DEGs | MMD vs control floor (shift models only for MMD)."""
    order = sorted(entries, key=lambda e: (e['kind'] != 'baseline', e['model']))
    names = [e['model'] for e in order]
    x = np.arange(len(names))
    colors = [C_BASE if e['kind'] == 'baseline' else C_SHIFT for e in order]
    fig, axes = plt.subplots(1, 3, figsize=(max(14, 1.6 * len(names)), 5))
    for ax, key, title in [(axes[0], 'auroc', 'Test AUROC'),
                           (axes[1], 'pearson', 'Pearson-on-DEGs')]:
        vals = [e[key] for e in order]
        ax.bar(x, vals, color=colors, edgecolor='black', width=0.6)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # MMD: model vs control floor (shift only)
    ax = axes[2]
    sh = [e for e in order if e['kind'] == 'shift' and not np.isnan(e['mmd'])]
    xs = np.arange(len(sh)); w = 0.4
    ax.bar(xs - w/2, [e['mmd'] for e in sh], w, label='model', color=C_SHIFT, edgecolor='black')
    ax.bar(xs + w/2, [e['mmd_floor'] for e in sh], w, label='control floor', color=C_CORR, edgecolor='black')
    ax.set_xticks(xs); ax.set_xticklabels([e['model'] for e in sh], rotation=40, ha='right', fontsize=8)
    ax.set_title('MMD: model vs control floor (lower=better)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out, 'metrics_panel.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    entries, refs = collect()
    if not entries:
        print("No eval CSVs found yet. Run the evals first."); return
    print(f"Found {len(entries)} model evals; refs={refs}")

    # combined summary table
    tab = pd.DataFrame([{k: e[k] for k in ('model', 'kind', 'auroc', 'pearson', 'mmd', 'mmd_floor')}
                        for e in entries])
    for k, v in refs.items():
        tab.loc[len(tab)] = {'model': f'[ref] {k}', 'kind': 'baseline', 'auroc': v,
                             'pearson': np.nan, 'mmd': np.nan, 'mmd_floor': np.nan}
    tab = tab.sort_values(['kind', 'auroc'], ascending=[True, False])
    tab.to_csv(os.path.join(args.out, 'comparison_summary.csv'), index=False)
    print(tab.to_string(index=False))

    for fn in (fig_headline, fig_box):
        p = fn(entries, refs, args.out)
        if p: print("wrote", p)
    p = fig_metrics_panel(entries, args.out)
    if p: print("wrote", p)
    print(f"\nFigures + comparison_summary.csv in {args.out}/")


if __name__ == '__main__':
    main()

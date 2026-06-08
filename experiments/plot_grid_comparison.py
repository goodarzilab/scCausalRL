"""Comparison figures for the exp8/exp9 DAG-revival + prior grids.

Reads each grid cell's Replogle held-out shift-eval, Norman external eval, and the
A-bypass ratio (parsed from the dag_grad_diagnostic log) + held-out-TF recovery
(dag_analysis scorecard). Produces:
  - dag_revival_fix.png : per exp8 cell, Replogle AUROC + A-bypass ratio (the
    "did removing lambda_sparse revive A without losing AUROC?" headline).
  - grid_indist_vs_norman.png : Replogle vs Norman AUROC for cells with both.
Re-runnable; skips cells whose evals haven't finished.
Usage: python plot_grid_comparison.py
"""
import os, glob, re, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
try: plt.style.use('seaborn-v0_8-whitegrid')
except Exception: pass
ROOT = os.path.dirname(os.path.abspath(__file__)); LOGS = os.path.join(ROOT, 'logs')

CELLS = [  # (label, exp_dir, cell_dir_name)
    ('exp8 sl0_ls10 (≈exp2)', 'exp8_dag_revival_grid', 'exp8_sl0p0_ls10p0'),
    ('exp8 sl0_ls0',          'exp8_dag_revival_grid', 'exp8_sl0p0_ls0p0'),
    ('exp8 sl0.1_ls10',       'exp8_dag_revival_grid', 'exp8_sl0p1_ls10p0'),
    ('exp8 sl0.1_ls0 ★',      'exp8_dag_revival_grid', 'exp8_sl0p1_ls0p0'),
    ('exp8 sl0.5_ls10',       'exp8_dag_revival_grid', 'exp8_sl0p5_ls10p0'),
    ('exp8 sl0.5_ls0',        'exp8_dag_revival_grid', 'exp8_sl0p5_ls0p0'),
    ('exp9 c1 unsup',         'exp9_prior_generalization', 'exp9_c1_unsup'),
    ('exp9 c2 heldtf',        'exp9_prior_generalization', 'exp9_c2_heldtf'),
    ('exp9 c4 biginit',       'exp9_prior_generalization', 'exp9_c4_biginit'),
]


def _auroc(p, col='test_model_fixed_auroc'):
    try: return float(pd.read_csv(p).iloc[0][col])
    except Exception: return np.nan

def _bypass(cell):
    fs = sorted(glob.glob(os.path.join(LOGS, f'diag_{cell}_*.out')), key=os.path.getmtime)
    if not fs: return np.nan
    txt = open(fs[-1]).read()
    m = re.search(r'‖z_post − \(ε\+S\)‖ / ‖ε\+S‖ = ([0-9.eE+-]+)', txt)
    return float(m.group(1)) if m else np.nan

def collect():
    rows = []
    for label, exp, cell in CELLS:
        base = os.path.join(ROOT, exp, 'results/grid', cell)
        sc = os.path.join(base, 'dag_analysis', 'scorecard.json')
        held = np.nan
        if os.path.exists(sc):
            try: held = float(json.load(open(sc)).get('grn_heldout_auroc', np.nan))
            except Exception: pass
        rows.append(dict(label=label, cell=cell,
                         repl=_auroc(os.path.join(base, 'shift_eval', 'summary.csv')),
                         norman=_auroc(os.path.join(base, 'norman_eval', 'summary.csv'), 'norman_model_fixed_auroc'),
                         bypass=_bypass(cell), heldout_tf=held))
    return pd.DataFrame(rows)


def main():
    out = os.path.join(ROOT, 'figures'); os.makedirs(out, exist_ok=True)
    df = collect()
    df.to_csv(os.path.join(out, 'grid_comparison_summary.csv'), index=False)
    print(df.to_string(index=False))

    # Fig 1: exp8 "did we fix it" — Replogle AUROC + A-bypass
    e8 = df[df.cell.str.startswith('exp8')].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(e8))
    cols = ['#41ab5d' if (b == b and b > 0.1) else '#bdbdbd' for b in e8.bypass]  # green if A active
    axes[0].bar(x, e8.repl, color=cols, edgecolor='black')
    for i, v in enumerate(e8.repl):
        if v == v: axes[0].text(i, v+0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[0].axhline(0.5, color='red', ls='--', lw=1); axes[0].set_ylim(0.45, 0.75)
    axes[0].set_ylabel('Replogle held-out AUROC'); axes[0].set_title('Prediction (green = A active)')
    axes[0].set_xticks(x); axes[0].set_xticklabels(e8.label, rotation=35, ha='right', fontsize=8)
    bp = np.where(e8.bypass.values > 0, e8.bypass.values, 1e-6)
    axes[1].bar(x, bp, color=cols, edgecolor='black', log=True)
    for i, v in enumerate(e8.bypass):
        if v == v: axes[1].text(i, v*1.3, f'{v:.0e}', ha='center', fontsize=8, fontweight='bold')
    axes[1].set_ylabel('A-bypass ratio ‖z_post−(ε+S)‖/‖ε+S‖ (log)')
    axes[1].set_title('Is A used? (exp2≈1e-5 dead → ~0.45 active)')
    axes[1].set_xticks(x); axes[1].set_xticklabels(e8.label, rotation=35, ha='right', fontsize=8)
    for a in axes: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.suptitle('exp8 DAG-revival grid: removing lambda_sparse revives A while keeping AUROC',
                 fontsize=13, fontweight='bold'); plt.tight_layout()
    plt.savefig(os.path.join(out, 'dag_revival_fix.png'), dpi=180, bbox_inches='tight'); plt.close()
    print('wrote dag_revival_fix.png')

    # Fig 2: Replogle vs Norman for cells with both
    both = df[(df.repl == df.repl) & (df.norman == df.norman)].copy()
    if len(both):
        x = np.arange(len(both)); w = 0.38
        fig, ax = plt.subplots(figsize=(max(8, 1.4*len(both)), 5))
        ax.bar(x-w/2, both.repl, w, color='#2171b5', edgecolor='black', label='Replogle held-out')
        ax.bar(x+w/2, both.norman, w, color='#41ab5d', edgecolor='black', label='Norman external')
        ax.axhline(0.5, color='red', ls='--', lw=1); ax.set_ylabel('AUROC'); ax.set_ylim(0.45, 0.8)
        ax.set_xticks(x); ax.set_xticklabels(both.label, rotation=25, ha='right', fontsize=8)
        ax.legend(); ax.set_title('Grid cells: in-distribution vs external generalization')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'grid_indist_vs_norman.png'), dpi=180, bbox_inches='tight'); plt.close()
        print('wrote grid_indist_vs_norman.png')
    print(f"\nfigures in {out}/")


if __name__ == '__main__':
    main()

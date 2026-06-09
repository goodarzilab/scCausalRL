"""Consolidated Norman-native eval figure for the Tier-1/Tier-2 models.

Left  : held-out SINGLE-pert AUROC (n=8 test genes — noisy, caveat) vs corr/co-exp.
Right : COMBO AUROC (n=107) — shift_add vs shift_uscore per model, with additive_obs
        (oracle) and correlation as reference lines. The well-powered panel.
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
try: plt.style.use('seaborn-v0_8-whitegrid')
except Exception: pass
ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = [  # (label, color, shift_eval_dir, combo_csv)
    ('exp11 additive\n(Tier-1)', '#2171b5',
     'exp11_norman_shift/results/phase1/shift_eval',
     'exp11_norman_shift/results/phase1/norman_combo/combo_per_pert.csv'),
    ('exp12 mut\nloading', '#fd8d3c',
     'exp12_shift_mutilate/results/grid/exp12_norman_loadclamp/shift_eval',
     'exp12_shift_mutilate/results/grid/exp12_norman_loadclamp/norman_combo/combo_per_pert.csv'),
    ('exp12 mut\nshift-native ★', '#41ab5d',
     'exp12_shift_mutilate/results/grid/exp12_norman_shiftclamp/shift_eval',
     'exp12_shift_mutilate/results/grid/exp12_norman_shiftclamp/norman_combo/combo_per_pert.csv'),
]


def msem(a):
    a = np.asarray([x for x in a if x == x])
    return (float(a.mean()), float(a.std()/np.sqrt(len(a)))) if len(a) else (np.nan, 0.0)


def main():
    out = os.path.join(ROOT, 'figures'); os.makedirs(out, exist_ok=True)
    fig, (axS, axC) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Panel A: held-out single AUROC (n=8) ----
    corr_a = pd.read_csv(os.path.join(ROOT, MODELS[0][2], 'test_corr.csv'))['auroc']
    coexp_a = pd.read_csv(os.path.join(ROOT, MODELS[0][2], 'test_coexp.csv'))['auroc']
    bars = [('correlation', '#bdbdbd', corr_a), ('co-exp nbr', '#969696', coexp_a)]
    for lbl, c, sed, _ in MODELS:
        a = pd.read_csv(os.path.join(ROOT, sed, 'test_model_fixed.csv'))['auroc']
        bars.append((lbl, c, a))
    n_single = len(corr_a)
    xs = np.arange(len(bars))
    for i, (lbl, c, a) in enumerate(bars):
        m, s = msem(a)
        axS.bar(i, m, 0.7, yerr=s, capsize=4, color=c, edgecolor='black')
        axS.text(i, m+s+0.005, f'{m:.3f}', ha='center', fontsize=9, fontweight='bold')
    axS.axhline(0.5, color='red', ls='--', lw=1, alpha=0.6)
    axS.set_xticks(xs); axS.set_xticklabels([b[0] for b in bars], fontsize=8)
    axS.set_ylabel('held-out single-pert AUROC', fontsize=12); axS.set_ylim(0.45, 1.0)
    axS.set_title(f'Held-out SINGLE perturbations (n={n_single} test genes — noisy)', fontsize=12, fontweight='bold')
    axS.spines['top'].set_visible(False); axS.spines['right'].set_visible(False)

    # ---- Panel B: combo AUROC (n=107), shift_add vs shift_uscore per model ----
    w = 0.34; xc = np.arange(len(MODELS))
    add_obs_vals, corr_vals, n_combo = [], [], 0
    for j, (lbl, c, _, ccsv) in enumerate(MODELS):
        df = pd.read_csv(os.path.join(ROOT, ccsv))
        n_combo = len(df)
        for k, (meth, hatch, col) in enumerate([('shift_add', '', '#6baed6'), ('shift_uscore', '//', c)]):
            m, s = msem(df[f'{meth}_auroc'])
            axC.bar(j + (k-0.5)*w, m, w, yerr=s, capsize=3, color=col, edgecolor='black',
                    hatch=hatch, label=meth if j == 0 else None)
            axC.text(j+(k-0.5)*w, m+s+0.004, f'{m:.3f}', ha='center', fontsize=8, fontweight='bold')
        add_obs_vals.append(msem(df['additive_obs_auroc'])[0]); corr_vals.append(msem(df['corr_auroc'])[0])
    axC.axhline(np.mean(add_obs_vals), color='#9e9ac8', ls='--', lw=1.6, label=f'additive_obs (oracle) {np.mean(add_obs_vals):.3f}')
    axC.axhline(np.mean(corr_vals), color='#bdbdbd', ls=':', lw=1.6, label=f'correlation {np.mean(corr_vals):.3f}')
    axC.set_xticks(xc); axC.set_xticklabels([m[0] for m in MODELS], fontsize=8)
    axC.set_ylabel('Norman combo AUROC', fontsize=12); axC.set_ylim(0.7, 1.0)
    axC.set_title(f'COMBO prediction (n={n_combo}) — zero-shot additive vs mutilation', fontsize=12, fontweight='bold')
    axC.legend(fontsize=8, loc='lower right'); axC.spines['top'].set_visible(False); axC.spines['right'].set_visible(False)

    fig.suptitle('Norman-native eval: additive (Tier-1) vs coherently-trained mutilation (Tier-2)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out, 'norman_native_eval_summary.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print('wrote', p)


if __name__ == '__main__':
    main()

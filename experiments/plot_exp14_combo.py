"""exp14 combo eval figures + the exp11/exp12/exp14 'did combo-training help' comparison.
Reads the on-disk summary.csv / combo_per_pert.csv. Saves to experiments/figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(R, 'figures')


def _summ(p):
    return pd.read_csv(p).iloc[0]


def fig_exp14():
    cs = _summ(f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/norman_combo/summary.csv')
    ss = _summ(f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/shift_eval/summary.csv')
    methods = ['corr', 'shift_add', 'shift_uscore', 'additive_obs']
    labs = ['corr\n(baseline)', 'shift_add\n(additive,0-shot)', 'shift_uscore\n(mutilation,0-shot)', 'additive_obs\n(obs oracle)']
    cols = ['#999', '#4C72B0', '#C44E52', '#55A868']
    au = [cs[f'{m}_auroc'] for m in methods]; sp = [cs[f'{m}_spearman'] for m in methods]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, vals, t in [(ax[0], au, 'Combo DEG-AUROC (n=107)'), (ax[1], sp, 'Combo Spearman(|effect|)')]:
        b = a.bar(range(4), vals, color=cols)
        for i, v in enumerate(vals): a.text(i, v+0.004, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        a.set_xticks(range(4)); a.set_xticklabels(labs, fontsize=8); a.set_title(t); a.grid(axis='y', alpha=0.3)
    ax[0].axhline(0.5, ls='--', c='gray', lw=0.8)
    fig.suptitle(f"exp14 Norman combo-trained world model (RL WM) — held-out singles AUROC = "
                 f"{ss['test_model_fixed_auroc']:.3f}  |  combos: mutilation {cs['shift_uscore_auroc']:.3f} > additive {cs['shift_add_auroc']:.3f}",
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95]); out = f'{FIG}/exp14_combo_eval.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); print('saved', out)


def fig_compare():
    # did combo-training help? exp11 additive (singles), exp12 mut-shiftnative (singles), exp14 (combo-trained)
    runs = [
        ('exp11\n(additive,\nsingles-only)',
         f'{R}/exp11_norman_shift/results/phase1/norman_combo/summary.csv',
         f'{R}/exp11_norman_shift/results/phase1/shift_eval/summary.csv'),
        ('exp12 shift-native\n(mutilation,\nsingles-only)',
         f'{R}/exp12_shift_mutilate/results/grid/exp12_norman_shiftclamp/norman_combo/summary.csv',
         f'{R}/exp12_shift_mutilate/results/grid/exp12_norman_shiftclamp/shift_eval/summary.csv'),
        ('exp14\n(combo-TRAINED)',
         f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/norman_combo/summary.csv',
         f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/shift_eval/summary.csv'),
    ]
    add = [_summ(c)['shift_add_auroc'] for _, c, _ in runs]
    mut = [_summ(c)['shift_uscore_auroc'] for _, c, _ in runs]
    sing = [_summ(s)['test_model_fixed_auroc'] for _, _, s in runs]
    x = np.arange(3); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x-w, sing, w, label='held-out SINGLES AUROC', color='#8172B3')
    ax.bar(x,   add,  w, label='combos: shift_add (additive)', color='#4C72B0')
    ax.bar(x+w, mut,  w, label='combos: shift_uscore (mutilation)', color='#C44E52')
    for i in range(3):
        for dx, v in [(-w, sing[i]), (0, add[i]), (w, mut[i])]:
            ax.text(i+dx, v+0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in runs]); ax.set_ylim(0.78, 0.95)
    ax.set_ylabel('DEG-AUROC'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.set_title('Did combo-training help? Norman, held-out evals\n'
                 'combo-training (exp14) lifts BOTH singles (0.891→0.918) and combos (0.896→0.910)', fontsize=10)
    fig.tight_layout(); out = f'{FIG}/exp11_12_14_combo_compare.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); print('saved', out)


if __name__ == '__main__':
    fig_exp14(); fig_compare()

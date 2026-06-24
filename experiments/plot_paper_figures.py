"""Regenerate the 4 result figures in a consistent research-paper style: single colour
hue, SEM error bars (from per-item CSVs), minimal text, no top/right spines.
  - exp14_combo_eval.png            (exp14 model card: held-out singles + combos)
  - exp11_12_14_combo_compare.png   (did combo-training help; held-out)
  - exp16_crosscell.png             (per-line held-out singles, model vs corr)
  - rl_value_add_ladder.png         (DE-match per planner, held-out combos)
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(R, 'figures')
HUE = '#3C5488'
SHADES = ['#A9B8D0', '#6F86B0', '#3C5488']
CK = f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/checkpoints/phase1_epoch150.pt'
TEST_FS = {frozenset(k) for k in torch.load(CK, map_location='cpu', weights_only=False)['config']['phase1']['combo_test_keys']}


def sem(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def _heldout(csv):
    d = pd.read_csv(csv); d['fs'] = d['combo'].map(lambda s: frozenset(s.split('+')))
    return d[d['fs'].isin(TEST_FS)]


def _clean(ax, ylo, hline=True):
    ax.set_ylim(ylo, 1.0); ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=0.25)
    if hline and ylo < 0.5:
        ax.axhline(0.5, ls='--', c='gray', lw=0.7)


def fig_exp14():
    se = f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/shift_eval'
    cp = _heldout(f'{R}/exp14_norman_combotrain/results/grid/exp14_norman_combotrain/norman_combo/combo_per_pert.csv')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # singles (per-pert AUROC): model / corr / coexp
    s_labs, s_m, s_e, s_h = [], [], [], []
    for lab, f, hatch in [('model', 'test_model_fixed.csv', ''), ('corr', 'test_corr.csv', '//'), ('coexp', 'test_coexp.csv', '//')]:
        try:
            m, e = sem(pd.read_csv(f'{se}/{f}')['auroc']); s_labs.append(lab); s_m.append(m); s_e.append(e); s_h.append(hatch)
        except Exception:
            pass
    bars = ax[0].bar(range(len(s_labs)), s_m, yerr=s_e, capsize=3, color=HUE, edgecolor='k', linewidth=0.6)
    for b, h in zip(bars, s_h):
        if h: b.set_hatch(h); b.set_alpha(0.55)
    for i, (m, e) in enumerate(zip(s_m, s_e)): ax[0].text(i, m+e+0.006, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
    ax[0].set_xticks(range(len(s_labs))); ax[0].set_xticklabels(s_labs); _clean(ax[0], 0.5)
    ax[0].set_ylabel('DEG-AUROC'); ax[0].set_title('Held-out SINGLES', fontsize=11)
    # combos (held-out): 4 predictors, AUROC
    c_specs = [('corr', '//'), ('shift_add', ''), ('shift_uscore', ''), ('additive_obs', '//')]
    c_m, c_e, c_h, c_l = [], [], [], []
    for m, h in c_specs:
        mu, e = sem(cp[f'{m}_auroc']); c_m.append(mu); c_e.append(e); c_h.append(h); c_l.append(m)
    bars = ax[1].bar(range(4), c_m, yerr=c_e, capsize=3, color=HUE, edgecolor='k', linewidth=0.6)
    for b, h in zip(bars, c_h):
        if h: b.set_hatch(h); b.set_alpha(0.55)
    for i, (m, e) in enumerate(zip(c_m, c_e)): ax[1].text(i, m+e+0.006, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(c_l, fontsize=8); _clean(ax[1], 0.5)
    ax[1].set_title(f'Held-out COMBOS (n={len(cp)})', fontsize=11)
    fig.suptitle('exp14 Norman combo-trained world model (RL WM)', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f'{FIG}/exp14_combo_eval.png', dpi=200, bbox_inches='tight')
    print('saved exp14_combo_eval.png')


def fig_compare():
    runs = [('exp11', 'exp11_norman_shift/results/phase1'),
            ('exp12-shift', 'exp12_shift_mutilate/results/grid/exp12_norman_shiftclamp'),
            ('exp14', 'exp14_norman_combotrain/results/grid/exp14_norman_combotrain')]
    series = [('held-out singles', 'auroc', 'single'), ('combo shift_add', 'shift_add_auroc', 'combo'),
              ('combo u-score', 'shift_uscore_auroc', 'combo')]
    M = np.zeros((3, 3)); E = np.zeros((3, 3))
    for j, (name, d) in enumerate(runs):
        sm, se_ = sem(pd.read_csv(f'{R}/{d}/shift_eval/test_model_fixed.csv')['auroc'])
        cp = _heldout(f'{R}/{d}/norman_combo/combo_per_pert.csv')
        for i, (_, col, kind) in enumerate(series):
            M[i, j], E[i, j] = (sm, se_) if kind == 'single' else sem(cp[col])
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (lab, _, _) in enumerate(series):
        ax.bar(x + (i-1)*w, M[i], w, yerr=E[i], capsize=3, color=SHADES[i], edgecolor='k', linewidth=0.5, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in runs]); ax.set_ylim(0.78, 0.95)
    ax.set_ylabel('DEG-AUROC'); ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=0.25)
    ax.set_title('Did combo-training help? (held-out singles + held-out combos)', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{FIG}/exp11_12_14_combo_compare.png', dpi=200, bbox_inches='tight')
    print('saved exp11_12_14_combo_compare.png')


def fig_exp16():
    d = pd.read_csv(f'{R}/exp16_multicellline/results/grid/exp16_multicellline/crosscell/per_pert.csv')
    lines = ['k562', 'hepg2', 'jurkat', 'rpe1']
    mm, me, cm, ce = [], [], [], []
    for L in lines:
        s = d[d['cell_line'] == L]
        a, b = sem(s['model_auroc']); mm.append(a); me.append(b)
        a, b = sem(s['corr_auroc']); cm.append(a); ce.append(b)
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, mm, w, yerr=me, capsize=3, color=HUE, edgecolor='k', linewidth=0.6, label='4-cell model')
    b2 = ax.bar(x + w/2, cm, w, yerr=ce, capsize=3, color=HUE, edgecolor='k', linewidth=0.6, label='corr baseline')
    for b in b2: b.set_hatch('//'); b.set_alpha(0.55)
    for i in range(4):
        ax.text(i-w/2, mm[i]+me[i]+0.005, f'{mm[i]:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([l.upper() for l in lines]); _clean(ax, 0.45)
    ax.set_ylabel('held-out single-pert DEG-AUROC'); ax.legend(fontsize=9, frameon=False)
    ax.set_title('exp16 — 4-cell-line model, per-line held-out singles', fontsize=11)
    fig.tight_layout(); fig.savefig(f'{FIG}/exp16_crosscell.png', dpi=200, bbox_inches='tight')
    print('saved exp16_crosscell.png')


def fig_rl():
    base = f'{R}/../experiments_rl'
    srcs = [('random', 'rl0_mpc/results/fullpool/random.csv', True),
            ('seq-PPO', 'rl1_seq_ppo/results/seqppo_test.csv', False),
            ('set-PPO', 'rl2_set_ppo/results/setppo_test.csv', False),
            ('MPC→distill', 'rl3_mpc_distill/results/distill_test.csv', False),
            ('top2-marginal', 'rl0_mpc/results/fullpool/top2_marginal.csv', True),
            ('greedy-1', 'rl0_mpc/results/fullpool/greedy1.csv', True),
            ('MPC', 'rl0_mpc/results/fullpool/MPC.csv', False)]
    labs, m, e, hatch = [], [], [], []
    for lab, rel, isb in srcs:
        p = f'{base}/{rel}'
        if not os.path.exists(p): continue
        mu, se_ = sem(pd.read_csv(p)['match']); labs.append(lab); m.append(mu); e.append(se_); hatch.append('//' if isb else '')
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(labs)), m, yerr=e, capsize=3, color=HUE, edgecolor='k', linewidth=0.6)
    for b, h in zip(bars, hatch):
        if h: b.set_hatch(h); b.set_alpha(0.55)
    for i, (mu, se_) in enumerate(zip(m, e)): ax.text(i, mu+se_+0.008, f'{mu:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel('gene-space DE-match'); _clean(ax, 0.0, hline=False)
    ax.set_title('RL value-add ladder — held-out combos, 8000-gene zero-shot pool\n(hatched = non-learned baselines)', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{FIG}/rl_value_add_ladder.png', dpi=200, bbox_inches='tight')
    print('saved rl_value_add_ladder.png')


if __name__ == '__main__':
    fig_exp14(); fig_compare(); fig_exp16(); fig_rl()

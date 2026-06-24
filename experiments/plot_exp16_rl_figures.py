"""Generate (1) exp16 cross-cell-line bar chart and (2) the RL value-add ladder figure
from the on-disk result CSVs. Saves into experiments/figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)


def fig_exp16():
    f = os.path.join(ROOT, 'exp16_multicellline/results/grid/exp16_multicellline/crosscell/summary.csv')
    d = pd.read_csv(f)
    lines = d['cell_line'].str.upper().tolist()
    x = np.arange(len(lines)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, d['model_auroc'], w, label='4-cell model', color='#C44E52')
    b2 = ax.bar(x + w/2, d['corr_auroc'], w, label='correlation baseline', color='#999999')
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{b.get_height():.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.axhline(0.5, ls='--', c='gray', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(lines, d['n_heldout'])])
    ax.set_ylabel('held-out single-pert DEG-AUROC'); ax.set_ylim(0.45, 0.78)
    ax.set_title('exp16 — 4-cell-line CRISPRi world model: per-line held-out singles\n'
                 '(joint training beats corr in every context; RPE1 rescued vs exp13 0.694≈0.679)',
                 fontsize=10)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); out = os.path.join(FIG, 'exp16_crosscell.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); print('saved', out)


def fig_rl_ladder():
    base = os.path.join(ROOT, '../experiments_rl')
    lad = pd.read_csv(os.path.join(base, 'rl0_mpc/results/fullpool/ladder.csv'))
    rows = {r['label']: (r['match'], r['recall']) for _, r in lad.iterrows()}
    for label, rel in [('seq_PPO', 'rl1_seq_ppo/results/seqppo_test.csv'),
                       ('set_PPO', 'rl2_set_ppo/results/setppo_test.csv'),
                       ('distill', 'rl3_mpc_distill/results/distill_test.csv')]:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            t = pd.read_csv(p); rows[label] = (t['match'].mean(), t['recall'].mean())
    order = ['random', 'seq_PPO', 'set_PPO', 'distill', 'top2_marginal', 'greedy1', 'MPC']
    order = [m for m in order if m in rows]
    vals = [rows[m][0] for m in order]
    fam = {'random': '#999', 'seq_PPO': '#4C72B0', 'set_PPO': '#4C72B0', 'distill': '#8172B3',
           'top2_marginal': '#55A868', 'greedy1': '#55A868', 'MPC': '#C44E52'}
    cols = [fam.get(m, '#777') for m in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    b = ax.bar(range(len(order)), vals, color=cols)
    for i, m in enumerate(order):
        ax.text(i, vals[i]+0.008, f'{vals[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(['random', 'seq-PPO', 'set-PPO', 'MPC→distill', 'top2\nmarginal', 'greedy-1', 'MPC\n(oracle)'][:len(order)])
    ax.set_ylabel('gene-space DE-match (held-out combos)')
    ax.set_title('RL value-add ladder — 8000-gene zero-shot pool (frozen exp14 world model)\n'
                 'learned policies (blue/purple) amortize; MPC/greedy (red/green) are the planning ceiling',
                 fontsize=10)
    ax.set_ylim(0, 0.9); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); out = os.path.join(FIG, 'rl_value_add_ladder.png')
    fig.savefig(out, dpi=150, bbox_inches='tight'); print('saved', out)


if __name__ == '__main__':
    fig_exp16()
    fig_rl_ladder()

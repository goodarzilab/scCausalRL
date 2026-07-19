"""Generate the Briggs-benchmark write-up figures (PNG) from the produced results, following the dataviz
method (validated palette, thin marks, recessive grid, direct labels, one axis, legend for >=2 series).
All numbers are the established results (sources cited inline). Writes results/briggs/figures/fig*.png.
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'briggs', 'figures')
os.makedirs(OUT, exist_ok=True)

# ---- dataviz reference palette (light surface) ----
BLUE, GREEN, MAG, YEL, AQUA, ORANGE, VIOLET, RED = (
    '#2a78d6', '#008300', '#e87ba4', '#eda100', '#1baf7a', '#eb6834', '#4a3aa7', '#e34948')
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.linewidth': 1.0, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
})


def finish(ax):
    ax.spines['left'].set_color(AXIS); ax.spines['bottom'].set_color(AXIS)
    ax.grid(axis='x', visible=False)


# ============ FIG 1: frozen-latent readout recovers high-k fate fidelity ============
# source: logs/readout (deg3_t25 base, random 80/20); readout_head.py by-k
def fig1():
    k = [2, 3, 4, 5, 6, 7]
    readout = [0.726, 0.790, 0.763, 0.714, 0.692, 0.631]
    raw = [0.190, 0.297, 0.406, 0.354, 0.310, 0.302]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(k, readout, '-o', color=BLUE, lw=2, ms=8, label='Ridge readout on frozen WM latent', zorder=3)
    ax.plot(k, raw, '-o', color=ORANGE, lw=2, ms=8, label='raw one-step fate-axis projection', zorder=3)
    for x, y in zip(k, readout): ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, 9), ha='center', color=BLUE, fontsize=9)
    for x, y in zip(k, raw): ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(0, -14), ha='center', color=ORANGE, fontsize=9)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel('cocktail size  k  (# TFs)'); ax.set_ylabel('Spearman(predicted, observed HSPC fate)')
    ax.set_title('Fig 1 · The frozen WM latent encodes fate at every k —\nthe raw one-step projection loses it', fontsize=12, loc='left')
    ax.set_ylim(-0.05, 0.9); ax.set_xticks(k); ax.legend(frameon=False, loc='center right', fontsize=10)
    finish(ax); fig.tight_layout(); fig.savefig(f'{OUT}/fig1_readout_by_k.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 2: model sweep — Briggs transfer ranking + regularization trend ============
# source: 33-model sweep_fidelity (sp_k3, reliable k<=3)
def fig2():
    ranking = [  # (tag, sp_k3) sorted desc
        ('deg3_t25_frozen_r32', 0.272), ('mmd0.25', 0.229), ('mmd1_Jcombo_ho50', 0.226), ('masked_go', 0.221),
        ('nlSCM_mmd2_deg2', 0.219), ('mmd0.5', 0.173), ('fthead_r64', 0.167), ('Jhs_L256_k32', 0.161),
        ('ft_fullcombo_fx', 0.161), ('ft_ixhead_frozen', 0.141), ('mmd2', 0.139), ('dag_only_nohead', 0.138),
        ('mmd5', 0.125), ('mmd1_deg2', 0.125), ('nlSCM_mmd1', 0.116), ('dag_head_r16', 0.116),
        ('fthead_r64_e80', 0.107), ('mmd1_deg1', 0.105), ('ft_fullcombo', 0.104), ('mmd1_deg5', 0.101),
        ('ft_combofrac0.1', 0.097), ('ft_ixhead_frozen_fx', 0.076), ('mmd1_deg3', 0.069),
        ('prod_deg8_all40', 0.055), ('mmd1_JNcombo_ho50', 0.051), ('deg10_frozen_r32', 0.051), ('mmd10', 0.050),
        ('mmd1_deg2_Jcombo_ho50', 0.041), ('deg5_frozen_r32', 0.036), ('deg8_frozen_r32', 0.032),
        ('mmd1_deg8', 0.028), ('fthead_r32', 0.021), ('J_hardself_k8_base', -0.019)]
    tags = [t for t, _ in ranking][::-1]; vals = [v for _, v in ranking][::-1]
    colors = []
    for t in tags:
        colors.append(BLUE if 'deg3_t25_frozen' in t else (RED if t == 'prod_deg8_all40' else MUTED))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 8), gridspec_kw={'width_ratios': [1.5, 1]})
    y = np.arange(len(tags))
    axL.barh(y, vals, color=colors, height=0.72, edgecolor=SURFACE, linewidth=0.5)
    axL.set_yticks(y); axL.set_yticklabels(tags, fontsize=7.5)
    axL.axvline(0, color=AXIS, lw=0.8)
    axL.set_xlabel('Briggs transfer — Spearman (reliable k≤3)')
    axL.set_title('Fig 2A · Best-on-Joung ≠ best-on-Briggs:\nproduction model ranks 24 / 33', fontsize=12, loc='left')
    axL.annotate('production\n(deg8, all-40)', (0.055, tags.index('prod_deg8_all40')), xytext=(0.14, tags.index('prod_deg8_all40')),
                 fontsize=8.5, color=RED, va='center', arrowprops=dict(arrowstyle='->', color=RED, lw=1))
    axL.annotate('chosen base\n(deg3_t25)', (0.272, tags.index('deg3_t25_frozen_r32')), xytext=(0.15, tags.index('deg3_t25_frozen_r32') - 3),
                 fontsize=8.5, color=BLUE, va='center', arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
    axL.grid(axis='y', visible=False); axL.spines['left'].set_color(AXIS); axL.spines['bottom'].set_color(AXIS)

    # panel B: MMD-weight trend (regularization anti-correlates with transfer)
    mmd_w = [0.25, 0.5, 2, 5, 10]; mmd_v = [0.229, 0.173, 0.139, 0.125, 0.050]
    axR.plot(mmd_w, mmd_v, '-o', color=BLUE, lw=2, ms=8)
    for x, yy in zip(mmd_w, mmd_v): axR.annotate(f'{yy:.2f}', (x, yy), textcoords='offset points', xytext=(0, 9), ha='center', color=BLUE, fontsize=9)
    axR.set_xscale('log'); axR.set_xticks(mmd_w); axR.set_xticklabels([str(w) for w in mmd_w])
    axR.set_xlabel('MMD loss weight (log)'); axR.set_ylabel('Briggs transfer — Spearman (k≤3)')
    axR.set_title('Fig 2B · Heavier regularization →\nworse transfer (overfits Joung DE)', fontsize=12, loc='left')
    axR.set_ylim(0, 0.28); finish(axR)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig2_model_sweep.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 3: readout generalization — Ridge vs MLP, random vs leave-TF-out ============
def fig3():
    groups = ['random 80/20', 'leave-TF-out\n(20 novel TFs)']
    ridge = [0.707, 0.562]; mlp = [0.693, 0.100]
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    b1 = ax.bar(x - w / 2 - 0.01, ridge, w, color=BLUE, edgecolor=SURFACE, linewidth=1, label='Ridge (linear)')
    b2 = ax.bar(x + w / 2 + 0.01, mlp, w, color=VIOLET, edgecolor=SURFACE, linewidth=1, label='MLP (nonlinear)')
    for b in list(b1) + list(b2):
        ax.annotate(f'{b.get_height():.2f}', (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel('Spearman(pred, obs fate), test')
    ax.set_title('Fig 3 · Use the LINEAR readout — MLP overfits training TFs,\ncollapses on novel TFs; Ridge generalizes', fontsize=11.5, loc='left')
    ax.set_ylim(0, 0.8); ax.legend(frameon=False, fontsize=10); finish(ax)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig3_readout_generalization.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 4: pessimism/support calibration — MOReL/MOPO fix ============
def fig4():
    labels = ['λ0 γ0\n(none)', 'λ3 γ0\n(DAG only)', 'λ1 γ2\n(support)', 'λ1 γ5', 'λ1 γ10']
    driver = [0.00, 0.09, 0.28, 0.25, 0.36]; maha = [26.0, 12.0, 1.7, 1.6, 1.55]
    x = np.arange(len(labels))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    cols = [MUTED, MUTED, GREEN, BLUE, BLUE]
    bA = axA.bar(x, driver, 0.62, color=cols, edgecolor=SURFACE, linewidth=1)
    axA.axhline(0.03, color=AXIS, lw=1, ls='--'); axA.annotate('chance 0.03', (len(x) - 1, 0.03), xytext=(0, 5), textcoords='offset points', ha='right', color=MUTED, fontsize=8.5)
    for b in bA: axA.annotate(f'{b.get_height():.2f}', (b.get_x() + b.get_width() / 2, b.get_height()), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9, color=INK2)
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9); axA.set_ylabel('HSPC-driver rate in top cocktails')
    axA.set_title('Fig 4A · Recovery of real biology', fontsize=11.5, loc='left'); axA.set_ylim(0, 0.42); finish(axA)
    bB = axB.bar(x, maha, 0.62, color=cols, edgecolor=SURFACE, linewidth=1)
    axB.set_yscale('log')
    for b in bB: axB.annotate(f'{b.get_height():.1f}', (b.get_x() + b.get_width() / 2, b.get_height()), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=9, color=INK2)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=9); axB.set_ylabel('Mahalanobis dist. to Briggs manifold (log)')
    axB.set_title('Fig 4B · On-manifold-ness (support)', fontsize=11.5, loc='left'); finish(axB); axB.grid(axis='x', visible=False)
    fig.suptitle('Support penalty (MOReL/MOPO) fixes reward exploitation — DAG-uncertainty alone does not',
                 fontsize=12.5, x=0.01, ha='left', y=1.02)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig4_pessimism_calibration.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 5: value-add ladder (M0/M2/M3) ============
def fig5():
    methods = ['M0 fate-MPC\n(planner)', 'M2 set-PPO\n(from scratch)', 'M3 distill\n(BC of planner)']
    rate = [0.28, 0.07, 0.47]; cols = [BLUE, RED, GREEN]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    b = ax.bar(x, rate, 0.6, color=cols, edgecolor=SURFACE, linewidth=1)
    ax.axhline(0.03, color=AXIS, lw=1, ls='--'); ax.annotate('chance 0.03', (2.4, 0.03), xytext=(0, 5), textcoords='offset points', ha='right', color=MUTED, fontsize=8.5)
    for bb in b: ax.annotate(f'{bb.get_height():.2f}', (bb.get_x() + bb.get_width() / 2, bb.get_height()), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=11, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel('HSPC-driver rate in top cocktails')
    ax.set_title('Fig 5 · Value-add ladder:  MPC ≈ distill ≫ from-scratch PPO\n(from-scratch RL fails on the sparse pessimistic reward)', fontsize=11.5, loc='left')
    ax.set_ylim(0, 0.55); finish(ax)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig5_value_add_ladder.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 6: which validated biology each method recovers (presence matrix) ============
def fig6():
    tfs = ['KLF2', 'TAL1', 'ZFP90', 'LMO2', 'RPA3', 'RNF138',        # P1
           'ZNF85', 'MIER1', 'DDIT3',                                 # P2
           'ETV2', 'RUNX1', 'KLF1', 'LYL1', 'IKZF1', 'GATA2', 'HOXA9', 'MEIS1']  # drivers
    methods = ['M0 fate-MPC', 'M2 set-PPO', 'M3 distill', 'de-novo']
    rec = {
        'M0 fate-MPC': {'KLF2', 'TAL1', 'ZFP90', 'ZNF85', 'ETV2', 'IKZF1', 'KLF1', 'RUNX1'},
        'M2 set-PPO': {'KLF2', 'RNF138', 'RPA3', 'TAL1', 'DDIT3', 'MIER1', 'ETV2', 'GATA2', 'HOXA9', 'LYL1', 'MEIS1'},
        'M3 distill': {'KLF2', 'LMO2', 'TAL1', 'MIER1', 'ETV2', 'IKZF1', 'KLF1', 'LYL1', 'RUNX1'},
        'de-novo': {'KLF2', 'TAL1', 'ZFP90', 'LMO2', 'RPA3', 'ZNF85', 'MIER1', 'ETV2', 'HOXA9', 'IKZF1', 'KLF1', 'LMO2', 'LYL1', 'RUNX1'},
    }
    M = np.array([[1 if t in rec[m] else 0 for m in methods] for t in tfs])
    fig, ax = plt.subplots(figsize=(6.2, 8))
    for i in range(len(tfs)):
        for j in range(len(methods)):
            c = (GREEN if i < 6 else (MAG if i < 9 else BLUE)) if M[i, j] else '#eeeeea'
            ax.add_patch(plt.Rectangle((j, len(tfs) - 1 - i), 0.92, 0.92, facecolor=c, edgecolor=SURFACE, lw=1.5))
    ax.set_xlim(-0.1, len(methods)); ax.set_ylim(-0.1, len(tfs))
    ax.set_xticks(np.arange(len(methods)) + 0.46); ax.set_xticklabels(methods, fontsize=9.5, rotation=20, ha='right')
    ax.set_yticks(np.arange(len(tfs)) + 0.46); ax.set_yticklabels(tfs[::-1], fontsize=9)
    ax.set_title('Fig 6 · Validated HSPC biology recovered by each method', fontsize=11.5, loc='left', pad=42)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.grid(False)
    leg = [Patch(facecolor=GREEN, label='P1 member (Briggs arrayed #1)'),
           Patch(facecolor=MAG, label='P2 member (arrayed #2)'),
           Patch(facecolor=BLUE, label='canonical HSPC driver')]
    ax.legend(handles=leg, frameon=False, fontsize=8.2, loc='lower left', bbox_to_anchor=(0, 1.005), ncol=3, columnspacing=1.2, handlelength=1.2)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig6_recovery_matrix.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 7: clean-control background lifts BOTH WM fidelity and RL recovery ============
def fig7():
    bg = ['screen-bulk\n(original)', 'joung_ctrl\n(pure Joung)', 'joung+briggs\n(pooled)']
    fid = [0.564, 0.789, 0.790]      # readout Ridge Spearman at k>=7 (Briggs-native axis)
    rec = [0.33, 0.50, 0.38]         # RL fate-MPC driver rate
    x = np.arange(len(bg)); cols = [MUTED, GREEN, BLUE]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, vals, ttl, ylab, ylim in [(axA, fid, 'Fig 7A · WM fidelity (readout, k≥7)', 'Spearman(pred, obs fate)', 0.9),
                                       (axB, rec, 'Fig 7B · RL recovery (fate-MPC)', 'HSPC-driver rate', 0.6)]:
        b = ax.bar(x, vals, 0.62, color=cols, edgecolor=SURFACE, linewidth=1)
        for bb in b: ax.annotate(f'{bb.get_height():.2f}', (bb.get_x() + bb.get_width() / 2, bb.get_height()), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=10, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(bg, fontsize=9); ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=11.5, loc='left'); ax.set_ylim(0, ylim); finish(ax)
    axB.axhline(0.03, color=AXIS, lw=1, ls='--'); axB.annotate('chance 0.03', (2.4, 0.03), xytext=(0, 4), textcoords='offset points', ha='right', color=MUTED, fontsize=8.5)
    fig.suptitle('A clean hESC / Joung-control origin lifts BOTH the world model and the planner', fontsize=12.5, x=0.01, ha='left', y=1.02)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig7_clean_background.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 8: Joung-RL in-distribution ladder ============
def fig8():
    meth = ['MPC\n(beam)', 'MPC-greedy', 'MPC-distill', 'set-PPO', 'seq-PPO', '1-driver\noracle', 'random']
    r1 = [0.50, 0.50, 0.50, 0.167, 0.167, 0.667, 0.162]
    resid = [0.860, 0.860, 0.860, 0.098, -0.042, 0.723, -0.140]
    x = np.arange(len(meth))
    cols = [BLUE, BLUE, GREEN, ORANGE, RED, MUTED, MUTED]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    bA = axA.bar(x, r1, 0.66, color=cols, edgecolor=SURFACE, linewidth=1)
    axA.axhline(0.167, color=AXIS, lw=1, ls='--'); axA.annotate('chance 0.17', (len(x) - 1, 0.167), xytext=(0, 4), textcoords='offset points', ha='right', color=MUTED, fontsize=8.5)
    for b in bA: axA.annotate(f'{b.get_height():.2f}', (b.get_x() + b.get_width() / 2, b.get_height()), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=9, color=INK2)
    axA.set_xticks(x); axA.set_xticklabels(meth, fontsize=8.5); axA.set_ylabel('nearest-lineage recall@1'); axA.set_ylim(0, 0.75)
    axA.set_title('Fig 8A · Reaching the right fate (recall@1)', fontsize=11.5, loc='left'); finish(axA)
    bB = axB.bar(x, resid, 0.66, color=cols, edgecolor=SURFACE, linewidth=1); axB.axhline(0, color=AXIS, lw=0.8)
    for b in bB: axB.annotate(f'{b.get_height():+.2f}', (b.get_x() + b.get_width() / 2, b.get_height()), textcoords='offset points', xytext=(0, 3 if b.get_height() >= 0 else -12), ha='center', fontsize=9, color=INK2)
    axB.set_xticks(x); axB.set_xticklabels(meth, fontsize=8.5); axB.set_ylabel('resid_match to lineage centroid'); axB.set_ylim(-0.25, 1.0)
    axB.set_title('Fig 8B · Fate-alignment (resid_match)', fontsize=11.5, loc='left'); finish(axB)
    fig.suptitle('Joung-RL ladder (in-distribution, 6 fetal lineages): MPC ≈ distill; set-PPO partial; seq-PPO fails; oracle leads', fontsize=12, x=0.01, ha='left', y=1.02)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig8_joung_rl_ladder.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 9: set-PPO resolution (scratch fails -> distill -> BC-init+PPO) ============
def fig9():
    meth = ['from-scratch\nPPO\n(7-config sweep)', 'distill\n(BC of planner)', 'BC-init\n+ PPO fine-tune']
    rate = [0.03, 0.21, 0.43]; div = [None, 46, 9]; cols = [RED, BLUE, GREEN]
    x = np.arange(len(meth))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    b = ax.bar(x, rate, 0.6, color=cols, edgecolor=SURFACE, linewidth=1)
    ax.axhline(0.03, color=AXIS, lw=1, ls='--'); ax.annotate('chance 0.03', (2.4, 0.03), xytext=(0, 4), textcoords='offset points', ha='right', color=MUTED, fontsize=8.5)
    for i, bb in enumerate(b):
        ax.annotate(f'{bb.get_height():.2f}', (bb.get_x() + bb.get_width() / 2, bb.get_height()), textcoords='offset points', xytext=(0, 4), ha='center', fontsize=11, color=INK)
        if div[i]: ax.annotate(f'{div[i]} distinct\ncocktails', (bb.get_x() + bb.get_width() / 2, bb.get_height() / 2), ha='center', fontsize=8.5, color=SURFACE, weight='bold')
    ax.set_xticks(x); ax.set_xticklabels(meth, fontsize=9.5); ax.set_ylabel('HSPC-driver rate (sampled, best-of-both WM)')
    ax.set_title('Fig 9 · Fixing set-PPO: reward-tuning can\'t (scratch fails all configs);\nwarm-start-then-fine-tune both unblocks PPO and beats distill', fontsize=11, loc='left')
    ax.set_ylim(0, 0.5); finish(ax)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig9_setppo_resolution.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 10: finer per-cell-type Joung ladder (16 targets) + BC-init+PPO ============
def fig10():
    meth = ['MPC\n(beam)', 'MPC-greedy', 'MPC-distill', 'BC-init\n+PPO', 'set-PPO', 'seq-PPO', '1-driver\noracle']
    r1 = [0.375, 0.438, 0.312, 0.312, 0.062, 0.062, 0.438]
    resid = [0.804, 0.794, 0.708, 0.721, 0.165, 0.133, 0.519]
    cols = [BLUE, BLUE, GREEN, AQUA, ORANGE, RED, MUTED]
    x = np.arange(len(meth))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    bA = axA.bar(x, r1, 0.66, color=cols, edgecolor=SURFACE, linewidth=1)
    axA.axhline(0.062, color=AXIS, lw=1, ls='--'); axA.annotate('chance 0.06', (len(x) - 1, 0.062), xytext=(0, 4), textcoords='offset points', ha='right', color=MUTED, fontsize=8.5)
    for b in bA: axA.annotate(f'{b.get_height():.2f}', (b.get_x() + b.get_width() / 2, b.get_height()), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=9, color=INK2)
    axA.set_xticks(x); axA.set_xticklabels(meth, fontsize=8.5); axA.set_ylabel('nearest-cell-type recall@1'); axA.set_ylim(0, 0.5)
    axA.set_title('Fig 10A · Reaching the right cell type (recall@1)', fontsize=11, loc='left'); finish(axA)
    bB = axB.bar(x, resid, 0.66, color=cols, edgecolor=SURFACE, linewidth=1)
    for b in bB: axB.annotate(f'{b.get_height():.2f}', (b.get_x() + b.get_width() / 2, b.get_height()), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=9, color=INK2)
    axB.set_xticks(x); axB.set_xticklabels(meth, fontsize=8.5); axB.set_ylabel('resid_match (fate alignment)'); axB.set_ylim(0, 0.9)
    axB.set_title('Fig 10B · Fate alignment — planners beat the oracle (0.80 vs 0.52)', fontsize=11, loc='left'); finish(axB)
    fig.suptitle('Finer Joung ladder (16 cell types): MPC leads; BC-init+PPO ≈ distill (dense reward); planners beat oracle on alignment, not top-1', fontsize=11.5, x=0.01, ha='left', y=1.02)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig10_joung_finer_ladder.png', dpi=150, bbox_inches='tight'); plt.close(fig)


# ============ FIG 11: diversity vs precision Pareto (entropy lever), prod WM ============
def fig11():
    ent = [0.02, 0.05, 0.10, 0.20]; distinct = [45, 659, 1999, 2000]; rate = [0.17, 0.00, 0.05, 0.06]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(distinct, rate, s=140, c=[BLUE, AQUA, ORANGE, RED], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    offs = {0.02: (6, -16), 0.05: (8, 6), 0.10: (8, 8), 0.20: (8, 6)}
    for e, dx, ry in zip(ent, distinct, rate):
        ax.annotate(f'ent={e} (BC+PPO)', (dx, ry), textcoords='offset points', xytext=offs[e], fontsize=8.5, color=INK2)
    ax.scatter([50], [0.18], s=170, marker='D', c=GREEN, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.annotate('distill (no PPO)', (50, 0.18), textcoords='offset points', xytext=(12, 8), fontsize=9, color=GREEN)
    ax.axhline(0.03, color=AXIS, lw=1, ls='--'); ax.annotate('chance 0.03', (2000, 0.03), xytext=(0, 4), textcoords='offset points', ha='right', color=MUTED, fontsize=8.5)
    ax.set_xscale('log'); ax.set_xlabel('distinct cocktails (diversity, log)'); ax.set_ylabel('HSPC-driver rate (precision)')
    ax.set_title('Fig 11 · Diversity↔precision Pareto (prod WM): entropy buys diversity,\ncosts precision; distill is the balanced operating point', fontsize=11, loc='left')
    ax.set_ylim(-0.01, 0.22); finish(ax)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig11_diversity_pareto.png', dpi=150, bbox_inches='tight'); plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11):
    f(); print('made', f.__name__)
print('figures in', OUT)

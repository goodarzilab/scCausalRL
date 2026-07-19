"""Consolidated comparison across the exp17 + exp19 Joung world-model ablations.

SCOPE: raw-trained models only (orig, esm, esm_con, mmd/mmdmse families, and the exp19
ugate/epsmix nonlinear-composition models). The residual-TRAINED variants (esm_resid*,
sephead*) are omitted here because the drift-removed metric double-subtracts for them
(non-comparable); their numbers are reported in text. l128 runs diverged (NaN) and are out.

Panel A — SINGLES specificity, ALL metrics on the DRIFT-REMOVED (TF-specific) residual Δ
  (raw Δ discrimination is ~0.50 for everyone because the shared differentiation drift
  dominates, hence we score the residual). Three metrics per model:
    discrimination_cos_resid : retrieval specificity (chance 0.50)
    deg_auroc_resid          : recover the TF's DEGs from |Δ_pred| (chance 0.50)
    effect_spearman_resid    : rank-corr of predicted vs observed Δ over genes (chance 0)
Panel B — COMBO prediction on the 40 held-out pairs (RAW combo Δ): the model do-operator
  vs MODEL-additive (Σ of the shift-encoder's PREDICTED single-TF Δ — no epistasis). Dotted
  line = the OBSERVED-singles additive oracle (Σ true single Δ; data-only, same for all),
  from the quartile eval — the upper bound a deployable model can't see.
"""
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

E17 = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/results'
E19 = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose/results'
HUE17, HUE19 = '#3C5488', '#E64B35'
# raw-trained family only (residual-trained + l128 excluded for metric-basis consistency)
KEEP = {'v2_orig','v2_esm','v2_esm_con','v2_esm_mmd','v2_esm_mmd_con','v2_esm_mmdmse',
        'v2_esm_mmdmse_con','v2_esm_con_ugate','v2_esm_con_epsmix','v2_esm_con_ugate_epsmix'}


def load(sub):
    out = {}
    for base, exp in [(E17, 'exp17'), (E19, 'exp19')]:
        for d in glob.glob(f'{base}/{sub}/*_ep60'):
            run = os.path.basename(d).replace('_ep60', '')
            s = os.path.join(d, 'summary.json')
            if run in KEEP and os.path.exists(s):
                out[run] = (exp, json.load(open(s)))
    return out


def oracle_pearson():
    """Overall observed-singles additive-oracle pearson (n-weighted over quartiles)."""
    j = json.load(open(f'{E19}/epistasis_quartile/epistasis_q/summary.json'))
    m = next(iter(j['models'].values()))
    num = den = 0.0
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        e = m[q]; num += e['addobs_pearson'] * e['n']; den += e['n']
    return num / den


held = load('heldout_eval'); combo = load('combo_eval')
order = sorted(held, key=lambda r: -held[r][1].get('discrimination_cos_resid', 0))

fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
# ---- Panel A: drift-removed metrics (DEG-AUROC dropped — saturated ~0.86 for all) ----
mets = [('discrimination_cos_resid', 'discrimination', '#3C5488'),
        ('effect_spearman_resid', 'effect-Spearman', '#00A087')]
x = np.arange(len(order)); w = 0.38
for i, (k, lab, c) in enumerate(mets):
    vals = [held[r][1].get(k, np.nan) for r in order]
    ax[0].bar(x + (i-0.5)*w, vals, w, color=c, edgecolor='k', linewidth=0.4, label=lab)
ax[0].axhline(0.5, ls='--', c='gray', lw=1, label='discrimination chance 0.50')
ax[0].axhline(0.0, c='k', lw=0.6)
ax[0].axhline(0.645, ls=':', c='green', lw=1, label='disc ridge ceiling 0.645')
ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=40, ha='right', fontsize=7.5)
for t, r in zip(ax[0].get_xticklabels(), order):
    t.set_color(HUE19 if held[r][0] == 'exp19' else 'black')
ax[0].set_ylabel('held-out metric (drift-removed residual Δ)')
ax[0].set_title('(A) Singles specificity — DRIFT-REMOVED residual Δ', fontsize=11)
ax[0].legend(fontsize=7.5, frameon=False, ncol=2); ax[0].grid(axis='y', alpha=0.25)
ax[0].spines[['top', 'right']].set_visible(False)

# ---- Panel B: do-op vs MODEL-additive (+ observed oracle line) ----
corder = sorted([r for r in combo if combo[r][1].get('pairs', {}).get('do_pearson_delta') is not None],
                key=lambda r: -combo[r][1]['pairs']['do_pearson_delta'])
xb = np.arange(len(corder)); w = 0.4
do = [combo[r][1]['pairs']['do_pearson_delta'] for r in corder]
ad = [combo[r][1]['pairs'].get('add_pearson_delta', np.nan) for r in corder]
ax[1].bar(xb - w/2, do, w, color=HUE17, edgecolor='k', linewidth=0.5, label='do-operator')
ax[1].bar(xb + w/2, ad, w, color='#cccccc', edgecolor='k', linewidth=0.5, hatch='//',
          label='MODEL-additive (Σ predicted singles)')
orc = oracle_pearson()
ax[1].axhline(orc, ls=':', c='darkred', lw=1.5, label=f'OBSERVED-singles oracle ({orc:.2f})')
ax[1].axhline(0, c='k', lw=0.6)
ax[1].set_xticks(xb); ax[1].set_xticklabels(corder, rotation=40, ha='right', fontsize=7.5)
for t, r in zip(ax[1].get_xticklabels(), corder):   # exp19 runs in red, exp17 in black
    t.set_color(HUE19 if combo[r][0] == 'exp19' else 'black')
ax[1].set_ylabel('combo pearson_delta — RAW combo Δ (40 held-out pairs)')
ax[1].set_title('(B) Combo: do-op vs MODEL-additive (raw Δ)', fontsize=11)
ax[1].legend(fontsize=7.5, frameon=False); ax[1].grid(axis='y', alpha=0.25)
ax[1].spines[['top', 'right']].set_visible(False)

fig.suptitle('Joung WM ablations (exp17 blue, exp19/epistasis red) — held-out test TFs, ep60\n'
             'A: drift-removed residual (TF-specific). B: raw combo Δ; additive = predicted singles, not observed.',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f'{E17}/v2_comparison.png'; fig.savefig(out, dpi=200, bbox_inches='tight'); print('saved', out)
print('oracle pearson =', round(orc, 3))
for r in order:
    j = held[r][1]
    print(f"  {r:26s} disc_resid={j.get('discrimination_cos_resid'):.3f} deg_auroc_resid={j.get('deg_auroc_resid'):.3f} eff_spear_resid={j.get('effect_spearman_resid'):.3f}")

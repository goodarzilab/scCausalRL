"""Old metric (disc_resid, drift-removed cosine discrimination) vs the NEW magnitude-crediting
metrics (Pnk = raw KO-excluded Pearson; deg_auroc = DEG recovery; eff_spear = effect-size ranking),
across ALL models. Shows the metrics DECOUPLE: disc_resid is low+variable (0.50-0.63) while the
magnitude metrics are high+stable (~0.9) — the models recover the biology well; disc_resid is the
harsh TF-specific discrimination axis. Reads results/figures/full_eval_table.csv.

Run: python evaluation/make_metrics_figure.py
"""
import os, csv, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FAM = {'learned':'#2a78d6','masked':'#1baf7a','frozen':'#eda100','localized':'#4a3aa7','loc+masked':'#c0392b'}
MET = {'test_disc':('#c0392b','disc_resid (OLD)'), 'test_Pnk':('#1baf7a','Pnk (raw KO-excl Pearson)'),
       'test_deg_auroc':('#2a78d6','DEG-recovery AUROC'), 'test_eff_spear':('#eda100','effect Spearman')}
rows = list(csv.DictReader(open('results/figures/full_eval_table.csv')))
g = lambda r,k: float(r.get(k,'nan') or 'nan')
rows = [r for r in rows if not np.isnan(g(r,'test_disc'))]
rows.sort(key=lambda r: g(r,'test_disc'))
x = np.arange(len(rows)); labs = [r['tag'].replace('v2b_','') for r in rows]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw={'width_ratios':[2.1,1]})
# Panel A: per-arm, all 4 metrics as dot-series sorted by disc_resid
for k,(c,lab) in MET.items():
    y = np.array([g(r,k) for r in rows]); ax.plot(y, x, 'o-', color=c, ms=5, lw=1.3, label=lab, alpha=0.9)
ax.axvline(0.5, ls='--', lw=1, color='#888'); ax.text(0.5, len(rows), ' chance', color='#888', fontsize=8, va='bottom')
ax.axvline(0.602, ls='--', lw=1, color='#d03b3b'); ax.text(0.602, len(rows), ' ESM-NN', color='#d03b3b', fontsize=8, va='bottom')
ax.set_yticks(x); ax.set_yticklabels(labs, fontsize=6.5); ax.set_xlim(0.35,1.0)
ax.set_xlabel('metric value', fontsize=11); ax.legend(loc='lower right', fontsize=9)
ax.set_title('OLD disc_resid vs NEW magnitude metrics — across all arms (sorted by disc_resid)', fontsize=12)
# Panel B: disc_resid (x) vs DEG-AUROC (y) per arm — decoupling; deg_auroc high+flat regardless of disc
for r in rows:
    c = FAM.get(r['family'],'#888')
    ax2.errorbar(g(r,'test_disc'), g(r,'test_deg_auroc'), xerr=g(r,'test_disc_sem'), yerr=g(r,'test_deg_auroc_sem'),
                 fmt='o', ms=7, mfc=c, mec='white', mew=0.6, ecolor=c, elinewidth=0.8, alpha=0.9)
ax2.axvline(0.5, ls='--', lw=1, color='#888'); ax2.axvline(0.602, ls='--', lw=1, color='#d03b3b')
ax2.set_xlabel('disc_resid (OLD, TF-specific discrimination)', fontsize=11)
ax2.set_ylabel('DEG-recovery AUROC (NEW)', fontsize=11)
ax2.set_title('metrics decouple: models recover DEGs (~0.9)\nregardless of disc_resid', fontsize=11)
ax2.grid(alpha=0.15)
ax2.legend(handles=[Line2D([0],[0],marker='o',color='w',mfc=c,ms=8,label=f) for f,c in FAM.items()],
           title='decoder', loc='lower left', fontsize=8)
fig.tight_layout(); out='results/figures/metrics_old_vs_new.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); print('wrote', out)

"""Appendix figure — GRN-database comparison (moved OUT of main Fig 4).
CAVEAT (stated in the caption + Limitations): the world model is trained prior-off on ALL TFs (no non-TF
genes), so comparing its projected graph / do-operator to a curated TF->target database (CollecTRI/DoRothEA)
is warped -- these databases are off-predictors in our training regime. Shown for completeness/honesty only.
Data:
  A  grn_auroc_stats.json             (projected-G vs co-expression AUROC + 95% CI)
  B  bio_target_recovery_results.json (do-operator direct-target recovery; A vs shuffled vs A=0 vs coexpr)
Run:  cd paper/figures && python fig_app_grn.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import paperstyle as ps

CV = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/docs/figures/causal_val"
AU_JSON = os.path.join(CV, "grn_auroc_stats.json")
BIO_JSON = os.path.join(CV, "bio_target_recovery_results.json")

# Verbatim mirrors of the two result files, kept in-file so the appendix figure still
# builds where the results tree is not mounted.  Values are the released ones.
_AU_CACHE = {
    "CollecTRI":      {"G_projected":  {"auroc": 0.5688, "ci95": [0.5608, 0.5769]},
                       "coexpression": {"auroc": 0.6009, "ci95": [0.5928, 0.6089]}},
    "DoRothEA (A/B)": {"G_projected":  {"auroc": 0.6223, "ci95": [0.6104, 0.6343]},
                       "coexpression": {"auroc": 0.7126, "ci95": [0.7006, 0.7245]}},
    "STRING PPI":     {"G_projected":  {"auroc": 0.6230, "ci95": [0.5868, 0.6593]},
                       "coexpression": {"auroc": 0.6808, "ci95": [0.6445, 0.7170]}},
}
_BIO_CACHE = {
    "CollecTRI":      {"A original": 0.5442, "A shuffled": 0.6236,
                       "A = 0 (no propagation)": 0.6278, "co-expression": 0.5721},
    "DoRothEA (A/B)": {"A original": 0.6240, "A shuffled": 0.6430,
                       "A = 0 (no propagation)": 0.6517, "co-expression": 0.6369},
    "_stats": {"CollecTRI": {"A_original_vs_A0": {"bh_q": 1e-12}}},
}
au = json.load(open(AU_JSON)) if os.path.exists(AU_JSON) else _AU_CACHE
bio = json.load(open(BIO_JSON)) if os.path.exists(BIO_JSON) else _BIO_CACHE

fig, (axA, axB) = ps.grid(1, 2, h=2.55, wspace=0.16)


def _legend_below(ax, ncol):
    """Legend parked *under* the axes.

    Both panels carry a two-line title, so the strip above the axes is spoken for; the
    band below the categorical tick labels is the only place a legend can live without
    touching either the title or the tops of the bars.  Constrained layout measures the
    legend as part of the axes, so the canvas grows to hold it.
    """
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol,
                     frameon=False, fontsize=ps.FS["legend"], columnspacing=1.2)

# ---------- A: projected-graph AUROC vs co-expression ----------
refs = ["CollecTRI", "DoRothEA (A/B)", "STRING PPI"]; lbl = ["CollecTRI", "DoRothEA", "STRING"]
xc = np.arange(len(refs)); bw = 0.25      # leaves a clean channel for the "chance" label
for j, (k, col, nm) in enumerate([("G_projected", ps.ACCENT["graph"], "projected $G$"),
                                   ("coexpression", ps.ACCENT["coexpr"], "co-expression")]):
    a = [au[r][k]["auroc"] for r in refs]
    lo = [au[r][k]["auroc"] - au[r][k]["ci95"][0] for r in refs]
    hi = [au[r][k]["ci95"][1] - au[r][k]["auroc"] for r in refs]
    axA.bar(xc + (j - 0.5) * bw, a, bw, yerr=[lo, hi], color=col, label=nm,
            edgecolor="white", linewidth=0.4, zorder=3,
            error_kw=dict(lw=0.7, capsize=1.6, ecolor=ps.INK, zorder=4))
axA.axhline(0.5, color=ps.INK, ls="--", lw=0.6, zorder=2)
# the chance label goes in the empty channel BETWEEN two bar groups, just above the line,
# so it touches neither a bar nor the axis frame
axA.text(0.5, 0.503, "chance", fontsize=ps.FS["small"], color=ps.INK_SOFT,
         va="bottom", ha="center")
axA.set_xticks(xc); axA.set_xticklabels(lbl); axA.set_ylabel("edge AUROC")
axA.set_ylim(0.48, 0.77)
ps.tidy(axA)
_legend_below(axA, ncol=2)
axA.set_title("Projected-graph edge AUROC\n($G{=}D A^{T} D^{T}$ vs co-expression)")

# ---------- B: do-operator direct-target recovery ----------
dbs = ["CollecTRI", "DoRothEA (A/B)"]; dlbl = ["CollecTRI", "DoRothEA"]
preds = [("A original", ps.ACCENT["graph"]), ("A shuffled", ps.VERMILION_L),
         ("A = 0 (no propagation)", ps.GREY), ("co-expression", ps.ACCENT["coexpr"])]
xd = np.arange(len(dbs)); bw = 0.19
for i, (p, col) in enumerate(preds):
    vals = [bio[d][p] for d in dbs]
    axB.bar(xd + (i - 1.5) * bw, vals, bw, color=col,
            label=p.replace(" (no propagation)", ""), edgecolor="white", linewidth=0.4,
            zorder=3)
axB.axhline(0.5, color=ps.INK, ls="--", lw=0.6, zorder=2)
axB.set_xticks(xd); axB.set_xticklabels(dlbl); axB.set_ylabel("direct-target AUROC")
axB.set_ylim(0.48, 0.73)
ps.tidy(axB)
_legend_below(axB, ncol=2)
# headline stat, parked in the clear band above every bar (it used to sit on the chance
# line, where it was both illegible and clipped by the left spine)
q = bio["_stats"]["CollecTRI"]["A_original_vs_A0"]["bh_q"]
axB.text(0.5, 0.695, f"$A$ orig $<$ $A{{=}}0$  {ps.stars(q)}", ha="center", va="top",
         fontsize=ps.FS["small"], color=ps.ACCENT["lo"])
axB.set_title("Do-operator direct-target recovery\n(decoded $D S_g$, NOT the graph)")

# panel letters, forced onto a common baseline (the two legends differ in height)
letters = ps.label_axes(fig, [axA, axB])
ytop = min(t.get_position()[1] for t in letters)
for t in letters:
    t.set_position((t.get_position()[0], ytop))

ps.save(fig, "fig7")
print("wrote fig_app_grn.pdf / .png")

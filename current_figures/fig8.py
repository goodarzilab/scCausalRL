"""Fig 9 (APPENDIX) — permutation-null support for the directed-A causal claim.

The two panels moved out of the main Fig 5 (edge-direction ablation):
  A  permutation-null histogram: the 1000 shuffled-A mean Δ-Pearsons form a tight spike
     near -0.16; the observed original-A mean (+0.54, 95% CI shaded) sits far in the right
     tail -> 0/1000 shuffles exceed it, Δ+0.69, p<0.001.
  B  per-held-out-TF paired scatter (n=66): original vs one representative shuffled-A draw;
     59/66 TFs sit below y=x (original > shuffled) -> shuffling A hurts prediction per TF.

Both panels are the verbatim code/data from the released Fig 5 C/D:
  results/briggs/edge_direction_null.json  (opt-in dump from evaluation/causal_wm_validation.py
  --dump_null_json; seed-0 recompute reproduces orig +0.536, null -0.156+/-0.010, Δ+0.69,
  Wilcoxon p=8.9e-11).  edge_direction_results.json carries the aggregate scalars.
Run:  cd paper/figures && python fig9.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import paperstyle as ps

CV = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/docs/figures/causal_val"
RES = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs"
ED_JSON = os.path.join(CV, "edge_direction_results.json")
NULL_JSON = os.path.join(RES, "edge_direction_null.json")           # full null + per-TF vectors

# Verbatim mirror of the edge_direction_results.json scalars the panels need, kept in-file so
# the figure still builds when the results tree is not mounted.  Live data always supersedes it.
_ED_CACHE = {
    "single": {
        "original": {"pearson": 0.5356, "n": 66},
        "shuffled": {"pearson": -0.1563, "n": 66},
    },
    "_stats": {
        "single": {"delta_orig_minus_shuffled": 0.69,
                   "shuffled_pearson_mean": -0.1563,
                   "orig_pearson_ci95": [0.4433, 0.6252], "n_shuffle": 1000},
    },
}
ed = json.load(open(ED_JSON)) if os.path.exists(ED_JSON) else _ED_CACHE
null = json.load(open(NULL_JSON)) if os.path.exists(NULL_JSON) else None

CO = ps.EDGE["original"]      # blue
CS = ps.EDGE["shuffled"]      # grey

fig, (axA, axB) = ps.grid(1, 2, h=2.6, wspace=0.24)
# thin side margins so the edge x-tick labels do not spill off the canvas
fig.get_layout_engine().set(rect=(0.02, 0, 0.96, 1))


# ============================================================ A: permutation-null histogram ===
if null is not None:
    nm = np.asarray(null["null_means"], float)
    n_shuf = int(null.get("n_shuffle", ed["_stats"]["single"].get("n_shuffle", len(nm))))
    n_exceed = int((nm >= ed["single"]["original"]["pearson"]).sum())
else:
    nm = None
    n_shuf = ed["_stats"]["single"].get("n_shuffle", 1000)
    n_exceed = 0
orig = ed["single"]["original"]["pearson"]
null_mean = ed["_stats"]["single"].get("shuffled_pearson_mean", ed["single"]["shuffled"]["pearson"])
ci_lo, ci_hi = ed["_stats"]["single"].get("orig_pearson_ci95", [np.nan, np.nan])
delta = ed["_stats"]["single"]["delta_orig_minus_shuffled"]

if nm is not None:
    xlo, xhi = nm.min() - 0.05, orig + 0.14
    bins = np.linspace(nm.min() - 0.006, nm.max() + 0.006, 26)
    axA.axvspan(ci_lo, ci_hi, color=ps.tint(CO, 0.80), zorder=0)       # observed 95% CI band
    axA.hist(nm, bins=bins, color=CS, edgecolor="white", linewidth=0.25, zorder=2)
    ymax0 = axA.get_ylim()[1]
    axA.set_ylim(0, ymax0 * 1.32)
    ymax = axA.get_ylim()[1]
    axA.axvline(orig, color=CO, lw=1.6, zorder=4)
    xmid = (orig + null_mean) / 2
    # gap arrow: null spike -> observed, labelled with the delta; the exceed-stat a line below
    yA = ymax * 0.46
    axA.annotate("", xy=(orig, yA), xytext=(null_mean, yA),
                 arrowprops=dict(arrowstyle="<->", color=ps.INK_SOFT, lw=0.7,
                                 shrinkA=0, shrinkB=0), zorder=3)
    axA.text(xmid, yA + ymax * 0.02, f"$\\Delta$+{delta:.2f}", ha="center", va="bottom",
             fontsize=ps.FS["annot"], color=ps.INK)
    axA.text(xmid, yA - ymax * 0.06, f"{n_exceed}/{n_shuf} exceed\n{ps.stars((1 + n_exceed) / (n_shuf + 1))}",
             ha="center", va="top", fontsize=ps.FS["small"], color=ps.INK_SOFT,
             linespacing=1.25)
    axA.text(null_mean, ymax * 0.985, "shuffled-$A$\nnull", ha="center", va="top",
             fontsize=ps.FS["small"], color=ps.INK_SOFT, linespacing=1.15)
    axA.text(orig - 0.012, ymax * 0.985, "original $A$\n$+%.2f$" % orig, ha="right", va="top",
             fontsize=ps.FS["annot"], color=CO, linespacing=1.15, fontweight="bold")
    axA.set_xlim(xlo, xhi)
    axA.set_xlabel("mean $\\Delta$-Pearson")
    axA.set_ylabel("shuffles")
    axA.set_yticks([t for t in axA.get_yticks() if 0 <= t <= ymax0])
    axA.set_title("Observed vs shuffled-$A$ null")
else:
    axA.text(0.5, 0.5, "data pending:\nedge_direction_null.json", ha="center", va="center",
             transform=axA.transAxes, fontsize=ps.FS["small"], color=ps.INK_SOFT)
ps.tidy(axA)


# ============================================================ B: per-TF paired scatter ===
if null is not None:
    po = np.array([np.nan if v is None else v for v in null["pearson_per_original"]], float)
    psh = np.array([np.nan if v is None else v for v in null["pearson_per_shuffled_repr"]], float)
    m = ~(np.isnan(po) | np.isnan(psh))
    po, psh = po[m], psh[m]
    n_tf = po.size
    n_below = int((po > psh).sum())         # below y=x  <=>  original > shuffled
    lim_lo = min(po.min(), psh.min()) - 0.06
    lim_hi = max(po.max(), psh.max()) + 0.06
    axB.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], lim_lo,
                     color=ps.tint(CO, 0.90), zorder=0)              # original-beats-shuffled region
    axB.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ":", color=ps.RULE, lw=0.9, zorder=1)
    axB.axhline(0, color=ps.tint(ps.RULE, 0.5), lw=0.5, zorder=1)
    axB.axvline(0, color=ps.tint(ps.RULE, 0.5), lw=0.5, zorder=1)
    axB.scatter(po, psh, s=13, color=CO, edgecolor="white", linewidth=0.3,
                alpha=0.85, zorder=3)
    axB.set_xlim(lim_lo, lim_hi); axB.set_ylim(lim_lo, lim_hi)
    axB.set_xticks([-0.5, 0.0, 0.5, 1.0]); axB.set_yticks([-0.5, 0.0, 0.5, 1.0])
    axB.set_xlabel("original")
    axB.set_ylabel("shuffled $A$")
    axB.text(0.04, 0.97, f"{n_below}/{n_tf} TFs below $y=x$\n(original $>$ shuffled)",
             transform=axB.transAxes, ha="left", va="top",
             fontsize=ps.FS["small"], color=CO, linespacing=1.3)
    axB.set_title("Shuffling $A$ hurts per TF")
else:
    axB.text(0.5, 0.5, "data pending:\nedge_direction_null.json", ha="center", va="center",
             transform=axB.transAxes, fontsize=ps.FS["small"], color=ps.INK_SOFT)
ps.tidy(axB)

ps.label_axes(fig, [axA, axB])
ps.save(fig, "fig8")
print("wrote fig9.pdf / .png")

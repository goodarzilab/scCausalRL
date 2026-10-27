"""Fig 5 (MAIN) — the directed causal graph is load-bearing FOR PREDICTION.
Solid, database-free causal validation on the deployed joint WM (jb_prod_wm ep60):

  A  single-TF edge-direction ablation   (original / reversed / symmetrized / shuffled A)
  B  combination edge-direction ablation  (same 4 variants)
       Source: edge_direction_results.json (1000-shuffle-A null; drift-removed Pearson_DEG + disc, same metrics as Fig 2; recall@1 in appendix).
  C  Top-10 causal-hub GO:BP programs of the latent DAG A (native redraw of the old raster's
       right half), bars = total |A| in+out connectivity, colour = # member genes.
       Source: panels/fig4B_latentA_data.npz (real A from ep60; GO-mask coherence 1.00).

The permutation-null histogram and per-held-out-TF paired scatter that supported A/B moved to
the appendix (Fig 9).  Run:  cd paper/figures && python make_fig4B_latentA.py && python fig5.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import paperstyle as ps

CV = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/docs/figures/causal_val"
ED_JSON = os.path.join(CV, "edge_direction_results.json")
NPZ = "panels/fig4B_latentA_data.npz"                               # C: real A / GO hubs

# Verbatim mirror of edge_direction_results.json, kept in-file so the figure still builds
# on a machine where the results tree is not mounted.  Values are the released ones; the
# recall@1 entries are the exact k/n fractions (n = 66 single TFs / 14 held-out combos).
_ED_CACHE = {
    "single": {
        "original":    {"pearson": 0.5356, "recall1": 48 / 66, "disc_auroc": 0.8444, "n": 66},
        "reversed":    {"pearson": -0.1305, "recall1": 1 / 66, "disc_auroc": 0.5918, "n": 66},
        "symmetrized": {"pearson": 0.1066, "recall1": 5 / 66, "disc_auroc": 0.6858, "n": 66},
        "shuffled":    {"pearson": -0.1563, "recall1": 1 / 66, "disc_auroc": 0.6004, "n": 66},
    },
    "combo": {
        "original":    {"pearson": 0.9768, "recall1": 6 / 14, "disc_auroc": 0.6264, "n": 14},
        "reversed":    {"pearson": 0.7967, "recall1": 1 / 14, "disc_auroc": 0.5495, "n": 14},
        "symmetrized": {"pearson": 0.9178, "recall1": 1 / 14, "disc_auroc": 0.5276, "n": 14},
        "shuffled":    {"pearson": 0.8751, "recall1": 0.8 / 14, "disc_auroc": 0.5382, "n": 14},
    },
    "_stats": {
        "single": {"delta_orig_minus_shuffled": 0.69,
                   "shuffled_pearson_mean": -0.1563,
                   "orig_pearson_ci95": [0.4433, 0.6252], "n_shuffle": 1000},
        "combo": {"delta_orig_minus_shuffled": 0.10},
    },
}
ed = json.load(open(ED_JSON)) if os.path.exists(ED_JSON) else _ED_CACHE

variants = ["original", "reversed", "symmetrized", "shuffled"]
METRICS = [("pearson", "$\\mathrm{Pearson}_{\\mathrm{DEG}}$"), ("disc_auroc", "disc.")]  # same drift-removed metrics as Fig 2
BW = 0.19


# ================================================================= A / B ===
def ablation(ax, grp, title, sig, bracket_pad):
    """Grouped bars for one ablation + a significance bracket on the Pearson group."""
    x = np.arange(len(METRICS))
    for i, v in enumerate(variants):
        vals = [ed[grp][v][m] for m, _ in METRICS]
        ax.bar(x + (i - 1.5) * BW, vals, BW, color=ps.EDGE[v], label=v,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=ps.INK, lw=0.6, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in METRICS])
    ax.set_ylabel("metric value")
    ax.set_ylim(-0.25, 1.30)
    ps.tidy(ax)
    # per-metric permutation stars vs the 1000-shuffle-A null (Pearson_DEG group also shows Delta)
    st = ed["_stats"][grp]; dp = st.get("delta_orig_minus_shuffled", 0.0)
    pkey = {"pearson": "permutation_p", "disc_auroc": "permutation_p_disc_auroc"}
    for xi, (m, _l) in enumerate(METRICS):
        pv = st.get(pkey.get(m)); star = ps.stars(pv) if pv is not None else sig
        if not star:
            continue
        top = max(ed[grp][v][m] for v in variants)
        lab = f"$\\Delta$+{dp:.2f}  {star}" if m == "pearson" else star
        ps.sig_bar(ax, xi - 1.5 * BW, xi + 1.5 * BW, top + bracket_pad, lab, dh=0.03)
    ax.text(0.995, 1.0, f"$n$={ed[grp]['original']['n']}", transform=ax.transAxes,
            ha="right", va="top", fontsize=ps.FS["small"], color=ps.INK_SOFT)
    ax.set_title(title)


# --------------------------------------------------------------------------- canvas: 2 rows
# row 1 = [A, B]  row 2 = C (full text width).  Explicit per-row gridspecs (not a single
# ps.grid) because C's 10 wrapped GO:BP hub labels need a WIDE left margin, which under one
# shared constrained-layout grid would steal horizontal space from column 0 and squeeze A.
# Native vector, one type ladder, exact TEXTWIDTH_IN width.
H = 3.6
fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, H))
gs1 = fig.add_gridspec(1, 2, left=0.088, right=0.985, top=0.885, bottom=0.660, wspace=0.26)
gs3 = fig.add_gridspec(1, 1, left=0.315, right=0.895, top=0.560, bottom=0.090)
axA, axB = fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1])
axC = fig.add_subplot(gs3[0, 0])
cax = fig.add_axes([0.912, 0.120, 0.015, 0.330])          # C colorbar (# member genes)

ablation(axA, "single", "Single-TF ablation", ps.stars(1 / 1001), 0.10)
ablation(axB, "combo", "Combination ablation", ps.stars(1 / 1001), 0.05)

# one shared legend for both ablations, parked above row 1 so it cannot touch a bar
handles, labels = axA.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.992),
           ncol=4, frameon=False, fontsize=ps.FS["legend"], columnspacing=1.3,
           handlelength=1.15, borderaxespad=0.0)


# ============================================================ C: Top-10 causal-hub GO:BP programs ===
# Native redraw (was the right half of the dropped raster).  Hub logic MIRRORS
# make_fig4B_latentA.py: top-10 programs by total |A| in+out connectivity, colour = #member genes.
if os.path.exists(NPZ):
    z = np.load(NPZ, allow_pickle=True)
    deg = z["deg"]; n_member = z["n_member"]; terms = z["terms"]; coherence = float(z["coherence"])
    top = np.argsort(-deg)[:10][::-1]                    # ascending -> largest hub at the top
    # NEVER truncate: 3 of the top-10 share the 'Negative Regulation Of ...' prefix, so wrap
    # wide enough that each keeps its discriminating tail (>=2 lines only when unavoidable).
    labels = [ps.wrap_label(str(terms[k]).split(" (GO:")[0], width=38, maxlines=2) for k in top]
    norm = mpl.colors.Normalize(vmin=float(n_member[top].min()),
                                vmax=float(n_member[top].max() + 1e-9))
    cmap = mpl.colormaps[ps.CMAP_SEQ]
    cols = [cmap(0.40 + 0.55 * norm(n_member[k])) for k in top]
    y = np.arange(len(top))
    axC.barh(y, deg[top], color=cols, edgecolor="white", linewidth=0.4, zorder=3, height=0.70)
    axC.set_yticks(y); axC.set_yticklabels(labels, fontsize=ps.FS["small"])
    axC.set_xlabel("total |$A$| connectivity (in+out edge weight)", labelpad=1)
    axC.set_title("Top-10 causal-hub GO:BP programs")
    axC.tick_params(axis="y", length=0)
    axC.xaxis.grid(True, color=ps.tint(ps.RULE, 0.5), lw=0.5, zorder=0)
    axC.set_xlim(0, float(deg[top].max()) * 1.02)
    axC.set_ylim(-0.62, len(top) - 0.38)
    ps.tidy(axC)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("# member genes", fontsize=ps.FS["small"], labelpad=2)
    cb.ax.tick_params(labelsize=ps.FS["small"], length=1.8)
    cb.outline.set_linewidth(0.5)
    axC.text(0.985, 0.05, f"GO-mask coherence {coherence:.2f}", transform=axC.transAxes,
             ha="right", va="bottom", fontsize=ps.FS["small"], color=ps.INK_SOFT)
else:
    axC.text(0.5, 0.5, "data pending: run make_fig4B_latentA.py", ha="center", va="center",
             transform=axC.transAxes, fontsize=ps.FS["small"], color=ps.INK_SOFT)
    ps.tidy(axC); cax.axis("off")

ps.label_axes(fig, [axA, axB, axC])
ps.save(fig, "fig4")
print("wrote fig5.pdf / .png")

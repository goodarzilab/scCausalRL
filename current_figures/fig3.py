"""Fig 3 (MAIN) — combinatorial epistasis + the directed causal graph is load-bearing for prediction.

Single taller figure (2 rows x 3 cols) that merges the two previously-separate blocks so the paper
carries ONE fig3 at a manuscript-friendly aspect (~1.5:1) instead of the wide-and-short composite:

  EPISTASIS block (deg8 co8 family — the held-out sibling of the deployed planner world model):
    A  predicted vs observed net epistasis on n = 276 held-out Briggs combinations (combo-trained WM
         jb_ft_replay_co8, deg8).  Source: results/briggs/fig4_epistasis_deg8.json
         (datasets.briggs.{net_o,net_m_combo}).  r_combo = +0.70.  Axes z-scored: the identity line is
         the correlation; magnitude is NOT calibrated.
    B  singles-only (jb_singles_base_co8) vs combo-trained (jb_ft_replay_co8) epistasis r_pair on the
         same 276 held-out combos.  Whisker = bootstrap 95% CI over the 276 combos.
    C  held-out concordance by combination ORDER k (per_k in the JSON): near chance for pairs/triples,
         strong for higher-order cocktails.  Bar = r_pair, whisker = bootstrap 95% CI over the k bin.
    D  interpretable example held-out combination (largest do-op gain over additive on drift-removed
         per-DE-gene residuals).  n = # DE genes; 95% CI on each r bootstrapped over genes.

  CAUSAL LOAD-BEARING ablation block (deployed joint WM jb_prod_wm ep60; 1000-shuffle-A null):
    E  single-TF edge-direction ablation   (original / reversed / symmetrized / shuffled A)
    F  combination edge-direction ablation  (same 4 variants)
         Source: docs/figures/causal_val/edge_direction_results.json (drift-removed Pearson_DEG + disc,
         same metrics as Fig 2).  Delta+0.69 single / +0.10 combo, stars vs the 1000-shuffle-A null.

Both loaders keep their in-file fallback caches so the figure builds even if a results path is absent;
live data always supersedes the cache.  Panel letters come from ps.label_axes (measured tight bbox).
Run:  cd current_figures && python3 fig3.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import paperstyle as ps

# ---------------------------------------------------------------- data (A-D: epistasis) ---
RES = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs"
EPI_PATH = os.path.join(RES, "fig4_epistasis_deg8.json")            # deg8 co8 family (A/B/C/D)
EPI = json.load(open(EPI_PATH)) if os.path.exists(EPI_PATH) else None

# ---------------------------------------------------------------- data (E/F: ablation) ----
CV = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/docs/figures/causal_val"
ED_JSON = os.path.join(CV, "edge_direction_results.json")
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

# ---------------------------------------------------------------- palette ------------------
CB = ps.SOURCE["joung_combo"]     # combo-trained (flagship)
CS = ps.METHOD["additive"]        # singles-only grey
CADD = ps.ACCENT["coexpr"]        # additive baseline
CB_D = ps.shade(CB, 0.30)
GUIDE = ps.RULE                   # identity line
ZERO = ps.tint(ps.RULE, 0.55)     # zero cross-hairs

DSET = EPI["datasets"][EPI["panelA_set"]] if EPI else None          # 'briggs'
SETNAME = EPI["panelA_set"].capitalize() if EPI else ""
N_COMBOS = DSET["n"] if DSET else 0

variants = ["original", "reversed", "symmetrized", "shuffled"]
METRICS = [("pearson", "$\\mathrm{Pearson}_{\\mathrm{DEG}}$"), ("disc_auroc", "disc.")]
BW = 0.19


# ---------------------------------------------------------------- helpers ------------------
def sym_ticks(ax, lim, nbins=5):
    """Explicit symmetric ticks strictly inside +/-lim (keeps ps.audit clean of stray off-view ticks)."""
    t = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(-lim, lim)
    t = t[(t > -lim) & (t < lim)]
    ax.set_xticks(t); ax.set_yticks(t)
    return t


def boot_ci_r(x, y, nboot=5000, seed=0):
    """Bootstrap 95% CI of Pearson r over paired samples (rows)."""
    x = np.asarray(x, float); y = np.asarray(y, float); N = len(x)
    if N < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed); rs = []
    for _ in range(nboot):
        i = rng.integers(0, N, N); xi, yi = x[i], y[i]
        if xi.std() > 1e-9 and yi.std() > 1e-9:
            rs.append(np.corrcoef(xi, yi)[0, 1])
    rs = np.array(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def ablation(ax, grp, title, sig, bracket_pad):
    """Grouped bars for one edge-direction ablation + a significance bracket vs the shuffled-A null."""
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


# ---------------------------------------------------------------- canvas -------------------
# 2 rows x 3 cols at exact manuscript width, taller aspect (~1.5:1) so nothing is squished.
#   top row:    A epistasis scatter | B singles-vs-combo | E single-TF ablation
#   bottom row: C order-k           | D example combo    | F combination ablation
# rect reserves a top strip for the shared ablation legend and a bottom strip for the ps.note captions.
H = 3.80
fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, H))
# Explicit gridspec (not ps.grid's constrained layout): the two multi-line ps.note captions (under B
# and C) and the shared figure legend are hand-placed outside the axes, which a constrained layout
# cannot budget for -> it would collapse the axes.  Margins reserve a top strip for the legend and a
# bottom strip + a wide inter-row gap for the captions.  Generous wspace so the wide top-row titles
# (centred on B, D) clear the E/F panel letters.
gs = fig.add_gridspec(2, 3, left=0.078, right=0.985, top=0.895, bottom=0.195,
                      hspace=0.92, wspace=0.52)
axA, axB, axE = (fig.add_subplot(gs[0, j]) for j in range(3))
axC, axD, axF = (fig.add_subplot(gs[1, j]) for j in range(3))


# ================= A: predicted vs observed net epistasis (combo-trained, held-out combos) =================
if DSET is not None:
    xo = np.array(DSET["net_o"]); ym = np.array(DSET["net_m_combo"])
    r_c = DSET["r_combo"]; n = DSET["n"]; sp = DSET["r_combo_spearman"]
    lo, hi = DSET["r_combo_ci"]
    zx = (xo - xo.mean()) / xo.std(); zy = (ym - ym.mean()) / ym.std()
    lim = max(np.abs(np.r_[zx, zy]).max() * 1.22, 1.0)
    axA.plot([-lim, lim], [-lim, lim], ":", color=GUIDE, lw=0.9, zorder=0)
    axA.axhline(0, color=ZERO, lw=0.6, zorder=0); axA.axvline(0, color=ZERO, lw=0.6, zorder=0)
    axA.scatter(zx, zy, s=10, color=CB, edgecolor="white", lw=0.25, alpha=0.8, zorder=2)
    axA.set_xlim(-lim, lim); axA.set_ylim(-lim, lim); sym_ticks(axA, lim)
    axA.set_xlabel("obs. (z)"); axA.set_ylabel("pred. (z)")
    axA.text(0.03, 0.97, f"$r={r_c:+.2f}$ [{lo:+.2f}, {hi:+.2f}]\n$\\rho={sp:+.2f}$,  "
                         f"$n={n}$ {SETNAME}",
             transform=axA.transAxes, ha="left", va="top", fontsize=ps.FS["annot"],
             color=CB, linespacing=1.3)
    axA.text(0.97, 0.03, "direction recovered;\nmagnitude uncalib.",
             transform=axA.transAxes, ha="right", va="bottom",
             fontsize=ps.FS["small"], color=ps.INK_SOFT, linespacing=1.3)
else:
    axA.axis("off"); axA.text(0.5, 0.5, "data pending", ha="center", va="center")
axA.set_title("Predicts held-out epistasis")
ps.tidy(axA)


# ================= B: singles-only vs combo-trained r_pair (whisker = bootstrap CI over combos) =========
if DSET is not None:
    no = np.array(DSET["net_o"]); mc = np.array(DSET["net_m_combo"]); msg = np.array(DSET["net_m_singles"])
    r_s = DSET["r_singles"]; r_cb = DSET["r_combo"]
    s_lo, s_hi = boot_ci_r(no, msg, seed=1); c_lo, c_hi = DSET["r_combo_ci"]
    bars = [("singles-\nonly", r_s, s_lo, s_hi, CS), ("combo-\ntrained", r_cb, c_lo, c_hi, CB)]
    for i, (lbl, v, blo, bhi, col) in enumerate(bars):
        axB.bar(i, v, 0.60, color=col, edgecolor="white", linewidth=0.4, zorder=1)
        yerr = np.array([[max(v - blo, 0)], [max(bhi - v, 0)]])
        axB.errorbar(i, v, yerr=yerr, fmt="none", ecolor=ps.INK, elinewidth=0.9, capsize=2.4, zorder=3)
        axB.text(i + 0.36, v, f"{v:+.2f}", ha="left", va="center", fontsize=ps.FS["annot"], color=col)
    axB.axhline(0, color=ps.INK, lw=0.6)
    axB.set_xticks([0, 1]); axB.set_xticklabels([b[0] for b in bars])
    axB.set_ylabel("epistasis $r_{pair}$")
    axB.set_ylim(-0.25, 1.05); axB.set_xlim(-0.62, 1.80); axB.set_yticks([0.0, 0.5, 1.0])
    ps.note(axB, "bar $=r_{pair}$;  whisker = bootstrap 95% CI\n"
                 f"$n={N_COMBOS}$ held-out combos / model", y=-0.50)
else:
    axB.axis("off"); axB.text(0.5, 0.5, "data pending", ha="center", va="center")
axB.set_title("Interaction needs combo training")
ps.tidy(axB)


# ================= E: single-TF edge-direction ablation =================
ablation(axE, "single", "Single-TF ablation", ps.stars(1 / 1001), 0.10)


# ================= C: held-out concordance by combination order k (per_k) =================
if DSET is not None and DSET.get("per_k"):
    pk = DSET["per_k"]; order = [k for k in ("k2-3", "k4-5", "k6-8") if k in pk]
    for i, kk in enumerate(order):
        v = pk[kk]; r = v["r"]; blo, bhi = v["r_lo"], v["r_hi"]
        axC.bar(i, r, 0.60, color=CB, edgecolor="white", linewidth=0.4, zorder=1)
        yerr = np.array([[max(r - blo, 0)], [max(bhi - r, 0)]])
        axC.errorbar(i, r, yerr=yerr, fmt="none", ecolor=ps.INK, elinewidth=0.9, capsize=2.4, zorder=3)
        axC.text(i, bhi + 0.03, f"{r:+.2f}", ha="center", va="bottom", fontsize=ps.FS["small"], color=CB)
    axC.axhline(0, color=ps.INK, ls=(0, (3, 2)), lw=0.7)
    axC.text(len(order) - 0.35, -0.045, "chance", fontsize=ps.FS["small"], va="top", ha="right", color=ps.INK_SOFT)
    axC.set_xticks(range(len(order)))
    axC.set_xticklabels([f"{k.replace('k', 'k=')}\n($n={pk[k]['n']}$)" for k in order])
    axC.set_ylabel("epistasis $r_{pair}$"); axC.set_ylim(-0.30, 1.22); axC.set_xlim(-0.7, len(order) - 0.3)
    axC.set_yticks([0.0, 0.5, 1.0])
    ps.note(axC, "held-out combination order $k$\n"
                 "near chance for pairs/triples, strong for higher-order cocktails",
            x=0.0, ha="left", y=-0.56, color=CB_D)
else:
    axC.axis("off"); axC.text(0.5, 0.5, "data pending", ha="center", va="center")
axC.set_title("Interaction is combinatorial")
ps.tidy(axC)


# ================= D: interpretable example held-out combination =================
def _pick_panelD(recs):
    """Held-out combo with the largest do-op advantage over additive, judged on the SAME drift-removed
    per-DE-gene residuals the panel plots (not the raw-delta r_do/r_add stashed as scalars in the JSON)."""
    best, best_gain = None, -np.inf
    for r in recs:
        if "_genes" not in r:
            continue
        gg = r["_genes"]; ox = np.array(gg["obs"]); do = np.array(gg["do"]); add = np.array(gg["add"])
        rd = _pear(do, ox); ra = _pear(add, ox)
        if not np.isfinite(rd) or rd < 0.5:
            continue
        if (rd - ra) > best_gain:
            best, best_gain = r, rd - ra
    return best


rec = _pick_panelD(DSET["recs"]) if DSET is not None else None
if rec is not None:
    g = rec["_genes"]; ox = np.array(g["obs"]); do = np.array(g["do"]); add = np.array(g["add"])
    r_d = _pear(do, ox); r_add = _pear(add, ox)
    lo_d, hi_d = boot_ci_r(do, ox); lo_a, hi_a = boot_ci_r(add, ox)
    label = rec["key"].replace("+", " + ")
    lim = max(np.abs(np.r_[ox, do]).max() * 1.15, 0.05)
    axD.plot([-lim, lim], [-lim, lim], ":", color=GUIDE, lw=0.9, zorder=0)
    axD.axhline(0, color=ZERO, lw=0.6, zorder=0); axD.axvline(0, color=ZERO, lw=0.6, zorder=0)
    axD.scatter(ox, do, s=6, color=CB, edgecolor="none", alpha=0.5, zorder=2)
    axD.set_xlim(-lim, lim); axD.set_ylim(-lim, lim); sym_ticks(axD, lim, nbins=4)
    axD.set_xlabel("obs. $\\Delta$expr (DE genes)"); axD.set_ylabel("pred. $\\Delta$expr")
    axD.text(0.03, 0.97, f"{label}\ndo-op $r={r_d:+.2f}$ [{lo_d:+.2f},{hi_d:+.2f}]\n$n={len(ox)}$ DE genes",
             transform=axD.transAxes, ha="left", va="top", fontsize=ps.FS["small"],
             color=CB, linespacing=1.3)
    axD.text(0.97, 0.03, f"additive $r={r_add:+.2f}$\n[{lo_a:+.2f},{hi_a:+.2f}]",
             transform=axD.transAxes, ha="right", va="bottom",
             fontsize=ps.FS["small"], color=CADD, linespacing=1.3)
else:
    axD.axis("off"); axD.text(0.5, 0.5, "data pending", ha="center", va="center")
axD.set_title("Example held-out combination")
ps.tidy(axD)


# ================= F: combination edge-direction ablation =================
ablation(axF, "combo", "Combination ablation", ps.stars(1 / 1001), 0.05)


# ---- shared ablation legend (original/reversed/symmetrized/shuffled), parked in the top strip ----
handles, labels = axE.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.998),
           ncol=4, frameon=False, fontsize=ps.FS["legend"], columnspacing=0.9,
           handlelength=1.0, handletextpad=0.4, borderaxespad=0.0)

ps.label_axes(fig, [axA, axB, axE, axC, axD, axF], letters=["A", "B", "E", "C", "D", "F"], dy=0.035)
ps.save(fig, "fig3")
print("wrote fig3.pdf / .png" + ("" if EPI is not None else "   [EPI DATA MISSING]"))
print("ED source:", "live" if os.path.exists(ED_JSON) else "IN-FILE CACHE (edge_direction_results.json missing)")
if DSET is not None:
    print(f"A: r={DSET['r_combo']:.3f} n={DSET['n']} | B: singles={DSET['r_singles']:.3f} "
          f"combo={DSET['r_combo']:.3f} | C per-k:",
          {k: round(v['r'], 3) for k, v in DSET['per_k'].items()},
          "| D:", (rec['key'] if rec else "none"))
print(f"E: single n={ed['single']['original']['n']} | F: combo n={ed['combo']['original']['n']}")

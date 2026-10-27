"""Fig 3 — Held-out perturbation prediction beats a fair additive baseline (panel figure, vector PDF).

All numbers read LIVE from the consolidated WM-eval JSONs (drift-removed, same-context additive).
ERROR BARS = SEM over the 3 held-out-split WM seeds (co8 / co8_s2 / co8_s3), ddof=1. Sample sizes (n =
number of held-out perturbations) are annotated on every panel. Metric = drift-removed disc / Pearson_DEG.

Checkpoints:
  - jb_prod_wm       = deployable prod WM (train_frac=1.0); its only held-out multi-TF set = the 222
                       Briggs answer-key combos (single deployable checkpoint -> no seed band on that bar).
  - jb_ft_replay_co8 (+ s2/s3) = the held-out-SPLIT model, 3 seeds -> all held-out claims + error bars.

Layout: 2 x 2 constrained-layout grid at exactly ps.TEXTWIDTH_IN.  Every legend is parked
ABOVE its axes (under the panel title) and every provenance caption sits BELOW its axes via
ps.note, so no text is ever drawn on a bar, a line or another label.  Panel letters come from
ps.label_axes, which measures each axes' tight bbox post-layout.
Run:  cd paper/figures && python fig2_prediction.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import paperstyle as ps

CANON = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/canon"
LIVE = os.path.isdir(CANON)
J = lambda f: json.load(open(os.path.join(CANON, f)))["groups"]
HO_SEEDS = ["consolidated_jb_ft_replay_co8.json",
            "consolidated_jb_ft_replay_co8_s2.json",
            "consolidated_jb_ft_replay_co8_s3.json"]
PROD = J("consolidated_jb_prod_wm.json") if LIVE else None
HO = J("consolidated_jb_ft_replay_co8.json") if LIVE else None
_GCACHE = {f: J(f) for f in HO_SEEDS} if LIVE else None
SRC = {"prod": PROD, "ho": HO}

# Verbatim mirror of the quantities the four panels draw out of
# consolidated_jb_prod_wm.json and consolidated_jb_ft_replay_co8{,_s2,_s3}.json, recovered
# from the released fig2_prediction.pdf (bar tops / marker centres / error-bar caps read off
# the vector content stream and calibrated against the y-tick geometry, ~1e-5 in data units;
# printed n / Delta / p / % strings taken verbatim from the PDF text layer).  Kept in-file so
# the figure still builds when the results tree is absent -- LIVE data always supersedes it.
_CACHE = {
    # 3-seed (mean, SEM) per group x method x metric.  Cross-check: the panel-D gaps below
    # equal WM - additive of the k-sweep entries to 1e-5.
    "seed_stat": {
        "joung_singles_train":   {"WM+head": {"disc": (0.986268, 0.001651),
                                              "allDEG": (0.876252, 0.003746)}},
        "joung_singles_heldout": {"WM+head": {"disc": (0.796898, 0.026776),
                                              "allDEG": (0.413710, 0.035364)}},
        "briggs_singles":        {"WM+head": {"disc": (0.712830, 0.003880),
                                              "allDEG": (0.196102, 0.002963)}},
        "briggs_combos_held-out_k2-3": {"WM+head": {"disc": (0.775383, 0.004779),
                                                    "allDEG": (0.383294, 0.008616)},
                                        "obs-add": {"disc": (0.610125, 0.012968),
                                                    "allDEG": (0.230542, 0.009688)}},
        "briggs_combos_held-out_k4-5": {"WM+head": {"disc": (0.834811, 0.004090),
                                                    "allDEG": (0.503161, 0.015053)},
                                        "obs-add": {"disc": (0.537565, 0.003087),
                                                    "allDEG": (0.241918, 0.008680)}},
        "briggs_combos_held-out_k6-8": {"WM+head": {"disc": (0.825153, 0.009192),
                                                    "allDEG": (0.375433, 0.005385)},
                                        "obs-add": {"disc": (0.523004, 0.009355),
                                                    "allDEG": (0.137218, 0.006571)}},
        # panel B draws the single-checkpoint bar height (see "value") and only the seed SEM,
        # so the 3-seed mean of this group is not on the page and is not mirrored.
        "briggs_combos_held-out": {"WM+head": {"allDEG": (None, 0.007748)},
                                   "obs-add": {"allDEG": (None, 0.003648)}},
    },
    # 3-seed (mean, SEM) of the per-seed difference WM+head - obs-add (panel D bars; the
    # briggs_combos_held-out row is panel B's caption, printed there as +0.226 +/- 0.005).
    "seed_gap": {
        "briggs_combos_held-out_k2-3": {"disc": (0.165253, 0.017743),
                                        "allDEG": (0.152746, 0.009133)},
        "briggs_combos_held-out_k4-5": {"disc": (0.297243, 0.006573),
                                        "allDEG": (0.261234, 0.006409)},
        "briggs_combos_held-out_k6-8": {"disc": (0.302146, 0.015370),
                                        "allDEG": (0.238216, 0.003191)},
        "briggs_combos_held-out": {"allDEG": (0.226, 0.005)},
    },
    # single-checkpoint drift-removed allDEG (panel B bar heights)
    "value": {
        "prod": {"briggs_combos_answerkey": {"WM+head": 0.391390, "obs-add": 0.207821}},
        "ho":   {"briggs_combos_held-out":  {"WM+head": 0.442328, "obs-add": 0.217391}},
    },
    # paired Wilcoxon summaries, exactly as printed on the released figure
    "paired": {
        "prod": {"briggs_combos_answerkey": {"delta": 0.184, "n": 222, "p": 3e-21}},
        "ho": {"briggs_combos_held-out":      {"delta": 0.225, "n": 304, "p": 8e-49},
               "briggs_combos_held-out_k2-3": {"win": 0.88},
               "briggs_combos_held-out_k4-5": {"win": 0.99},
               "briggs_combos_held-out_k6-8": {"win": 0.95, "p": 3e-15}},
    },
    # n = finite held-out perturbations per group (printed under every x tick)
    "n": {"joung_singles_train": 426, "joung_singles_heldout": 63, "briggs_singles": 57,
          "briggs_combos_held-out_k2-3": 80, "briggs_combos_held-out_k4-5": 138,
          "briggs_combos_held-out_k6-8": 86},
}


def seeds(gkey, method, metric):
    """per-seed scalar across the 3 held-out-split seeds."""
    vs = []
    for f in HO_SEEDS:
        g = _GCACHE[f]
        try: vs.append(float(g[gkey][method][metric]))
        except (KeyError, TypeError): pass
    return np.array(vs, float)


def msem(vs):
    n = len(vs)
    return (vs.mean() if n else np.nan, vs.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0, n)


def nitems(gkey, metric="allDEG", src=HO, method="WM+head"):
    a = np.asarray(src[gkey][method].get(metric + "_items", []), float)
    return int(np.isfinite(a).sum())


def paired(g, metric, m1="WM+head", m2="obs-add"):
    a = np.asarray(g[m1][metric + "_items"], float); b = np.asarray(g[m2][metric + "_items"], float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    p = wilcoxon(a, b).pvalue if len(a) else np.nan
    return (a - b).mean(), int(len(a)), p, float((a > b).mean())


# ---- thin accessors: live results tree first, cached PDF geometry only as a fallback ----
def stat(gkey, method, metric):
    """3-seed (mean, SEM)."""
    if LIVE:
        m, s, _ = msem(seeds(gkey, method, metric)); return m, s
    return _CACHE["seed_stat"][gkey][method][metric]


def gap(gkey, metric):
    """3-seed (mean, SEM) of the per-seed advantage WM+head - obs-add."""
    if LIVE:
        m, s, _ = msem(seeds(gkey, "WM+head", metric) - seeds(gkey, "obs-add", metric))
        return m, s
    return _CACHE["seed_gap"][gkey][metric]


def n_of(gkey):
    return nitems(gkey) if LIVE else _CACHE["n"][gkey]


def value(src, gkey, metric="allDEG"):
    """(WM+head, obs-add) for a single deployable checkpoint."""
    if LIVE:
        g = SRC[src][gkey]; return g["WM+head"][metric], g["obs-add"][metric]
    v = _CACHE["value"][src][gkey]; return v["WM+head"], v["obs-add"]


def pair_stat(src, gkey, metric="allDEG"):
    """dict(delta=, n=, p=, win=) for the paired WM-vs-additive Wilcoxon test."""
    if LIVE:
        d, n, p, w = paired(SRC[src][gkey], metric)
        return dict(delta=d, n=n, p=p, win=w)
    return _CACHE["paired"][src][gkey]


C_WM, C_ADD = ps.METHOD["policy"], ps.METHOD["additive"]
C_DISC, C_DEG = ps.METHOD["oracle"], ps.ACCENT["wm"]
EK = dict(lw=0.7, capsize=1.8, ecolor=ps.INK)
TPAD1 = 17.0        # title pad that clears a ONE-row legend parked above the axes
TPAD2 = 29.0        # ... and a TWO-row legend (panel C has four series)

fig, axes = ps.grid(2, 2, h=3.55, hspace=0.22, wspace=0.10)
axA, axB, axC, axD = axes.ravel()

# ---------------- A: held-out single-TF prediction — WM vs retrieval baselines ----------------
# WM+head vs two model-free retrieval copies (ESM-kNN, co-expr-kNN), ALL scored on the SAME held-out
# single-TF sets with ONE identical drift-removed metric (evaluation/panelA_baselines.py). This is a k=1
# singles readout on that exact set, so WM+head here (Joung-HO allDEG 0.497) is intentionally NOT the
# combo analyses of panels B–D. Error bars = percentile bootstrap 95% CI over the held-out TFs (_items).
PA = json.load(open(os.path.join(os.path.dirname(CANON), "panelA_baselines_co8.json")))["groups"]
C_WMh, C_ESM, C_CX = ps.METHOD["policy"], ps.GREY, ps.GREY_L      # WM (blue) vs retrieval (grey / light grey)
methA = [("WM+head", C_WMh, "WM + head"), ("esm_knn", C_ESM, "ESM-kNN"), ("coexp_knn", C_CX, "co-expr-kNN")]
grpA = [("joung_singles_heldout", "Joung\nheld-out"), ("briggs_singles", "Briggs\ncross-screen")]
nA = [PA[k]["n"] for k, _ in grpA]


def _bootci(items, seed, nb=5000):
    """value + (lower_err, upper_err): 95% percentile bootstrap over the finite held-out TFs."""
    a = np.asarray(items, float); a = a[np.isfinite(a)]
    v = float(a.mean()) if len(a) else np.nan
    if len(a) < 2:
        return v, 0.0, 0.0
    bs = a[np.random.default_rng(seed).integers(0, len(a), size=(nb, len(a)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return v, max(v - lo, 0.0), max(hi - v, 0.0)


cx = {("allDEG", 0): 0.0, ("allDEG", 1): 1.15, ("disc", 0): 2.75, ("disc", 1): 3.90}
bw = 0.30; _seed = 0; printA = []
for mi, (mkey, col, mlab) in enumerate(methA):
    for (metric, gi), xc in cx.items():
        gkey = grpA[gi][0]; v = PA[gkey][mkey][metric]
        bv, elo, ehi = _bootci(PA[gkey][mkey]["_items"][metric], _seed); _seed += 1
        axA.bar(xc + (mi - 1) * bw, v, bw, yerr=[[elo], [ehi]], color=col, edgecolor="white",
                linewidth=0.4, error_kw=EK, label=mlab if (metric, gi) == ("allDEG", 0) else None)
        printA.append((metric, grpA[gi][1].replace("\n", " "), mkey, v, bv, elo, ehi))
# paired Wilcoxon star above the WM+head bar: WM+head vs the STRONGEST retrieval baseline,
# over the held-out TFs (same items the bootstrap CI uses). ns if WM does not beat it.
for (_m, _gi), _xc in cx.items():
    _gk = grpA[_gi][0]
    _wm = np.asarray(PA[_gk]["WM+head"]["_items"][_m], float)
    _cands = [(np.nanmean(np.asarray(PA[_gk][mk]["_items"][_m], float)),
               np.asarray(PA[_gk][mk]["_items"][_m], float))
              for mk, *_ in methA if mk != "WM+head"]
    _bl = max(_cands, key=lambda t: t[0])[1]
    if _wm.shape != _bl.shape:
        continue
    _fm = np.isfinite(_wm) & np.isfinite(_bl)
    if _fm.sum() < 2:
        continue
    try:
        _pv = wilcoxon(_wm[_fm], _bl[_fm]).pvalue
    except ValueError:
        continue
    _eh = _bootci(_wm, 999)[2]
    axA.text(_xc - bw, PA[_gk]["WM+head"][_m] + min(_eh, 0.05) + 0.02, ps.stars(_pv),
             ha="center", va="bottom", fontsize=ps.FS["annot"], color=ps.INK)
axA.axhline(0, color=ps.INK, ls=(0, (3, 2)), lw=0.7, zorder=1)                 # allDEG chance = 0
axA.plot([cx[("disc", 0)] - 0.55, cx[("disc", 1)] + 0.55], [0.5, 0.5],         # disc chance = 0.5
         color=ps.INK_SOFT, ls=(0, (1, 1.4)), lw=0.9, zorder=1)
axA.text(cx[("disc", 1)] + 0.58, 0.5, "0.5", ha="left", va="center",
         fontsize=ps.FS["small"], color=ps.INK_SOFT)
axA.axvline(1.95, color=ps.RULE, lw=0.6, zorder=0)                             # allDEG | disc divider
xt = [cx[("allDEG", 0)], cx[("allDEG", 1)], cx[("disc", 0)], cx[("disc", 1)]]
axA.set_xticks(xt)
axA.set_xticklabels([f"{grpA[g][1]}\n$n$={nA[g]}" for g in (0, 1, 0, 1)], fontsize=ps.FS["small"])
tr = axA.get_xaxis_transform()
axA.text(0.575, -0.52, "$\\mathrm{Pearson}_{\\mathrm{DEG}}$", transform=tr, ha="center", va="top",
         fontsize=ps.FS["annot"], fontweight="bold", color=ps.INK)
axA.text(3.325, -0.52, "disc.", transform=tr, ha="center", va="top",
         fontsize=ps.FS["annot"], fontweight="bold", color=ps.INK)
_cd, _ca = stat("joung_singles_train", "WM+head", "disc")[0], stat("joung_singles_train", "WM+head", "allDEG")[0]
axA.text(0.02, 0.995, "Joung-train ceiling\ndisc %.2f / $\\mathrm{Pearson}_{\\mathrm{DEG}}$ %.2f" % (_cd, _ca),
         transform=axA.transAxes, ha="left", va="top", fontsize=ps.FS["small"],
         color=ps.INK_SOFT, linespacing=1.25)
axA.set_ylabel("drift-removed metric")
axA.set_xlim(-0.6, 4.9); axA.set_ylim(-0.08, 1.12); axA.set_yticks(np.arange(0, 1.01, 0.2))
ps.legend_above(axA, ncol=3)
axA.set_title("Held-out single-TF: WM vs retrieval", pad=TPAD1)
ps.tidy(axA)
print("[A] metric | set | method | plotted(v) | boot-mean | -CI/+CI  (v must match JSON)")
for _m, _s, _k, _v, _bv, _lo, _hi in printA:
    print("    %-6s %-13s %-10s v=%+.3f boot=%+.3f  -%.3f/+%.3f" % (_m, _s, _k, _v, _bv, _lo, _hi))

# ---------------- B: held-out combos beat additive ----------------
pA_ = pair_stat("prod", "briggs_combos_answerkey")
pH_ = pair_stat("ho", "briggs_combos_held-out")
d3m, d3s = gap("briggs_combos_held-out", "allDEG")
# per-set bars: WM vs additive; WM held-out gets 3-seed SEM, answer-key single-ckpt
setsB = [("answer-key\n$n$=%d" % pA_["n"], value("prod", "briggs_combos_answerkey"),
          pA_["delta"], pA_["p"], 0.0, 0.0),
         ("held-out\n$n$=%d" % pH_["n"], value("ho", "briggs_combos_held-out"),
          pH_["delta"], pH_["p"],
          stat("briggs_combos_held-out", "WM+head", "allDEG")[1],
          stat("briggs_combos_held-out", "obs-add", "allDEG")[1])]
x = np.arange(len(setsB)); bw = 0.34
capB = 0.0
for xi, (lbl, (wm, add), dlt, pv, wsem, asem) in enumerate(setsB):
    axB.bar(xi - bw / 2, wm, bw, yerr=wsem, color=C_WM, edgecolor="white", linewidth=0.4,
            error_kw=EK, label="WM + head" if xi == 0 else None)
    axB.bar(xi + bw / 2, add, bw, yerr=asem, color=C_ADD, edgecolor="white", linewidth=0.4,
            error_kw=EK, label="additive (same-ctxt)" if xi == 0 else None)
    ytop = max(wm, add) + wsem
    ps.sig_bar(axB, xi - bw / 2, xi + bw / 2, ytop + 0.05, ps.stars(pv), dh=0.02)
    axB.text(xi, ytop + 0.225, f"$\\Delta$+{dlt:.3f}\n$p$={pv:.0e}", ha="center", va="bottom",
             fontsize=ps.FS["small"], color=ps.INK, linespacing=1.25)
    capB = max(capB, ytop + 0.225)
axB.set_xticks(x); axB.set_xticklabels([s[0] for s in setsB], fontsize=ps.FS["small"])
axB.set_ylabel("$\\mathrm{Pearson}_{\\mathrm{DEG}}$ (drift-rm.)")
axB.set_ylim(0, capB + 0.22)
ps.legend_above(axB, ncol=2)
ps.note(axB, f"held-out: 3-seed $\\Delta$+{d3m:.3f}$\\pm${d3s:.3f} (SEM);\n"
              "answer-key: 1 deployable ckpt", y=-0.44)
axB.set_title("Combos beat additive", pad=TPAD1)
ps.tidy(axB)

# ---------------- C: k-sweep (WM vs additive), 3-seed SEM bands ----------------
kkeys = ["briggs_combos_held-out_k2-3", "briggs_combos_held-out_k4-5", "briggs_combos_held-out_k6-8"]
klbl = ["2–3", "4–5", "6–8"]
xk = np.arange(len(kkeys))
series = [("disc", "WM+head", C_WM, "o-", "WM disc."), ("disc", "obs-add", C_ADD, "o-", "add. disc."),
          ("allDEG", "WM+head", C_WM, "^--", "WM $\\mathrm{Pearson}_{\\mathrm{DEG}}$"), ("allDEG", "obs-add", C_ADD, "^--", "add. $\\mathrm{Pearson}_{\\mathrm{DEG}}$")]
for metric, method, col, sty, lab in series:
    ms = [stat(k, method, metric) for k in kkeys]
    axC.errorbar(xk, [m for m, _ in ms], yerr=[s for _, s in ms], fmt=sty, color=col,
                 lw=1.2 if "WM" in lab else 1.0, markersize=3.6, capsize=1.8, elinewidth=0.7, label=lab)
nk = [n_of(k) for k in kkeys]
axC.set_xticks(xk); axC.set_xticklabels([f"{l}\n$n$={n}" for l, n in zip(klbl, nk)],
                                        fontsize=ps.FS["small"])
axC.set_xlabel("perturbation order $k$"); axC.set_ylabel("drift-removed metric")
axC.set_ylim(0, 1.0); axC.set_xlim(-0.28, 2.28)
ps.legend_above(axC, ncol=2)
axC.set_title("Prediction vs perturbation order", pad=TPAD2)
ps.tidy(axC)

# ---------------- D: non-additivity grows with order (3-seed SEM) ----------------
xk = np.arange(len(kkeys)); bw = 0.38
capD = 0.0
for mi, (metric, col, lab) in enumerate([("disc", C_DISC, "$\\Delta$ disc."), ("allDEG", C_DEG, "$\\Delta$ $\\mathrm{Pearson}_{\\mathrm{DEG}}$")]):
    gp = [gap(k, metric) for k in kkeys]
    axD.bar(xk + (mi - 0.5) * bw, [m for m, _ in gp], bw, yerr=[s for _, s in gp],
            color=col, edgecolor="white", linewidth=0.4, error_kw=EK, label=lab)
for xi, k in enumerate(kkeys):
    pst = pair_stat("ho", k, "allDEG")
    top = max(sum(gap(k, m)) for m in ("disc", "allDEG"))
    axD.text(xi, top + 0.022, f"{ps.stars(pst.get('p'))}  {pst.get('win', 0) * 100:.0f}%",
             ha="center", va="bottom", fontsize=ps.FS["small"], color=ps.INK_SOFT)
    capD = max(capD, top + 0.022)
axD.set_xticks(xk); axD.set_xticklabels([f"{l}\n$n$={n}" for l, n in zip(klbl, nk)],
                                        fontsize=ps.FS["small"])
axD.set_xlabel("perturbation order $k$"); axD.set_ylabel("advantage $\\Delta$ (WM$-$add.)")
axD.set_ylim(0, capD + 0.17)
ps.legend_above(axD, ncol=2)
axD.text(0.03, 0.97, "% = held-out pairs WM$>$add", transform=axD.transAxes, ha="left",
         va="top", fontsize=ps.FS["small"], color=ps.INK_SOFT)
axD.set_title("Non-additivity grows with order", pad=TPAD1)
ps.tidy(axD)

ps.label_axes(fig, [axA, axB, axC, axD])
ps.save(fig, "fig2")
print("wrote fig2_prediction.pdf / .png" + ("" if LIVE else "   [cached PDF-recovered inputs]"))
print("A n:", nA, "| B held-out n:", pH_["n"], "answer-key n:", pA_["n"],
      "| 3-seed dH:", round(d3m, 3), "+/-", round(d3s, 3))

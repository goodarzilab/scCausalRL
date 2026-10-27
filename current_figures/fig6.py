"""Fig 6 - Cell-fate design results (flagship conditional GFlowNet, MLP head, ensemble pessimism lambda=0.25;
FINAL_REC eval, folds 0/1/2 held-out = n=41 Briggs fates; elite top-10%-by-reward readout).  Data: fig6_data.json.
5-panel: A Briggs recovery across metrics vs budget-matched baselines (aggregate, no sig marks),
B per-fate elite recall@20 policy-minus-baseline (significance shown through the graph),
C diversity, D HSPC arrayed TFs the policy proposes (elite, canonical highlighted),
E HSPC arrayed-key coverage vs baselines (elite recall@50 + union; diversity-coverage advantage).
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import paperstyle as ps

HERE = os.path.dirname(__file__)
D = json.load(open(os.path.join(HERE, "fig6_data.json")))
BS, BM, HS, DV, AR = D["by_source"], D["briggs_metrics"], D["hspc_pertf"], D["diversity"], D["hspc_arrayed"]
EK = dict(lw=0.6, capsize=1.6, ecolor=ps.INK)
SRC = ["joung_single", "joung_combo", "briggs", "HSPC"]
SLBL = ["Joung\nsingle", "Joung\ncombo", "Briggs", "HSPC"]
METH = [("policy", "policy"), ("MPC_exact", "MPC"), ("additive_greedy", "additive"), ("random_shortlist", "random")]
MCOL = {"policy": ps.METHOD["policy"], "MPC_exact": ps.METHOD["MPC"],
        "additive_greedy": ps.METHOD["additive"], "random_shortlist": ps.METHOD["random"]}
GREY_L = ps.GREY_L
P1C, P2C = ps.METHOD["policy"], ps.GREEN

fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, 4.35), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06)
LEG = dict(frameon=False, fontsize=ps.FS["legend"], handlelength=0.8,
           handletextpad=0.4, columnspacing=0.7, borderpad=0.2, labelspacing=0.22)
gs = fig.add_gridspec(2, 6, hspace=0.72, wspace=1.1, height_ratios=[0.82, 1.0])
axA = fig.add_subplot(gs[0, 0:3]); axB = fig.add_subplot(gs[0, 3:6])
axC = fig.add_subplot(gs[1, 0:2]); axD = fig.add_subplot(gs[1, 2:4]); axE = fig.add_subplot(gs[1, 4:6])

def _stars(p):
    s = ps.stars(p); return s if s else "ns"

# ---------- A: Briggs recovery across metrics vs baselines (elite readout, n=41; no sig marks, see B) ----------
EL = D["briggs_elite"]; MEL = EL["metrics"]; nEL = EL["n"]
ORDA = ["recall@5", "recall@20", "soft_recall@20", "precision@5"]
ALAB = ["recall\n@5", "recall\n@20", "soft-rec\n@20", "prec\n@5"]
methE = [("policy", "policy"), ("MPC", "MPC"), ("additive", "additive"), ("random", "random")]
MCOLE = {"policy": ps.METHOD["policy"], "MPC": ps.METHOD["MPC"],
         "additive": ps.METHOD["additive"], "random": ps.METHOD["random"]}
xa = np.arange(len(ORDA)); bw = 0.20
for i, (lab, key) in enumerate(methE):
    v = [MEL[m][key][0] for m in ORDA]; e = [MEL[m][key][1] for m in ORDA]
    axA.bar(xa + (i - 1.5) * bw, v, bw, yerr=e, color=MCOLE[lab], label=lab, edgecolor="white", linewidth=0.3, error_kw=EK)
axA.set_xticks(xa); axA.set_xticklabels(ALAB, fontsize=ps.FS["small"])
axA.set_ylabel("driver recovery (elite)"); axA.set_ylim(0, 1.02); axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axA.set_title(f"Briggs recovery vs baselines ($n$={nEL})"); ps.tidy(axA)
# bracket policy vs MPC only where significant (recall@5, precision@5); recall@20/soft are parity (ns)
for xi, m in enumerate(ORDA):
    if MEL[m]["p_MPC"] < 0.05:
        xpol, xmpc = xa[xi] - 1.5 * bw, xa[xi] - 0.5 * bw
        top = max(MEL[m]["policy"][0] + MEL[m]["policy"][1], MEL[m]["MPC"][0] + MEL[m]["MPC"][1])
        ps.sig_bar(axA, xpol, xmpc, top + 0.04, ps.stars(MEL[m]["p_MPC"]), dh=0.014, fs=ps.FS["annot"], color=ps.METHOD["policy"])
axA.legend(loc="upper right", ncol=2, **LEG)
ps.note(axA, "elite (top-10%-by-reward) readout; brackets = policy vs MPC", y=-0.24)

# ---------- B: per-fate elite recall@20, policy minus each baseline (n=41) ----------
PF = D["briggs_elite_perfate"]; polpf = np.array(PF["policy"], float)
pR20 = MEL["recall@20"]
BB = [("vs MPC", "MPC", ps.METHOD["MPC"], pR20["p_MPC"]),
      ("vs add.", "additive", ps.METHOD["additive"], pR20["p_additive"]),
      ("vs rand.", "random", ps.METHOD["random"], pR20["p_random"])]
rng = np.random.default_rng(1)
for i, (lab, key, col, pv) in enumerate(BB):
    diff = polpf - np.array(PF[key], float); diff = diff[~np.isnan(diff)]; m = len(diff)
    xj = i + (rng.random(m) - 0.5) * 0.30
    axB.scatter(xj, diff, s=7, color=col, alpha=0.65, edgecolor="white", linewidth=0.15, zorder=3)
    mu = diff.mean(); se = diff.std(ddof=1) / np.sqrt(m)
    axB.plot([i - 0.22, i + 0.22], [mu, mu], color=ps.INK, lw=1.5, zorder=4)
    axB.plot([i, i], [mu - 1.96 * se, mu + 1.96 * se], color=ps.INK, lw=0.9, zorder=4)
    st = _stars(pv)
    axB.text(i, 0.96, st, ha="center", va="top", fontsize=ps.FS["annot"],
             color=(ps.INK_SOFT if st == "ns" else ps.METHOD["policy"]), transform=axB.get_xaxis_transform())
axB.axhline(0, color=ps.INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
axB.set_xticks(range(3)); axB.set_xticklabels([b[0] for b in BB], fontsize=ps.FS["small"])
axB.set_ylabel("per-fate recall@20\n(policy $-$ baseline)"); axB.set_ylim(-0.55, 1.05)
axB.set_title(f"Per held-out fate ($n$={len(polpf)})"); ps.tidy(axB)
ps.note(axB, "each dot = one fate; bar = mean, whisker = 95% CI", y=-0.24)

# ---------- C: diversity at fixed quality ----------
dl = [("policy", DV["policy"], ps.METHOD["policy"]), ("beam-256", DV["beam256"], ps.METHOD["MPC"]),
      ("random", DV["random"], ps.METHOD["random"])]
xc = np.arange(len(dl))
axC.bar(xc, [v for _, v, _ in dl], 0.62, color=[c for *_, c in dl], edgecolor="white", linewidth=0.3)
for xi, (_, v, _) in zip(xc, dl):
    axC.text(xi, v + 4, str(int(v)), ha="center", va="bottom", fontsize=ps.FS["small"], color=ps.INK)
axC.set_xticks(xc); axC.set_xticklabels([l for l, *_ in dl], fontsize=ps.FS["small"])
axC.set_ylabel("distinct high-quality\ncocktails (of 256)"); axC.set_ylim(0, 172); axC.set_yticks([0, 50, 100, 150])
axC.set_title("Diversity at quality"); ps.tidy(axC)
ps.note(axC, "HSPC pool, $n$=256 proposals", y=-0.26)

# ---------- D: HSPC arrayed TFs the policy proposes (elite; which TFs) ----------
from matplotlib.patches import Patch
HF = D["hspc_arrayed_final"]; SURF = HF["surfaced"]; nrun = HF["n_runs"]
CAN_C, OTH_C = ps.METHOD["policy"], ps.GREEN
order = SURF[::-1]                                    # highest freq at top of the axis
yy = np.arange(len(order))
for y, (t, fr, can) in zip(yy, order):
    axD.barh(y, fr, 0.66, color=(CAN_C if can else OTH_C), edgecolor="white", linewidth=0.3, zorder=2)
axD.set_yticks(yy); axD.set_yticklabels([t for t, _, _ in order], fontsize=ps.FS["small"])
for tick, (t, _, can) in zip(axD.get_yticklabels(), order):
    tick.set_color(CAN_C if can else OTH_C)
axD.set_xlim(0, 1.0); axD.set_ylim(-0.7, len(order) - 0.3); axD.set_xticks([0, 0.5, 1.0])
axD.set_xlabel(f"frac. of $n$={nrun} runs\nproposing the TF")
axD.set_title("HSPC TFs proposed"); ps.tidy(axD)
axD.legend(handles=[Patch(fc=CAN_C, label="canonical hematop."), Patch(fc=OTH_C, label="other")],
           loc="lower right", **LEG)

# ---------- E: HSPC arrayed-key coverage vs baselines (elite; diversity coverage) ----------
AM = [("policy", ps.METHOD["policy"]), ("MPC", ps.METHOD["MPC"]), ("additive", ps.METHOD["additive"]), ("random", ps.METHOD["random"])]
xe = np.arange(2); bwe = 0.20
for i, (m, col) in enumerate(AM):
    vals = [HF[m]["r50"][0], HF[m]["union50"] / HF["n_tf"]]
    errs = [HF[m]["r50"][1], 0.0]
    axE.bar(xe + (i - 1.5) * bwe, vals, bwe, yerr=errs, color=col, label=m, edgecolor="white", linewidth=0.3, error_kw=EK)
axE.set_xticks(xe); axE.set_xticklabels(["recall\n@50", f"union\nhits/{HF['n_tf']}"])
axE.set_ylabel("fraction of\narrayed key"); axE.set_ylim(0, 0.32); axE.set_yticks([0, 0.1, 0.2, 0.3])
axE.set_title("HSPC arrayed coverage"); ps.tidy(axE)
axE.legend(loc="upper right", ncol=2, **LEG)
ps.note(axE, f"{HF['n_tf']}-TF arrayed key; policy $>$ baselines", y=-0.26)

ps.label_axes(fig, [axA, axB, axC, axD, axE], dy=0.028)
ps.save(fig, os.path.join(HERE, "fig6"))
print("wrote fig6 (5-panel)")

"""Fig 9 - Planner training and generalization (flagship conditional GFlowNet, MLP head, lambda=0.25).
House style (paperstyle).
  A trajectory-balance loss + distinct-TF count over training (real flagship log)
  B in-domain vs held-out driver recall@20 (elite) by source -> Briggs generalizes; small Joung sources (6/2 fates) near floor
"""
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import paperstyle as ps

# ---- A: training curve from the flagship run log ----
LOG = None
for p in ("/home/sgoudarzi/scCausalRL/experiments_rl/logs/rl_mk_cond_FLAG_mlp_l025_f1_s0_*.out",):
    g = sorted(glob.glob(p), key=os.path.getmtime)
    if g:
        LOG = g[-1]; break
its, tb, dist = [], [], []
if LOG:
    for ln in open(LOG, errors="ignore"):
        m = re.search(r"\[setppo\] it\s+(\d+) .* TB_loss\s+([\d.]+) logZ\s+-?[\d.]+ distinct\s+(\d+)", ln)
        if m:
            its.append(int(m[1]) / 1e6); tb.append(float(m[2])); dist.append(int(m[3]))
its, tb, dist = np.array(its), np.array(tb), np.array(dist)

# ---- B: in-domain vs held-out elite recall@20 by source (flagship dumps, 11 runs) ----
GEN = [("Joung\nsingle", 0.043, 0.016, 0.023, 0.022),
       ("Joung\ncombo", 0.066, 0.021, 0.075, 0.044),
       ("Briggs", 0.562, 0.014, 0.562, 0.028)]

fig, (axA, axB) = ps.grid(1, 2, h=2.2, wspace=0.34)

# A
axA.plot(its, tb, color=ps.METHOD["policy"], lw=1.1)
axA.set_yscale("log"); axA.set_xlabel(r"training steps ($\times10^{6}$)")
axA.set_ylabel("trajectory-balance loss", color=ps.METHOD["policy"])
axA.tick_params(axis="y", labelcolor=ps.METHOD["policy"]); ps.tidy(axA)
a2 = axA.twinx()
a2.plot(its, dist, color=ps.GREEN, lw=1.1)
a2.set_ylabel("distinct TFs sampled", color=ps.GREEN); a2.set_ylim(240, 286)
a2.tick_params(axis="y", labelcolor=ps.GREEN); a2.spines["top"].set_visible(False)
axA.set_title("GFlowNet training (flagship)")
if len(tb):
    axA.text(0.97, 0.94, f"loss {tb[0]:.0f}$\\to${tb[-1]:.1f}\ndistinct $\\to$280/280",
             transform=axA.transAxes, ha="right", va="top", fontsize=ps.FS["small"],
             color=ps.INK_SOFT, linespacing=1.3)

# B
x = np.arange(len(GEN)); bw = 0.38
axB.bar(x - bw / 2, [g[1] for g in GEN], bw, yerr=[g[2] for g in GEN], color=ps.tint(ps.METHOD["policy"], 0.45),
        label="in-domain", edgecolor="white", linewidth=0.3, error_kw=dict(lw=0.6, capsize=1.6, ecolor=ps.INK))
axB.bar(x + bw / 2, [g[3] for g in GEN], bw, yerr=[g[4] for g in GEN], color=ps.METHOD["policy"],
        label="held-out", edgecolor="white", linewidth=0.3, error_kw=dict(lw=0.6, capsize=1.6, ecolor=ps.INK))
axB.set_xticks(x); axB.set_xticklabels([g[0] for g in GEN])
axB.set_ylabel("driver recall@20 (elite)"); axB.set_ylim(0, 0.66)
axB.set_title("Held-out tracks in-domain"); ps.tidy(axB)
axB.legend(loc="upper left", frameon=False, fontsize=ps.FS["legend"], handlelength=0.9)
axB.annotate("generalizes", xy=(2, 0.56), xytext=(1.4, 0.6), fontsize=ps.FS["small"], color=ps.INK_SOFT)

ps.label_axes(fig, [axA, axB])
ps.save(fig, os.path.join(os.path.dirname(__file__), "fig9"))
print("wrote fig9")

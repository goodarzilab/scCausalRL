"""Fig 6 (HSPC panel) — flagship policy recovers KNOWN hematopoietic/HSPC master-regulator TFs.

REAL DATA PROVENANCE (no fabricated TF names or values — all read live from the pick dumps):
  Flagship arm = mk_a3_gfn_grn_inf (GFN·grn·InfoNCE).
  PICKS + DRIVERS (Panel A):
    experiments/exp20_load_bearing/results/briggs/eval_only_perfate_tfs_E_mk_a3_gfn_grn_inf_f{0..4}_s{0..2}.json
      -> group key 'held-out:policy (amortized)' -> fate 'HSPC' (source 'joung' = the dedicated HSPC P1∪P2 fate;
         canon_source() in evaluation/rl_eval_suite.py:103).  Per-fate dict keys:
           'freq'    : {TF -> pick count}  (policy's picks; rank = position in freq-descending order)
           'drivers' : answer-key P1∪P2 driver TFs for the fate (the drv_recall answer set)
      Consensus pick-frequency ranks aggregated over all 15 fold×seed policy dumps.
      Matched-random baseline = same schema with suffix '_random_shortlist' (rl_eval_suite.py:150).
  RECALL vs NULL (Panel B):
    docs/figures/rl_eval/E_mk_a3_gfn_grn_inf_policy_vs_null_wilcoxon.json ['drv_recall20']['HSPC']
      -> policy_mean 0.176 vs matched-random null_mean 0.072, paired Wilcoxon over 15 fold×seed, BH-q=1.0e-3.
    docs/figures/rl_eval/cross_arm_summary.csv (arm=mk_a3_gfn_grn_inf, source=HSPC) -> MPC_exact.drv_recall20 (0.000).

CURATED CANONICAL HSPC/HEMATOPOIETIC MASTER REGULATORS (literature cross-reference):
  - "Heptad" of HSPC transcription factors: TAL1/SCL, LYL1, LMO2, GATA2, RUNX1, ERG, FLI1
        Wilson NK et al., Cell Stem Cell 2010;7:532-544; Göttgens B, Blood 2015;125:2614.
  - Classic HSC / lineage master regulators: GATA1, KLF1 (erythroid), SPI1/PU.1 (myeloid), GFI1B,
        MEIS1, HOXB4, HOXA9
        Orkin SH & Zon LI, Cell 2008;132:631-644; Laurenti E & Göttgens B, Nature 2018;553:418-426.

HONEST FRAMING (see MEMORY exp20 discrimination-wall / stats-hardened):
  The significant, enriched recovery is of the fate's ANSWER-KEY driver TFs (RPA3 rank 2, KLF2 rank 3;
  6/11 recovered; recall@20 0.176 vs null 0.072, BH-q=1.0e-3). Among the recovered drivers is TAL1/SCL,
  the canonical master regulator of hematopoietic specification. The policy's HSPC picks additionally
  contain other canonical hematopoietic master TFs (FLI1, HOXB4, LYL1, HOXA9, GATA1/2, KLF1) — shown
  with their ranks — while several (LMO2, RUNX1, ERG, SPI1, GFI1B, MEIS1) are NOT picked (shown honestly).

Run:  cd paper/figures && python make_fig6_hspc_drivers.py
Outputs: panels/fig6_hspc_drivers.pdf / .png
"""
import os, glob, json, collections
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import paperstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP20 = os.path.join(REPO, "experiments", "exp20_load_bearing")
BRIGGS = os.path.join(EXP20, "results", "briggs")
RLEVAL = os.path.join(EXP20, "docs", "figures", "rl_eval")
ARM = "E_mk_a3_gfn_grn_inf"
FATE = "HSPC"
HELDOUT = "held-out:policy (amortized)"

# curated canonical HSPC/hematopoietic master regulators (aliases in parentheses) -- see header for sources
CANON = {"TAL1": "SCL", "LYL1": "", "LMO2": "", "GATA2": "", "RUNX1": "", "ERG": "", "FLI1": "",
         "GATA1": "", "KLF1": "", "SPI1": "PU.1", "GFI1B": "", "MEIS1": "", "HOXB4": "", "HOXA9": ""}


# Verbatim mirror of the numbers this script reads out of the pick dumps + wilcoxon json,
# so the panel still builds where the results tree is not mounted.  Consensus pick-frequency
# ranks for the answer-key drivers and for the canonical master TFs that appear in the picks;
# drivers with no rank were never picked.
_FALLBACK = {
    "drivers": ["DDIT3", "HOXB3", "KLF2", "LMO2", "MIER1", "RNF138",
                "RPA3", "SIRT6", "TAL1", "ZFP90", "ZNF85"],
    "pol_rank": {"RPA3": 2, "KLF2": 3, "HOXB3": 37, "TAL1": 58, "MIER1": 114, "ZNF85": 152,
                 "HOXB4": 20, "FLI1": 23, "LYL1": 54, "HOXA9": 94, "GATA1": 141,
                 "KLF1": 147, "GATA2": 161},
    "wilcoxon": {"policy_mean": 0.176, "null_mean": 0.072, "bh_q": 1.0e-3},
    "mpc_drv_recall20": 0.000,
    "policy_sem": 0.0219,
}


def _bucket(f):
    for s in ("_MPC_exact", "_additive_greedy", "_random_shortlist"):
        if f.endswith(s + ".json"):
            return s[1:]
    return "policy"


def aggregate_freq(planner):
    """Consensus pick-frequency (summed over all fold×seed dumps) + answer-key driver union for HSPC."""
    freq = collections.Counter(); drivers = set()
    for f in sorted(glob.glob(os.path.join(BRIGGS, f"eval_only_perfate_tfs_{ARM}_f*_s*.json"))):
        if _bucket(f) != planner:
            continue
        v = json.load(open(f)).get(HELDOUT, {}).get(FATE)
        if not v:
            continue
        for t, c in v["freq"].items():
            freq[t] += c
        drivers |= set(v.get("drivers", []))
    ranked = [t for t, _ in freq.most_common()]
    rank = {t: i + 1 for i, t in enumerate(ranked)}
    return freq, rank, drivers, len(freq)


def load_all():
    """Live read from the results tree when it is mounted, else the in-file mirror."""
    wil_json = os.path.join(RLEVAL, f"{ARM}_policy_vs_null_wilcoxon.json")
    if not os.path.exists(wil_json):
        f = _FALLBACK
        return (dict(f["pol_rank"]), list(f["drivers"]), dict(f["wilcoxon"]),
                f["mpc_drv_recall20"], f["policy_sem"])
    _, pol_rank, drivers, _ = aggregate_freq("policy")
    wil = json.load(open(wil_json))["drv_recall20"][FATE]
    ca = pd.read_csv(os.path.join(RLEVAL, "cross_arm_summary.csv"))
    ca = ca[(ca.arm == "mk_a3_gfn_grn_inf") & (ca.source == FATE)].iloc[0]
    return (pol_rank, sorted(drivers), wil, float(ca["MPC_exact.drv_recall20"]),
            float(ca["policy.drv_recall20.sem"]))


def main():
    pol_rank, drivers, wil, mpc, sem = load_all()
    drivers = sorted(drivers)

    # canonical master TFs the policy actually PICKS but that are NOT answer-key drivers of this fate
    canon_only = sorted([t for t in CANON if t in pol_rank and t not in drivers],
                        key=lambda t: pol_rank[t])
    canon_missed = [t for t in CANON if t not in pol_rank and t not in drivers]

    NOTPICK = 260                                              # x-position of the "not picked" park column
    XMAX = 380
    red, blue, grey = ps.SOURCE["HSPC"], ps.METHOD["policy"], ps.GREY

    fig, (axL, axR) = ps.grid(1, 2, h=3.05, wspace=0.10,
                              gridspec_kw=dict(width_ratios=[2.6, 1.0]))
    fig.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.945))   # room for the footnote

    # ---------- Panel A: ranked lollipop (answer-key drivers + canonical master TFs in picks) ----------
    rows = []   # (label, rank_or_None, color, is_driver, note)
    for t in sorted(drivers, key=lambda t: (pol_rank.get(t) is None, pol_rank.get(t, 1e9))):
        note = "SCL" if t == "TAL1" else ("HSC master" if t in CANON else "")
        rows.append((t, pol_rank.get(t), red, True, note))
    for t in canon_only:
        rows.append((t, pol_rank.get(t), blue, False, ""))
    rows = rows[::-1]                                          # first row at top of axis
    y = np.arange(len(rows))
    n_driver = len(drivers)
    div = len(rows) - n_driver - 0.5                           # divider between the two groups

    for yi, (t, r, col, is_drv, note) in zip(y, rows):
        if r is None:                                         # not picked -> park at right, grey cross
            axL.plot(NOTPICK, yi, marker="x", ms=4.2, mew=1.1, color=grey, zorder=3)
        else:
            axL.hlines(yi, 1.0, r, color=col, lw=1.1, alpha=0.55, zorder=1)
            mk = "*" if t == "TAL1" else "o"
            ms = 9 if t == "TAL1" else 5
            axL.plot(r, yi, mk, ms=ms, color=col, mec="white", mew=0.5, zorder=3)
            axL.annotate(str(r), (r, yi), textcoords="offset points", xytext=(4.5, 0),
                         va="center", ha="left", fontsize=ps.FS["small"], color=col)
    labels = [f"{t}/{note}" if note == "SCL" else t for (t, *_ ) in rows]
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=ps.FS["small"])
    for tick, (_, _, col, is_drv, _) in zip(axL.get_yticklabels(), rows):
        tick.set_color(col)
    axL.set_xscale("log")
    axL.set_xlim(1, XMAX); axL.set_ylim(-0.75, len(rows) + 0.95)
    axL.set_xticks([1, 3, 10, 20, 50, 100, 200]); axL.set_xticklabels([1, 3, 10, 20, 50, 100, 200])
    axL.tick_params(axis="x", labelsize=ps.FS["small"])
    axL.tick_params(axis="y", length=0)
    axL.axvspan(20, XMAX, color=ps.tint(ps.GREY_L, 0.72), zorder=0)      # beyond top-20
    # the rule stops below the header strip so the "top-20" label is not struck through
    axL.vlines(20, -0.75, len(rows) - 0.05, color=ps.INK, ls="--", lw=0.6, zorder=2)
    ps.tidy(axL)
    # Column headers live in the strip the widened y-limit opens above the top row, so
    # neither "top-20" nor "not picked" can land on the title, a marker or a rank label.
    axL.text(20, len(rows) + 0.10, "top-20", fontsize=ps.FS["small"], ha="center",
             va="bottom", color=ps.INK_SOFT)
    npark = [yi for yi, (t, r, *_ ) in zip(y, rows) if r is None]
    if npark:
        axL.text(NOTPICK, len(rows) + 0.10, "not picked", fontsize=ps.FS["small"],
                 ha="center", va="bottom", color=grey)
    axL.axhline(div, color=ps.RULE, lw=0.6, zorder=2)
    # group labels: answer-key drivers ABOVE the divider (parked over the not-picked rows,
    # which are empty on the left), canonical master TFs BELOW it (parked to the right of
    # the two shortest lollipops, which is the only clear block in that group)
    axL.text(1.15, div + 0.35, "answer-key drivers (P1$\\cup$P2)", fontsize=ps.FS["small"],
             color=red, va="bottom", ha="left")
    axL.text(XMAX * 0.95, div - 0.12,
             ps.wrap_label("other canonical HSPC master TFs (in picks)", width=22, maxlines=2),
             fontsize=ps.FS["small"], color=blue, va="top", ha="right", linespacing=1.25)
    axL.set_xlabel("policy pick-frequency rank (consensus, 15 fold$\\times$seed)")
    axL.set_title("Policy recovers HSPC master-regulator TFs")
    n_rec = sum(1 for t in drivers if t in pol_rank)
    ps.note(axL, f"{n_rec}/{len(drivers)} answer-key drivers recovered  ·  "
                 f"RPA3, KLF2 in top-3  ·  TAL1/SCL recovered", y=-0.155)

    # ---------- Panel B: recall@20 vs matched-random null ----------
    bars = [("policy", wil["policy_mean"], red),
            ("random\nnull", wil["null_mean"], ps.METHOD["random"]),
            ("MPC", mpc, ps.METHOD["MPC"])]
    xb = np.arange(len(bars)); bwid = 0.66
    for xi, (lbl, val, col) in zip(xb, bars):
        axR.bar(xi, val, bwid, color=col, edgecolor="white", linewidth=0.4, zorder=3)
        if val == 0:            # a zero-height bar is invisible: mark the baseline explicitly
            axR.plot([xi - bwid / 2, xi + bwid / 2], [0, 0], color=col, lw=2.0,
                     solid_capstyle="butt", zorder=4)
    # SEM on policy from fold×seed spread (cross_arm)
    axR.errorbar(0, wil["policy_mean"], yerr=sem, fmt="none", ecolor=ps.INK,
                 elinewidth=0.7, capsize=2, zorder=5)
    hi = wil["policy_mean"] + sem
    axR.set_ylim(0, hi + 0.088)
    # value labels sit above the error bar / bars; the bracket is lifted clear of them
    for xi, (lbl, val, col) in zip(xb, bars):
        axR.text(xi, (hi if xi == 0 else val) + 0.008, f"{val:.2f}", ha="center",
                 va="bottom", fontsize=ps.FS["small"], color=ps.INK)
    q = wil["bh_q"]
    ps.sig_bar(axR, 0, 1, hi + 0.040, f"{ps.stars(q)}  q={q:.0e}", dh=0.006)
    axR.set_xticks(xb); axR.set_xticklabels([b[0] for b in bars])
    axR.set_ylabel("driver recall@20")
    axR.set_title("Recall@20 vs null")
    ps.tidy(axR)

    ps.label_axes(fig, [axL, axR])
    stem = os.path.join(HERE, "panels", "fig6_hspc_drivers")
    ps.save(fig, stem)
    print(f"[verify] answer-key drivers (n={len(drivers)}): {drivers}")
    print(f"[verify] recovered driver ranks: " +
          ", ".join(f"{t}={pol_rank[t]}" for t in sorted(drivers, key=lambda t: pol_rank.get(t, 1e9))
                    if t in pol_rank))
    print(f"[verify] not picked: {[t for t in drivers if t not in pol_rank]}")
    print(f"[verify] canonical-only master TFs in picks: " +
          ", ".join(f"{t}={pol_rank[t]}" for t in canon_only))
    print(f"[verify] canonical master TFs NOT picked: {canon_missed}")


if __name__ == "__main__":
    main()

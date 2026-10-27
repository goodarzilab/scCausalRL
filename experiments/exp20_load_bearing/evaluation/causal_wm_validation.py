"""Causal validation of the frozen single-cell TF-perturbation world model (WM).

ONE modular, publication-quality validation of the deployable Joung+Briggs production
world model (jb_prod_wm) — a frozen causal-VAE with a latent DAG A (d=357), a
shift-mutilate do-operator, and a masked-GO ("SENA-style") decoder D: gene = D·z.

Three analyses (pick with --analyses edge,grn,bio or 'all'):

  edge  EDGE-DIRECTION ABLATION on Δ-perturbation prediction. Re-solve the SCM with
        4 variants of A — original / reversed (Aᵀ, flipped causal direction) /
        symmetrized ((A+Aᵀ)/2) / shuffled (off-diagonal entries permuted, sparsity
        preserved) — and score held-out single-TF and combo Δ prediction (drift-removed
        Pearson-on-DEGs + discrimination AUROC). The SHUFFLED arm is the correctness null
        (memory: judge A by correctness-gain over a shuffled-A null, not cos(Δ)).

  grn   PROJECTED GRAPH + GO MEMBERSHIP + AUROC vs known networks. Form the gene-gene
        graph G = D·Aᵀ·Dᵀ through the masked-GO decoder; visualise A + the GO programs
        of the top |A|-hub latent factors; benchmark |G| at recovering CollecTRI,
        DoRothEA(A/B), STRING-PPI, and control co-expression (AUROC vs random 0.5),
        against a control-coexpression predictor baseline.

  bio   BIOLOGICAL VALIDATION. Do-operator single-TF Δ predictions recover known TF
        targets (CollecTRI / DoRothEA) better than chance — and REQUIRE the real A
        (original A ≫ shuffled A ≈ A=0 ≈ coexpression), a direct load-bearing test in
        gene-regulatory space.

Figures → experiments/exp20_load_bearing/docs/figures/causal_val/.

Run:
  python experiments/exp20_load_bearing/evaluation/causal_wm_validation.py --analyses all
"""
import os
import sys
import json
import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import torch

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
EXP20 = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP20))
RL = os.path.join(REPO, "experiments_rl")
for p in (RL, EXP20):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_CKPT = os.path.join(
    EXP20, "results", "grid", "jb_prod_wm", "checkpoints", "phase1_epoch60.pt")
VOCAB = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
DEFAULT_SINGLES = os.path.join(VOCAB, "joung_singles_proc_v2b.h5ad")
DEFAULT_COMBO = os.path.join(VOCAB, "joung_combo_proc_v2b.h5ad")
OUT_DIR = os.path.join(EXP20, "docs", "figures", "causal_val")

# ---- validated categorical palette (from the dataviz skill reference instance) ----
PAL = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e2"


# ======================================================================
# matplotlib publication style
# ======================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _bars(ax, groups, series, values, colors, ylabel, title, ylim=None,
          chance=None, chance_label="chance", annotate=True, rot=0):
    """Grouped bar chart. groups: x categories; series: legend entries;
    values: [n_series][n_groups]; colors: per-series hex."""
    import numpy as np
    ng, ns = len(groups), len(series)
    x = np.arange(ng)
    w = 0.8 / max(ns, 1)
    for s in range(ns):
        xs = x + (s - (ns - 1) / 2) * w
        vals = values[s]
        b = ax.bar(xs, vals, w * 0.92, label=series[s], color=colors[s],
                   edgecolor="white", linewidth=0.6, zorder=3)
        if annotate:
            for xi, v in zip(xs, vals):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                ax.text(xi, v + (0.012 if v >= 0 else -0.03) * (ylim[1] - ylim[0] if ylim else 1),
                        f"{v:.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
    if chance is not None:
        ax.axhline(chance, ls="--", lw=1.2, color=INK2, zorder=2)
        ax.text(ng - 0.5, chance, f" {chance_label}", va="center", ha="left",
                fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rot, ha="center" if rot == 0 else "right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    if ns >= 2:
        ax.legend(frameon=False, fontsize=9, ncol=min(ns, 3), loc="upper right")


# ======================================================================
# stack setup
# ======================================================================
def build(args):
    import torch as T
    from core.setup_stack import build_stack
    dev = T.device(args.device or ("cuda" if T.cuda.is_available() else "cpu"))
    sargs = SimpleNamespace(
        dataset="joung", checkpoint=args.checkpoint, wm_root=EXP20,
        singles_path=args.singles_path, combo_path=args.combo_path,
        curated_topk=256, n_ctrl=args.n_ctrl)
    print(f"[setup] device={dev}  ckpt={os.path.basename(args.checkpoint)}", flush=True)
    model, meta, op, data = build_stack(sargs, dev, n_dag=args.n_dag)
    A = op.mean_A().to(T.float32)
    D = model.decoder.effective_D().detach().to(T.float32)     # [n_genes, d] (masked)
    print(f"[setup] latent_dim d={A.shape[0]}  n_genes={D.shape[0]}  "
          f"prior_off (lambda_prior=0)  intervene={getattr(model,'pert_intervene_mode','?')}",
          flush=True)
    # train-single drift (gene space, A-independent) — the drift-removed baseline
    trs = data.single_split["train"]
    drift = np.mean([data.de_single[g]["lfc"] for g in trs], 0).astype(np.float32)
    return SimpleNamespace(model=model, meta=meta, op=op, data=data, dev=dev,
                           A=A, D=D, drift=drift,
                           gene_names=list(map(str, data.gene_names)),
                           g2i=data.gene_to_idx, thresh=float(data.thresh))


# ======================================================================
# A-variant construction + Δ prediction
# ======================================================================
def spectral_norm(M):
    try:
        return float(torch.linalg.matrix_norm(M, ord=2))
    except Exception:
        return float(M.norm())


def make_variant(A, kind, rng=None, match_spectral=True):
    """original / reversed(Aᵀ) / symmetrized((A+Aᵀ)/2) / shuffled(perm off-diag entries)."""
    d = A.shape[0]
    off = ~torch.eye(d, dtype=torch.bool, device=A.device)
    if kind == "original":
        return A.clone()
    if kind == "reversed":
        V = A.t().clone()
    elif kind == "symmetrized":
        V = 0.5 * (A + A.t())
    elif kind == "shuffled":
        V = torch.zeros_like(A)
        vals = A[off]                                   # off-diagonal entries (incl. structural zeros)
        perm = torch.as_tensor(rng.permutation(vals.numel()), device=A.device)
        V[off] = vals[perm]                             # same multiset → sparsity/#edges preserved
    else:
        raise ValueError(kind)
    if match_spectral:                                  # match ‖A‖₂ so magnitude isn't a confound
        sn = spectral_norm(V)
        if sn > 1e-8:
            V = V * (spectral_norm(A) / sn)
    return V


def ctrl_recon(op, data, A):
    z0 = op.scm_forward(data.eps_ctrl_mean.unsqueeze(0), A)
    return op.decode(z0)[0].detach()


def pred_delta(op, data, idx_list, A, cr):
    """Predicted gene-space Δ (pred_pert − pred_ctrl) for a gene set under adjacency A."""
    z = op.apply_sets(data.eps_ctrl_mean, [list(idx_list)], A)      # [1,d]
    return (op.decode(z)[0].detach() - cr)


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.nan
    return float((a * b).sum() / (na * nb))


# ---- statistics helpers (hardening pass) --------------------------------------------
def bh_qvalues(pvals):
    """Benjamini-Hochberg q-values (same routine as briggs_cao_annotate.py:60-70)."""
    p = np.asarray(pvals, float)
    o = np.argsort(p); m = len(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.clip(q, 0, 1)
    return out


def bootstrap_ci(vals, n_boot=5000, seed=0, alpha=0.05):
    """Percentile bootstrap CI on the nan-mean of `vals` (replicate = perturbation)."""
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)])
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def paired_wilcoxon(a, b, alternative="greater"):
    """Paired Wilcoxon signed-rank on matched (a_i, b_i); drops pairs with a nan.
    Returns dict(stat, p, n_pairs, median_diff, mean_diff)."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    diff = a - b
    n = int(np.sum(np.abs(diff) > 0))
    if n < 1 or np.allclose(diff, 0):
        return dict(stat=float("nan"), p=float("nan"), n_pairs=int(a.size),
                    median_diff=float(np.median(diff)) if a.size else float("nan"),
                    mean_diff=float(np.mean(diff)) if a.size else float("nan"))
    try:
        st, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    except Exception:
        return dict(stat=float("nan"), p=float("nan"), n_pairs=int(a.size),
                    median_diff=float(np.median(diff)), mean_diff=float(np.mean(diff)))
    return dict(stat=float(st), p=float(p), n_pairs=int(a.size),
                median_diff=float(np.median(diff)), mean_diff=float(np.mean(diff)))


# ======================================================================
# ANALYSIS 1 — edge-direction ablation
# ======================================================================
def collect_perts(S, kind, cap=None, seed=0):
    """Return list of (label, gene_idx_tuple, observed_lfc[np], target_idx_set)."""
    data = S.data; g2i = S.g2i
    rows = []
    if kind == "single":
        keys = list(S.data.single_split.get("test", []))
        if not keys:
            keys = list(S.data.de_single.keys())
        for g in keys:
            if g in data.de_single and g in g2i:
                rows.append((g, (g2i[g],), np.asarray(data.de_single[g]["lfc"], np.float32), {g2i[g]}))
    else:
        keys = list(data.combo_split.get("test", [])) + list(data.combo_split.get("triples", []))
        if not keys:
            keys = list(data.de_combo.keys())
        for key in keys:
            if key not in data.de_combo:
                continue
            idx = tuple(g2i[g] for g in key if g in g2i)
            if len(idx) < 2:
                continue
            rows.append(("+".join(key), idx, np.asarray(data.de_combo[key]["lfc"], np.float32), set(idx)))
    if cap and len(rows) > cap:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), cap, replace=False))]
    return rows


def eval_variant(S, perts, A, disc=True):
    """Drift-removed Pearson-on-DEGs (correctness) + discrimination (recall@1, AUROC).

    Batches all perturbations through one `apply_sets` solve (fast enough for a
    >=1000-shuffle null). `pearson_per` = per-perturbation Pearson (paired-test replicate
    axis). `disc=False` skips the O(N^2) discrimination loop for the null draws."""
    op, data, drift, thr = S.op, S.data, S.drift, S.thresh
    cr = ctrl_recon(op, data, A).cpu().numpy()
    sets = [list(idx) for _, idx, _, _ in perts]
    Z = op.apply_sets(data.eps_ctrl_mean, sets, A)                 # [N, d]
    preds = op.decode(Z).detach().cpu().numpy() - cr[None, :]       # [N, n_genes]
    resid_pred, resid_obs, sels, corr = [], [], [], []
    for j, (_, idx, lfc, tgt) in enumerate(perts):
        rp = preds[j] - drift
        ro = lfc - drift
        sel = np.abs(lfc) > thr
        for t in tgt:
            sel[t] = False
        resid_pred.append(rp); resid_obs.append(ro); sels.append(sel)
        corr.append(_pearson(rp[sel], ro[sel]) if int(sel.sum()) >= 3 else np.nan)
    corr = np.asarray(corr, float)
    out = dict(pearson=float(np.nanmean(corr)), pearson_per=corr,
               n=int(np.sum(~np.isnan(corr))))
    if not disc:
        out["recall1"] = np.nan; out["disc_auroc"] = np.nan
        return out
    resid_pred = np.stack(resid_pred)
    N = len(perts)
    valid = [j for j in range(N) if int(sels[j].sum()) >= 3]
    Sm = np.full((N, N), -1e9, np.float32)
    for j in valid:
        sel = sels[j]
        o = resid_obs[j][sel]; o = o - o.mean(); on = np.linalg.norm(o) + 1e-9
        P = resid_pred[:, sel]; P = P - P.mean(1, keepdims=True)
        pn = np.linalg.norm(P, axis=1) + 1e-9
        Sm[:, j] = (P @ o) / (pn * on)
    r1 = np.mean([np.argmax(Sm[i, valid]) == valid.index(i) for i in valid]) if valid else np.nan
    auc = []
    for i in valid:
        others = [j for j in valid if j != i]
        if not others:
            continue
        wins = sum((Sm[i, i] > Sm[i, j]) + 0.5 * (Sm[i, i] == Sm[i, j]) for j in others)
        auc.append(wins / len(others))
    out["recall1"] = float(r1)
    out["disc_auroc"] = float(np.mean(auc)) if auc else np.nan
    return out


def analysis_edge(S, args):
    plt = _mpl()
    rng = np.random.default_rng(args.seed)
    singles = collect_perts(S, "single", cap=args.max_perts, seed=args.seed)
    combos = collect_perts(S, "combo", cap=args.max_perts, seed=args.seed)
    print(f"[edge] singles={len(singles)}  combos={len(combos)}", flush=True)
    variants = ["original", "reversed", "symmetrized", "shuffled"]
    results = {"single": {}, "combo": {}}
    edge_stats = {}
    n_disc = min(args.n_shuffle, args.n_disc)          # #shuffles that also get the disc metrics (bar means)
    for kind, perts in (("single", singles), ("combo", combos)):
        if not perts:
            continue
        # --- original / reversed / symmetrized: single full evaluation each ---
        orig_per = None
        for v in ("original", "reversed", "symmetrized"):
            res = eval_variant(S, perts, make_variant(S.A, v, rng=rng, match_spectral=args.match_spectral))
            if v == "original":
                orig_per = res["pearson_per"]                    # per-pert pearson (paired-test replicate axis)
            results[kind][v] = {k: res[k] for k in ("pearson", "recall1", "disc_auroc", "n")}
        orig_mean = float(np.nanmean(orig_per))
        # --- shuffled-A null: >=1000 draws (permutation p + paired Wilcoxon + CI) ---
        N = len(perts)
        null_means = np.empty(args.n_shuffle)
        per_sum = np.zeros(N); per_cnt = np.zeros(N)
        disc_accs = []
        shuf_per_repr = None                                # ONE representative shuffled draw's per-pert pearson
        for k in range(args.n_shuffle):
            Ash = make_variant(S.A, "shuffled",
                               rng=np.random.default_rng(args.seed + 100000 + k),
                               match_spectral=args.match_spectral)
            want = k < n_disc
            a = eval_variant(S, perts, Ash, disc=want)
            null_means[k] = a["pearson"]
            pp = a["pearson_per"]; ok = ~np.isnan(pp)
            per_sum[ok] += pp[ok]; per_cnt[ok] += 1
            if k == 0:
                shuf_per_repr = np.asarray(a["pearson_per"], float).copy()
            if want:
                disc_accs.append(a)
        shuf_per = np.where(per_cnt > 0, per_sum / np.maximum(per_cnt, 1), np.nan)
        results[kind]["shuffled"] = dict(
            pearson=float(np.mean(null_means)),
            recall1=float(np.mean([a["recall1"] for a in disc_accs])) if disc_accs else np.nan,
            disc_auroc=float(np.mean([a["disc_auroc"] for a in disc_accs])) if disc_accs else np.nan,
            n=disc_accs[0]["n"] if disc_accs else int(np.sum(~np.isnan(orig_per))))
        # permutation p: P(shuffled-A mean Pearson >= original)
        perm_p = float((1 + int(np.sum(null_means >= orig_mean))) / (args.n_shuffle + 1))
        # per-metric permutation p for recall@1 / disc-AUROC over the n_disc shuffles that computed disc
        if disc_accs:
            _nr = np.array([a["recall1"] for a in disc_accs], float)
            _na = np.array([a["disc_auroc"] for a in disc_accs], float)
            _or = float(results[kind]["original"]["recall1"]); _oa = float(results[kind]["original"]["disc_auroc"])
            _mr = int(np.sum(~np.isnan(_nr))); _ma = int(np.sum(~np.isnan(_na)))
            perm_p_recall1 = float((1 + int(np.nansum(_nr >= _or))) / (_mr + 1)) if _mr else float("nan")
            perm_p_disc = float((1 + int(np.nansum(_na >= _oa))) / (_ma + 1)) if _ma else float("nan")
            _ndisc = max(_mr, _ma)
        else:
            perm_p_recall1 = perm_p_disc = float("nan"); _ndisc = 0
        wil = paired_wilcoxon(orig_per, shuf_per, alternative="greater")
        ci = bootstrap_ci(orig_per, seed=args.seed)
        edge_stats[kind] = dict(
            orig_pearson=orig_mean, orig_pearson_ci95=list(ci),
            shuffled_pearson_mean=float(np.mean(null_means)),
            shuffled_pearson_std=float(np.std(null_means, ddof=1)) if args.n_shuffle > 1 else float("nan"),
            delta_orig_minus_shuffled=orig_mean - float(np.mean(null_means)),
            n_shuffle=int(args.n_shuffle),
            permutation_p=perm_p,
            permutation_p_recall1=perm_p_recall1, permutation_p_disc_auroc=perm_p_disc, n_disc=int(_ndisc),
            wilcoxon_p=wil["p"], wilcoxon_stat=wil["stat"], wilcoxon_n_pairs=wil["n_pairs"],
            wilcoxon_median_diff=wil["median_diff"])
        # --- OPT-IN evidence dump (single-TF only): full null distribution + per-TF paired vectors ---
        if args.dump_null_json and kind == "single":
            def _clean(v):                                  # NaN -> None so the JSON is valid & loadable
                v = float(v)
                return None if (v != v) else v
            dump = dict(
                analysis="single",
                checkpoint=os.path.basename(args.checkpoint),
                n_shuffle=int(args.n_shuffle),
                seed=int(args.seed),
                match_spectral=bool(args.match_spectral),
                orig_pearson_mean=orig_mean,
                orig_pearson_ci95=list(ci),
                shuffled_pearson_mean=float(np.mean(null_means)),
                shuffled_pearson_std=float(np.std(null_means, ddof=1)) if args.n_shuffle > 1 else float("nan"),
                delta_orig_minus_shuffled=orig_mean - float(np.mean(null_means)),
                permutation_p=perm_p,
                n_exceed=int(np.sum(null_means >= orig_mean)),
                tf_labels=[p[0] for p in perts],
                null_means=[float(x) for x in null_means],
                pearson_per_original=[_clean(x) for x in orig_per],
                pearson_per_shuffled_repr=[_clean(x) for x in shuf_per_repr],
                shuffled_repr_seed=int(args.seed + 100000))
            os.makedirs(os.path.dirname(os.path.abspath(args.dump_null_json)), exist_ok=True)
            json.dump(dump, open(args.dump_null_json, "w"), indent=2)
            print(f"[edge] dumped single-TF null+per-TF evidence -> {args.dump_null_json}", flush=True)
        for v in variants:
            res = results[kind][v]
            print(f"[edge] {kind:6s} {v:12s} pearson={res['pearson']:.3f} "
                  f"disc_auroc={res['disc_auroc']:.3f} recall@1={res['recall1']:.3f} (n={res['n']})",
                  flush=True)
        print(f"[edge-STATS] {kind}: orig={orig_mean:.3f} (95%CI {ci[0]:.3f},{ci[1]:.3f}) "
              f"shuffled={np.mean(null_means):.3f} Δ={orig_mean-np.mean(null_means):+.3f} | "
              f"perm-p={perm_p:.2e} (n_shuffle={args.n_shuffle}) | "
              f"paired-Wilcoxon p={wil['p']:.2e} (n_pairs={wil['n_pairs']}, median Δ={wil['median_diff']:+.3f})",
              flush=True)
    results["_stats"] = edge_stats

    labels = {"original": "original A", "reversed": "reversed Aᵀ",
              "symmetrized": "symmetrized", "shuffled": "shuffled"}
    cols = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["red"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, metric, ylab, ttl, ch in (
            (axes[0], "pearson", "Pearson on DEGs (drift-removed)",
             "Δ-prediction correctness", 0.0),
            (axes[1], "disc_auroc", "discrimination AUROC",
             "Δ-prediction discrimination", 0.5)):
        groups, series, vals, ser_cols = [], [], [], []
        present = [k for k in ("single", "combo") if results[k]]
        for si, kind in enumerate(present):
            series.append("single-TF" if kind == "single" else "combo")
            vals.append([results[kind][v][metric] for v in variants])
            ser_cols.append([PAL["blue"], PAL["violet"]][si])
        _bars(ax, [labels[v] for v in variants], series, vals, ser_cols,
              ylab, ttl, chance=ch, chance_label="chance", rot=20)
    fig.suptitle("Edge-direction ablation — is the directionality of the latent DAG A "
                 "load-bearing for perturbation prediction?", fontsize=13, fontweight="bold", y=1.02)
    # correctness-gain annotation vs shuffled null
    note = []
    for kind in ("single", "combo"):
        if results[kind]:
            g = results[kind]["original"]["pearson"] - results[kind]["shuffled"]["pearson"]
            note.append(f"{kind}: +{g:.3f} correctness over shuffled-A null")
    fig.text(0.5, -0.04, "   |   ".join(note), ha="center", fontsize=9, color=INK2)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig1_edge_direction_ablation.png")
    fig.savefig(out); plt.close(fig)
    print(f"[edge] saved {out}", flush=True)
    json.dump(results, open(os.path.join(OUT_DIR, "edge_direction_results.json"), "w"), indent=2)
    return results


# ======================================================================
# reference networks
# ======================================================================
def load_references(gene_names):
    """Return dict name -> set of (src_idx, tgt_idx) directed pairs present in axis."""
    g2iU = {g.upper(): i for i, g in enumerate(gene_names)}
    refs = {}

    def _pairs(net):
        out = set()
        for s, t in zip(net["source"].astype(str).str.upper(), net["target"].astype(str).str.upper()):
            i, j = g2iU.get(s), g2iU.get(t)
            if i is not None and j is not None and i != j:
                out.add((i, j))
        return out

    import decoupler as dc
    try:
        refs["CollecTRI"] = _pairs(dc.op.collectri(organism="human"))
    except Exception as e:
        print(f"[grn] CollecTRI skipped: {e}")
    try:
        do = dc.op.dorothea(organism="human")
        do = do[do["confidence"].isin(["A", "B"])]
        refs["DoRothEA (A/B)"] = _pairs(do)
    except Exception as e:
        print(f"[grn] DoRothEA skipped: {e}")
    # STRING PPI (undirected → both directions), via omnipath; optional
    try:
        import omnipath
        oi = omnipath.interactions.AllInteractions()
        net = oi.get(genesymbols=True)
        col = "source_genesymbol" if "source_genesymbol" in net.columns else "source"
        colt = "target_genesymbol" if "target_genesymbol" in net.columns else "target"
        sel = net[net.get("sources", "").astype(str).str.contains("STRING", case=False, na=False)] \
            if "sources" in net.columns else net.iloc[0:0]
        pp = set()
        for s, t in zip(sel[col].astype(str).str.upper(), sel[colt].astype(str).str.upper()):
            i, j = g2iU.get(s), g2iU.get(t)
            if i is not None and j is not None and i != j:
                pp.add((i, j)); pp.add((j, i))
        if len(pp) > 200:
            refs["STRING PPI"] = pp
        else:
            print(f"[grn] STRING skipped: only {len(pp)} in-axis edges")
    except Exception as e:
        print(f"[grn] STRING skipped (omnipath): {str(e)[:120]}")
    for k, v in refs.items():
        print(f"[grn] reference {k}: {len(v)} in-axis directed edges", flush=True)
    return refs


def coexpr_matrix(S):
    X = S.data._ctrl_X.detach().cpu().numpy().astype(np.float32)   # [n_ctrl, n_genes]
    C = np.corrcoef(X.T)
    return np.nan_to_num(C, nan=0.0)


def auroc_over_pairs(score_mat, positives, rows, n_genes):
    """AUROC of |score| recovering `positives` over the universe rows×(all genes), i!=j."""
    from sklearn.metrics import roc_auc_score
    rows = np.asarray(sorted(rows))
    Sabs = np.abs(score_mat[rows])                       # [R, n_genes]
    y = np.zeros_like(Sabs, dtype=np.int8)
    rpos = {r: k for k, r in enumerate(rows)}
    for (i, j) in positives:
        if i in rpos:
            y[rpos[i], j] = 1
    diag = np.zeros_like(y, dtype=bool)
    for k, r in enumerate(rows):
        diag[k, r] = True
    mask = ~diag
    yy = y[mask]; ss = Sabs[mask]
    if yy.sum() == 0 or yy.sum() == yy.size:
        return np.nan, int(yy.sum()), 0
    return float(roc_auc_score(yy, ss)), int(yy.sum()), int(yy.size - yy.sum())


def auroc_normal_test(auc, npos, nneg):
    """Hanley–McNeil / Mann–Whitney normal-approx test of AUROC > 0.5.
    SE_0 = sqrt((n1+n2+1)/(12 n1 n2)); returns (z, one-sided p, 95%CI). Approximate:
    ignores row-correlation in the pooled label vector (flagged in the report)."""
    if not (auc == auc) or npos < 1 or nneg < 1:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    from scipy.stats import norm
    se = float(np.sqrt((npos + nneg + 1) / (12.0 * npos * nneg)))
    z = (auc - 0.5) / se
    p = float(norm.sf(z))
    return float(z), p, (float(auc - 1.96 * se), float(auc + 1.96 * se))


# ======================================================================
# ANALYSIS 2 — projected graph + GO membership + AUROC
# ======================================================================
def go_terms_for_columns(gene_names, dec_mask, lam=0.3):
    """Reconstruct the GO:BP term name per latent column (deterministic greedy mask)."""
    try:
        from utils.bio_priors import build_go_bp_mask
        M, terms = build_go_bp_mask(gene_names, max_term_size=30, min_genes_in_axis=5,
                                    max_overlap=0.5, leak=lam, max_pathways=None, verbose=False)
        if M.shape[1] == dec_mask.shape[1]:
            agree = float(((M > lam) == (dec_mask.cpu().numpy() > lam)).mean())
            print(f"[grn] GO term reconstruction: {len(terms)} terms, mask agreement {agree:.3f}")
            return terms
        print(f"[grn] GO term count {M.shape[1]} != dec_mask {dec_mask.shape[1]}; using indices")
    except Exception as e:
        print(f"[grn] GO term reconstruction failed: {e}")
    return [f"program {k}" for k in range(dec_mask.shape[1])]


def analysis_grn(S, args):
    plt = _mpl()
    import numpy as np
    A = S.A.cpu().numpy(); D = S.D.cpu().numpy()
    d = A.shape[0]; n = D.shape[0]
    dec_mask = getattr(S.model.decoder, "dec_mask", None)
    terms = go_terms_for_columns(S.gene_names, dec_mask) if dec_mask is not None \
        else [f"program {k}" for k in range(d)]
    n_member = (dec_mask.cpu().numpy() > 0.99).sum(0) if dec_mask is not None else np.zeros(d)

    # ---- projected gene-gene graph G = D Aᵀ Dᵀ ----
    G = D @ A.T @ D.T                                     # [n, n]
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- FIGURE 2a: latent A + GO hubs ----------
    fig = plt.figure(figsize=(15.5, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.5)
    ax0 = fig.add_subplot(gs[0])
    vmax = float(np.percentile(np.abs(A[A != 0]), 99)) if (A != 0).any() else 1.0
    im = ax0.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax0.set_title(f"Latent DAG A  ({d}×{d}), signed edges")
    ax0.set_xlabel("latent program (child)"); ax0.set_ylabel("latent program (parent)")
    cb = fig.colorbar(im, ax=ax0, orientation="horizontal", fraction=0.05, pad=0.14)
    cb.set_label("edge weight", fontsize=9)
    nnz = int((A != 0).sum())
    ax0.text(0.02, 0.98, f"{nnz} edges  ({100*nnz/(d*d-d):.1f}% density)",
             transform=ax0.transAxes, va="top", fontsize=9, color=INK,
             bbox=dict(boxstyle="round", fc="white", ec=GRID))

    ax1 = fig.add_subplot(gs[1])
    deg = np.abs(A).sum(0) + np.abs(A).sum(1)             # total |A| connectivity per program
    top = np.argsort(-deg)[:20][::-1]
    def _short(t, m=34):
        t = t.split(" (GO:")[0]
        return t if len(t) <= m else t[:m - 1] + "…"
    ylabels = [f"[{k}] {_short(terms[k])}" for k in top]
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=float(n_member[top].min()), vmax=float(n_member[top].max() + 1e-9))
    cmap = matplotlib.cm.get_cmap("Blues")
    barcols = [cmap(0.35 + 0.6 * norm(n_member[k])) for k in top]
    ax1.barh(range(len(top)), deg[top], color=barcols, edgecolor="white", linewidth=0.5, zorder=3)
    ax1.set_yticks(range(len(top))); ax1.set_yticklabels(ylabels, fontsize=7.5)
    ax1.set_xlabel("total |A| connectivity (in+out edge weight)")
    ax1.set_title("Top-20 causal-hub latent programs\n(labeled by GO:BP term; color = # member genes)")
    ax1.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb2 = fig.colorbar(sm, ax=ax1, fraction=0.04, pad=0.02); cb2.set_label("# member genes", fontsize=9)
    fig.suptitle("Latent causal graph A and its GO:BP program identities (masked-GO decoder)",
                 fontsize=13, fontweight="bold", y=1.02)
    out = os.path.join(OUT_DIR, "fig2a_latentA_and_GO.png")
    fig.savefig(out); plt.close(fig)
    print(f"[grn] saved {out}", flush=True)

    # ---------- FIGURE 2b: AUROC vs reference networks ----------
    refs = load_references(S.gene_names)
    Gout = {"G_projected": G}
    coex = None
    if refs:
        coex = coexpr_matrix(S)
        Gout["coexpression"] = coex
    # universe rows = union of sources across regulatory refs, present in axis
    reg_rows = set()
    for name, pos in refs.items():
        for (i, j) in pos:
            reg_rows.add(i)
    if not reg_rows:
        reg_rows = set(S.data.screened_idx)
    reg_rows = sorted(reg_rows)
    print(f"[grn] AUROC universe: {len(reg_rows)} source rows × {n} genes", flush=True)

    auroc = {}
    grn_stats = {}
    for name, pos in refs.items():
        auroc[name] = {}
        grn_stats[name] = {}
        for pname, mat in Gout.items():
            if pname == "coexpression" and name in ("STRING PPI",) and coex is None:
                continue
            a, npos, nneg = auroc_over_pairs(mat, pos, reg_rows, n)
            auroc[name][pname] = a
            z, pval, ci = auroc_normal_test(a, npos, nneg)
            grn_stats[name][pname] = dict(auroc=a, z=z, p_vs_0p5=pval, ci95=list(ci),
                                          n_pos=int(npos), n_neg=int(nneg))
            print(f"[grn] AUROC {name:16s} | {pname:12s} = {a:.3f}  "
                  f"(95%CI {ci[0]:.3f},{ci[1]:.3f}; z={z:.1f}, p_vs_0.5={pval:.2e}; pos={npos})", flush=True)
        # honest head-to-head: does the causal projected graph beat the co-expression control?
        gp = auroc[name].get("G_projected"); ce = auroc[name].get("coexpression")
        if gp is not None and ce is not None and gp == gp and ce == ce:
            verdict = "G_projected > coexpr" if gp > ce else "coexpr >= G_projected (G NOT better)"
            grn_stats[name]["G_vs_coexpr"] = dict(delta=float(gp - ce), verdict=verdict)
            print(f"[grn-STATS] {name:16s} G_projected−coexpr = {gp-ce:+.3f}  -> {verdict}", flush=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ref_names = list(auroc.keys())
    predictors = [p for p in ("G_projected", "coexpression") if any(p in auroc[r] for r in ref_names)]
    plabels = {"G_projected": "projected graph  G = D·Aᵀ·Dᵀ", "coexpression": "control co-expression"}
    pcols = {"G_projected": PAL["blue"], "coexpression": PAL["orange"]}
    vals = [[auroc[r].get(p, np.nan) for r in ref_names] for p in predictors]
    _bars(ax, ref_names, [plabels[p] for p in predictors], vals,
          [pcols[p] for p in predictors],
          "AUROC (recover reference edges)",
          "Projected graph G recovers known gene–gene / regulatory networks",
          ylim=(0.4, 1.0), chance=0.5, chance_label="random", rot=0)
    out = os.path.join(OUT_DIR, "fig2b_grn_auroc.png")
    fig.savefig(out); plt.close(fig)
    print(f"[grn] saved {out}", flush=True)
    json.dump({k: {p: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                   for p, v in d_.items()} for k, d_ in auroc.items()},
              open(os.path.join(OUT_DIR, "grn_auroc_results.json"), "w"), indent=2)
    json.dump(grn_stats, open(os.path.join(OUT_DIR, "grn_auroc_stats.json"), "w"), indent=2)
    return {"auroc": auroc, "stats": grn_stats, "references": {k: len(v) for k, v in refs.items()}}


# ======================================================================
# ANALYSIS 3 — do-operator single-TF target recovery (biology)
# ======================================================================
def analysis_bio(S, args):
    plt = _mpl()
    from sklearn.metrics import roc_auc_score
    import decoupler as dc
    g2iU = {g.upper(): i for i, g in enumerate(S.gene_names)}

    def _targets(net_call, conf=None):
        try:
            net = net_call()
            if conf is not None:
                net = net[net["confidence"].isin(conf)]
        except Exception as e:
            print(f"[bio] reference failed: {e}")
            return {}
        tgt = {}
        for s, t in zip(net["source"].astype(str).str.upper(), net["target"].astype(str).str.upper()):
            i, j = g2iU.get(s), g2iU.get(t)
            if i is not None and j is not None and i != j:
                tgt.setdefault(i, set()).add(j)
        return tgt

    refs = {"CollecTRI": _targets(lambda: dc.op.collectri(organism="human")),
            "DoRothEA (A/B)": _targets(lambda: dc.op.dorothea(organism="human"), conf=["A", "B"])}
    refs = {k: v for k, v in refs.items() if v}
    if not refs:
        print("[bio] no reference TF→target networks available; skipping")
        return {}

    coex = coexpr_matrix(S)
    A0 = torch.zeros_like(S.A)
    rng = np.random.default_rng(args.seed)
    # A-shuffled null: average per-TF AUROC over >=3 independent shuffles (was 1)
    n_bshuf = max(int(getattr(args, "bio_n_shuffle", 5)), 3)
    A_shufs = [make_variant(S.A, "shuffled",
                            rng=np.random.default_rng(args.seed + 200000 + k),
                            match_spectral=args.match_spectral) for k in range(n_bshuf)]
    cr_orig = ctrl_recon(S.op, S.data, S.A)
    cr_A0 = ctrl_recon(S.op, S.data, A0)
    cr_shufs = [ctrl_recon(S.op, S.data, Av) for Av in A_shufs]
    print(f"[bio] A-shuffled null averaged over {n_bshuf} shuffles per TF", flush=True)

    n = len(S.gene_names)
    results = {}
    per_tf = {}                                  # rname -> {predictor: [per-TF AUROC]}
    PREDS = ["A original", "A shuffled", "A = 0 (no propagation)", "co-expression"]
    for rname, tgt in refs.items():
        # eval TFs = sources with >=5 in-axis targets AND a WM shift available (screened set is safest)
        tfs = [i for i, ts in tgt.items() if len(ts) >= 5]
        rng.shuffle(tfs)
        if args.max_perts:
            tfs = tfs[:max(args.max_perts, 60)]
        per = {p: [] for p in PREDS}
        for tf in tfs:
            pos = np.zeros(n, np.int8)
            for j in tgt[tf]:
                pos[j] = 1
            pos[tf] = 0
            keep = np.ones(n, bool); keep[tf] = False
            if pos[keep].sum() < 3:
                continue
            try:
                pd_o = pred_delta(S.op, S.data, [tf], S.A, cr_orig).cpu().numpy()
                pd_0 = pred_delta(S.op, S.data, [tf], A0, cr_A0).cpu().numpy()
                a_o = roc_auc_score(pos[keep], np.abs(pd_o)[keep])
                a_0 = roc_auc_score(pos[keep], np.abs(pd_0)[keep])
                a_sh = np.mean([roc_auc_score(
                    pos[keep], np.abs(pred_delta(S.op, S.data, [tf], A_shufs[k], cr_shufs[k]).cpu().numpy())[keep])
                    for k in range(n_bshuf)])
                a_ce = roc_auc_score(pos[keep], np.abs(coex[tf])[keep])
            except Exception:
                continue
            per["A original"].append(a_o); per["A shuffled"].append(a_sh)
            per["A = 0 (no propagation)"].append(a_0); per["co-expression"].append(a_ce)
        per_tf[rname] = {p: np.asarray(v, float) for p, v in per.items()}
        results[rname] = {p: (float(np.mean(v)) if v else np.nan) for p, v in per.items()}
        results[rname]["n_tfs"] = len(per["A original"])
        print(f"[bio] {rname}: " + "  ".join(
            f"{p}={results[rname][p]:.3f}" for p in PREDS) +
            f"  (n_tf={results[rname]['n_tfs']})", flush=True)

    # ── paired Wilcoxon across per-TF AUROCs (A original vs shuffled / vs A=0) + BH across DBs ──
    tests = []                                   # (ref, comparison, wilcoxon-dict)
    for rname in results:
        w_sh = paired_wilcoxon(per_tf[rname]["A original"], per_tf[rname]["A shuffled"],
                               alternative="two-sided")
        w_a0 = paired_wilcoxon(per_tf[rname]["A original"], per_tf[rname]["A = 0 (no propagation)"],
                               alternative="two-sided")
        tests.append((rname, "A_original_vs_shuffled", w_sh))
        tests.append((rname, "A_original_vs_A0", w_a0))
    qvals = bh_qvalues([t[2]["p"] for t in tests])
    bio_stats = {}
    for (rname, comp, w), q in zip(tests, qvals):
        bio_stats.setdefault(rname, {})[comp] = dict(
            wilcoxon_p=w["p"], bh_q=float(q), n_pairs=w["n_pairs"],
            median_diff=w["median_diff"], mean_diff=w["mean_diff"])
        print(f"[bio-STATS] {rname:16s} {comp:26s} "
              f"median Δ(orig−ref)={w['median_diff']:+.4f}  Wilcoxon p={w['p']:.3g}  BH-q={q:.3g}  "
              f"(n_tf={w['n_pairs']})  {'orig HIGHER' if w['median_diff']>0 else 'orig LOWER/equal'}",
              flush=True)
    results["_stats"] = bio_stats

    preds = ["A original", "A shuffled", "A = 0 (no propagation)", "co-expression"]
    pcols = [PAL["blue"], PAL["red"], PAL["yellow"], PAL["orange"]]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ref_names = [r for r in results.keys() if r != "_stats"]
    vals = [[results[r].get(p, np.nan) for r in ref_names] for p in preds]
    _bars(ax, ref_names, preds, vals, pcols,
          "mean per-TF target-recovery AUROC",
          "Do-operator single-TF predictions recover known TF targets above chance\n"
          "(canonical-target recovery does not require A propagation)",
          ylim=(0.4, 0.85), chance=0.5, chance_label="random", rot=0)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig3_do_operator_target_recovery.png")
    fig.savefig(out); plt.close(fig)
    print(f"[bio] saved {out}", flush=True)
    json.dump(results, open(os.path.join(OUT_DIR, "bio_target_recovery_results.json"), "w"), indent=2)
    return results


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--singles_path", default=DEFAULT_SINGLES)
    ap.add_argument("--combo_path", default=DEFAULT_COMBO)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_dag", type=int, default=16)
    ap.add_argument("--n_ctrl", type=int, default=500)
    ap.add_argument("--analyses", default="all", help="comma list: edge,grn,bio or 'all'")
    ap.add_argument("--max_perts", type=int, default=150, help="cap perts/TFs per group for speed (0=all)")
    ap.add_argument("--n_shuffle", type=int, default=1000,
                    help="shuffled-A null draws for the edge-direction permutation p (>=1000 for a valid tail-p)")
    ap.add_argument("--n_disc", type=int, default=25,
                    help="how many of the null draws also get the (slower) discrimination metrics for the bar means")
    ap.add_argument("--bio_n_shuffle", type=int, default=5,
                    help="shuffled-A draws to average per-TF AUROC in the bio target-recovery null (>=3)")
    ap.add_argument("--match_spectral", action="store_true", default=True,
                    help="rescale variants to ‖A‖₂ so magnitude is not a confound")
    ap.add_argument("--no_match_spectral", dest="match_spectral", action="store_false")
    ap.add_argument("--dump_null_json", default="",
                    help="OPT-IN: if set, dump the SINGLE-TF full null_means array + per-TF "
                         "original/representative-shuffled pearson_per to this path "
                         "(evidence for the permutation-null histogram + paired scatter figure; "
                         "does NOT change any existing output)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.max_perts == 0:
        args.max_perts = None
    todo = ["edge", "grn", "bio"] if args.analyses == "all" else \
        [a.strip() for a in args.analyses.split(",") if a.strip()]

    # Reproducibility: op.mean_A() -> DAGModule._spectral_clamp() estimates ‖A‖₂ by an
    # UNSEEDED power iteration (dag.py:152, v=torch.randn), so the eval-mode A picks up a
    # slightly different global scale each run and the (I-Aᵀ)⁻¹ solve amplifies it into a
    # ~0.04 drift in the original-A Pearson. Seed the global torch/numpy RNG so the released
    # numbers (single orig≈+0.536, Δ≈+0.69) reproduce deterministically. Same seed-0 the
    # standalone panel builder (paper/figures/make_fig4B_latentA.py) already pins.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    S = build(args)
    summary = {}
    if "edge" in todo:
        summary["edge"] = analysis_edge(S, args)
    if "grn" in todo:
        summary["grn"] = analysis_grn(S, args)
    if "bio" in todo:
        summary["bio"] = analysis_bio(S, args)
    print("\n==== DONE ====  figures in", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()

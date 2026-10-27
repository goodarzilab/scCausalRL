"""Four validation diagnostics on the WM-search vs additive_baseline result (per task):
  1. Self-match leak fix: exclude pool-genes from the target DEG mask; re-rank; report GATA2 move.
  2. Full recall@k curve (k=10/20/50/100), P1/P2/Lange, both methods, both targets; name recovered P1;
     cap P2 by pool coverage.
  3. Rank stability: 20 single-DAG draws (structural uncertainty, conservative) + 50 DEG-mask bootstraps;
     median rank + 95% CI per key TF; score-distribution plot (are the 118 scores near-tied?).
  4. Direct TAL1xLMO2 / TAL1xETV2 synergy: rho = D_pair - (D_a + D_b); ||rho||/||D_pair|| and
     ||S_int||/||S_singles||; null over 500 random pairs (upper-tail?). Training-coverage: none of these
     TFs are in ANY of the 57 Joung training combos -> failure is extrapolation, not architecture.
"""
import os, sys, itertools, argparse
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack
from core.scoring import torch_match
from analysis.briggs_additive_vs_planner import (
    additive_fit, rank_of, recall_at_k, build_cao_target,
    P1, P2, LANGE, CANON, EPISTATIC, MARKER_KEEP, MARKER_EXCL, EXP20, H17, BRIGGS_SUP)

FIG = f'{EXP20}/results/briggs/figures'


def wm_delta(op, eps0, ctrl_recon, sets, A, bs=512):
    out = []
    for s in range(0, len(sets), bs):
        z = op.apply_sets(eps0, sets[s:s + bs], A)
        out.append((op.decode(z) - ctrl_recon).cpu())
    return torch.cat(out, 0)


def wm_ranking(DELTA, cand_tfs, allsets, gene_names, y, sel, topk_cocktails=100):
    """TF ranking = selection frequency in the top-K WM-scored cocktails (tie-break: best containing match)."""
    yt = torch.tensor(y, dtype=torch.float32); sel_t = torch.tensor(sel, dtype=torch.long)
    match = torch_match(DELTA, yt[sel_t], sel_t).numpy()
    order = np.argsort(-match)[:topk_cocktails]
    freq = {t: 0 for t in cand_tfs}; best = {t: -9.0 for t in cand_tfs}
    for si, st in enumerate(allsets):
        for gi in st:
            g = gene_names[gi]
            if g in best and match[si] > best[g]:
                best[g] = float(match[si])
    for oi in order:
        for gi in allsets[oi]:
            g = gene_names[gi]
            if g in freq:
                freq[g] += 1
    ranked = sorted(cand_tfs, key=lambda t: (-freq[t], -best[t]))
    return ranked, freq, best, match


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    ap.add_argument('--n_seeds', type=int, default=20); ap.add_argument('--n_boot', type=int, default=50)
    ap.add_argument('--n_null', type=int, default=500)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    gene_names = list(map(str, data.gene_names)); g2i = data.gene_to_idx

    iso = pd.concat([pd.read_excel(f'{BRIGGS_SUP}/mmc5.xlsx', s)['TF'] for s in ['Pool A', 'Pool B', 'Pool C', 'Pool D']]).astype(str)
    briggs_morf = set(iso.str.split('|').str[0].str.strip())
    cand_tfs = sorted(briggs_morf & set(data.de_single))
    cand_idx = [g2i[t] for t in cand_tfs]; pool_gene_set = set(cand_idx)
    D = np.stack([np.asarray(data.de_single[t]['lfc'], np.float32) for t in cand_tfs], axis=1)

    eps0 = data.eps_ctrl_mean.to(dev); A = op.mean_A(); ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]
    singles = [[i] for i in cand_idx]
    pairs = [[i, j] for i, j in itertools.combinations(cand_idx, 2)]
    allsets = singles + pairs

    # ---- targets with LEAK FIX (#1): drop any pool-gene from the DEG mask ----
    cao = pd.read_csv(f'{H17}/cao_type_mean_logexpr.csv', index_col=0); cao.columns = cao.columns.astype(str)
    tgt_hsc, _ = build_cao_target(cao, gene_names, 'Liver-Hematopoietic stem cells')
    thr = data.thresh

    def leakfix(sel):
        excl = sorted({gene_names[int(i)] for i in sel if int(i) in pool_gene_set})
        keep = np.array([int(i) for i in sel if int(i) not in pool_gene_set])
        return keep, excl

    sel_cao_raw = np.where(np.abs(tgt_hsc) > thr)[0]
    sel_cao, excl_cao = leakfix(sel_cao_raw)
    sel_mk_raw = np.array(sorted(g2i[g] for g in MARKER_KEEP if g in g2i and g not in MARKER_EXCL))
    sel_mk, excl_mk = leakfix(sel_mk_raw)
    targets = {'Cao-HSC': (tgt_hsc, sel_cao), 'marker-relaxed': (tgt_hsc, sel_mk)}
    print("===== #1 · SELF-MATCH LEAK FIX =====", flush=True)
    print(f"Cao-HSC: {len(sel_cao_raw)}->{len(sel_cao)} mask genes | excluded {len(excl_cao)} pool-genes: {excl_cao}", flush=True)
    print(f"marker-relaxed: {len(sel_mk_raw)}->{len(sel_mk)} mask genes | excluded pool-genes: {excl_mk}", flush=True)
    print(f"  (marker-relaxed effective target genes now: {[gene_names[i] for i in sel_mk]})", flush=True)

    DELTA = wm_delta(op, eps0, ctrl_recon, allsets, A)               # mean_A point estimate

    # ---- #1 + #2: rankings + recall@k on leak-free targets ----
    OLD = {'Cao-HSC': {'GATA2': 16}, 'marker-relaxed': {'GATA2': 114}}   # leaky (stage-2) for comparison
    print("\n===== #2 · FULL recall@k (LEAK-FREE targets) =====", flush=True)
    for tname, (y, sel) in targets.items():
        # additive
        wabs, _, _ = additive_fit(D, y, sel)
        add_ranked = [cand_tfs[i] for i in np.argsort(-wabs)]
        # WM
        wm_ranked, freq, best, match = wm_ranking(DELTA, cand_tfs, allsets, gene_names, y, sel)
        print(f"\n--- {tname} ---", flush=True)
        for mname, ranked in [('additive', add_ranked), ('WM-search', wm_ranked)]:
            line = " ".join(f"{t}={rank_of(ranked, t)}" for t in EPISTATIC)
            rk = {k: {nm: recall_at_k(ranked, key, k) for nm, key in [('P1', P1), ('P2', P2), ('Lange', LANGE)]}
                  for k in (10, 20, 50, 100)}
            print(f"  {mname:9s} ranks: {line}", flush=True)
            print(f"    recall@k  " + " | ".join(
                f"k{k}: P1 {rk[k]['P1']:.2f} P2 {rk[k]['P2']:.2f} Lange {rk[k]['Lange']:.2f}" for k in (10, 20, 50, 100)), flush=True)
        # which P1 recovered by WM (top-20); P2 pool cap
        p1_prop = [t for t in P1 if t in cand_tfs]; p2_prop = [t for t in P2 if t in cand_tfs]
        print(f"  WM top-20 ∩ P1: {sorted(set(wm_ranked[:20]) & set(P1))} | P1 proposable={p1_prop}", flush=True)
        print(f"  P2 proposable={len(p2_prop)}/6 (missing {sorted(set(P2)-set(cand_tfs))}) -> max P2 recall=1.0 over {len(p2_prop)}", flush=True)
        print(f"  GATA2 rank: leaky {OLD[tname]['GATA2']} -> leak-free {rank_of(wm_ranked,'GATA2')}", flush=True)

    # ---- #3: stability (Cao-HSC) — PROPER: each replicate is an N-DAG-AVERAGED score (the deployed
    # estimator), NOT a single draw. Single-DAG SD is ~sqrt(N)x too large. Report N=8 (deployed) and N=64. ----
    print("\n===== #3 · RANK STABILITY (Cao-HSC) — N-averaged estimator =====", flush=True)
    y, sel = targets['Cao-HSC']
    KEYTFS = EPISTATIC + [t for t in (P1 + P2) if t in cand_tfs]
    KEYTFS = list(dict.fromkeys(KEYTFS))

    def dag_avg_delta(N, seed):
        torch.manual_seed(seed)
        acc = None
        for _ in range(N):
            Ak = op.sample_dags(1)[0]
            Dk = wm_delta(op, eps0, ctrl_recon, allsets, Ak)
            acc = Dk if acc is None else acc + Dk
        return acc / N

    dag_ci = {}
    for N in (8, 64):
        ranks_N = {t: [] for t in KEYTFS}
        for rep in range(a.n_seeds):
            Dr = dag_avg_delta(N, seed=2000 + N * 1000 + rep)
            rk, _, _, _ = wm_ranking(Dr, cand_tfs, allsets, gene_names, y, sel)
            for t in KEYTFS: ranks_N[t].append(rank_of(rk, t))
        dag_ci[N] = ranks_N
        print(f"  [N={N} averaged, {a.n_seeds} replicates] computed", flush=True)
    # mask bootstrap (reuse mean_A DELTA)
    boot_ranks = {t: [] for t in KEYTFS}
    rng = np.random.default_rng(0)
    for _ in range(a.n_boot):
        selb = rng.choice(sel, size=len(sel), replace=True)
        rk, _, _, _ = wm_ranking(DELTA, cand_tfs, allsets, gene_names, y, selb)
        for t in KEYTFS: boot_ranks[t].append(rank_of(rk, t))

    def ci(arr):
        a_ = np.array(arr); return f"{int(np.median(a_)):3d} [{int(np.percentile(a_,2.5)):3d},{int(np.percentile(a_,97.5)):3d}]"
    print(f"{'TF':8s} {'N=8 med[CI]':16s} {'N=64 med[CI]':16s} {'mask-boot med[CI]':18s}", flush=True)
    for t in KEYTFS:
        print(f"{t:8s} {ci(dag_ci[8][t]):16s} {ci(dag_ci[64][t]):16s} {ci(boot_ranks[t]):18s}", flush=True)
    # score-distribution plot: best-match-per-TF sorted
    _, _, best, _ = wm_ranking(DELTA, cand_tfs, allsets, gene_names, y, sel)
    bvals = np.sort([best[t] for t in cand_tfs])[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(bvals) + 1), bvals, '-o', ms=3, color='#2a78d6')
    ax.set_xlabel('TF rank (1-118)'); ax.set_ylabel('best cocktail match (Pearson on mask)')
    ax.set_title('WM-search Cao-HSC: best-match-per-TF across ranks (tightness of the landscape)')
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    fig.savefig(f'{FIG}/fig14_wm_rank_scores.png', dpi=150); print(f"saved -> {FIG}/fig14_wm_rank_scores.png", flush=True)

    # ---- #4: direct synergy TAL1xLMO2, TAL1xETV2 + null ----
    print("\n===== #4 · DIRECT SYNERGY (extrapolation: none of these TFs in the 57 training combos) =====", flush=True)
    has_head = getattr(model, 'interaction_U', None) is not None
    print(f"interaction head present: {has_head}", flush=True)

    def synergy(a_tf, b_tf):
        ai, bi = g2i[a_tf], g2i[b_tf]
        Da = (op.decode(op.apply_sets(eps0, [[ai]], A)) - ctrl_recon)[0]
        Db = (op.decode(op.apply_sets(eps0, [[bi]], A)) - ctrl_recon)[0]
        Dp = (op.decode(op.apply_sets(eps0, [[ai, bi]], A)) - ctrl_recon)[0]
        rho = Dp - (Da + Db)
        frac = float(rho.norm() / (Dp.norm() + 1e-9))
        gi = torch.as_tensor([ai, bi], device=dev)
        S_int = model.compute_interaction(gi); S_sing = op.shift_for([ai, bi]).sum(0)
        sratio = float(S_int.norm() / (S_sing.norm() + 1e-9))
        return frac, sratio

    # null distribution over random pairs
    rng = np.random.default_rng(1)
    null_frac, null_sr = [], []
    for _ in range(a.n_null):
        i, j = rng.choice(cand_idx, 2, replace=False)
        Da = (op.decode(op.apply_sets(eps0, [[int(i)]], A)) - ctrl_recon)[0]
        Db = (op.decode(op.apply_sets(eps0, [[int(j)]], A)) - ctrl_recon)[0]
        Dp = (op.decode(op.apply_sets(eps0, [[int(i), int(j)]], A)) - ctrl_recon)[0]
        null_frac.append(float((Dp - Da - Db).norm() / (Dp.norm() + 1e-9)))
        gi = torch.as_tensor([int(i), int(j)], device=dev)
        null_sr.append(float(model.compute_interaction(gi).norm() / (op.shift_for([int(i), int(j)]).sum(0).norm() + 1e-9)))
    null_frac = np.array(null_frac); null_sr = np.array(null_sr)
    print(f"null (n={a.n_null} random pairs): rho-frac median {np.median(null_frac):.3f} [{np.percentile(null_frac,5):.3f},{np.percentile(null_frac,95):.3f}] | "
          f"S_int/S_sing median {np.median(null_sr):.3f} [{np.percentile(null_sr,5):.3f},{np.percentile(null_sr,95):.3f}]", flush=True)
    for a_tf, b_tf in [('TAL1', 'LMO2'), ('TAL1', 'ETV2')]:
        frac, sratio = synergy(a_tf, b_tf)
        pf = float((null_frac < frac).mean()); ps = float((null_sr < sratio).mean())
        print(f"  {a_tf}x{b_tf}: rho-frac={frac:.3f} (pctile {pf*100:.0f}) | S_int/S_sing={sratio:.3f} (pctile {ps*100:.0f})", flush=True)
    print("  interpretation: pctile>=95 -> specific synergy; ~50 -> generic magnitude; rho~0 -> effectively additive", flush=True)

    # ---- #5: COVERAGE DOSE-RESPONSE — does the interaction head engage more on TRAINED pairs? ----
    # Split pairs into IN-coverage (present in the 57 training combos) vs OUT-of-coverage; compare the
    # non-additivity (rho-frac) and head magnitude (S_int/S_singles). If in-coverage >> out (and HSPC ~ out),
    # the head generalizes only within pair-training coverage -> "57 pairs is not enough" (methods result).
    print("\n===== #5 · COVERAGE DOSE-RESPONSE (learned-interaction generalization vs pair-training) =====", flush=True)
    import re, anndata as ad
    C = ad.read_h5ad(a.combo_path, backed='r')
    train_pairs = set()
    for ts in set(map(str, C.obs['TF_set'].values)):
        genes = [p.split('-')[-1] for p in ts.split(',')]              # "TFORF0098-MSGN1" -> "MSGN1"
        genes = [g for g in genes if g in g2i]
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                train_pairs.add(frozenset((genes[i], genes[j])))
    train_pairs = [tuple(p) for p in train_pairs]
    print(f"in-coverage pairs (from 57 training combos, both TFs in gene space): {len(train_pairs)}", flush=True)

    def pair_metrics(a_tf, b_tf):
        ai, bi = g2i[a_tf], g2i[b_tf]
        Da = (op.decode(op.apply_sets(eps0, [[ai]], A)) - ctrl_recon)[0]
        Db = (op.decode(op.apply_sets(eps0, [[bi]], A)) - ctrl_recon)[0]
        Dp = (op.decode(op.apply_sets(eps0, [[ai, bi]], A)) - ctrl_recon)[0]
        frac = float((Dp - Da - Db).norm() / (Dp.norm() + 1e-9))
        gi = torch.as_tensor([ai, bi], device=dev)
        sr = float(model.compute_interaction(gi).norm() / (op.shift_for([ai, bi]).sum(0).norm() + 1e-9))
        return frac, sr

    grp = {}
    grp['in-coverage'] = [pair_metrics(*p) for p in train_pairs]
    # out-of-coverage: random pairs NOT co-occurring in training, over the same de_single TF universe
    de_universe = sorted(set(data.de_single))
    trainset = set(train_pairs) | {(b, a) for a, b in train_pairs}
    rng2 = np.random.default_rng(7); oc = []
    while len(oc) < a.n_null:
        i, j = rng2.choice(de_universe, 2, replace=False)
        if (i, j) not in trainset and (j, i) not in trainset:
            oc.append(pair_metrics(str(i), str(j)))
    grp['out-of-coverage'] = oc
    grp['HSPC-canonical'] = [pair_metrics(a_, b_) for a_, b_ in
                             [('TAL1', 'LMO2'), ('TAL1', 'ETV2'), ('GATA2', 'ETV2'), ('GATA2', 'TAL1'), ('LMO2', 'ETV2'), ('GATA2', 'LMO2')]]
    print(f"{'group':16s} {'n':>4s} {'rho-frac med[IQR]':24s} {'S_int/S_sing med[IQR]':24s}", flush=True)
    for gname, vals in grp.items():
        rf = np.array([v[0] for v in vals]); sr = np.array([v[1] for v in vals])
        print(f"{gname:16s} {len(vals):4d} "
              f"{np.median(rf):.3f} [{np.percentile(rf,25):.3f},{np.percentile(rf,75):.3f}]      "
              f"{np.median(sr):.3f} [{np.percentile(sr,25):.3f},{np.percentile(sr,75):.3f}]", flush=True)
    # dose-response percentile: where do in-coverage / HSPC sit in the out-of-coverage null?
    oc_rf = np.array([v[0] for v in grp['out-of-coverage']]); oc_sr = np.array([v[1] for v in grp['out-of-coverage']])
    for gname in ['in-coverage', 'HSPC-canonical']:
        rf = np.array([v[0] for v in grp[gname]]); sr = np.array([v[1] for v in grp[gname]])
        p_rf = float((oc_rf[None, :] < rf[:, None]).mean()) * 100
        p_sr = float((oc_sr[None, :] < sr[:, None]).mean()) * 100
        print(f"  {gname}: mean rho-frac pctile vs out-of-cov = {p_rf:.0f} | mean S_int/S_sing pctile = {p_sr:.0f}", flush=True)
    # plot the dose-response distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mi, lab in [(axes[0], 0, 'rho-frac (gene-space non-additivity)'), (axes[1], 1, 'S_int/S_singles (head magnitude)')]:
        ax.boxplot([[v[mi] for v in grp[g]] for g in grp], labels=list(grp.keys()), showfliers=False)
        ax.set_title(lab); ax.tick_params(axis='x', rotation=15)
    fig.suptitle('Coverage dose-response: interaction-head engagement by pair-training coverage')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig15_coverage_doseresponse.png', dpi=150)
    print(f"saved -> {FIG}/fig15_coverage_doseresponse.png", flush=True)
    print("  RULE: in-coverage >> out-of-coverage AND HSPC ~ out-of-coverage => head generalizes only within "
          "pair-training coverage (57 pairs insufficient) -> methods contribution, not architecture failure.", flush=True)
    print("\n[diag done]", flush=True)


if __name__ == '__main__':
    main()

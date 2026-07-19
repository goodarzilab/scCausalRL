"""Is the projected latent-DAG GRN (G = D·A·Dᵀ) a real causal graph, or CollecTRI memorization?

The CollecTRI prior (margin-ranking, λ=10) actively pushes G[TF,target] above background during
training, so a high CollecTRI AUROC is EXPECTED and by itself proves nothing. The real questions,
addressed by controls here:
  - AUROC vs CollecTRI (TF rows)                  — memorization upper bound (prior target)
  - shuffled-A AUROC (permute DAG columns)        — does the DAG carry the AUROC, or just D + prior?
  - random-D AUROC                                — control projection
  - control gene-gene CORRELATION GRN AUROC       — does the DAG beat plain co-expression?
  - DoRothEA-minus-CollecTRI AUROC (HELD-OUT)     — generalization beyond the trained prior
  - latent-DAG sparsity/density/degree/spec-radius/acyclicity
  - known master-TF target ranks + GSEA(Hallmark) on each master TF's predicted targets

Run: python evaluation/grn_analysis.py [--checkpoint ...]
"""
import os, sys, json, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
EXP19 = ROOT
DEF_CKPT = f'{EXP19}/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt'
SINGLES = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
OUT = f'{HERE}/../results/grn_analysis'; os.makedirs(OUT, exist_ok=True)
MASTER_TFS = ['FLI1', 'MSGN1', 'GATA1', 'GATA4', 'NEUROD1', 'MYOD1', 'CDX2', 'HNF4A', 'TBX5', 'SPI1',
              'PAX6', 'SOX2', 'RUNX1', 'FOXA2', 'TAL1', 'GRHL3']


def auroc_tfrows(score_tf, pair_set, tf_rows, n_genes):
    """AUROC over the TF-row submatrix: y=1 for (tf,target) in pair_set. score_tf [n_tf,n_genes]."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.zeros(score_tf.shape, dtype=np.int8)
    rowpos = {r: k for k, r in enumerate(tf_rows)}
    npos = 0
    for (ti, gi) in pair_set:
        if ti in rowpos and gi < n_genes:
            y[rowpos[ti], gi] = 1; npos += 1
    if npos == 0 or y.sum() == y.size:
        return None, None, npos
    yf, sf = y.ravel(), score_tf.ravel()
    return float(roc_auc_score(yf, sf)), float(average_precision_score(yf, sf)), int(npos)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', default=DEF_CKPT)
    ap.add_argument('--n_ctrl', type=int, default=3000); ap.add_argument('--n_perm', type=int, default=10)
    a = ap.parse_args(); dev = 'cpu'; torch.set_grad_enabled(False)
    sys.path.insert(0, EXP19)
    from evaluation.run_shift_eval_norman import load_shift_model
    import decoupler as dc
    model, gene_names, ckpt = load_shift_model(a.checkpoint, dev)
    gene_names = [str(g) for g in gene_names]; n = len(gene_names)
    up2i = {g.upper(): i for i, g in enumerate(gene_names)}
    D = model.decoder.D.detach().cpu().numpy().astype(np.float64)          # [n, d]
    A = model.dag.get_edge_probs().detach().cpu().numpy().astype(np.float64)  # [d, d]
    d = A.shape[0]
    print(f"[grn] WM={os.path.basename(os.path.dirname(os.path.dirname(a.checkpoint)))} | genes={n} d={d}")

    # ---- networks ----
    ct = dc.op.collectri(organism='human'); dor = dc.op.dorothea(organism='human', levels=['A', 'B'])
    def pairs(net):
        out = set()
        for s, t in zip(net['source'].astype(str).str.upper(), net['target'].astype(str).str.upper()):
            if s in up2i and t in up2i and up2i[s] != up2i[t]:
                out.add((up2i[s], up2i[t]))
        return out
    ct_pairs = pairs(ct); dor_pairs = pairs(dor); held = dor_pairs - ct_pairs   # DoRothEA not in CollecTRI
    tf_rows = sorted({ti for ti, _ in ct_pairs} | {ti for ti, _ in held})
    Dtf = D[tf_rows]                                                       # [n_tf, d]
    print(f"[grn] CollecTRI pairs in-axis={len(ct_pairs)} | DoRothEA held-out (not in CollecTRI)={len(held)} | TF rows={len(tf_rows)}")

    def Gtf(Dr, Aa, Dall):                                                 # TF-row GRN block
        return np.abs(Dr @ Aa @ Dall.T)
    G_tf = Gtf(Dtf, A, D)

    res = {}
    res['collectri_auroc'], res['collectri_auprc'], _ = auroc_tfrows(G_tf, ct_pairs, tf_rows, n)
    res['dorothea_heldout_auroc'], res['dorothea_heldout_auprc'], res['n_heldout'] = auroc_tfrows(G_tf, held, tf_rows, n)

    # ---- controls: shuffled-A (permute cols), random-D ----
    rng = np.random.default_rng(0); sh, rd = [], []
    for _ in range(a.n_perm):
        Ap = A[:, rng.permutation(d)]
        sh.append(auroc_tfrows(Gtf(Dtf, Ap, D), ct_pairs, tf_rows, n)[0])
        Q, _ = np.linalg.qr(rng.standard_normal((n, d)))                   # random orthonormal D'
        rd.append(auroc_tfrows(Gtf(Q[tf_rows], A, Q), ct_pairs, tf_rows, n)[0])
    res['collectri_auroc_shuffledA'] = (float(np.mean(sh)), float(np.std(sh)))
    res['collectri_auroc_randomD'] = (float(np.mean(rd)), float(np.std(rd)))

    # ---- control gene-gene correlation GRN (does the DAG beat co-expression?) ----
    try:
        import anndata as ad, scipy.sparse as sp
        sys.path.insert(0, '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas')
        from data.load_joung import _lognorm
        S = ad.read_h5ad(SINGLES); ctrl = np.where(S.obs['is_control'].values.astype(bool))[0]
        sel = rng.choice(ctrl, min(a.n_ctrl, len(ctrl)), replace=False)
        Xc = np.asarray(_lognorm(S.X)[sel].todense(), np.float64)          # [n_ctrl, n]
        Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
        corr_tf = np.abs((Xc[:, tf_rows].T @ Xc) / Xc.shape[0])           # [n_tf, n]
        res['collectri_auroc_corrGRN'] = auroc_tfrows(corr_tf, ct_pairs, tf_rows, n)[0]
        res['dorothea_heldout_auroc_corrGRN'] = auroc_tfrows(corr_tf, held, tf_rows, n)[0]
    except Exception as e:
        print(f"[grn] corr-GRN skipped: {e}"); res['collectri_auroc_corrGRN'] = None

    # ---- latent-DAG sparsity / topology ----
    A_hard = (model.dag.sample_A(n_samples=1).squeeze(0).detach().cpu().numpy() if hasattr(model.dag, 'sample_A') else (np.abs(A) > 1e-3).astype(float))
    nz = np.abs(A) > 1e-3
    eig = np.abs(np.linalg.eigvals(A))
    res['latent_dag'] = {'d': d, 'density_prob>1e-3': float(nz.mean()), 'n_edges': int(nz.sum()),
                         'mean_in_degree': float(nz.sum(0).mean()), 'mean_out_degree': float(nz.sum(1).mean()),
                         'max_out_degree': int(nz.sum(1).max()), 'spectral_radius': float(eig.max()),
                         'hard_gate_density': float((np.abs(A_hard) > 1e-6).mean()),
                         'G_feat_effective_rank': int(np.linalg.matrix_rank(D @ A @ D.T, tol=1e-6))}

    # ---- known master-TF target ranks (within each TF's G_feat row) + top targets ----
    ct_by_tf = {}
    for ti, gi in ct_pairs:
        ct_by_tf.setdefault(ti, set()).add(gi)
    known = {}
    for tf in MASTER_TFS:
        i = up2i.get(tf)
        if i is None: continue
        row = np.abs(D[i] @ A @ D.T)                                       # [n]
        order = np.argsort(-row); rankpos = {g: r for r, g in enumerate(order)}
        top = [gene_names[j] for j in order[:8] if j != i][:6]
        tgts = ct_by_tf.get(i, set())
        med_pct = float(np.median([1 - rankpos[g] / n for g in tgts])) if tgts else None
        known[tf] = {'n_collectri_targets': len(tgts), 'median_target_percentile': med_pct, 'top6_predicted': top}
    res['known_master_tfs'] = known

    # ---- GSEA (Hallmark) on a few master-TF predicted-target vectors ----
    try:
        import gseapy as gp, pandas as pd
        hm = dc.op.hallmark(organism='human'); gs = {k: list(v) for k, v in hm.groupby('source')['target'].apply(list).items()}
        gsea = {}
        for tf in ['FLI1', 'MSGN1', 'NEUROD1', 'HNF4A']:
            i = up2i.get(tf)
            if i is None: continue
            rnk = pd.Series(D[i] @ A @ D.T, index=[g.upper() for g in gene_names]).groupby(level=0).max().sort_values(ascending=False)
            try:
                pre = gp.prerank(rnk=rnk.reset_index(), gene_sets=gs, min_size=10, max_size=500,
                                 permutation_num=200, seed=0, no_plot=True, outdir=None, threads=4)
                t = pre.res2d.sort_values('NES', ascending=False).head(3)
                gsea[tf] = [(r['Term'], round(float(r['NES']), 2), float(r['FDR q-val'])) for _, r in t.iterrows()]
            except Exception as e:
                gsea[tf] = f"prerank failed: {str(e)[:80]}"
        res['gsea_hallmark_top3'] = gsea
    except Exception as e:
        print(f"[grn] GSEA skipped: {e}"); res['gsea_hallmark_top3'] = None

    json.dump(res, open(f'{OUT}/grn_analysis.json', 'w'), indent=2, default=str)
    print("\n================ GRN ANALYSIS ================")
    print(f"CollecTRI AUROC (real)        : {res['collectri_auroc']:.3f}  AUPRC {res['collectri_auprc']:.3f}")
    print(f"  shuffled-A control          : {res['collectri_auroc_shuffledA'][0]:.3f} ± {res['collectri_auroc_shuffledA'][1]:.3f}")
    print(f"  random-D control            : {res['collectri_auroc_randomD'][0]:.3f} ± {res['collectri_auroc_randomD'][1]:.3f}")
    print(f"  corr-GRN baseline           : {res.get('collectri_auroc_corrGRN')}")
    print(f"DoRothEA HELD-OUT AUROC (real): {res['dorothea_heldout_auroc']:.3f}  (corr-GRN {res.get('dorothea_heldout_auroc_corrGRN')})  n={res['n_heldout']}")
    print(f"latent DAG: {res['latent_dag']}")
    print(f"known master-TF median target percentile:")
    for tf, v in known.items():
        if v['median_target_percentile'] is not None:
            print(f"  {tf:8s} pct={v['median_target_percentile']:.3f} (n={v['n_collectri_targets']})  top: {v['top6_predicted']}")
    print(f"GSEA Hallmark: {res['gsea_hallmark_top3']}")
    print(f"\nsaved -> {OUT}/grn_analysis.json")


if __name__ == '__main__':
    main()

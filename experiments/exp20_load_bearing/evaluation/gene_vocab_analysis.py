"""Gene-vocabulary analysis: does restricting to TF-responsive genes raise the discrimination ceiling?

Joung et al. motivation: only ~27% of TFs significantly move the transcriptome, and the paper's own
discriminative analyses use 1-2k variable genes (not the 9923-HVG transcriptome). Our HVG vocab is
selected for TOTAL variance = dominated by the shared pluripotency->differentiation drift + housekeeping,
NOT for genes that discriminate BETWEEN TFs. Hypothesis: most of the 9923 genes carry drift+noise and
DILUTE the TF-specific residual, capping disc_resid at 0.672.

Per-gene TF-specificity (drift-removed, split-half):
  reliability[g]      = corr over TFs of (r_A[:,g], r_B[:,g])   -- does gene g's cross-TF response reproduce?
  between_var[g]      = var over TFs of mean residual           -- does gene g move differently across TFs?
  informative score   = reliability * sqrt(between_var)
Then the DECISIVE test: recompute the split-half disc_resid CEILING (r_A retrieves r_B across TFs, cosine)
restricted to the top-K most-informative genes, for K in a sweep. If the ceiling RISES as noise genes are
dropped then plateaus/falls, the peak K is the vocab size that maximizes achievable discrimination.

Run: python evaluation/gene_vocab_analysis.py --device cpu
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from data.load_joung import load_joung


def zrow(M):
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def disc_ceiling(rA, rB):
    """r_A[t] retrieves r_B[t] (same TF) over other TFs, by cosine. Chance 0.5."""
    ZA, ZB = zrow(rA), zrow(rB)
    C = ZA @ ZB.T
    diag = np.diag(C)[:, None]
    T = rA.shape[0]
    return float(((diag > C).sum(1) / (T - 1)).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--min_cells', type=int, default=60)
    ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dd = load_joung(n_hvg=None); de = dd['de_single']
    gn = np.asarray(dd['gene_names']); G = len(gn)
    X = dd['adata'].X; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); ctrl_mean = X[ctrl].mean(0)
    sp_map = dd['single_perts']; train = [g for g in dd['pert_split']['train'] if g in sp_map and g in de]
    rng = np.random.default_rng(0)

    drift = np.stack([np.asarray(de[g], np.float32) for g in train]).mean(0)
    tfs, rA, rB = [], [], []
    for g in train:
        idx = np.asarray(sp_map[g])
        if len(idx) < a.min_cells:
            continue
        idx = rng.permutation(idx); h = len(idx) // 2
        rA.append((X[idx[:h]].mean(0) - ctrl_mean) - drift)
        rB.append((X[idx[h:2 * h]].mean(0) - ctrl_mean) - drift)
        tfs.append(g)
    rA = np.stack(rA); rB = np.stack(rB); T = len(tfs)
    print(f"=== GENE VOCAB ANALYSIS === train TFs>={a.min_cells} cells: {T}, genes: {G}", flush=True)

    # per-gene reproducibility of the cross-TF residual response + between-TF spread
    rel = np.array([np.corrcoef(rA[:, g], rB[:, g])[0, 1] if np.std(rA[:, g]) > 1e-9 and np.std(rB[:, g]) > 1e-9
                    else 0.0 for g in range(G)])
    rel = np.nan_to_num(rel)
    rmean = (rA + rB) / 2
    between_var = rmean.var(0)                                   # variance across TFs of the residual
    score = rel * np.sqrt(np.clip(between_var, 0, None))          # informative = reproducible AND TF-variable
    order = np.argsort(-score)                                    # most informative first

    # baseline ceiling on all genes
    base = disc_ceiling(rA, rB)
    print(f"\nfull vocab ({G} genes): disc_resid ceiling = {base:.3f}", flush=True)
    print(f"\ndisc_resid CEILING vs curated vocab size (top-K by TF-specific reproducible signal):", flush=True)
    print(f"  {'K':>6} {'ceiling':>9} {'vs full':>8} {'cum.signal%':>12}", flush=True)
    tot_sig = float(np.clip(score, 0, None).sum())
    best = (0, base)
    for K in [200, 400, 600, 1000, 1500, 2000, 3000, 4000, 6000, 8000, G]:
        if K > G:
            continue
        gsel = order[:K]
        cval = disc_ceiling(rA[:, gsel], rB[:, gsel])
        cum = 100 * float(np.clip(score[gsel], 0, None).sum()) / (tot_sig + 1e-9)
        flag = "  <-- peak" if cval > best[1] else ""
        if cval > best[1]:
            best = (K, cval)
        print(f"  {K:>6} {cval:>9.3f} {cval-base:>+8.3f} {cum:>11.1f}%{flag}", flush=True)
    print(f"\nBEST: K={best[0]} genes -> ceiling {best[1]:.3f} (full={base:.3f}, gain {best[1]-base:+.3f})", flush=True)

    # how concentrated is the signal? genes for X% of reproducible TF-specific signal
    s = np.clip(score[order], 0, None); cum = np.cumsum(s) / (s.sum() + 1e-9)
    print(f"\nsignal concentration (genes for fraction of reproducible TF-specific signal):", flush=True)
    for thr in (0.5, 0.8, 0.9, 0.95):
        print(f"  {int(thr*100)}%: {int(np.searchsorted(cum, thr)+1)} genes", flush=True)
    print(f"\nreliability distribution: rel>0.3: {(rel>0.3).sum()} genes | rel>0.5: {(rel>0.5).sum()} | "
          f"rel>0.7: {(rel>0.7).sum()}  (of {G})", flush=True)
    # top informative genes (sanity check they're TF-effectors not housekeeping)
    print(f"\ntop 30 most TF-informative genes: {', '.join(gn[order[:30]])}", flush=True)
    # save the ranking for building the vocab
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'gene_vocab_ranked.npz')
    np.savez(out, genes=gn[order], score=score[order], reliability=rel[order], between_var=between_var[order])
    print(f"\nsaved ranked vocab -> {out}", flush=True)


if __name__ == '__main__':
    main()

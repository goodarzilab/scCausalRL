"""HSPC-similarity scorers for the Briggs 2026 benchmark + a de-risking verification.

Two STATE-based cosine scorers (Briggs's headline metric is cosine of a cell state to an HSPC centroid):
  (A) Briggs-native : centroid = mean lognorm profile of Briggs's own primary-HSPC cells (obs.sample=='pHSC').
  (B) Cao fetal-liver: 'Liver-Hematopoietic stem cells' deviation signature from cao_type_mean_logexpr.csv
      (peak-mean-centered over the 165 fetal cell types, discriminative half, z-scored) — the exp17 recipe.
Both score a lognorm expression STATE (control_mean + predicted Δ for model preds; the cell state for observed
cells), z-scored, cosine on the shared-gene intersection.

SMOKE (__main__): load the GEO all-cells h5ad, build both centroids, score every pHSC and pHep cell, and report
mean±sd pHSC vs pHep + AUROC separability + the cross-correlation of the two centroids' per-cell scores.
If AUROC(pHSC vs pHep) is not clearly >0.5 the metric is broken and the whole fate-eval premise fails — run this first.

Run: python evaluation/briggs/hspc_sim.py --h5ad <decompressed all_cells .h5ad>
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, anndata as ad
CAO = "/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/human_dev_annotation/cao_type_mean_logexpr.csv"
HSC_ROW = "Liver-Hematopoietic stem cells"


def lognorm(X):
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)


def zrow(v):  # z-score a 1-D profile
    v = np.asarray(v, np.float32); return (v - v.mean()) / (v.std() + 1e-8)


def build_cao_hsc_signature(model_genes):
    """Return (sig_vector, gene_idx_into_model) for the Cao HSC deviation signature on shared genes."""
    import pandas as pd
    df = pd.read_csv(CAO, index_col=0)                       # [165 celltypes x 6557 genes]
    shared = [g for g in model_genes if g in df.columns]
    S = df[shared].values.astype(np.float32)                # [165, n_shared]
    dev = S[list(df.index).index(HSC_ROW)] - S.mean(0)      # peak-mean-centered deviation signature for HSC
    disc = np.var(S - S.mean(0, keepdims=True), axis=0)     # discriminative half (var over celltypes)
    keep = disc > np.median(disc)
    idx = np.array([model_genes.index(g) for g in shared])
    return zrow(dev[keep]), idx[keep]


def cao_score(state_lognorm, sig, sig_idx):
    """cosine of z(state[sig_idx]) to the HSC deviation signature. state_lognorm: [G] or [N,G]."""
    X = np.atleast_2d(state_lognorm)[:, sig_idx]
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return (Xz @ sig) / len(sig)


def briggs_score(state_lognorm, phsc_centroid_z, cidx):
    """cosine of z(state) to z(pHSC mean) on the pHSC-centroid gene set. state_lognorm: [G] or [N,G]."""
    X = np.atleast_2d(state_lognorm)[:, cidx]
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return (Xz @ phsc_centroid_z) / len(phsc_centroid_z)


def fate_axes(Xln, samp, genes):
    """DIRECTION-based scoring. Build the HSPC fate axis d = HSPC_pseudobulk - background, in gene space,
    for both references. Score a displacement Δ (combo_mean - background, or decode(z_post)-decode(z_ctrl))
    by cosine alignment with d — NOT the absolute-position cosine (which is dominated by the hESC baseline).
    Returns dict with the two axes + their gene-index sets + the background (bulk) mean."""
    bulk = Xln.mean(0)
    phsc = np.array([s.lower() == 'phsc' for s in samp])
    pm = Xln[phsc].mean(0)
    bidx = np.where((pm > 0.01) | (bulk > 0.01))[0]          # expressed genes
    d_briggs = (pm - bulk)[bidx]                             # pHSC-vs-bulk fate direction
    d_cao, cidx = build_cao_hsc_signature(genes)            # Cao HSC deviation signature (already HSC - celltype mean)
    return dict(bulk=bulk, bidx=bidx, d_briggs=d_briggs, cidx=cidx, d_cao=d_cao)


def project(delta, ax):
    """cosine of a gene-space displacement `delta` [G] with each fate axis (Briggs-native, Cao)."""
    db, dc = delta[ax['bidx']], delta[ax['cidx']]
    cb = float(np.dot(db, ax['d_briggs']) / (np.linalg.norm(db) * np.linalg.norm(ax['d_briggs']) + 1e-9))
    cc = float(np.dot(dc, ax['d_cao']) / (np.linalg.norm(dc) * np.linalg.norm(ax['d_cao']) + 1e-9))
    return cb, cc


def auroc(pos, neg):
    from itertools import product
    pos, neg = np.asarray(pos), np.asarray(neg)
    # Mann-Whitney U / (n_pos*n_neg)
    allv = np.concatenate([pos, neg]); order = allv.argsort(); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--h5ad', required=True)
    ap.add_argument('--sample_col', default='sample'); a = ap.parse_args()
    A = ad.read_h5ad(a.h5ad)
    genes = list(map(str, A.var_names)); samp = A.obs[a.sample_col].astype(str).values
    print(f"[briggs] {A.shape[0]} cells x {A.shape[1]} genes; samples={sorted(set(samp))[:25]}", flush=True)
    phsc_m = np.isin(samp, [s for s in set(samp) if s.lower() == 'phsc'])
    phep_m = np.isin(samp, [s for s in set(samp) if s.lower() == 'phep'])
    print(f"[briggs] pHSC cells={phsc_m.sum()}  pHep cells={phep_m.sum()}", flush=True)
    if phsc_m.sum() < 10 or phep_m.sum() < 10:
        print("!! pHSC/pHep not found under this sample_col — inspect obs columns:", list(A.obs.columns)); return
    Xln = lognorm(A.X)                                       # [N, G] lognorm states

    # (A) Briggs-native centroid = DEVIATION of pHSC mean from the background (all-cell mean), z-scored,
    # on the discriminative half. A raw cosine to the pHSC mean rewards general cell-ness (pHSC~pHep);
    # the deviation isolates HSC-SPECIFIC genes — mirrors Briggs's discriminative scNym-latent metric.
    phsc_mean = Xln[phsc_m].mean(0); bg_mean = Xln.mean(0)
    cidx = np.where(phsc_mean > 0.01)[0]                     # pHSC-expressed genes
    dev = phsc_mean[cidx] - bg_mean[cidx]                    # HSPC-vs-background deviation
    disc = dev ** 2 > np.median(dev ** 2)                    # discriminative half
    cidx = cidx[disc]
    phsc_cent_z = zrow(phsc_mean[cidx] - bg_mean[cidx])
    # (B) Cao HSC deviation signature on shared genes
    sig, sig_idx = build_cao_hsc_signature(genes)
    print(f"[genes] Briggs-native centroid genes={len(cidx)} | Cao HSC signature genes={len(sig_idx)}", flush=True)

    for name, m in [('pHSC', phsc_m), ('pHep', phep_m)]:
        b = briggs_score(Xln[m], phsc_cent_z, cidx); c = cao_score(Xln[m], sig, sig_idx)
        print(f"  {name}: Briggs-native {b.mean():+.3f}±{b.std():.3f} | Cao-HSC {c.mean():+.3f}±{c.std():.3f}", flush=True)
    bp = briggs_score(Xln[phsc_m], phsc_cent_z, cidx); bn = briggs_score(Xln[phep_m], phsc_cent_z, cidx)
    cp = cao_score(Xln[phsc_m], sig, sig_idx); cn = cao_score(Xln[phep_m], sig, sig_idx)
    print(f"\n[SEPARABILITY pHSC vs pHep] AUROC  Briggs-native={auroc(bp, bn):.3f}  Cao-HSC={auroc(cp, cn):.3f}", flush=True)
    # cross-correlation of the two metrics over a mixed sample of cells
    rng = np.random.default_rng(0); idx = rng.choice(A.shape[0], min(8000, A.shape[0]), replace=False)
    xb = briggs_score(Xln[idx], phsc_cent_z, cidx); xc = cao_score(Xln[idx], sig, sig_idx)
    r = np.corrcoef(xb, xc)[0, 1]
    print(f"[cross-check] per-cell corr(Briggs-native, Cao-HSC) over {len(idx)} cells = {r:+.3f}", flush=True)
    print("done", flush=True)


if __name__ == '__main__':
    main()

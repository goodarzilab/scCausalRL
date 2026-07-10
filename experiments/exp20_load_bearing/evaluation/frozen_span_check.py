"""Does the frozen decoder's column space SPAN the TF-discriminative signal?

The frozen-64 arm trains stably (pert loss converges) but discriminates at chance. Hypothesis:
its fixed basis D = [GO:BP | PCA-of-expression] does not contain the drift-removed per-TF residual
directions, so no objective can route them. Test: fraction of per-TF RESIDUAL energy captured by
span(D_current) vs span(residual-PCA-64). If current << residual-PCA, rebuild D from residuals.

Run: python evaluation/frozen_span_check.py --device cpu
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from data.load_joung import load_joung


def captured(R, B):
    """Fraction of Frobenius energy of rows of R captured by orthonormal basis of columns of B."""
    Q, _ = np.linalg.qr(B)                          # [G, k] orthonormal
    proj = R @ Q                                    # [T, k] coords
    return float((proj ** 2).sum() / ((R ** 2).sum() + 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu')
    ap.add_argument('--min_cells', type=int, default=50); a = ap.parse_args()
    dd = load_joung(n_hvg=None); de = dd['de_single']; gn = dd['gene_names']; G = len(gn)
    X = dd['adata'].X; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    sp_map = dd['single_perts']; train = [g for g in dd['pert_split']['train'] if g in sp_map and g in de]
    ctrl = np.array(dd['control_idx']); ctrl_mean = X[ctrl].mean(0)

    # per-TF drift-removed residual (the discriminative signal we want A to route)
    drift = np.stack([np.asarray(de[g], np.float32) for g in train]).mean(0)
    R = np.stack([np.asarray(de[g], np.float32) - drift for g in train])          # [T, G]
    print(f"=== FROZEN DECODER SPAN CHECK === train TFs={len(train)}, genes={G}", flush=True)

    # current frozen D = [GO:BP | PCA-of-expression], as built in train_vae.py
    from utils.bio_priors import build_go_bp_mask
    Mgo, _ = build_go_bp_mask(gn, max_term_size=30, min_genes_in_axis=5, max_overlap=0.5,
                              leak=0.0, max_pathways=192, verbose=False)
    Mgo = np.asarray(Mgo, np.float32)
    Kgo = Mgo.shape[1]; npca = 64 - Kgo if Kgo < 64 else 0
    print(f"  current D: GO cols={Kgo}, expr-PCA cols={max(npca,0)} (latent≈{Kgo+max(npca,0)})", flush=True)
    Xs = X[np.random.default_rng(0).choice(X.shape[0], min(20000, X.shape[0]), replace=False)]
    Xs = Xs - Xs.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    Dexpr = Vt[:max(npca, 64)].T                                                  # expr-PCA cols

    # residual-PCA basis (the proposed fix): top-k directions of the residual matrix itself
    _, _, VtR = np.linalg.svd(R - R.mean(0, keepdims=True), full_matrices=False)
    Dres64 = VtR[:64].T

    D_cur = np.concatenate([Mgo, Dexpr[:, :max(npca, 1)]], axis=1) if npca > 0 else Mgo
    print(f"\n  RESIDUAL energy captured by span(D):", flush=True)
    print(f"    current  [GO | expr-PCA]      : {100*captured(R, D_cur):5.1f}%", flush=True)
    print(f"    GO cols only                  : {100*captured(R, Mgo):5.1f}%", flush=True)
    print(f"    expr-PCA 64                   : {100*captured(R, Dexpr[:, :64]):5.1f}%", flush=True)
    print(f"    residual-PCA 64 (PROPOSED)    : {100*captured(R, Dres64):5.1f}%", flush=True)
    print(f"    residual-PCA 128              : {100*captured(R, VtR[:128].T):5.1f}%", flush=True)
    # how many residual-PCA dims to reach 90%?
    energy = (VtR ** 2).sum(1) if VtR.ndim > 1 else None
    s = np.linalg.svd(R - R.mean(0, keepdims=True), compute_uv=False)
    cum = np.cumsum(s**2) / (s**2).sum()
    for thr in (0.5, 0.8, 0.9):
        k = int(np.searchsorted(cum, thr) + 1)
        print(f"    residual-PCA dims for {int(thr*100)}% energy: {k}", flush=True)


if __name__ == '__main__':
    main()

"""Is the TF-specific (drift-removed) residual REAL SIGNAL or NOISE — and is our objective exploiting it?
Pure-data analyses (no model needed) + target-transform magnitudes.

(1) Split-half reliability of each TF's drift-removed residual: split a TF's cells in half, r_A/r_B =
    (half-mean − ctrl) − drift. corr(r_A, r_B) over genes = how reproducible the residual is.
(2) Reliability-DISCRIMINATION (the data ceiling for disc_resid): does r_A[g] retrieve r_B[g] (same TF)
    over r_B[g'] (other TFs)? chance 0.5. This is the BEST any model could do zero-shot on this data.
(3) Variance budget: fraction of Δ variance in shared (drift) vs TF-specific residual vs project-out ⊥.
(4) Target magnitude through the R4 transforms (raw → drift-removed → project-out → pseudobulk).

Run: python evaluation/objective_dissection.py --device cpu
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from data.load_joung import load_joung


def zrow(M):
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--min_cells', type=int, default=50)
    ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dd = load_joung(n_hvg=None); de = dd['de_single']
    gn = dd['gene_names']; G = len(gn)
    X = dd['adata'].X; X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)
    ctrl = np.array(dd['control_idx']); ctrl_mean = X[ctrl].mean(0)
    sp_map = dd['single_perts']; train = [g for g in dd['pert_split']['train'] if g in sp_map]
    rng = np.random.default_rng(0)

    # per-TF full residual (drift-removed) + split halves
    drift = np.stack([np.asarray(de[g], np.float32) for g in train if g in de]).mean(0)
    tfs, rA, rB, rfull = [], [], [], []
    for g in train:
        idx = np.asarray(sp_map[g])
        if len(idx) < a.min_cells:
            continue
        idx = rng.permutation(idx); h = len(idx) // 2
        mA = X[idx[:h]].mean(0); mB = X[idx[h:2 * h]].mean(0); mF = X[idx].mean(0)
        rA.append((mA - ctrl_mean) - drift); rB.append((mB - ctrl_mean) - drift)
        rfull.append((mF - ctrl_mean) - drift); tfs.append(g)
    rA = np.stack(rA); rB = np.stack(rB); rfull = np.stack(rfull); T = len(tfs)

    # (1) split-half reliability of the residual
    rel = np.array([np.corrcoef(rA[i], rB[i])[0, 1] for i in range(T)])
    # (2) reliability-discrimination (data ceiling for disc_resid): r_A retrieves r_B of same TF
    ZA, ZB = zrow(rA), zrow(rB); C = ZA @ ZB.T
    diag = np.diag(C)[:, None]
    disc_ceiling = float(((diag > C).sum(1) / (T - 1)).mean())
    top1_ceiling = float((C.argmax(1) == np.arange(T)).mean())
    # (3) variance budget on the observed Δ = drift + residual
    dobs = np.stack([np.asarray(de[g], np.float32) for g in tfs])          # per-TF full Δ (drift+resid)
    var_shared = np.var(drift) * G                                         # proxy: shared-direction energy
    # project-out shared direction
    u = drift / (np.linalg.norm(drift) + 1e-9)
    resid_perp = rfull - (rfull @ u)[:, None] * u
    def enorm(M): return float(np.mean(np.linalg.norm(M, axis=1)))
    # (4) target magnitude through transforms
    raw = np.stack([X[np.asarray(sp_map[g])].mean(0) - ctrl_mean for g in tfs])   # raw Δ (what sparse_con fits)
    print("=== OBJECTIVE DISSECTION (train TFs, Joung) ===", flush=True)
    print(f"  train TFs with >={a.min_cells} cells: {T}", flush=True)
    print(f"\n(1) split-half reliability of drift-removed residual:  mean corr(r_A,r_B) = {np.nanmean(rel):.3f}  "
          f"(median {np.nanmedian(rel):.3f}, frac>0.2: {(rel>0.2).mean():.2f})", flush=True)
    print(f"\n(2) DATA CEILING for disc_resid (r_A retrieves r_B):  discrimination_cos = {disc_ceiling:.3f}  "
          f"(chance 0.5) | retrieval_top1 = {top1_ceiling:.3f} (chance {1/T:.3f})", flush=True)
    print(f"    -> the best ANY model could score zero-shot is bounded by this reliability.", flush=True)
    print(f"\n(3) target magnitude through R4 transforms (mean L2 per TF):", flush=True)
    print(f"    raw Δ (sparse_con target)      ‖·‖ = {enorm(raw):.3f}", flush=True)
    print(f"    drift-removed residual         ‖·‖ = {enorm(rfull):.3f}   ({100*enorm(rfull)/enorm(raw):.0f}% of raw)", flush=True)
    print(f"    + project-out shared (R4)      ‖·‖ = {enorm(resid_perp):.3f}   ({100*enorm(resid_perp)/enorm(rfull):.0f}% of residual)", flush=True)
    print(f"    shared drift ‖·‖ = {np.linalg.norm(drift):.3f}  (the mode the model finds easy)", flush=True)
    # how much of the residual is ALONG the shared direction (removed by project-out)?
    along = np.abs(rfull @ u).mean(); tot = np.linalg.norm(rfull, axis=1).mean()
    print(f"\n(4) residual component ALONG shared dir (removed by project-out): {along:.3f} of {tot:.3f} "
          f"({100*along/tot:.0f}% of residual energy) -> project-out discards this much SIGNAL", flush=True)


if __name__ == '__main__':
    main()

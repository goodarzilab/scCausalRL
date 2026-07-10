"""TF-selection analysis: does restricting to TFs that ACTUALLY drive a response raise the ceiling?

Joung et al.: only ~27% of TFs significantly move pseudotime. An inert TF's drift-removed residual is
pure noise -> two inert TFs are indistinguishable in principle, so they set an unbeatable floor on
disc_resid. Our split-half reliability of 0.31 is likely a bimodal average (effective TFs high, inert ~0).

Per-TF (drift-removed, split-half r_A/r_B over that TF's own cells):
  magnitude[t]   = ||mean residual||         (effect size; selection criterion, LEAST circular)
  reliability[t] = corr(r_A[t], r_B[t])      (is the effect reproducible)
Sweep: keep the top-N TFs by each criterion, recompute the split-half disc_resid CEILING among them.
disc metric = fraction of OTHER TFs with lower cosine than the true match -> chance 0.5 for ANY N (comparable).

Run: python evaluation/tf_selection_analysis.py --device cpu
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from data.load_joung import load_joung


def zrow(M):
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def disc_ceiling(rA, rB):
    ZA, ZB = zrow(rA), zrow(rB); C = ZA @ ZB.T
    diag = np.diag(C)[:, None]; T = rA.shape[0]
    if T < 3:
        return float('nan')
    return float(((diag > C).sum(1) / (T - 1)).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--min_cells', type=int, default=60)
    ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dd = load_joung(n_hvg=None); de = dd['de_single']
    gn = np.asarray(dd['gene_names'])
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
    print(f"=== TF SELECTION ANALYSIS === train TFs>={a.min_cells} cells: {T}", flush=True)

    rmean = (rA + rB) / 2
    magnitude = np.linalg.norm(rmean, axis=1)                              # effect size per TF
    reliability = np.array([np.corrcoef(rA[t], rB[t])[0, 1] for t in range(T)])
    reliability = np.nan_to_num(reliability)

    base = disc_ceiling(rA, rB)
    print(f"\nfull set ({T} TFs): disc_resid ceiling = {base:.3f}", flush=True)
    print(f"reliability: mean {reliability.mean():.3f} median {np.median(reliability):.3f} | "
          f">0.3: {(reliability>0.3).sum()} ({100*(reliability>0.3).mean():.0f}%) | "
          f">0.5: {(reliability>0.5).sum()} | >0.7: {(reliability>0.7).sum()}", flush=True)
    # is reliability bimodal (effective vs inert)? report magnitude percentiles
    print(f"effect magnitude percentiles: 10%={np.percentile(magnitude,10):.2f} 50%={np.percentile(magnitude,50):.2f} "
          f"90%={np.percentile(magnitude,90):.2f}", flush=True)

    for name, key in [('MAGNITUDE (effect size)', magnitude), ('RELIABILITY (reproducible)', reliability)]:
        order = np.argsort(-key)
        print(f"\ndisc_resid CEILING vs top-N TFs by {name}:", flush=True)
        print(f"  {'N':>6} {'%kept':>6} {'ceiling':>9} {'vs full':>8} {'mean_rel':>9}", flush=True)
        for N in [50, 100, 200, 300, 400, 500, 700, 1000, T]:
            if N > T:
                continue
            sel = order[:N]
            print(f"  {N:>6} {100*N/T:>5.0f}% {disc_ceiling(rA[sel][:,:],rB[sel]):>9.3f} "
                  f"{disc_ceiling(rA[sel],rB[sel])-base:>+8.3f} {reliability[sel].mean():>9.3f}", flush=True)

    # how many TFs are effectively inert (residual indistinguishable from noise)?
    inert = (reliability < 0.1)
    print(f"\n~inert TFs (split-half reliability < 0.1): {inert.sum()} of {T} ({100*inert.mean():.0f}%)", flush=True)
    print(f"  -> these are indistinguishable in principle and cap disc_resid.", flush=True)
    # top effective TFs (sanity: known master regulators?)
    top = np.argsort(-reliability)[:30]
    print(f"\ntop 30 most reproducible TFs: {', '.join(np.asarray(tfs)[top])}", flush=True)
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'tf_effect_ranked.npz')
    np.savez(out, tfs=np.asarray(tfs), magnitude=magnitude, reliability=reliability)
    print(f"\nsaved TF ranking -> {out}", flush=True)


if __name__ == '__main__':
    main()

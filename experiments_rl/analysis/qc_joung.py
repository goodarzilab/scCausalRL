"""THOROUGH QC of the Joung TF-atlas singles used for world-model training.
Motivated by CausalGRN-style perturbation QC: is each TF overexpression (a) well-sampled,
(b) on-target (the TF's own mRNA actually up), (c) transcriptionally impactful (DEG yield)?
And is the 9,923-gene vocab high-quality (HVG ∪ TF ∪ RBP) or noise?

Two passes over joung_singles_proc.h5ad (dense, chunked):
  pass 1 (controls): ctrl_mean, ctrl_std per gene
  pass 2 (per-TF):   per-TF mean  → on-target Cohen's D, DEG count, ||Δ||
Effect size = Cohen's D vs control:  d_g = (mean_pert_g − ctrl_mean_g) / ctrl_std_g.
DEG = |d_g| > 0.5 (excluding the perturbed gene itself).

Outputs: analysis/qc_joung_perts.csv (per-TF) + printed summary.
"""
import os, sys, numpy as np, pandas as pd
import anndata as ad, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
PROC = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
CHUNK = 50000
DEG_D = 0.5          # |Cohen's D| threshold for a DEG


def dense(x):
    return (x.toarray() if sp.issparse(x) else np.asarray(x)).astype(np.float64)


def main():
    a = ad.read_h5ad(PROC, backed='r')
    n, g = a.shape
    genes = np.array([str(x).upper() for x in a.var_names])
    g2i = {x: i for i, x in enumerate(genes)}
    is_ctrl = (a.obs['is_control'].astype(str) == 'True').values
    tf = a.obs['TF_gene'].astype(str).str.upper().values
    ctrl_rows = np.where(is_ctrl)[0]
    print(f"[data] {n:,} cells × {g:,} genes | controls={len(ctrl_rows):,} | perturbed={n-len(ctrl_rows):,} | TFs={pd.unique(tf[~is_ctrl]).size}")

    # ---- pass 1: control mean/std ----
    s = np.zeros(g); ss = np.zeros(g)
    for i in range(0, len(ctrl_rows), CHUNK):
        X = dense(a.X[np.sort(ctrl_rows[i:i+CHUNK])])
        s += X.sum(0); ss += (X**2).sum(0)
    nc = len(ctrl_rows)
    ctrl_mean = s / nc
    ctrl_std = np.sqrt(np.maximum(ss/nc - ctrl_mean**2, 0)) + 1e-6
    print("[pass1] control mean/std done")

    # ---- pass 2: per-TF sums (all cells, group by TF; controls skipped) ----
    from collections import defaultdict
    tf_sum = defaultdict(lambda: np.zeros(g)); tf_n = defaultdict(int)
    order = np.arange(n)
    for i in range(0, n, CHUNK):
        rows = order[i:i+CHUNK]
        X = dense(a.X[rows])
        tfs = tf[rows]; ic = is_ctrl[rows]
        for t in np.unique(tfs[~ic]):
            m = (tfs == t) & (~ic)
            tf_sum[t] += X[m].sum(0); tf_n[t] += int(m.sum())
        if (i // CHUNK) % 5 == 0:
            print(f"  pass2 {i:,}/{n:,}", flush=True)

    rows = []
    for t in tf_sum:
        nt = tf_n[t]
        mean_t = tf_sum[t] / nt
        d = (mean_t - ctrl_mean) / ctrl_std               # per-gene Cohen's D
        ti = g2i.get(t, -1)
        on_d = float(d[ti]) if ti >= 0 else np.nan          # on-target (TF's own gene)
        on_lfc = float(mean_t[ti] - ctrl_mean[ti]) if ti >= 0 else np.nan
        dd = d.copy()
        if ti >= 0: dd[ti] = 0.0                             # exclude self from DEG count
        rows.append(dict(TF=t, n_cells=nt, in_vocab=(ti >= 0),
                         ontarget_cohenD=on_d, ontarget_lfc=on_lfc,
                         n_DEG=int((np.abs(dd) > DEG_D).sum()),
                         delta_norm=float(np.linalg.norm(mean_t - ctrl_mean))))
    df = pd.DataFrame(rows).sort_values('n_cells', ascending=False)
    df.to_csv(f'{HERE}/qc_joung_perts.csv', index=False)

    # ---- vocab composition: TF fraction ----
    perturbed_tfs = set(pd.unique(tf[~is_ctrl]))
    tf_in_vocab = sum(t in g2i for t in perturbed_tfs)
    try:
        import decoupler as dc
        ct = dc.op.collectri(organism='human')
        all_tf = set(ct['source'].astype(str).str.upper())
        vocab_tf = sum(x in all_tf for x in genes)
    except Exception as e:
        vocab_tf = None; print("collectri TF census skipped:", e)

    def pct(mask): return f"{mask.sum()} ({100*mask.mean():.1f}%)"
    print("\n" + "="*78 + "\nJOUNG TF-ATLAS QC SUMMARY\n" + "="*78)
    print(f"controls: {nc:,}   perturbed TFs: {len(df)}")
    print("\n--- cells per perturbation ---")
    print(f"  median={int(df.n_cells.median())}  mean={df.n_cells.mean():.0f}  min={df.n_cells.min()}  max={df.n_cells.max()}")
    for thr in (5, 20, 50, 100, 150):
        print(f"  TFs with >= {thr:>3} cells: {pct(df.n_cells>=thr)}")
    print("\n--- on-target OE efficacy (TF's own gene up vs control) ---")
    iv = df[df.in_vocab]
    print(f"  TFs whose gene IS in the 9,923 vocab: {tf_in_vocab}/{len(perturbed_tfs)} ({100*tf_in_vocab/len(perturbed_tfs):.1f}%)")
    print(f"  of those: on-target Cohen's D  median={iv.ontarget_cohenD.median():.2f}")
    for thr in (0.2, 0.5, 1.0):
        print(f"    on-target |D| > {thr}: {pct(iv.ontarget_cohenD.abs()>thr)}   (D>{thr}, i.e. UP: {pct(iv.ontarget_cohenD>thr)})")
    print("\n--- downstream impact (DEG yield, |Cohen's D|>0.5, excl. self) ---")
    print(f"  median DEGs/pert={int(df.n_DEG.median())}  mean={df.n_DEG.mean():.0f}")
    for thr in (10, 50, 100):
        print(f"  TFs with >= {thr:>3} DEGs: {pct(df.n_DEG>=thr)}")
    print(f"  median ||Δ|| (effect magnitude) = {df.delta_norm.median():.2f}")
    print("\n--- gene vocab composition ---")
    print(f"  vocab size: {g}")
    if vocab_tf is not None:
        print(f"  vocab genes that are CollecTRI TFs: {vocab_tf} ({100*vocab_tf/g:.1f}%)")
    print(f"  perturbed-TF genes present in vocab: {tf_in_vocab} ({100*tf_in_vocab/g:.1f}% of vocab)")
    # combined 'CausalGRN-style' high-quality perturbation set
    hq = df[(df.n_cells >= 50) & (df.in_vocab) & (df.ontarget_cohenD > 0.2) & (df.n_DEG >= 10)]
    print(f"\n>>> HIGH-QUALITY perts (>=50 cells & in-vocab & on-target D>0.2 & >=10 DEGs): {len(hq)} / {len(df)}")
    hq2 = df[(df.n_cells >= 100) & (df.n_DEG >= 10)]
    print(f">>> looser (>=100 cells & >=10 DEGs): {len(hq2)} / {len(df)}")
    print(f"saved per-TF QC -> {HERE}/qc_joung_perts.csv")


if __name__ == '__main__':
    main()

"""Rebuild the Joung singles proc file on the vocab-v2 gene axis (4000 seurat_v3 HVG U fate markers).

Our current 9923 proc overlaps vocab-v2 by only 61%, so we rebuild from the raw counts h5ad rather than
subset. Output matches the load_joung contract: X = raw counts (sparse), obs['TF_gene'] (gene symbol,
isoform merged), obs['is_control'] (GFP/mCherry). load_joung lognorms downstream.
"""
import os, numpy as np, scipy.sparse as sp, anndata as ad, pandas as pd

RAW = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/GSE217460_210322_TFAtlas_raw.h5ad"
VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
OUT = os.path.join(VDIR, "joung_singles_proc_v2.h5ad")
CTRL = {"GFP", "mCherry", "mcherry", "EGFP"}


def main():
    import argparse
    _ap=argparse.ArgumentParser(); _ap.add_argument("--vocab", default=None); _ap.add_argument("--out", default=None); _A=_ap.parse_args()
    vocab = [g.strip() for g in open(_A.vocab or os.path.join(VDIR, "vocab_v2_genes.txt")) if g.strip()]
    vset = set(vocab)
    print(f"[v2] vocab-v2 genes: {len(vocab)}", flush=True)
    print(f"[v2] loading raw {RAW} to memory (counts)...", flush=True)
    A = ad.read_h5ad(RAW)                                   # full to memory (high-mem node)
    print(f"[v2] raw shape {A.shape}", flush=True)
    mask = np.array([str(g) in vset for g in A.var_names])
    print(f"[v2] genes kept: {int(mask.sum())} of {A.shape[1]} (vocab genes present: "
          f"{len(vset & set(map(str, A.var_names)))}/{len(vocab)})", flush=True)
    # raw X is DENSE (37528 cols); subset to vocab cols then convert to sparse. Extract obs/var and
    # free the 172GB dense array before building the AnnData.
    Xsub = A.X[:, mask]
    Xsub = (Xsub.toarray() if sp.issparse(Xsub) else np.asarray(Xsub)).astype(np.float32)
    var_names = A.var_names[mask].astype(str)
    tf = A.obs['TF'].astype(str).values
    obs_names = A.obs_names
    del A; import gc; gc.collect()
    X = sp.csr_matrix(Xsub); del Xsub; gc.collect()
    var = pd.DataFrame(index=var_names)
    tf_gene = np.array([t.split('-', 1)[1] if '-' in t else t for t in tf])
    is_ctrl = np.array([g in CTRL for g in tf_gene])
    obs = pd.DataFrame({'TF_gene': tf_gene, 'is_control': is_ctrl}, index=obs_names)
    v2 = ad.AnnData(X=X, obs=obs, var=var)
    print(f"[v2] built {v2.shape} | controls={int(is_ctrl.sum())} | TFs={pd.Series(tf_gene[~is_ctrl]).nunique()}", flush=True)
    v2.write_h5ad(_A.out or OUT)
    print(f"[v2] wrote {_A.out or OUT} ({os.path.getsize(_A.out or OUT)/1e9:.2f} GB)", flush=True)


if __name__ == '__main__':
    main()

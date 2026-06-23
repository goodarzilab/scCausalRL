"""Recall@k head-to-head on the SAME Joung shifts: S6 aggregate vs cell-level matched-norm
reference (both plain mean-expression cosine, OE-axis removed). No compression — all cell types
ranked, scored against per-TF biologically-correct types (recall_at_k.TARGETS).
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from data.load_joung import _lognorm
sys.path.insert(0, HERE)
from recall_at_k import TARGETS
D = '/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'


def cosmat(Ydf, sigdf, drift_pcs=4):
    """cosine(TF shift, cell-type signature) over shared genes, after OE-axis removal."""
    shared = sigdf.index.intersection(Ydf.columns)
    X = sigdf.loc[shared].values.astype(np.float64); Dd = X - X.mean(1, keepdims=True)
    keep = Dd.var(1) > np.quantile(Dd.var(1), 0.5); Dd, sh = Dd[keep], shared[keep]
    Dz = (Dd - Dd.mean(0)) / (Dd.std(0) + 1e-8)
    Y = Ydf[sh].values.astype(np.float64); Yc = Y - Y.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False); k = min(drift_pcs, len(S))
    Yc = Yc - (U[:, :k] * S[:k]) @ Vt[:k]
    Yz = (Yc - Yc.mean(1, keepdims=True)) / (Yc.std(1, keepdims=True) + 1e-8)
    return pd.DataFrame((Yz @ Dz) / Yz.shape[1], index=Ydf.index, columns=sigdf.columns)


def recall(cos, name):
    cos.index = cos.index.astype(str).str.upper()
    types = np.array([str(c) for c in cos.columns]); C = len(types); tl = np.array([t.lower() for t in types])
    gt = {k.upper(): v for k, v in TARGETS.items()}
    r1 = r5 = r10 = 0; ranks = []; n = 0
    for tf in [t for t in gt if t in cos.index]:
        acc = np.array([any(kw in tl[i] for kw in gt[tf]) for i in range(C)])
        if acc.sum() == 0: continue
        order = np.argsort(-cos.loc[tf].values); best = int(np.where(acc[order])[0].min()) + 1
        ranks.append(best); r1 += best <= 1; r5 += best <= 5; r10 += best <= 10; n += 1
    print(f"\n[{name}]  {C} types, {n} TFs:  recall@1={r1/n:.2f}  recall@5={r5/n:.2f}  "
          f"recall@10={r10/n:.2f}  median-rank={int(np.median(ranks))}/{C}", flush=True)


def main():
    a = ad.read_h5ad(JOUNG); a.var_names = [str(v).upper() for v in a.var_names]; a.X = _lognorm(a.X)
    tf = a.obs['TF_gene'].astype(str).values; ctrl = np.asarray(a.obs['is_control'].values, bool)
    base = np.asarray(a.X[ctrl].mean(0)).ravel(); rows, names = [], []
    for g in sorted(set(tf)):
        m = (tf == g) & (~ctrl)
        if m.sum() >= 50: rows.append(np.asarray(a.X[m].mean(0)).ravel() - base); names.append(g)
    Ydf = pd.DataFrame(np.vstack(rows), index=names, columns=a.var_names)
    Ydf = Ydf.loc[:, ~Ydf.columns.duplicated()]
    print(f'Joung shifts: {Ydf.shape}', flush=True)
    # S6 aggregate signatures
    g = pd.read_csv(f'{D}/GSE156793_S2_Metadata_genes.txt.gz'); id2s = dict(zip(g.gene_id, g.gene_short_name.astype(str).str.upper()))
    S6 = pd.read_csv(f'{D}/GSE156793_S6_gene_expression_celltype.txt.gz', index_col=0)
    S6.index = S6.index.map(lambda e: id2s.get(e)); S6 = S6[S6.index.notna()]; S6 = S6[~S6.index.duplicated()]
    # cell-level matched-norm signatures (log1p CP10k means)
    cl = pd.read_csv(f'{HERE}/cao_type_mean_logexpr.csv', index_col=0).T
    cl.index = cl.index.astype(str).str.upper(); cl = cl[~cl.index.duplicated()]
    recall(cosmat(Ydf, S6), 'S6 aggregate (paper)')
    recall(cosmat(Ydf, cl), 'cell-level matched-norm (log1p CP10k)')


if __name__ == '__main__':
    main()

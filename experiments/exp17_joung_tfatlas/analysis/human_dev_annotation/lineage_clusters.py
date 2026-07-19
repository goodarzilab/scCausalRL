"""Data-driven coarse lineages (Leiden) over the Cao cell-type signatures — replaces the
hand-written keyword map in fate_separability.coarse_fate(). We cluster the 165 cell-level
reference signatures (cao_type_mean_logexpr) by transcriptional similarity, so a "lineage" is
an emergent group of cell types, not a manual bucket. Each TF inherits the lineage of its
annotated fine cell type (joung_human_annotation_celllevel.csv).

Outputs:
  cao_celltype_to_lineage.csv   cell_type -> lineage_id, lineage_name
  joung_tf_lineage.csv          TF -> fine_celltype, lineage_id, lineage_name, fdr_qval
Run: python lineage_clusters.py [resolution]
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CL = f'{HERE}/cao_type_mean_logexpr.csv'
ANN = f'{HERE}/joung_human_annotation_celllevel.csv'


def main():
    import scanpy as sc, anndata as ad
    res = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    # cell-type signatures: file is types × genes (index = cell types)
    M = pd.read_csv(CL, index_col=0)
    M.index = M.index.astype(str)
    X = M.values.astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)                       # drift-remove (cross-type mean)
    keep = Xc.var(0) > np.quantile(Xc.var(0), 0.5)          # high-var genes (match cosmat)
    Xz = (Xc[:, keep] - Xc[:, keep].mean(0)) / (Xc[:, keep].std(0) + 1e-8)

    A = ad.AnnData(Xz.astype(np.float32), obs=pd.DataFrame(index=M.index))
    sc.pp.pca(A, n_comps=min(50, A.shape[0] - 1))
    sc.pp.neighbors(A, n_neighbors=15, metric='cosine', use_rep='X_pca')
    sc.tl.leiden(A, resolution=res, key_added='lineage', flavor='igraph', n_iterations=2, directed=False)
    lin = A.obs['lineage'].astype(int).values
    types = np.array(M.index)
    K = lin.max() + 1

    # name each lineage by its 3 most central cell types (closest to cluster centroid)
    Xpca = A.obsm['X_pca']; names = {}
    for L in range(K):
        idx = np.where(lin == L)[0]; c = Xpca[idx].mean(0)
        order = idx[np.argsort(((Xpca[idx] - c) ** 2).sum(1))]
        short = [types[i].split('-')[-1].replace(' positive cells', '+').replace(' cells', '') for i in order[:3]]
        names[L] = f"L{L}:" + "/".join(short)

    ct2lin = pd.DataFrame({'cell_type': types, 'lineage_id': lin,
                           'lineage_name': [names[l] for l in lin]}).sort_values('lineage_id')
    ct2lin.to_csv(f'{HERE}/cao_celltype_to_lineage.csv', index=False)

    # map TFs via their annotated fine cell type
    ann = pd.read_csv(ANN); ann['TF'] = ann['TF'].astype(str)
    lmap = dict(zip(ct2lin['cell_type'], ct2lin['lineage_id']))
    nmap = dict(zip(ct2lin['cell_type'], ct2lin['lineage_name']))
    ann['lineage_id'] = ann['top_celltype'].map(lmap)
    ann['lineage_name'] = ann['top_celltype'].map(nmap)
    miss = ann['lineage_id'].isna().sum()
    out = ann[['TF', 'top_celltype', 'lineage_id', 'lineage_name', 'fdr_qval']].dropna(subset=['lineage_id'])
    out['lineage_id'] = out['lineage_id'].astype(int)
    out.to_csv(f'{HERE}/joung_tf_lineage.csv', index=False)

    print(f"Leiden resolution {res} → {K} lineages over {len(types)} cell types "
          f"({miss} TF fine-types unmapped)")
    conf = out[out['fdr_qval'] < 0.10]
    print(f"\nlineage | #celltypes | #TFs(all) | #TFs(q<0.10) | name")
    for L in range(K):
        nct = int((lin == L).sum()); nall = int((out['lineage_id'] == L).sum())
        ncf = int((conf['lineage_id'] == L).sum())
        print(f"  {L:2d}    |   {nct:3d}      |   {nall:4d}    |   {ncf:3d}        | {names[L]}")
    print(f"\nconfident TFs total (q<0.10): {len(conf)} across {conf['lineage_id'].nunique()} lineages "
          f"(>=3 TFs: { (conf['lineage_id'].value_counts()>=3).sum() } lineages)")
    print("saved -> cao_celltype_to_lineage.csv, joung_tf_lineage.csv")


if __name__ == '__main__':
    main()

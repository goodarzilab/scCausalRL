"""Train a CellTypist model on the Cao 2020 human fetal reference (Organ_cell_lineage = the
same 172-type label space our cosine annotation uses), restricted to the Joung HVG axis.

Saves: the trained model (.pkl), per-cell-type mean log-expression (for mean-deviation
signatures), and the gene list — consumed by celltypist_custom_compare.py for the head-to-head
vs our shift-cosine annotation.

Auto-detects the loom↔S1 cell-id join key and the gene row-attr (robust to attr naming).
Run (high-mem):  sbatch analysis/human_dev_annotation/celltypist_custom_train.sbatch
"""
import os, sys, json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, h5py
import loompy, celltypist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from data.load_joung import _lognorm
D = '/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
LOOM = f'{D}/GSE156793_S3_gene_count.loom'; S1 = f'{D}/GSE156793_S1_metadata_cells.txt.gz'
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
LABEL = 'Organ_cell_lineage'; N_PER_TYPE = 250; MIN_TYPE_CELLS = 50
OUT = HERE


def joung_symbols():
    with h5py.File(JOUNG, 'r') as f:
        vn = f['var/_index'][:]
    return [str(x.decode() if isinstance(x, bytes) else x).upper() for x in vn]


def main():
    joung = joung_symbols(); jset = set(joung)
    print(f'Joung genes: {len(joung)}', flush=True)
    s1 = pd.read_csv(S1, usecols=['sample', LABEL]).dropna(subset=[LABEL])
    lab_by_id = s1.set_index('sample')[LABEL]
    print(f'S1: {len(lab_by_id)} labelled cells, {lab_by_id.nunique()} {LABEL} types', flush=True)

    with loompy.connect(LOOM, 'r') as ds:
        print('loom shape (genes x cells):', ds.shape, flush=True)
        # detect cell-id col attr (max overlap with S1 'sample')
        samp = set(lab_by_id.index.astype(str).values)
        cid_key = max(ds.ca.keys(), key=lambda k: len(set(str(x) for x in ds.ca[k][:5000]) & samp))
        cids = np.array([str(x) for x in ds.ca[cid_key]])
        # detect gene symbol row attr (max overlap with Joung)
        g_key = max(ds.ra.keys(), key=lambda k: len(set(str(x).upper() for x in ds.ra[k][:]) & jset))
        gsym = np.array([str(x).upper() for x in ds.ra[g_key]])
        gene_keep = np.isin(gsym, joung); gene_idx = np.where(gene_keep)[0]
        print(f'cid_key={cid_key} gene_key={g_key} | genes kept (loom∩Joung): {int(gene_keep.sum())}', flush=True)

        lab = lab_by_id.reindex(cids).values            # label per loom cell (NaN if absent)
        valid = np.where(pd.notna(lab))[0]
        # subsample up to N_PER_TYPE per type
        rng = np.random.default_rng(0); sel = []
        ser = pd.Series(valid, index=lab[valid])
        for L, idxs in ser.groupby(level=0):
            iv = idxs.values
            if len(iv) < MIN_TYPE_CELLS: continue
            sel += list(rng.choice(iv, min(N_PER_TYPE, len(iv)), replace=False))
        sel = np.sort(np.array(sel)); labels_sel = lab[sel]
        print(f'training cells: {len(sel)} across {pd.Series(labels_sel).nunique()} types', flush=True)

    # Read the selected cells in SEQUENTIAL column blocks via h5py. The loom is float32 with
    # (64,64) gzip chunks (~1TB uncompressed); scattered ds[:, sel] reads decompress a huge
    # number of chunks (the hang). Streaming blocks decompresses each chunk at most once.
    print('reading loom in sequential column blocks (h5py)...', flush=True)
    import h5py
    B = 16384; blocks = []
    with h5py.File(LOOM, 'r') as f:
        Mat = f['matrix']                               # [genes x cells]
        for start in range(0, Mat.shape[1], B):
            end = min(start + B, Mat.shape[1])
            cin = sel[(sel >= start) & (sel < end)]
            if len(cin) == 0: continue
            blk = Mat[:, start:end]                      # [genes x b] (chunks decompressed once)
            blocks.append(blk[gene_idx][:, cin - start])
            if (start // B) % 50 == 0: print(f'  block @ {start}/{Mat.shape[1]}', flush=True)
    M = np.concatenate(blocks, axis=1)                  # [G_keep x N_sel]
    genes_keep = gsym[gene_idx]

    X = sp.csr_matrix(M.T.astype(np.float32)); del M    # cells x genes
    adt = ad.AnnData(X=X, obs=pd.DataFrame({'label': labels_sel}),
                     var=pd.DataFrame(index=pd.Index(genes_keep, name='gene')))
    adt = adt[:, ~adt.var_names.duplicated()].copy()
    adt.X = _lognorm(adt.X)                              # log1p CP10k (CellTypist input)
    print(f'training AnnData: {adt.shape}', flush=True)

    print('training CellTypist (SGD LR, no feature selection → coefs span all genes)...', flush=True)
    model = celltypist.train(adt, labels='label', n_jobs=8, use_SGD=True, mini_batch=True,
                             batch_size=1000, epochs=10, feature_selection=False)
    model.write(f'{OUT}/cao_celltypist_model.pkl')

    # per-type mean log-expression (for the mean-deviation signature baseline in compare)
    df = pd.DataFrame(adt.X.toarray() if sp.issparse(adt.X) else adt.X,
                      columns=adt.var_names, index=adt.obs['label'].values)
    typemean = df.groupby(level=0).mean()                # [types x genes]
    typemean.to_csv(f'{OUT}/cao_type_mean_logexpr.csv')
    json.dump({'n_train_cells': int(adt.n_obs), 'n_types': int(typemean.shape[0]),
               'n_genes': int(adt.n_vars), 'cid_key': cid_key, 'gene_key': g_key,
               'label': LABEL}, open(f'{OUT}/cao_train_meta.json', 'w'), indent=2)
    print(f'\nsaved model + type-mean ({typemean.shape}) + meta → {OUT}/', flush=True)


if __name__ == '__main__':
    main()

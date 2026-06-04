import scanpy as sc
import numpy as np
import scipy.sparse


def preprocess(adata, n_hvg=2000, force_include=None):
    """
    Shared preprocessing pipeline for all datasets.

    Steps:
    1. Store raw counts in adata.layers['counts']
    2. Normalize: normalize_total(target_sum=1e4), then log1p
    3. Flag top n_hvg highly variable genes (seurat_v3); OR in `force_include` so
       any externally-specified genes (e.g. Norman's perturbed gene set) are
       guaranteed to survive HVG filtering even if they're below the n_hvg cutoff.
    4. Subset adata to the union mask.
    5. Compute PCA (50 components)
    6. Return processed AnnData

    force_include: iterable of gene names (matched against adata.var_names) that
        MUST be kept regardless of variance. Genes not present in adata are silently
        skipped. Useful for cross-dataset eval where the eval set's perturbed genes
        must appear in the training HVG axis.
    """
    # Store raw counts before normalization
    if scipy.sparse.issparse(adata.X):
        adata.layers['counts'] = adata.X.copy()
    else:
        adata.layers['counts'] = adata.X.copy()

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Find HVGs - handle edge cases
    n_genes = adata.n_vars
    n_hvg_actual = min(n_hvg, n_genes)

    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvg_actual, flavor='seurat_v3', layer='counts'
        )
        if force_include is not None and len(force_include) > 0:
            force_set = set(force_include)
            forced_mask = np.asarray(adata.var_names.isin(force_set))
            hvg_mask = adata.var['highly_variable'].values.copy()
            union_mask = hvg_mask | forced_mask
            adata.var['highly_variable'] = union_mask
            n_req = int(forced_mask.sum())
            n_added = int(union_mask.sum() - hvg_mask.sum())
            print(f"  HVG: {hvg_mask.sum()} from seurat_v3 + {n_added} force-included "
                  f"(of {n_req} requested present in this dataset; "
                  f"{n_req - n_added} were already in HVG)")
        adata = adata[:, adata.var['highly_variable']].copy()
    except Exception as e:
        print(f"  HVG selection failed ({e}), using all {n_genes} genes")

    # PCA
    n_comps = min(50, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=n_comps)

    return adata

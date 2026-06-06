import os
import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad
import scanpy as sc
from .preprocess import preprocess


def load_replogle(path=None, n_hvg=2000, cell_line='k562', max_cells=50000,
                  train_frac=0.8, val_frac=0.1, split_seed=42,
                  force_include=None):
    """
    Load Replogle et al. 2022 genome-wide Perturb-seq dataset.

    Subsets to requested cell line and optionally downsamples for memory.

    Returns dict with:
        adata: preprocessed AnnData
        control_idx: indices of control cells ('non-targeting')
        single_perts: dict gene_name -> cell indices
        combo_perts: dict (gene_a, gene_b) -> cell indices (empty for Replogle)
        gene_names: list of gene names
        pert_genes: list of all perturbed genes
        pert_split: dict {'train', 'val', 'test'} of gene name sets
    """
    from data.load_norman import split_single_perts
    if path is None:
        path = "/large_storage/goodarzilab/shervin/CausalBioRL/Replogle/"

    # Resolve file: accept direct .h5ad path, or directory containing train_v5.h5ad
    # (preferred — pretrain_data layout) or legacy replogle.h5ad
    if path.endswith('.h5ad'):
        h5ad_file = path
    elif os.path.isfile(os.path.join(path, "train_v5.h5ad")):
        h5ad_file = os.path.join(path, "train_v5.h5ad")
    else:
        h5ad_file = os.path.join(path, "replogle.h5ad")
    print(f"Loading Replogle dataset from {h5ad_file}...")
    print(f"  File size: {os.path.getsize(h5ad_file) / 1e9:.2f} GB")

    adata = ad.read_h5ad(h5ad_file)
    print(f"  Full shape: {adata.shape}")

    # K562_essential / K562_gwps unfiltered files have ENSG IDs as var_names
    # but carry the human-readable symbol in var['gene_name']. Translate so the
    # downstream pipeline (and any cross-dataset Norman alignment) sees symbols.
    if 'gene_name' in adata.var.columns and adata.var_names[0].startswith('ENSG'):
        new_names = adata.var['gene_name'].astype(str).values
        # Some ENSG IDs map to the same symbol — make unique by suffixing
        n_before = len(set(new_names))
        adata.var_names = new_names
        adata.var_names_make_unique()
        print(f"  Renamed var_names from ENSG → gene symbol "
              f"({n_before} unique symbols, {adata.n_vars - n_before} suffixed for duplicates)")

    # Some Replogle files (e.g. K562_gwps_filtered.h5ad) are already cell-line-
    # specific and do not carry a `cell_line` column. Only filter if the column
    # is present.
    if 'cell_line' in adata.obs.columns:
        print(f"  Cell lines present: {adata.obs['cell_line'].unique().tolist()}")
        mask = adata.obs['cell_line'].str.lower() == cell_line.lower()
        adata = adata[mask].copy()
        print(f"  After {cell_line} filter: {adata.shape[0]} cells")
    else:
        print(f"  No `cell_line` column — assuming file is already {cell_line}-only "
              f"(skipping cell-line filter).")

    # Downsample only if explicitly requested AND smaller than the cell-line subset.
    # Default for K562 is to keep all 186k cells — the dataset is already a well-defined
    # cell-line slice and the cell-sentence loader handles its own per-block sampling.
    if max_cells is not None and adata.n_obs > max_cells:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        print(f"  After downsampling to {max_cells}: {adata.shape[0]} cells")
    else:
        print(f"  No downsampling (using all {adata.shape[0]} {cell_line} cells)")

    # The X matrix is already expression (dense float32)
    # Check if it's raw counts or already normalized
    # For Replogle, X appears to be processed; use as-is or re-normalize
    print(f"  X dtype: {adata.X.dtype}, sparse: {scipy.sparse.issparse(adata.X)}")

    # If X_hvg exists, use it (subset of HVG expression)
    if 'X_hvg' in adata.obsm:
        print(f"  X_hvg shape: {adata.obsm['X_hvg'].shape}")

    # Preprocess
    print("  Preprocessing...")
    adata = preprocess(adata, n_hvg=n_hvg, force_include=force_include)
    print(f"  After preprocessing: {adata.shape}")

    # Build perturbation indices
    gene_col = 'gene'  # perturbation gene column

    control_mask = adata.obs[gene_col] == 'non-targeting'
    control_idx = np.where(control_mask)[0].tolist()

    single_perts = {}
    for i, gene in enumerate(adata.obs[gene_col]):
        if gene != 'non-targeting' and pd.notna(gene):
            if gene not in single_perts:
                single_perts[gene] = []
            single_perts[gene].append(i)

    pert_genes = sorted(list(single_perts.keys()))

    print(f"  Control cells: {len(control_idx)}")
    print(f"  Single perturbations: {len(single_perts)} genes")

    # Compute differential expression
    X_dense = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)

    if len(control_idx) > 0:
        ctrl_mean = X_dense[control_idx].mean(axis=0)
    else:
        ctrl_mean = X_dense.mean(axis=0)

    print("  Computing differential expression...")
    adata.uns['de_single'] = {}
    for gene, idxs in single_perts.items():
        if len(idxs) > 0:
            pert_mean = X_dense[idxs].mean(axis=0)
            adata.uns['de_single'][gene] = pert_mean - ctrl_mean

    adata.uns['de_combo'] = {}

    # Gene-level train/val/test split (same logic as Norman)
    pert_split = split_single_perts(single_perts, train_frac=train_frac,
                                     val_frac=val_frac, seed=split_seed)
    print(f"  Pert split: {len(pert_split['train'])} train, "
          f"{len(pert_split['val'])} val, {len(pert_split['test'])} test genes")

    # Build gem_group -> batch_id map (contiguous integers, deterministic order).
    # Replogle K562 has 48 unique gem_groups; the encoder will use these for additive
    # batch-effect conditioning. Stored in adata.obs['batch_id'] so downstream cell-
    # sentence batching can pair cells from the same plate.
    if 'gem_group' in adata.obs.columns:
        unique_gg = sorted(adata.obs['gem_group'].unique().tolist())
        gem_group_to_idx = {gg: i for i, gg in enumerate(unique_gg)}
        adata.obs['batch_id'] = adata.obs['gem_group'].map(gem_group_to_idx).astype(np.int64)
        n_batches = len(unique_gg)
        print(f"  gem_groups: {n_batches} unique (batch-effect conditioning enabled)")
    else:
        gem_group_to_idx = {}
        adata.obs['batch_id'] = 0
        n_batches = 0
        print(f"  No gem_group column found — batch conditioning disabled")

    return {
        'adata': adata,
        'control_idx': control_idx,
        'single_perts': single_perts,
        'combo_perts': {},
        'gene_names': list(adata.var_names),
        'pert_genes': pert_genes,
        'pert_split': pert_split,
        'gem_group_to_idx': gem_group_to_idx,
        'n_batches': n_batches,
    }

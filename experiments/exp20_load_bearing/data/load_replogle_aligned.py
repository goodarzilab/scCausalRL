"""Load Replogle dataset aligned to a target gene set (e.g., Norman's HVGs).

Used for external validation: forces Replogle data into the same gene coordinate
system as the trained model. Genes in target set but missing from Replogle are
zero-imputed. Genes in Replogle but not in target set are dropped.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse


def load_replogle_aligned(path, target_gene_names, max_cells=None, min_cells_per_pert=5):
    """
    Load Replogle (v5) and align to a fixed gene set.

    Args:
        path: directory containing train_v5.h5ad
        target_gene_names: list of gene names (e.g., from Norman-trained checkpoint)
        max_cells: optional downsample (default: no limit)
        min_cells_per_pert: minimum cells per perturbation to compute DE

    Returns: data_dict with keys matching load_norman/load_replogle structure
    """
    import scanpy as sc
    import anndata as ad

    h5_path = os.path.join(path, 'train_v5.h5ad')
    print(f"Loading Replogle v5 from {h5_path}")
    adata = sc.read_h5ad(h5_path)
    print(f"  Raw shape: {adata.shape}")

    # Subset to K562 cells only
    if 'cell_type' in adata.obs.columns:
        k562_mask = adata.obs['cell_type'].astype(str) == 'K562'
        n_k562 = int(k562_mask.sum())
        if n_k562 > 0:
            adata = adata[k562_mask].copy()
            print(f"  After K562 filter: {adata.shape}")

    # Optional downsample BEFORE normalization (saves memory)
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  After downsample: {adata.shape}")

    # Library-size normalization + log1p (same as Norman preprocessing, NO HVG)
    print("  Normalizing + log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Align to target gene set
    target_set = set(target_gene_names)
    existing = set(adata.var_names)
    overlap = target_set & existing
    missing = target_set - existing

    print(f"  Gene alignment: {len(overlap)}/{len(target_gene_names)} in Replogle, "
          f"{len(missing)} zero-imputed")

    # Build aligned matrix in target order
    n_cells = adata.n_obs
    n_target = len(target_gene_names)
    X_aligned = np.zeros((n_cells, n_target), dtype=np.float32)
    gene_to_pos = {g: i for i, g in enumerate(adata.var_names)}

    # Extract columns in chunks (memory-efficient for sparse matrices)
    X_full = adata.X
    is_sparse = scipy.sparse.issparse(X_full)

    for j, g in enumerate(target_gene_names):
        if g in gene_to_pos:
            col = X_full[:, gene_to_pos[g]]
            if is_sparse:
                col = col.toarray().flatten()
            else:
                col = np.asarray(col).flatten()
            X_aligned[:, j] = col.astype(np.float32)

    # Build new AnnData with aligned genes
    adata_aligned = ad.AnnData(
        X=X_aligned,
        obs=adata.obs.reset_index(drop=True).copy(),
        var=pd.DataFrame(index=target_gene_names),
    )

    # Parse perturbations — Replogle uses 'gene' column
    if 'gene' not in adata_aligned.obs.columns:
        raise ValueError("Expected 'gene' column in Replogle obs")

    gene_labels = adata_aligned.obs['gene'].astype(str).values

    control_idx = np.where(gene_labels == 'non-targeting')[0].tolist()
    print(f"  Control cells (non-targeting): {len(control_idx)}")

    # Collect single perturbations — only keep those whose gene is in target set
    # (we can't evaluate a perturbation on a gene we don't have in our gene axis)
    single_perts = {}
    for i, g in enumerate(gene_labels):
        if g == 'non-targeting' or g == 'nan':
            continue
        if g not in target_set:
            continue  # Can't evaluate — gene not in model's output
        single_perts.setdefault(g, []).append(i)

    # Filter to perturbations with enough cells
    single_perts = {g: idxs for g, idxs in single_perts.items()
                    if len(idxs) >= min_cells_per_pert}
    print(f"  Single perturbations with >={min_cells_per_pert} cells: {len(single_perts)}")

    # Compute de_single (mean perturbed - mean control)
    if len(control_idx) > 0:
        ctrl_mean = X_aligned[control_idx].mean(axis=0)
    else:
        ctrl_mean = X_aligned.mean(axis=0)

    print("  Computing differential expression...")
    adata_aligned.uns['de_single'] = {}
    for gene, idxs in single_perts.items():
        pert_mean = X_aligned[idxs].mean(axis=0)
        adata_aligned.uns['de_single'][gene] = pert_mean - ctrl_mean

    # Replogle has no combos
    adata_aligned.uns['de_combo'] = {}

    return {
        'adata': adata_aligned,
        'control_idx': control_idx,
        'single_perts': single_perts,
        'combo_perts': {},
        'gene_names': list(target_gene_names),
        'pert_genes': sorted(single_perts.keys()),
    }


if __name__ == '__main__':
    # Smoke test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/large_storage/goodarzilab/causal_state/pretrain_data/replogle')
    parser.add_argument('--max_cells', type=int, default=20000)
    args = parser.parse_args()

    # Use a dummy target gene set for testing
    dummy_target = ['TP53', 'MYC', 'CEBPA', 'GATA1', 'SPI1']
    dd = load_replogle_aligned(args.path, dummy_target, max_cells=args.max_cells)
    print(f"\nData dict keys: {list(dd.keys())}")
    print(f"  adata shape: {dd['adata'].shape}")
    print(f"  perturbed genes: {dd['pert_genes']}")
    print(f"  de_single keys: {list(dd['adata'].uns['de_single'].keys())}")

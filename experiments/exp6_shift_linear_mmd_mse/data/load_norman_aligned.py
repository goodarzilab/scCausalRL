"""Load Norman dataset aligned to a target gene set (e.g., Replogle's HVGs).

Used for external validation: forces Norman data into the same gene coordinate
system as the trained model. Genes in target set but missing from Norman are
zero-imputed. Genes in Norman but not in target set are dropped.

Used to validate a Replogle-trained model on Norman perturbations (cross-modality
test: CRISPRi-trained → CRISPRa eval, same cell line K562).
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse


def load_norman_aligned(path, target_gene_names, min_cells_per_pert=5, src_n_hvg=2000):
    """
    Load Norman and align to a fixed gene set.

    Args:
        path: directory with Norman MTX files (barcodes.tsv.gz, genes.tsv.gz,
              matrix.mtx.gz, cell_identities.csv.gz)
        target_gene_names: list of gene names (from Replogle-trained checkpoint)
        min_cells_per_pert: minimum cells per perturbation for DE computation

    Returns: data_dict with keys matching load_norman structure (including combos)
    """
    from data.load_norman import load_norman
    import anndata as ad

    # Reuse Norman's full loader (handles MTX parsing, perturbation classification,
    # HVG selection, DE precomputation). We'll then re-align gene axis.
    print(f"Loading Norman from {path}...")
    # force_include the target genes so EVERY K562-axis gene present in raw Norman
    # survives HVG filtering (otherwise only Norman's top-n_hvg ∩ target survive,
    # which zero-imputes ~90% of the axis and cripples the external eval).
    norman = load_norman(path=path, n_hvg=src_n_hvg, force_include=list(target_gene_names))

    src_adata = norman['adata']
    src_gene_names = list(src_adata.var_names)
    src_gene_to_idx = {g: i for i, g in enumerate(src_gene_names)}

    # Align to target gene axis
    target_set = set(target_gene_names)
    src_set = set(src_gene_names)
    overlap = target_set & src_set
    missing = target_set - src_set

    print(f"  Gene alignment: {len(overlap)}/{len(target_gene_names)} in Norman HVGs, "
          f"{len(missing)} zero-imputed")

    # Build aligned matrix in target order
    X_src = src_adata.X.toarray() if scipy.sparse.issparse(src_adata.X) else np.array(src_adata.X)
    X_src = X_src.astype(np.float32)
    n_cells = X_src.shape[0]
    n_target = len(target_gene_names)
    X_aligned = np.zeros((n_cells, n_target), dtype=np.float32)
    for j, g in enumerate(target_gene_names):
        if g in src_gene_to_idx:
            X_aligned[:, j] = X_src[:, src_gene_to_idx[g]]

    adata_aligned = ad.AnnData(
        X=X_aligned,
        obs=src_adata.obs.reset_index(drop=True).copy(),
        var=pd.DataFrame(index=target_gene_names),
    )
    adata_aligned.obs.index = adata_aligned.obs.index.astype(str)

    # Carry over perturbation indices (cell indices are unchanged — only columns subset)
    control_idx = list(norman['control_idx'])

    # Filter perturbations: keep only those whose perturbed gene is in target set
    # (otherwise we can't evaluate — the dx vector has no signal at the perturbed gene)
    target_gene_set = set(target_gene_names)
    single_perts = {g: idxs for g, idxs in norman['single_perts'].items()
                    if g in target_gene_set and len(idxs) >= min_cells_per_pert}

    combo_perts = {}
    for (ga, gb), idxs in norman['combo_perts'].items():
        if ga in target_gene_set and gb in target_gene_set and len(idxs) >= min_cells_per_pert:
            combo_perts[(ga, gb)] = idxs

    print(f"  Evaluable single perts (genes in target set): "
          f"{len(single_perts)}/{len(norman['single_perts'])}")
    print(f"  Evaluable combo perts (both genes in target set): "
          f"{len(combo_perts)}/{len(norman['combo_perts'])}")

    # Recompute DE on aligned gene axis (controls are same cells; new gene columns)
    if len(control_idx) > 0:
        ctrl_mean = X_aligned[control_idx].mean(axis=0)
    else:
        ctrl_mean = X_aligned.mean(axis=0)

    print("  Computing differential expression on aligned axis...")
    adata_aligned.uns['de_single'] = {}
    for gene, idxs in single_perts.items():
        adata_aligned.uns['de_single'][gene] = X_aligned[idxs].mean(axis=0) - ctrl_mean

    adata_aligned.uns['de_combo'] = {}
    for key, idxs in combo_perts.items():
        adata_aligned.uns['de_combo'][key] = X_aligned[idxs].mean(axis=0) - ctrl_mean

    return {
        'adata': adata_aligned,
        'control_idx': control_idx,
        'single_perts': single_perts,
        'combo_perts': combo_perts,
        'gene_names': list(target_gene_names),
        'pert_genes': sorted(single_perts.keys()),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--norman_path', default='/large_storage/goodarzilab/shervin/scCausalRL/Norman/')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Dummy target gene set for smoke test
    dummy_target = ['CEBPA', 'CEBPB', 'CEBPE', 'GATA1', 'TP53', 'MYC', 'KLF1']
    dd = load_norman_aligned(args.norman_path, dummy_target)
    print(f"\nData dict: adata={dd['adata'].shape}, "
          f"single={len(dd['single_perts'])}, combo={len(dd['combo_perts'])}")

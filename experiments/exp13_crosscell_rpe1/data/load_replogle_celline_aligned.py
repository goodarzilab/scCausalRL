"""Load a Replogle cell line (RPE1 / HepG2 / Jurkat) aligned to a target gene axis.

Cross-cell-line external validation: a model trained on K562 (CRISPRi) tested on a
DIFFERENT cell line of the SAME Replogle CRISPRi screen. Same modality (knockdown)
→ no CRISPRi↔CRISPRa sign flip (unlike Norman), so AUROC AND Pearson are both
meaningful. Aligns the source onto the checkpoint's exact gene axis (missing genes
zero-imputed), mirroring load_norman_aligned.
"""
import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad


def load_replogle_celline_aligned(path, target_gene_names, cell_line='rpe1',
                                  max_cells=200000, min_cells_per_pert=5, src_n_hvg=10000):
    from data.load_replogle import load_replogle
    print(f"Loading Replogle {cell_line} from {path} ...")
    src = load_replogle(path=path, n_hvg=src_n_hvg, cell_line=cell_line,
                        max_cells=max_cells, force_include=list(target_gene_names))
    src_adata = src['adata']
    src_idx = {g: i for i, g in enumerate(src_adata.var_names)}
    target = list(target_gene_names)
    overlap = set(target) & set(src_adata.var_names)
    print(f"  Gene alignment: {len(overlap)}/{len(target)} target genes present in {cell_line}, "
          f"{len(target) - len(overlap)} zero-imputed")

    X_src = (src_adata.X.toarray() if scipy.sparse.issparse(src_adata.X) else np.asarray(src_adata.X)).astype(np.float32)
    X_aligned = np.zeros((X_src.shape[0], len(target)), dtype=np.float32)
    for j, g in enumerate(target):
        if g in src_idx:
            X_aligned[:, j] = X_src[:, src_idx[g]]

    adata_aligned = ad.AnnData(X=X_aligned, obs=src_adata.obs.reset_index(drop=True).copy(),
                               var=pd.DataFrame(index=target))
    adata_aligned.obs.index = adata_aligned.obs.index.astype(str)

    control_idx = list(src['control_idx'])
    tset = set(target)
    single_perts = {g: idxs for g, idxs in src['single_perts'].items()
                    if g in tset and len(idxs) >= min_cells_per_pert}
    print(f"  Evaluable single perts (target gene on axis): {len(single_perts)}/{len(src['single_perts'])}")
    ctrl_mean = X_aligned[control_idx].mean(0) if control_idx else X_aligned.mean(0)
    adata_aligned.uns['de_single'] = {g: X_aligned[idxs].mean(0) - ctrl_mean for g, idxs in single_perts.items()}

    return {
        'adata': adata_aligned, 'control_idx': control_idx, 'single_perts': single_perts,
        'combo_perts': {}, 'gene_names': target, 'pert_genes': sorted(single_perts.keys()),
        'pert_split': {'train': set(), 'val': set(), 'test': set(single_perts.keys())},
    }

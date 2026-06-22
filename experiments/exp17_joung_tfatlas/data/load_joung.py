"""Loader for the preprocessed Joung TF Atlas singles (world-model training).

Reads the compact, axis-aligned h5ad written by preprocess_joung.py (counts on the
shared axis), library-normalizes + log1p, and returns the standard training data_dict:
control = GFP/mCherry cells; single_perts keyed by TF GENE symbol (isoforms merged);
pert_split is a deterministic gene-level split. Combos/triples are held out entirely
(evaluated separately from joung_combo_proc.h5ad). Single-modality (overexpression), so
no modality embedding — same contract as load_norman.
"""
import os
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd

from data.load_norman import split_single_perts

DEFAULT_PROC = os.path.join(os.path.dirname(__file__), 'joung_singles_proc.h5ad')


def _lognorm(X, target=1e4):
    """Library-size normalize to `target` + log1p, preserving sparsity."""
    X = X.tocsr().astype(np.float32) if sp.issparse(X) else sp.csr_matrix(X, dtype=np.float32)
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1.0
    X = X.multiply(1.0 / lib[:, None] * target).tocsr()
    X.data = np.log1p(X.data)
    return X


def load_joung(proc_path=DEFAULT_PROC, n_hvg=None, train_frac=0.8, val_frac=0.1,
               split_seed=42, min_cells_per_pert=5, **kwargs):
    adata = ad.read_h5ad(proc_path)
    adata.X = _lognorm(adata.X)
    gene_names = list(map(str, adata.var_names))
    gset = set(gene_names)

    is_ctrl = adata.obs['is_control'].values.astype(bool)
    control_idx = list(np.where(is_ctrl)[0])
    tfg = adata.obs['TF_gene'].astype(str).values

    single_perts = {}
    for i in np.where(~is_ctrl)[0]:
        g = tfg[i]
        if g in gset:
            single_perts.setdefault(g, []).append(int(i))
    # keep TFs with enough cells AND whose gene is on the axis (has a pert_label)
    single_perts = {g: idx for g, idx in single_perts.items() if len(idx) >= min_cells_per_pert}

    Xd = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    ctrl_mean = Xd[control_idx].mean(0)
    de_single = {g: Xd[idx].mean(0) - ctrl_mean for g, idx in single_perts.items()}
    adata.uns['de_single'] = de_single

    pert_split = split_single_perts(single_perts, train_frac=train_frac,
                                    val_frac=val_frac, seed=split_seed)
    print(f"[load_joung] {adata.shape}  TFs={len(single_perts)} (train/val/test="
          f"{len(pert_split['train'])}/{len(pert_split['val'])}/{len(pert_split['test'])})  "
          f"control={len(control_idx)}")
    return {
        'adata': adata, 'gene_names': gene_names, 'control_idx': control_idx,
        'single_perts': single_perts, 'combo_perts': {},
        'pert_genes': sorted(single_perts.keys()), 'pert_split': pert_split,
        'de_single': de_single,
    }

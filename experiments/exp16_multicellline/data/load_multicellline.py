"""Load 4 Replogle+Nadig CRISPRi cell lines on a shared gene axis.

K562 (Replogle) + RPE1 (Replogle) + HepG2 (Nadig/GSE264667) + Jurkat (Nadig), all
CRISPRi single-gene Perturb-seq. ~2053 perturbations are shared across all 4 lines,
so the model sees the SAME perturbation in 4 cell contexts → can learn cell-line-
invariant regulatory structure (+ a per-cell-line shift scale). Cell line is encoded
as the `modality_id` covariate (reusing the modality machinery; all are CRISPRi so the
modality slot is free for cell line).

Perturbations namespaced 'GENE__k562' etc.; gem_group offset per line so control
pairing stays within a cell line. Genes aligned to the intersection of the 4 axes.
"""
import numpy as np
import pandas as pd
import anndata as ad


def _subset_reorder(adata, genes):
    """Return adata with columns reordered to `genes` (all assumed present)."""
    idx = [adata.var_names.get_loc(g) for g in genes]
    import scipy.sparse
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    return ad.AnnData(X=X[:, idx].astype(np.float32),
                      obs=adata.obs.reset_index(drop=True).copy(),
                      var=pd.DataFrame(index=list(genes)))


def load_multicellline(files_dict, max_cells_per_line=150000, n_hvg=50000,
                       train_frac=0.8, val_frac=0.1, split_seed=42):
    from data.load_replogle import load_replogle
    print(f"[multicellline] loading {len(files_dict)} cell lines: {list(files_dict)}")
    loaded = {}
    for cl, fn in files_dict.items():
        # n_hvg huge ⇒ keep all genes (so the intersection is the true shared axis).
        loaded[cl] = load_replogle(path=fn, n_hvg=n_hvg, cell_line=cl, max_cells=max_cells_per_line)
        print(f"  {cl}: {loaded[cl]['adata'].shape}  {len(loaded[cl]['single_perts'])} perts  "
              f"{len(loaded[cl]['control_idx'])} ctrl")

    shared = sorted(set.intersection(*[set(d['adata'].var_names) for d in loaded.values()]))
    print(f"  shared gene axis (intersection): {len(shared)} genes")

    adatas, control_idx, single_perts = [], [], {}
    modality_to_id = {cl: i for i, cl in enumerate(files_dict)}
    offset, gg_offset = 0, 0
    bid_all, mod_all = [], []
    for cl, d in loaded.items():
        a = _subset_reorder(d['adata'], shared)
        n = a.n_obs
        bid = (a.obs['batch_id'].values.astype(np.int64) if 'batch_id' in a.obs.columns
               else np.zeros(n, dtype=np.int64)) + gg_offset
        bid_all.append(bid)
        mod_all.append(np.full(n, modality_to_id[cl], dtype=np.int64))
        control_idx.extend([offset + j for j in d['control_idx']])
        for g, idxs in d['single_perts'].items():
            single_perts[f"{g}__{cl}"] = [offset + j for j in idxs]
        offset += n
        gg_offset = int(bid.max()) + 1
        adatas.append(a)

    combined = ad.concat(adatas, join='inner', index_unique=None)
    combined.var = pd.DataFrame(index=list(shared))
    combined.obs = combined.obs.reset_index(drop=True)
    combined.obs['batch_id'] = np.concatenate(bid_all)
    combined.obs['modality_id'] = np.concatenate(mod_all)
    gene_names = list(shared)
    n_batches = int(combined.obs['batch_id'].max()) + 1
    print(f"  combined: {combined.shape}  {len(single_perts)} namespaced perts  "
          f"{len(control_idx)} ctrl  {n_batches} gem_groups  {len(modality_to_id)} cell lines")

    # per-namespaced DE
    import scipy.sparse
    X = combined.X.toarray() if scipy.sparse.issparse(combined.X) else np.asarray(combined.X)
    X = X.astype(np.float32)
    ctrl_mean = X[control_idx].mean(0) if control_idx else X.mean(0)
    combined.uns['de_single'] = {k: X[idxs].mean(0) - ctrl_mean for k, idxs in single_perts.items()}

    # gene-level split (un-namespaced) — deterministic
    from data.load_norman import split_single_perts
    base_perts = {}
    for k, idxs in single_perts.items():
        g = k.split('__')[0]
        base_perts.setdefault(g, []).extend(idxs)
    pert_split = split_single_perts(base_perts, train_frac=train_frac, val_frac=val_frac, seed=split_seed)

    return {
        'adata': combined, 'control_idx': control_idx, 'single_perts': single_perts,
        'combo_perts': {}, 'gene_names': gene_names,
        'pert_genes': sorted(base_perts.keys()), 'pert_split': pert_split,
        'modality_map': modality_to_id, 'n_modalities': len(modality_to_id),
        'n_batches': n_batches,
    }

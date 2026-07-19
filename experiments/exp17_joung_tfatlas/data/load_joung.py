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


_COMBO_FILLERS = {'GFP', 'mCherry', 'mcherry', 'EGFP'}


def _parse_tf_set(s):
    out = []
    for tok in str(s).split(','):
        sym = tok.split('-')[-1].strip()
        if sym and sym not in _COMBO_FILLERS:
            out.append(sym)
    return out


def _read_gene_list(x):
    """Accept a path (newline-separated), a list/set, or None -> returns set or None."""
    if x is None:
        return None
    if isinstance(x, (set, list, tuple)):
        return set(map(str, x))
    return {g.strip() for g in open(x) if g.strip()}


def load_joung(proc_path=DEFAULT_PROC, n_hvg=None, train_frac=0.8, val_frac=0.1,
               split_seed=42, min_cells_per_pert=5, load_combos=False,
               combo_proc_path=None, min_combo_cells=20,
               gene_vocab=None, effective_tfs=None, **kwargs):
    # gene_vocab: path/list of genes to SUBSET the axis to (vocab-v2). effective_tfs: path/list of
    # TF genes to KEEP as perturbations (drop inert TFs from the pert vocab; their cells still serve
    # reconstruction). Both default None = legacy behavior (all proc genes / all TFs).
    adata = ad.read_h5ad(proc_path)
    gene_vocab = _read_gene_list(gene_vocab)
    if gene_vocab is not None:
        keep = [g for g in map(str, adata.var_names) if g in gene_vocab]
        miss = len(gene_vocab) - len(keep)
        adata = adata[:, keep].copy()
        print(f"[load_joung] gene_vocab: subset to {len(keep)} genes "
              f"({miss} vocab genes absent from proc axis)")
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

    # effective-TF filter: keep only TFs that actually drive a response (drop inert). Cells of
    # dropped TFs remain in `adata` (reconstruction) but are not perturbation training targets.
    eff = _read_gene_list(effective_tfs)
    if eff is not None:
        before = len(single_perts)
        single_perts = {g: idx for g, idx in single_perts.items() if g in eff}
        print(f"[load_joung] effective_tfs: kept {len(single_perts)}/{before} TFs "
              f"({before-len(single_perts)} inert/absent dropped from pert vocab)")

    Xd = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    ctrl_mean = Xd[control_idx].mean(0)
    de_single = {g: Xd[idx].mean(0) - ctrl_mean for g, idx in single_perts.items()}
    adata.uns['de_single'] = de_single
    del Xd

    pert_split = split_single_perts(single_perts, train_frac=train_frac,
                                    val_frac=val_frac, seed=split_seed)

    # ── Optional: merge combo PAIRS for few-shot combo training (Tier-3). Combo cells
    # live in a separate, axis-aligned h5ad; concatenate them so combo_perts index into
    # the merged matrix. Pairs only (the dataset pads gene_idx to [2]); triples stay
    # zero-shot eval. The TRAIN/TEST combo split is taken later via combo_train_frac.
    combo_perts, de_combo = {}, {}
    if load_combos:
        cpath = combo_proc_path or os.path.join(os.path.dirname(proc_path), 'joung_combo_proc.h5ad')
        C = ad.read_h5ad(cpath)
        assert list(map(str, C.var_names)) == gene_names, "combo gene axis != singles axis"
        C.X = _lognorm(C.X)
        ts = C.obs['TF_set'].astype(str).values
        groups = {}
        for i, s in enumerate(ts):
            tfs = tuple(sorted(set(t for t in _parse_tf_set(s) if t in gset)))
            if len(tfs) == 2:                       # pairs only
                groups.setdefault(tfs, []).append(i)
        groups = {k: v for k, v in groups.items() if len(v) >= min_combo_cells}
        n0 = adata.n_obs
        Xc = C.X.toarray() if sp.issparse(C.X) else np.asarray(C.X)
        for k, local in groups.items():
            combo_perts[k] = [n0 + i for i in local]
            de_combo[k] = Xc[np.asarray(local)].mean(0) - ctrl_mean
        del Xc
        adata = ad.concat([adata, C], join='outer')
        adata.uns['de_single'] = de_single
        adata.uns['de_combo'] = de_combo
        print(f"[load_joung] merged {len(combo_perts)} combo pairs "
              f"({sum(len(v) for v in combo_perts.values())} cells) for combo training")

    print(f"[load_joung] {adata.shape}  TFs={len(single_perts)} (train/val/test="
          f"{len(pert_split['train'])}/{len(pert_split['val'])}/{len(pert_split['test'])})  "
          f"control={len(control_idx)}  combo_pairs={len(combo_perts)}")
    return {
        'adata': adata, 'gene_names': gene_names, 'control_idx': control_idx,
        'single_perts': single_perts, 'combo_perts': combo_perts,
        'pert_genes': sorted(single_perts.keys()), 'pert_split': pert_split,
        'de_single': de_single, 'de_combo': de_combo,
    }

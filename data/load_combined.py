"""Combined Norman (CRISPRa) + Replogle (CRISPRi) loader for joint Phase 1 training.

Returns a unified data_dict on a SHARED HVG axis with a per-cell `modality` label.
Per-dataset DE vectors are stored separately (modality-aware): Norman DE has
positive direction (overexpression), Replogle DE has negative direction (knockdown).

Perturbations are namespaced as 'GENE__norman' or 'GENE__replogle' so the
training loop can sample interventional batches from either dataset and use
the modality-matched dx.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse


# Modality IDs — used by encoder embedding lookup
MODALITY_CONTROL = 0
MODALITY_CRISPRA = 1
MODALITY_CRISPRI = 2

MODALITY_NAMES = {
    MODALITY_CONTROL: 'control',
    MODALITY_CRISPRA: 'CRISPRa',
    MODALITY_CRISPRI: 'CRISPRi',
}


def subset_adata_to_genes(adata, gene_list):
    """Subset AnnData to a target gene list (in given order). Genes missing
    from `adata` are zero-imputed."""
    import anndata as ad

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)

    existing = {g: i for i, g in enumerate(adata.var_names)}
    n_cells = X.shape[0]
    n_target = len(gene_list)
    X_sub = np.zeros((n_cells, n_target), dtype=np.float32)
    for j, g in enumerate(gene_list):
        if g in existing:
            X_sub[:, j] = X[:, existing[g]]

    new_adata = ad.AnnData(
        X=X_sub,
        obs=adata.obs.reset_index(drop=True).copy(),
        var=pd.DataFrame(index=gene_list),
    )
    return new_adata


def split_genes(all_genes, train_frac=0.8, val_frac=0.1, seed=42):
    """Gene-level split. Same gene gets the same split assignment regardless of dataset."""
    rng = np.random.default_rng(seed)
    genes = sorted(set(all_genes))
    n = len(genes)
    perm = rng.permutation(n)
    n_train = int(np.round(n * train_frac))
    n_val = int(np.round(n * val_frac))
    return {
        'train': set(genes[perm[i]] for i in range(n_train)),
        'val': set(genes[perm[i]] for i in range(n_train, n_train + n_val)),
        'test': set(genes[perm[i]] for i in range(n_train + n_val, n)),
    }


def load_combined(
    norman_path,
    replogle_path,
    n_hvg=2000,
    replogle_max_cells=100000,
    train_frac=0.8,
    val_frac=0.1,
    split_seed=42,
):
    """Load Norman + Replogle on a shared HVG axis with modality labels.

    Returns data_dict with:
      adata          : AnnData on shared HVG axis, with obs['modality'] in
                       {'control', 'CRISPRa', 'CRISPRi'} and obs['modality_id']
                       in {0, 1, 2}
      control_idx    : indices of all control cells (combined)
      single_perts   : dict {namespaced_key: [cell_indices]} where
                       namespaced_key = 'GENE__norman' or 'GENE__replogle'
      combo_perts    : dict {(ga, gb): [cell_indices]} — Norman only
      gene_names     : list of shared HVGs
      pert_genes     : sorted unique GENE names (un-namespaced)
      pert_split     : {'train', 'val', 'test'} sets of GENE names (un-namespaced)
      modality_map   : str -> int mapping for encoder embedding lookup

    adata.uns['de_single'] keyed by namespaced_key (modality-matched).
    adata.uns['de_combo']  keyed by (ga, gb) tuple from Norman.
    """
    import anndata as ad
    from data.load_norman import load_norman
    from data.load_replogle import load_replogle

    print("=" * 70)
    print("LOADING COMBINED NORMAN + REPLOGLE DATASET")
    print("=" * 70)

    # Step 1: Load each separately (each does its own HVG selection / preprocessing)
    print("\n[1/8] Loading Norman...")
    norman = load_norman(
        path=norman_path, n_hvg=n_hvg,
        train_frac=train_frac, val_frac=val_frac, split_seed=split_seed,
    )
    print(f"\n[2/8] Loading Replogle (max {replogle_max_cells} cells)...")
    replogle = load_replogle(
        path=replogle_path, n_hvg=n_hvg, max_cells=replogle_max_cells,
    )

    # Step 2: Find shared HVG axis (intersection)
    norman_genes = set(norman['gene_names'])
    replogle_genes = set(replogle['gene_names'])
    shared_genes = sorted(norman_genes & replogle_genes)
    print(f"\n[3/8] Shared HVG axis: {len(shared_genes)} genes "
          f"(Norman {len(norman_genes)}, Replogle {len(replogle_genes)})")

    # Step 3: Subset both AnnDatas to shared genes (in same order)
    print(f"\n[4/8] Subsetting Norman to shared genes...")
    norman_adata = subset_adata_to_genes(norman['adata'], shared_genes)
    norman_adata.obs['dataset'] = 'norman'

    print(f"[4/8] Subsetting Replogle to shared genes...")
    replogle_adata = subset_adata_to_genes(replogle['adata'], shared_genes)
    replogle_adata.obs['dataset'] = 'replogle'

    # Step 4: Add per-cell modality labels
    # Norman: control cells get 'control', perturbed cells get 'CRISPRa'
    # Replogle: control cells get 'control', perturbed cells get 'CRISPRi'
    print(f"\n[5/8] Assigning modality labels...")
    n_norman = norman_adata.n_obs
    n_replogle = replogle_adata.n_obs

    norman_modality = np.full(n_norman, 'CRISPRa', dtype=object)
    norman_control_set = set(norman['control_idx'])
    for i in range(n_norman):
        if i in norman_control_set:
            norman_modality[i] = 'control'
    norman_adata.obs['modality'] = norman_modality

    replogle_modality = np.full(n_replogle, 'CRISPRi', dtype=object)
    replogle_control_set = set(replogle['control_idx'])
    for i in range(n_replogle):
        if i in replogle_control_set:
            replogle_modality[i] = 'control'
    replogle_adata.obs['modality'] = replogle_modality

    # Step 5: Concatenate (Norman first, Replogle second)
    print(f"\n[6/8] Concatenating: Norman ({n_norman}) + Replogle ({n_replogle})")
    combined = ad.concat([norman_adata, replogle_adata], join='outer', index_unique=None)
    # Rebuild a clean integer index
    combined.obs = combined.obs.reset_index(drop=True)
    combined.obs.index = combined.obs.index.astype(str)

    # Add integer modality_id for encoder embedding
    modality_to_id = {'control': MODALITY_CONTROL,
                      'CRISPRa': MODALITY_CRISPRA,
                      'CRISPRi': MODALITY_CRISPRI}
    combined.obs['modality_id'] = combined.obs['modality'].map(modality_to_id).astype(int)

    n_combined = combined.n_obs
    print(f"  Combined shape: {combined.shape}")
    print(f"  Modality counts:")
    for mod, count in combined.obs['modality'].value_counts().items():
        print(f"    {mod:10s}: {count}")

    # Step 6: Re-index control_idx and single_perts with namespacing
    # Indices: Norman cells 0..n_norman-1, Replogle cells n_norman..n_norman+n_replogle-1
    print(f"\n[7/8] Building combined index dicts (namespaced)...")
    norman_offset = 0
    replogle_offset = n_norman

    # Combined control set
    control_idx = []
    control_idx.extend([norman_offset + i for i in norman['control_idx']])
    control_idx.extend([replogle_offset + i for i in replogle['control_idx']])

    # Namespaced single_perts: 'GENE__norman' / 'GENE__replogle'
    single_perts = {}
    for gene, idxs in norman['single_perts'].items():
        single_perts[f'{gene}__norman'] = [norman_offset + i for i in idxs]
    for gene, idxs in replogle['single_perts'].items():
        single_perts[f'{gene}__replogle'] = [replogle_offset + i for i in idxs]

    # Combo perts: Norman only (Replogle has none)
    combo_perts = {}
    for key, idxs in norman['combo_perts'].items():
        combo_perts[key] = [norman_offset + i for i in idxs]

    print(f"  Single perturbations (namespaced): {len(single_perts)}")
    print(f"  Combo perturbations (Norman only):  {len(combo_perts)}")
    print(f"  Control cells (combined):          {len(control_idx)}")

    # Step 7: Per-namespaced DE (modality-matched)
    # Reuse the precomputed de_single from each dataset (computed against
    # same-dataset controls), but project onto shared gene axis.
    print(f"\n[8/8] Building modality-matched DE vectors...")
    norman_gene_to_idx = {g: i for i, g in enumerate(norman['gene_names'])}
    replogle_gene_to_idx = {g: i for i, g in enumerate(replogle['gene_names'])}
    shared_idx_in_norman = np.array([norman_gene_to_idx[g] for g in shared_genes])
    shared_idx_in_replogle = np.array([replogle_gene_to_idx[g] for g in shared_genes])

    de_single_combined = {}
    for gene, dx_full in norman['adata'].uns.get('de_single', {}).items():
        de_single_combined[f'{gene}__norman'] = np.asarray(dx_full, dtype=np.float32)[shared_idx_in_norman]
    for gene, dx_full in replogle['adata'].uns.get('de_single', {}).items():
        de_single_combined[f'{gene}__replogle'] = np.asarray(dx_full, dtype=np.float32)[shared_idx_in_replogle]

    de_combo_combined = {}
    for key, dx_full in norman['adata'].uns.get('de_combo', {}).items():
        de_combo_combined[key] = np.asarray(dx_full, dtype=np.float32)[shared_idx_in_norman]

    combined.uns['de_single'] = de_single_combined
    combined.uns['de_combo'] = de_combo_combined

    # Step 8: Gene-level split (un-namespaced gene names)
    # Same gene gets same split in both datasets — fair held-out comparison
    all_pert_genes = set()
    for k in single_perts.keys():
        gene = k.split('__')[0]
        all_pert_genes.add(gene)
    pert_split = split_genes(all_pert_genes, train_frac=train_frac,
                             val_frac=val_frac, seed=split_seed)
    print(f"\nPert split (gene-level, applies to both modalities):")
    print(f"  train: {len(pert_split['train'])} genes")
    print(f"  val:   {len(pert_split['val'])} genes")
    print(f"  test:  {len(pert_split['test'])} genes")

    return {
        'adata': combined,
        'control_idx': control_idx,
        'single_perts': single_perts,         # namespaced keys
        'combo_perts': combo_perts,            # Norman only
        'gene_names': shared_genes,
        'pert_genes': sorted(all_pert_genes),  # un-namespaced
        'pert_split': pert_split,              # un-namespaced gene names
        'modality_map': modality_to_id,
        'n_modalities': 3,
    }


if __name__ == '__main__':
    # Smoke test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--norman_path', default='/large_storage/goodarzilab/shervin/CausalBioRL/Norman/')
    parser.add_argument('--replogle_path', default='/large_storage/goodarzilab/causal_state/pretrain_data/replogle')
    parser.add_argument('--n_hvg', type=int, default=2000)
    parser.add_argument('--max_cells', type=int, default=20000)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    dd = load_combined(
        norman_path=args.norman_path,
        replogle_path=args.replogle_path,
        n_hvg=args.n_hvg,
        replogle_max_cells=args.max_cells,
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST — COMBINED DATA DICT")
    print("=" * 70)
    print(f"adata.shape: {dd['adata'].shape}")
    print(f"gene_names: {len(dd['gene_names'])} genes")
    print(f"single_perts (namespaced): {len(dd['single_perts'])} keys")
    print(f"  Examples: {list(dd['single_perts'].keys())[:5]}")
    print(f"combo_perts: {len(dd['combo_perts'])} combos")
    print(f"control_idx: {len(dd['control_idx'])} cells")
    print(f"de_single keys: {len(dd['adata'].uns['de_single'])}")
    print(f"  Sample shapes: {next(iter(dd['adata'].uns['de_single'].values())).shape}")
    print(f"pert_split: {[(k, len(v)) for k, v in dd['pert_split'].items()]}")
    print(f"modality_id distribution:")
    print(dd['adata'].obs['modality_id'].value_counts())

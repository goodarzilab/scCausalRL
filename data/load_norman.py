import os
import gzip
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import anndata as ad
import scanpy as sc
from .preprocess import preprocess


def _parse_guide_identity(guide_str):
    """
    Parse Norman guide_identity string into perturbation type and gene names.

    Format: "PART1__PART2" where PART1 and PART2 are identical
    Each PART is "GENE1_GENE2" where:
      - if GENE2 contains "NegCtrl" -> single perturbation of GENE1
      - if neither contains "NegCtrl" -> combo of GENE1 + GENE2
      - if GENE1 also contains "NegCtrl" -> control

    Returns: (pert_type, gene_list)
      pert_type: 'control', 'single', 'combo'
      gene_list: [] for control, [gene] for single, [gene1, gene2] for combo
    """
    if pd.isna(guide_str):
        return 'control', []

    # Take first half (before __)
    first_part = guide_str.split('__')[0]
    parts = first_part.split('_')

    if len(parts) == 0:
        return 'control', []

    # Check each part for NegCtrl
    real_genes = [p for p in parts if 'NegCtrl' not in p and p != '']
    neg_ctrl_parts = [p for p in parts if 'NegCtrl' in p]

    if len(real_genes) == 0:
        return 'control', []
    elif len(real_genes) == 1:
        return 'single', real_genes
    else:
        # All parts are real genes -> combo
        return 'combo', real_genes[:2]  # take first two


def split_single_perts(single_perts, train_frac=0.8, val_frac=0.1, seed=42):
    """Split single perturbation genes into train/val/test sets.

    Split is by GENE (not by cell) — all cells for a given gene go to the same set.
    This prevents data leakage: the perturbation consistency loss only trains on
    train-split genes; val/test genes are never seen during training.

    Returns: dict with 'train', 'val', 'test' keys → sets of gene names
    """
    rng = np.random.default_rng(seed)
    genes = sorted(single_perts.keys())
    n = len(genes)
    perm = rng.permutation(n)
    n_train = int(np.round(n * train_frac))
    n_val = int(np.round(n * val_frac))
    return {
        'train': set(genes[perm[i]] for i in range(n_train)),
        'val': set(genes[perm[i]] for i in range(n_train, n_train + n_val)),
        'test': set(genes[perm[i]] for i in range(n_train + n_val, n)),
    }


def load_norman(path=None, n_hvg=2000, **kwargs):
    """
    Load Norman et al. 2019 K562 Perturb-seq dataset from raw MTX format.

    kwargs: train_frac (0.8), val_frac (0.1), split_seed (42) for gene split.

    Returns dict with:
        adata: preprocessed AnnData
        control_idx: indices of control cells
        single_perts: dict gene_name -> cell indices
        combo_perts: dict (gene_a, gene_b) -> cell indices
        gene_names: list of gene names in expression matrix
        pert_genes: list of all perturbed genes (action space)
    """
    if path is None:
        path = "/large_storage/goodarzilab/shervin/CausalBioRL/Norman/"

    print("Loading Norman dataset from MTX format...")

    # Load barcodes
    barcodes_file = os.path.join(path, "GSE133344_filtered_barcodes.tsv.gz")
    with gzip.open(barcodes_file, 'rt') as f:
        barcodes = [line.strip() for line in f]
    print(f"  Loaded {len(barcodes)} barcodes")

    # Load genes
    genes_file = os.path.join(path, "GSE133344_filtered_genes.tsv.gz")
    gene_ids = []
    gene_names = []
    with gzip.open(genes_file, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            gene_ids.append(parts[0])
            gene_names.append(parts[1] if len(parts) > 1 else parts[0])
    print(f"  Loaded {len(gene_names)} genes")

    # Load matrix
    matrix_file = os.path.join(path, "GSE133344_filtered_matrix.mtx.gz")
    print("  Loading MTX matrix (this may take a moment)...")
    with gzip.open(matrix_file, 'rb') as f:
        mat = scipy.io.mmread(f).tocsc()
    # MTX format: rows=genes, cols=cells -> transpose to cells x genes
    mat = mat.T.tocsr()
    print(f"  Matrix shape (cells x genes): {mat.shape}")

    # Load cell identities
    cell_id_file = os.path.join(path, "GSE133344_filtered_cell_identities.csv.gz")
    cell_meta = pd.read_csv(cell_id_file, compression='gzip')
    print(f"  Cell identities shape: {cell_meta.shape}")
    print(f"  Columns: {list(cell_meta.columns)}")

    # Build AnnData
    # The matrix has 111,668 barcodes but cell_meta may have 111,445 -> align by barcode
    obs_df = pd.DataFrame({'barcode': barcodes})
    obs_df.index = obs_df['barcode']

    # Merge cell metadata on barcode
    if 'cell_barcode' in cell_meta.columns:
        cell_meta_indexed = cell_meta.set_index('cell_barcode')
    else:
        cell_meta_indexed = cell_meta.set_index(cell_meta.columns[0])

    obs_merged = obs_df.join(cell_meta_indexed, how='left')

    var_df = pd.DataFrame({'gene_id': gene_ids, 'gene_name': gene_names})
    var_df.index = gene_names

    adata = ad.AnnData(X=mat, obs=obs_merged, var=var_df)
    adata.obs_names = barcodes
    adata.var_names = gene_names

    print(f"  AnnData shape: {adata.shape}")

    # Filter cells with good coverage
    if 'good_coverage' in adata.obs.columns:
        good_mask = adata.obs['good_coverage'].fillna(False).astype(bool)
        adata = adata[good_mask].copy()
        print(f"  After good_coverage filter: {adata.shape[0]} cells")

    # Parse guide identities
    print("  Parsing guide identities...")
    pert_types = []
    gene_lists = []

    guide_col = 'guide_identity' if 'guide_identity' in adata.obs.columns else adata.obs.columns[1]

    for guide in adata.obs[guide_col]:
        ptype, genes = _parse_guide_identity(guide)
        pert_types.append(ptype)
        gene_lists.append(genes)

    adata.obs['pert_type'] = pert_types
    adata.obs['pert_genes_list'] = [','.join(g) for g in gene_lists]

    # Make gene name unique if duplicates exist
    adata.var_names_make_unique()

    # Preprocess
    print("  Preprocessing (HVG selection, normalization, PCA)...")
    adata = preprocess(adata, n_hvg=n_hvg)
    print(f"  After preprocessing: {adata.shape}")

    # Build index dicts
    control_idx = np.where(adata.obs['pert_type'] == 'control')[0].tolist()

    single_perts = {}
    for i, (ptype, gl_str) in enumerate(zip(adata.obs['pert_type'], adata.obs['pert_genes_list'])):
        if ptype == 'single':
            gene = gl_str.strip()
            if gene:
                if gene not in single_perts:
                    single_perts[gene] = []
                single_perts[gene].append(i)

    combo_perts = {}
    for i, (ptype, gl_str) in enumerate(zip(adata.obs['pert_type'], adata.obs['pert_genes_list'])):
        if ptype == 'combo':
            parts = gl_str.split(',')
            if len(parts) == 2:
                gene_a, gene_b = parts[0].strip(), parts[1].strip()
                key = tuple(sorted([gene_a, gene_b]))
                if key not in combo_perts:
                    combo_perts[key] = []
                combo_perts[key].append(i)

    pert_genes = sorted(list(single_perts.keys()))

    print(f"  Control cells: {len(control_idx)}")
    print(f"  Single perturbations: {len(single_perts)} genes")
    print(f"  Combo perturbations: {len(combo_perts)} combos")

    # Compute differential expression
    X_dense = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)

    if len(control_idx) > 0:
        ctrl_mean = X_dense[control_idx].mean(axis=0)
    else:
        ctrl_mean = X_dense.mean(axis=0)

    print("  Computing differential expression for single perturbations...")
    adata.uns['de_single'] = {}
    for gene, idxs in single_perts.items():
        if len(idxs) > 0:
            pert_mean = X_dense[idxs].mean(axis=0)
            adata.uns['de_single'][gene] = pert_mean - ctrl_mean

    print("  Computing differential expression for combo perturbations...")
    adata.uns['de_combo'] = {}
    for (ga, gb), idxs in combo_perts.items():
        if len(idxs) > 0:
            pert_mean = X_dense[idxs].mean(axis=0)
            adata.uns['de_combo'][(ga, gb)] = pert_mean - ctrl_mean

    # Gene-level train/val/test split for single perturbations
    pert_split = split_single_perts(single_perts,
                                     train_frac=kwargs.get('train_frac', 0.8),
                                     val_frac=kwargs.get('val_frac', 0.1),
                                     seed=kwargs.get('split_seed', 42))
    print(f"  Pert split: {len(pert_split['train'])} train, "
          f"{len(pert_split['val'])} val, {len(pert_split['test'])} test genes")

    # Map gemgroup → contiguous integer batch_id so the encoder can do additive
    # batch-effect conditioning and PairedCellSentenceDataset can pair ctrl/pert
    # cells within the same plate (mirrors data/load_replogle.py).
    if 'gemgroup' in adata.obs.columns:
        unique_gg = sorted(adata.obs['gemgroup'].dropna().unique().tolist())
        gem_group_to_idx = {gg: i for i, gg in enumerate(unique_gg)}
        adata.obs['batch_id'] = (adata.obs['gemgroup']
                                  .map(gem_group_to_idx)
                                  .fillna(0)
                                  .astype(np.int64))
        n_batches = len(unique_gg)
        print(f"  gemgroups: {n_batches} unique (batch-effect conditioning enabled)")
    else:
        gem_group_to_idx = {}
        adata.obs['batch_id'] = 0
        n_batches = 0
        print(f"  No gemgroup column found — batch conditioning disabled")

    return {
        'adata': adata,
        'control_idx': control_idx,
        'single_perts': single_perts,
        'combo_perts': combo_perts,
        'gene_names': list(adata.var_names),
        'pert_genes': pert_genes,
        'pert_split': pert_split,
        'gem_group_to_idx': gem_group_to_idx,
        'n_batches': n_batches,
    }

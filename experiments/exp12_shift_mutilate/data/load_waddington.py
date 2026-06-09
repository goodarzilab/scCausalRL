import os
import gzip
import numpy as np
import pandas as pd
import scipy.sparse
import anndata as ad
import scanpy as sc
from .preprocess import preprocess


def load_waddington(path=None, n_hvg=2000):
    """
    Load Waddington-OT mouse embryonic reprogramming time course dataset.

    Files are in SCP295/ subdirectory:
    - SCP295/expression/Ex.Mat.dsmpl.15k.txt.gz: genes x cells TSV
    - SCP295/metadata/cell_days.txt: cell barcode + day (with 2 header rows)
    - SCP295/other/cell_sets.gmt: cell type annotations (GMT format)

    Returns dict with:
        adata: preprocessed AnnData
        timepoints: sorted list of unique days
        cells_by_day: dict day -> cell indices
        cell_sets: dict set_name -> cell indices
    """
    if path is None:
        path = "/large_storage/goodarzilab/shervin/scCausalRL/Waddington_OT/"

    scp_path = os.path.join(path, "SCP295")
    expr_file = os.path.join(scp_path, "expression", "Ex.Mat.dsmpl.15k.txt.gz")
    days_file = os.path.join(scp_path, "metadata", "cell_days.txt")
    gmt_file = os.path.join(scp_path, "other", "cell_sets.gmt")

    print("Loading Waddington-OT dataset...")

    # Load expression matrix (genes x cells)
    print("  Loading expression matrix (this may take a moment)...")
    expr_df = pd.read_csv(expr_file, sep='\t', index_col=0, compression='gzip')
    print(f"  Expression matrix shape (genes x cells): {expr_df.shape}")
    print(f"  Sample gene names: {list(expr_df.index[:5])}")
    print(f"  Sample cell names: {list(expr_df.columns[:3])}")

    # Check if we need to transpose (should be genes x cells -> transpose to cells x genes for AnnData)
    # If rows are genes (mouse gene names like Xkr4), keep as is and transpose
    # If rows are cells, don't transpose
    # Mouse gene check: look for lowercase gene symbols
    first_row_name = str(expr_df.index[0])
    # Mouse genes tend to be Titlecase (Xkr4, Sox17), human are UPPERCASE
    if first_row_name[0].isupper() and len(first_row_name) > 2 and first_row_name[1].islower():
        # Likely mouse gene names -> genes x cells -> transpose
        print("  Detected genes x cells orientation, transposing...")
        expr_df = expr_df.T  # now cells x genes

    print(f"  Expression matrix (cells x genes): {expr_df.shape}")

    # Load cell days
    print("  Loading cell days...")
    # File has 2 header rows: "NAME\tday" then "TYPE\tgroup" then data
    days_df = pd.read_csv(days_file, sep='\t', skiprows=2, header=None, names=['cell_id', 'day'])
    days_df['day'] = pd.to_numeric(days_df['day'], errors='coerce')
    days_df = days_df.dropna(subset=['day'])
    days_dict = dict(zip(days_df['cell_id'], days_df['day']))
    print(f"  Loaded {len(days_dict)} cell day annotations")
    print(f"  Unique days: {sorted(days_df['day'].unique())}")

    # Load cell sets (GMT format)
    print("  Loading cell sets...")
    cell_sets_raw = {}
    with open(gmt_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                set_name = parts[0]
                # description = parts[1]
                cells = parts[2:]
                cell_sets_raw[set_name] = set(cells)
    print(f"  Cell sets: {list(cell_sets_raw.keys())}")

    # Build AnnData from expression dataframe
    # cells x genes
    X = expr_df.values.astype(np.float32)
    obs_df = pd.DataFrame(index=expr_df.index)
    obs_df.index = obs_df.index.astype(str)
    var_df = pd.DataFrame(index=expr_df.columns)

    # Add day annotation
    obs_df['day'] = [days_dict.get(cid, np.nan) for cid in obs_df.index]

    # Add cell type annotation from GMT
    for set_name, cell_set in cell_sets_raw.items():
        obs_df[f'celltype_{set_name}'] = obs_df.index.isin(cell_set)

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.obs_names = list(expr_df.index)
    adata.var_names = list(expr_df.columns)
    print(f"  AnnData shape: {adata.shape}")

    # Filter cells with day annotation
    has_day = ~adata.obs['day'].isna()
    adata = adata[has_day].copy()
    print(f"  After day filter: {adata.shape[0]} cells")

    # Preprocess
    print("  Preprocessing (HVG selection, normalization, PCA)...")
    adata = preprocess(adata, n_hvg=n_hvg)
    print(f"  After preprocessing: {adata.shape}")

    # Build output dicts
    timepoints = sorted(adata.obs['day'].dropna().unique().tolist())

    cells_by_day = {}
    for day in timepoints:
        idxs = np.where(adata.obs['day'] == day)[0].tolist()
        cells_by_day[day] = idxs

    # Map cell set names to indices in final adata
    final_cell_ids = set(adata.obs_names)
    cell_sets_out = {}
    for set_name, cell_set in cell_sets_raw.items():
        # Find indices of cells in this set
        idxs = [i for i, cid in enumerate(adata.obs_names) if cid in cell_set]
        cell_sets_out[set_name] = idxs

    print(f"  Timepoints: {timepoints}")
    for day, idxs in cells_by_day.items():
        print(f"    Day {day}: {len(idxs)} cells")

    return {
        'adata': adata,
        'timepoints': timepoints,
        'cells_by_day': cells_by_day,
        'cell_sets': cell_sets_out,
    }

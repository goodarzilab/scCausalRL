"""
Fast helper to list Norman's perturbed gene names without loading the full MTX.

Only reads `GSE133344_filtered_cell_identities.csv.gz` and parses guide identities
with the same logic as data/load_norman.py. Useful for feeding `force_include`
into Replogle HVG selection at load time without a full Norman load.
"""
import os
import pandas as pd

from data.load_norman import _parse_guide_identity


def get_norman_perturbed_genes(path=None, include_singles=True, include_combos=True):
    """Return the set of gene names perturbed in Norman (singles + combo pairs).

    Reads only the cell_identities CSV; does NOT load the expression matrix.

    Args:
        path: Norman dataset directory (default /large_storage/.../Norman/)
        include_singles: include single-pert genes (default True)
        include_combos:  include combo gene pairs, both genes (default True)

    Returns:
        sorted list[str] of unique gene names perturbed in Norman that survive
        the same good_coverage filter applied during data/load_norman.py.
    """
    if path is None:
        path = "/large_storage/goodarzilab/shervin/scCausalRL/Norman/"
    cell_id_file = os.path.join(path, "GSE133344_filtered_cell_identities.csv.gz")
    if not os.path.isfile(cell_id_file):
        raise FileNotFoundError(f"Norman cell_identities not found at {cell_id_file}")

    df = pd.read_csv(cell_id_file, compression='gzip')

    if 'good_coverage' in df.columns:
        df = df[df['good_coverage'].fillna(False).astype(bool)]

    guide_col = 'guide_identity' if 'guide_identity' in df.columns else df.columns[1]

    genes = set()
    for guide in df[guide_col]:
        ptype, gl = _parse_guide_identity(guide)
        if ptype == 'single' and include_singles:
            genes.update(gl)
        elif ptype == 'combo' and include_combos:
            genes.update(gl)
    return sorted(genes)


if __name__ == "__main__":
    # Quick CLI: print counts when run directly
    g = get_norman_perturbed_genes()
    print(f"Norman perturbed genes (singles + combo pairs): {len(g)}")
    print(f"Sample: {g[:10]} ...")

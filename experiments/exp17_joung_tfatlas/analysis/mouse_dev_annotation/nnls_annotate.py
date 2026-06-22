"""Developmental annotation of the Joung TF atlas against the Cao 2019 mouse
organogenesis reference, via shift alignment to developmental cell-type signatures.

For each overexpressed TF we ask which mouse developmental cell-type DIRECTION its
induced expression shift points toward:

  Δ_TF ∈ R^G : induced shift = mean(OE cells) − mean(control)   (NOT absolute profile —
               all cells are hESC-derived, so absolute profiles share one huge baseline)
  D_c  ∈ R^G : cell-type deviation signature = mean(type c) − mean over cell types
  score_c = align(Δ_TF, D_c)   (cosine, primary; NNLS mixture also reported)

Three corrections make this work (see README / the diagnostic report):
  1. ORTHOLOGY: mouse↔human via the MGI 1:1 homology table (not uppercasing) → ~75-80%
     gene retention vs ~50%.
  2. SHARED OE-AXIS REMOVAL: every lentiviral-OE population shares a dominant
     selection/stress/proliferation axis (TF-shift pairwise corr 0.9+). We centre per TF
     and strip the top-K (default 4) SVD components — a simple perturbation factor model.
     Without this, EVERY estimator (OLS/ridge/NNLS/cosine) collapses to one answer.
  3. EARLY-PROGENITOR REFERENCE: hESC + ~7d OE reaches early progenitor states, not
     E9.5-13.5 terminal types. We restrict to ~18 progenitor/early Main_cell_types, so
     "didn't map to Myocytes/Hepatocytes" is expected, not a failure.
And we report only TFs with ≥50 cells (small-n pseudobulks are noise).

Run:  python nnls_annotate.py --ref mouse_ref_Main_cell_type
"""
import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
MGI = os.path.join(HERE, 'mgi_homologene.rpt')

# ~18 early/progenitor Main_cell_types (hESC+7d OE reaches these, not terminal types).
# Restricting here means "no map to Myocytes/Hepatocytes" is expected, not a bug.
EARLY_MAIN_TYPES = {
    'Chondroctye progenitors', 'Connective tissue progenitors', 'Early mesenchyme',
    'Intermediate Mesoderm', 'Limb mesenchyme', 'Stromal cells', 'Epithelial cells',
    'Endothelial cells', 'Neural progenitor cells', 'Neural Tube', 'Notochord cells',
    'Radial glia', 'Oligodendrocyte Progenitors', 'Inhibitory neuron progenitors',
    'Isthmic organizer cells', 'Jaw and tooth progenitors', 'Schwann cell precursor',
    'Postmitotic premature neurons', 'Premature oligodendrocyte', 'Primitive erythroid lineage',
}


def load_orthologs():
    """MGI 1:1 mouse→human symbol map (18.8k pairs)."""
    m = pd.read_csv(MGI, sep='\t')
    cnt = m.groupby(['DB Class Key', 'Common Organism Name']).size().unstack(fill_value=0)
    one2one = cnt[(cnt.get('mouse, laboratory', 0) == 1) & (cnt.get('human', 0) == 1)].index
    mm = m[m['DB Class Key'].isin(one2one)]
    mouse = mm[mm['Common Organism Name'] == 'mouse, laboratory'].set_index('DB Class Key')['Symbol']
    human = mm[mm['Common Organism Name'] == 'human'].set_index('DB Class Key')['Symbol']
    d = pd.concat([mouse.rename('m'), human.rename('h')], axis=1).dropna()
    return dict(zip(d['m'], d['h']))


def load_reference(ref_prefix, early_only=True):
    """Load X (genes×celltypes); map mouse→human symbols (MGI 1:1); optionally restrict
    columns to early/progenitor cell types."""
    path = ref_prefix if ref_prefix.endswith('.csv.gz') else ref_prefix + '.csv.gz'
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    X = pd.read_csv(path, index_col=0)
    X = X[~X.index.duplicated(keep='first')]
    omap = load_orthologs()
    X.index = X.index.map(lambda g: omap.get(g, None))    # mouse symbol → human ortholog
    X = X[X.index.notna()]
    X = X[~X.index.duplicated(keep='first')]
    print(f'   ortholog-mapped reference: {X.shape[0]} human genes × {X.shape[1]} cell types')
    if early_only:
        cols = [c for c in X.columns if c in EARLY_MAIN_TYPES]
        if len(cols) >= 5:
            X = X[cols]
            print(f'   restricted to {len(cols)} early/progenitor cell types')
        else:
            print(f'   (early-type whitelist matched only {len(cols)} cols — keeping all)')
    return X                                          # DataFrame [G_human × C]


def joung_pseudobulk(adata_path, min_cells=50):
    """Per-TF INDUCED SHIFT Δ = mean(OE cells) − mean(control cells) → [TF × G_human].

    The shift (not the absolute profile) is what carries TF identity: all Joung cells are
    hESC-derived, so absolute per-TF pseudobulks are dominated by the shared pluripotency
    baseline and look identical. Subtracting the control baseline isolates each TF's effect."""
    import scipy.sparse as sp
    a = ad.read_h5ad(adata_path)                      # X already lognorm (load_joung)
    X = a.X
    genes = list(a.var_names)
    tf = a.obs['TF_gene'].astype(str).values
    ctrl = np.asarray(a.obs['is_control'].values, bool) if 'is_control' in a.obs else \
        np.zeros(len(tf), bool)
    def mean_rows(mask):
        sub = X[mask]
        return np.asarray(sub.mean(0)).ravel() if sp.issparse(sub) else np.asarray(sub.mean(0)).ravel()
    base = mean_rows(ctrl)                             # hESC control baseline
    rows, names = [], []
    for g in sorted(set(tf)):
        m = (tf == g) & (~ctrl)
        if m.sum() < min_cells:
            continue
        rows.append(mean_rows(m) - base)              # Δ = induced shift
        names.append(g)
    return pd.DataFrame(np.vstack(rows), index=names, columns=genes)   # [TF × G_human]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='mouse_ref_Main_cell_type')
    ap.add_argument('--joung', default=JOUNG)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--remove_pcs', type=int, default=4,
                    help='remove this many shared nuisance PCs from the TF-shift matrix')
    ap.add_argument('--all_types', action='store_true',
                    help='keep all cell types (default: early/progenitor only)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(HERE, f'joung_devannot_{os.path.basename(args.ref)}')

    print('[1/4] loading mouse reference (MGI orthology, early types)...', flush=True)
    Xdf = load_reference(args.ref, early_only=not args.all_types)   # index already human symbols
    Xdf = Xdf[~Xdf.index.duplicated(keep='first')]

    print('[2/4] pseudobulking Joung TFs...', flush=True)
    Ydf = joung_pseudobulk(args.joung)
    Ydf.columns = Ydf.columns.astype(str).str.upper()
    Ydf = Ydf.loc[:, ~Ydf.columns.duplicated(keep='first')]
    print(f'   atlas: {Ydf.shape[0]} TFs × {Ydf.shape[1]} genes')

    # --- remove shared nuisance axes from the TF-shift matrix -----------------
    # All hESC-derived OE populations share strong common axes (selection/OE-stress,
    # library size) that correlate 0.9+ across unrelated TFs and otherwise dominate the
    # fit. Centre each TF's shift against the (unweighted) mean TF shift, then strip the
    # top-`remove_pcs` SVD components of the residual. What remains is TF-SPECIFIC.
    cell_types = list(Xdf.columns)
    Ymat = Ydf.values.astype(np.float64)
    Yc = Ymat - Ymat.mean(0, keepdims=True)
    if args.remove_pcs > 0:
        U, S, Vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False)
        k = min(args.remove_pcs, len(S))
        Yc = Yc - (U[:, :k] * S[:k]) @ Vt[:k]
        print(f'   removed top-{k} shared PC(s) from TF shifts', flush=True)
    Ydf = pd.DataFrame(Yc, index=Ydf.index, columns=Ydf.columns)

    # --- cell-type deviation signatures on discriminative shared genes --------
    shared = Xdf.index.intersection(Ydf.columns)
    print(f'[3/4] shared genes (mouse∩human): {len(shared)}', flush=True)
    X = Xdf.loc[shared].values.astype(np.float64)
    D = X - X.mean(1, keepdims=True)                   # [G × C] deviation signatures
    gvar = D.var(1)
    keep = gvar > np.quantile(gvar, 0.5)               # drop the flat (non-discriminative) half
    D, shared_kept = D[keep], shared[keep]
    print(f'   discriminative genes kept: {keep.sum()}', flush=True)
    Dn = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)   # unit-norm cols (for NNLS)
    Dz = (D - D.mean(0)) / (D.std(0) + 1e-8)           # per-celltype z-scored (for cosine)

    # --- score each TF: cosine alignment (robust, primary) + NNLS (mixture) ---
    print('[4/4] scoring TF→cell-type alignment (cosine + NNLS)...', flush=True)
    rows, Wcos, Wnnls = [], [], []
    for tf in Ydf.index:
        y = Ydf.loc[tf, shared_kept].values.astype(np.float64)
        yz = (y - y.mean()) / (y.std() + 1e-8)
        cos = (Dz.T @ yz) / len(yz)                    # [C] cosine-like alignment
        beta, _ = nnls(Dn, np.clip(y, 0, None))        # NNLS on the positive part (mixture)
        s = beta.sum(); wn = beta / s if s > 0 else beta
        Wcos.append(cos); Wnnls.append(wn)
        order = np.argsort(cos)[::-1][:args.topk]
        rows.append({
            'TF': tf,
            'top_celltype': cell_types[order[0]],
            'top_cos': float(cos[order[0]]),
            'specificity': float(cos[order[0]] - cos.mean()),   # peak above background
            'annotation': '; '.join(f'{cell_types[j]} ({cos[j]:+.2f})' for j in order),
        })
    pd.DataFrame(np.vstack(Wcos), index=Ydf.index, columns=cell_types).to_csv(out + '_cos.csv')
    pd.DataFrame(np.vstack(Wnnls), index=Ydf.index, columns=cell_types).to_csv(out + '_nnls.csv')
    annot = pd.DataFrame(rows).sort_values('specificity', ascending=False)
    annot.to_csv(out + '_annotation.csv', index=False)
    print(f'\nsaved → {out}_annotation.csv  (+ _cos.csv, _nnls.csv)')
    print('\n=== most cell-type-specific TFs (peak cosine above background) ===')
    print(annot.head(20).to_string(index=False))


if __name__ == '__main__':
    main()

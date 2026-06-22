"""Developmental annotation of the Joung TF atlas against the HUMAN fetal cell atlas
(Cao et al. 2020, Science; GEO GSE156793 / Descartes), the same reference family the Joung
paper used. Replaces the mouse MOCA reference → NO cross-species orthology step (direct
symbol match), so ~all 9,923 Joung HVGs are usable instead of the ~15k mouse-orthology subset.

Reference = S6_gene_expression_celltype: [63,562 genes (Ensembl) × 172 organ-cell-types],
mean expression per fetal organ-cell-type. We score which cell-type DIRECTION each TF's
induced shift Δ_TF points toward, after stripping the shared OE nuisance axes.

Method shoot-out (the README's open question, re-asked on the human ref): cosine vs NNLS vs
OLS vs ridge. Reports per-method confident-call counts + agreement so we can pick the best.

Run (high-mem):  sbatch scripts/human_annotate.sbatch
"""
import os, json, argparse
import numpy as np, pandas as pd
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = '/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
S6 = os.path.join(REF_DIR, 'GSE156793_S6_gene_expression_celltype.txt.gz')   # genes × celltypes
S2 = os.path.join(REF_DIR, 'GSE156793_S2_Metadata_genes.txt.gz')             # ensembl → symbol
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'


def load_human_reference():
    """[genes(symbol) × 172 organ-cell-types] mean-expression reference (no orthology step)."""
    g = pd.read_csv(S2)                                   # gene_id, gene_type, gene_short_name
    id2sym = dict(zip(g['gene_id'], g['gene_short_name'].astype(str).str.upper()))
    X = pd.read_csv(S6, index_col=0)                      # RowID(ensembl) × celltypes
    X.index = X.index.map(lambda e: id2sym.get(e))
    X = X[X.index.notna()]
    X = X[~X.index.duplicated(keep='first')]
    print(f'   human reference: {X.shape[0]} genes × {X.shape[1]} organ-cell-types', flush=True)
    return X


def joung_pseudobulk(adata_path, min_cells=50):
    """Per-TF induced shift Δ = mean(OE) − mean(control) → [TF × G]."""
    import anndata as ad, scipy.sparse as sp
    a = ad.read_h5ad(adata_path)
    X = a.X; genes = list(a.var_names)
    tf = a.obs['TF_gene'].astype(str).values
    ctrl = np.asarray(a.obs['is_control'].values, bool)
    mean_rows = lambda m: np.asarray(X[m].mean(0)).ravel()
    base = mean_rows(ctrl)
    rows, names = [], []
    for gname in sorted(set(tf)):
        m = (tf == gname) & (~ctrl)
        if m.sum() >= min_cells:
            rows.append(mean_rows(m) - base); names.append(gname)
    return pd.DataFrame(np.vstack(rows), index=names, columns=[str(x).upper() for x in genes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--remove_pcs', type=int, default=4)
    ap.add_argument('--min_cells', type=int, default=50)
    ap.add_argument('--ridge', type=float, default=1.0)
    ap.add_argument('--conf_thresh', type=float, default=0.12)  # mouse-comparable heuristic gate
    ap.add_argument('--n_perm', type=int, default=100)          # gene-permutations/TF for the pooled null
    args = ap.parse_args()

    print('[1/4] loading human fetal reference...', flush=True)
    Xdf = load_human_reference()
    print('[2/4] pseudobulking Joung TFs (induced shifts)...', flush=True)
    Ydf = joung_pseudobulk(JOUNG, args.min_cells)
    Ydf = Ydf.loc[:, ~Ydf.columns.duplicated(keep='first')]
    print(f'   atlas: {Ydf.shape[0]} TFs × {Ydf.shape[1]} genes', flush=True)

    # shared OE-axis removal (strip top-k SVD comps of the centred TF-shift matrix)
    Ymat = Ydf.values.astype(np.float64); Yc = Ymat - Ymat.mean(0, keepdims=True)
    if args.remove_pcs > 0:
        U, S, Vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False)
        k = min(args.remove_pcs, len(S)); Yc = Yc - (U[:, :k] * S[:k]) @ Vt[:k]
        print(f'   removed top-{k} shared PCs from TF shifts', flush=True)
    Ydf = pd.DataFrame(Yc, index=Ydf.index, columns=Ydf.columns)

    cell_types = list(Xdf.columns)
    shared = Xdf.index.intersection(Ydf.columns)
    print(f'[3/4] shared genes (human ref ∩ Joung HVG): {len(shared)}', flush=True)
    Xs = Xdf.loc[shared].values.astype(np.float64)
    D = Xs - Xs.mean(1, keepdims=True)                    # [G × C] deviation signatures
    keep = D.var(1) > np.quantile(D.var(1), 0.5)          # discriminative half
    D, shared_kept = D[keep], shared[keep]
    print(f'   discriminative genes kept: {int(keep.sum())}', flush=True)
    Dn = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)
    Dz = (D - D.mean(0)) / (D.std(0) + 1e-8)
    DtD = D.T @ D
    ridge_inv = np.linalg.inv(DtD + args.ridge * np.eye(D.shape[1]))
    ols_pinv = np.linalg.pinv(D)                          # [C × G]
    condD = float(np.linalg.cond(DtD))

    print(f'[4/4] scoring 4 methods (cosine/NNLS/OLS/ridge); cond(DᵀD)={condD:.1e}', flush=True)
    methods = ['cosine', 'nnls', 'ols', 'ridge']
    scores = {m: [] for m in methods}; rows = []
    rng_p = np.random.default_rng(0); pool_null = []     # permutation null for cosine specificity
    for tf in Ydf.index:
        y = Ydf.loc[tf, shared_kept].values.astype(np.float64)
        yz = (y - y.mean()) / (y.std() + 1e-8)
        cos = (Dz.T @ yz) / len(yz)
        # gene-permutation null: reorder yz (breaks gene↔signature alignment), recompute specificity.
        # yz is standardized so the null is ~exchangeable across TFs → pool draws for high resolution.
        idx = np.argsort(rng_p.random((args.n_perm, len(yz))), axis=1)        # [B, G] permutations
        ncos = (Dz.T @ yz[idx].T) / len(yz)                                   # [C, B]
        pool_null.append(ncos.max(0) - ncos.mean(0))                         # [B] null specificities
        beta_nnls, _ = nnls(Dn, np.clip(y, 0, None)); s = beta_nnls.sum(); beta_nnls = beta_nnls / s if s > 0 else beta_nnls
        beta_ols = ols_pinv @ y
        beta_ridge = ridge_inv @ (D.T @ y)
        sc = {'cosine': cos, 'nnls': beta_nnls, 'ols': beta_ols, 'ridge': beta_ridge}
        for m in methods:
            scores[m].append(sc[m])
        # specificity per method = peak − mean (cosine in cos units; coeffs z-less but comparable within method)
        row = {'TF': tf}
        for m in methods:
            v = sc[m]; top = int(np.argmax(v))
            row[f'{m}_top'] = cell_types[top]
            row[f'{m}_spec'] = float(v[top] - v.mean())
        row['top_celltype'] = row['cosine_top']; row['specificity'] = row['cosine_spec']
        rows.append(row)

    # ── permutation p-value (pooled null) + Benjamini-Hochberg FDR over all TFs ──
    pool = np.sort(np.concatenate(pool_null))            # [n_TF * n_perm] shared null
    obs = np.array([r['cosine_spec'] for r in rows])
    pvals = (1 + (len(pool) - np.searchsorted(pool, obs, side='left'))) / (len(pool) + 1)
    order = np.argsort(pvals); n = len(pvals)
    qs = np.minimum.accumulate((pvals[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    qvals = np.empty(n); qvals[order] = np.clip(qs, 0, 1)
    for r, pv, qv in zip(rows, pvals, qvals):
        r['perm_pval'] = float(pv); r['fdr_qval'] = float(qv)
    annot = pd.DataFrame(rows).sort_values('specificity', ascending=False)
    annot.to_csv(os.path.join(HERE, 'joung_human_annotation.csv'), index=False)
    for m in methods:
        pd.DataFrame(np.vstack(scores[m]), index=Ydf.index, columns=cell_types).to_csv(
            os.path.join(HERE, f'joung_human_{m}.csv'))

    # ── statistics ──────────────────────────────────────────────────────────
    nT = len(annot)
    stats = {'n_TFs': nT, 'n_celltypes': len(cell_types), 'shared_genes': int(len(shared)),
             'discriminative_genes': int(keep.sum()), 'cond_DtD': condD,
             'methods': {}, 'cosine_specificity': {
                 'median': float(annot['specificity'].median()),
                 'q90': float(annot['specificity'].quantile(0.90)),
                 'max': float(annot['specificity'].max())}}
    for m in methods:
        sp = annot[f'{m}_spec']
        stats['methods'][m] = {f'n_conf_gt_{args.conf_thresh}': int((sp > args.conf_thresh).sum()),
                               'n_conf_gt_0.05': int((sp > 0.05).sum()),
                               'median_spec': float(sp.median())}
    # cross-method agreement on the top call (cosine vs each)
    for m in ['nnls', 'ols', 'ridge']:
        stats['methods'][m]['agree_top_with_cosine'] = float((annot['cosine_top'] == annot[f'{m}_top']).mean())
    # FDR-controlled confidence (BH on the gene-permutation p-values) + percentile fallback
    q = annot['fdr_qval'].values
    stats['fdr'] = {'n_perm_per_tf': args.n_perm, 'pool_null_size': int(len(pool)),
                    'n_fdr_0.01': int((q < 0.01).sum()), 'n_fdr_0.05': int((q < 0.05).sum()),
                    'n_fdr_0.10': int((q < 0.10).sum()),
                    'min_pval': float(annot['perm_pval'].min())}
    stats['percentile_thresholds'] = {  # fallback: spec cutoff at each top-percentile
        'p90_spec': float(annot['specificity'].quantile(0.90)), 'n_top10pct': int(np.ceil(0.10 * nT)),
        'p95_spec': float(annot['specificity'].quantile(0.95)), 'n_top5pct': int(np.ceil(0.05 * nT))}
    json.dump(stats, open(os.path.join(HERE, 'human_annotation_stats.json'), 'w'), indent=2)

    print('\n===== HUMAN-FETAL ANNOTATION STATS =====')
    print(f"TFs (n>={args.min_cells}): {nT} | cell types: {len(cell_types)} | shared genes: {len(shared)} | cond(DᵀD): {condD:.1e}")
    f = stats['fdr']
    print(f"FDR-controlled (gene-permutation null, {f['pool_null_size']:,} draws): "
          f"q<0.01 → {f['n_fdr_0.01']} TFs | q<0.05 → {f['n_fdr_0.05']} | q<0.10 → {f['n_fdr_0.10']}  (min p={f['min_pval']:.1e})")
    print(f"\n{'method':8s} {'n_conf(>%.2f)'%args.conf_thresh:>12s} {'n_conf(>0.05)':>13s} {'median_spec':>11s} {'agree_w_cos':>11s}")
    for m in methods:
        d = stats['methods'][m]
        print(f"{m:8s} {d['n_conf_gt_%s'%args.conf_thresh]:>12d} {d['n_conf_gt_0.05']:>13d} {d['median_spec']:>11.3f} "
              f"{d.get('agree_top_with_cosine', 1.0):>11.2f}")
    print(f"\ncosine specificity: median={stats['cosine_specificity']['median']:.3f} "
          f"q90={stats['cosine_specificity']['q90']:.3f} max={stats['cosine_specificity']['max']:.3f}")
    print('\n=== top-20 most cell-type-specific TFs (cosine; q = BH-FDR) ===')
    print(annot.head(20)[['TF', 'top_celltype', 'specificity', 'perm_pval', 'fdr_qval']].to_string(index=False))
    print('\nsaved → joung_human_annotation.csv (+ per-method csvs, human_annotation_stats.json)')


if __name__ == '__main__':
    main()

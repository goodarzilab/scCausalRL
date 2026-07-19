"""Production annotation of all 1,084 Joung TFs with the method validated by the Joung-paper
ground truth: cosine_celllevel (TF shift Δ vs cell-level matched-norm Cao signatures, OE-axis
removed). Writes:
  joung_human_annotation_celllevel.csv   per-TF top cell type + specificity + perm-FDR + top5
  joung_human_cosine_celllevel.csv       full TF×celltype cosine matrix
  human_annotation_stats_v2.json         production stats + paper-validated recall table (slides)
  ../../figures/joung/exp17_annotation_recall.png   recall@k bar figure (paper GT, reachable pool)
Run: python finalize_annotation.py
"""
import os, sys, json
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from data.load_joung import _lognorm
D = '/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
FIGJ = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'figures', 'joung')); os.makedirs(FIGJ, exist_ok=True)


def cosine_parts(Ydf, sigdf, drift_pcs=4):
    """Return (Yz [TF×gene], Dz [gene×type], types) for cosine after OE-axis removal."""
    shared = sigdf.index.intersection(Ydf.columns)
    X = sigdf.loc[shared].values.astype(np.float64); Dd = X - X.mean(1, keepdims=True)
    keep = Dd.var(1) > np.quantile(Dd.var(1), 0.5); Dd, sh = Dd[keep], shared[keep]
    Dz = (Dd - Dd.mean(0)) / (Dd.std(0) + 1e-8)
    Y = Ydf[sh].values.astype(np.float64); Yc = Y - Y.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False); k = min(drift_pcs, len(S))
    Yc = Yc - (U[:, :k] * S[:k]) @ Vt[:k]
    Yz = (Yc - Yc.mean(1, keepdims=True)) / (Yc.std(1, keepdims=True) + 1e-8)
    return Yz / Yz.shape[1], Dz, np.array([str(c) for c in sigdf.columns]), Ydf.index


def main():
    a = ad.read_h5ad(JOUNG); a.var_names = [str(v).upper() for v in a.var_names]; a.X = _lognorm(a.X)
    tf = a.obs['TF_gene'].astype(str).values; ctrl = np.asarray(a.obs['is_control'].values, bool)
    base = np.asarray(a.X[ctrl].mean(0)).ravel(); rows, names = [], []
    for g in sorted(set(tf)):
        m = (tf == g) & (~ctrl)
        if m.sum() >= 50: rows.append(np.asarray(a.X[m].mean(0)).ravel() - base); names.append(g)
    Ydf = pd.DataFrame(np.vstack(rows), index=names, columns=a.var_names); Ydf = Ydf.loc[:, ~Ydf.columns.duplicated()]

    CL = pd.read_csv(f'{HERE}/cao_type_mean_logexpr.csv', index_col=0).T
    CL.index = CL.index.astype(str).str.upper(); CL = CL[~CL.index.duplicated()]
    Yz, Dz, types, tfnames = cosine_parts(Ydf, CL)
    cos = Yz @ Dz                                            # [TF × type]
    C = cos.shape[1]
    cosdf = pd.DataFrame(cos, index=tfnames, columns=types)
    cosdf.to_csv(f'{HERE}/joung_human_cosine_celllevel.csv')

    spec = cos.max(1) - cos.mean(1)
    order = np.argsort(-cos, 1)
    top5 = [';'.join(types[order[i, :5]]) for i in range(len(tfnames))]

    # permutation-FDR: shuffle reference gene rows (Dz) B times → null spec pool, BH-correct
    rng = np.random.default_rng(0); B = 100; null = []
    for _ in range(B):
        Dp = Dz[rng.permutation(Dz.shape[0])]
        cp = Yz @ Dp; null.append(cp.max(1) - cp.mean(1))
    null = np.concatenate(null)                              # 1084*B null specs
    pval = np.array([(null >= s).mean() for s in spec]); pval = np.maximum(pval, 1.0 / (len(null) + 1))
    o = np.argsort(pval); ranked = pval[o]; m = len(pval)
    q = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    qval = np.empty(m); qval[o] = np.clip(q, 0, 1)

    ann = pd.DataFrame({'TF': tfnames, 'top_celltype': types[order[:, 0]], 'specificity': spec,
                        'perm_pval': pval, 'fdr_qval': qval, 'top5_celltypes': top5}).sort_values('fdr_qval')
    ann.to_csv(f'{HERE}/joung_human_annotation_celllevel.csv', index=False)

    # ── recall figure from the paper-validated re-score (reachable pool) ──
    rc = pd.read_csv(f'{HERE}/paper_gt_rescore.csv')
    disp = {'cosine_S6 (shipped)': 'cosine_S6', 'cosine_celllevel (proposed)': 'cosine_celllevel ★',
            'celltypist_LRsig': 'celltypist_LRsig', 'celltypist_percell ΔP': 'celltypist_perΔP'}
    rc = rc[rc['method'].isin(disp)].copy(); rc['m'] = rc['method'].map(disp)
    chance = {'reachable_r@1': 0.07, 'reachable_r@5': 0.26, 'reachable_r@10': 0.42}
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); ks = ['reachable_r@1', 'reachable_r@5', 'reachable_r@10']
    labs = ['recall@1', 'recall@5', 'recall@10']; x = np.arange(len(ks)); w = 0.2
    colors = ['#999999', '#2171B5', '#27AE60', '#C0392B']
    for i, (_, r) in enumerate(rc.iterrows()):
        ax.bar(x + (i - 1.5) * w, [r[k] for k in ks], w, label=r['m'], color=colors[i % 4])
    ax.plot(x, [chance[k] for k in ks], 'k--', lw=1.3, label='chance'); ax.scatter(x, [chance[k] for k in ks], c='k', s=18)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('recall'); ax.set_ylim(0, 1)
    ax.set_title('TF→cell-type annotation vs Joung paper-validated ground truth\n(16 validated TFs, reachable cell types, n=82)', fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc='upper left'); fig.tight_layout()
    figp = f'{FIGJ}/exp17_annotation_recall.png'; fig.savefig(figp, dpi=140); plt.close(fig)

    # ── stats JSON for the slides ──
    def grab(name, col):
        v = rc[rc['method'] == name][col]; return float(v.iloc[0]) if len(v) else None
    paper_gt = {'n_validated_TFs': 18, 'n_scored_TFs': int(rc['reachable_nTF'].max()),
                'n_cell_types': 8, 'pool_reachable': int(rc['n_reachable'].max()), 'pool_all': int(rc['n_types'].max()),
                'chance': chance,
                'methods': {disp[m]: {'r@1': grab(m, 'reachable_r@1'), 'r@5': grab(m, 'reachable_r@5'),
                                      'r@10': grab(m, 'reachable_r@10'), 'medrank': int(rc[rc.method==m]['reachable_medrank'].iloc[0]),
                                      'all_r@1': grab(m, 'all_r@1'), 'all_medrank': int(rc[rc.method==m]['all_medrank'].iloc[0])}
                            for m in disp}}
    conf_qval = qval; conf_spec = spec
    stats = {'method': 'cosine_celllevel', 'n_TFs': int(len(tfnames)), 'n_celltypes': int(C),
             'production': {'median_spec': float(np.median(conf_spec)), 'q90_spec': float(np.quantile(conf_spec, 0.9)),
                            'max_spec': float(conf_spec.max()), 'n_fdr_0.05': int((conf_qval < 0.05).sum()),
                            'n_fdr_0.10': int((conf_qval < 0.10).sum()), 'n_top10pct': int((conf_spec >= np.quantile(conf_spec, 0.9)).sum())},
             'paper_gt': paper_gt, 'recall_fig': 'experiments/figures/joung/exp17_annotation_recall.png'}
    json.dump(stats, open(f'{HERE}/human_annotation_stats_v2.json', 'w'), indent=2)

    print(f'annotated {len(tfnames)} TFs × {C} types (cosine_celllevel)')
    print(f'  median spec {np.median(spec):.3f} | q<0.05 {int((qval<0.05).sum())} | q<0.10 {int((qval<0.10).sum())} | top-decile {int((spec>=np.quantile(spec,0.9)).sum())}')
    print(f'  paper-GT reachable recall@1: cosine_celllevel={grab("cosine_celllevel (proposed)","reachable_r@1")} '
          f'LRsig={grab("celltypist_LRsig","reachable_r@1")} S6={grab("cosine_S6 (shipped)","reachable_r@1")} (chance 0.07)')
    print(f'  wrote annotation csv, cosine matrix, stats_v2.json, {figp}')


if __name__ == '__main__':
    main()

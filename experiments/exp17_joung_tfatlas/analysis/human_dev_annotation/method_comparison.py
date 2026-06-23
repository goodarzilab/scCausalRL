"""ONE table comparing every TF→cell-type method on the SAME benchmark: 38 ground-truth master
TFs, rank all reference cell types, score recall@k + median rank of a biologically-correct type
(recall_at_k.TARGETS) + how many DISTINCT top-calls across the 38 TFs (does it differentiate TFs?).

Methods:
  cosine_S6        : shift Δ vs paper S6 mean-expression signatures (shipped)
  cosine_celllevel : shift Δ vs my cell-level log1p-CP10k mean signatures (proposed upgrade)
  celltypist_LRsig : shift Δ vs CellTypist's logistic-regression coefficient signatures
  celltypist_percell: CellTypist per-cell prediction, ΔP = meanP(OE) − meanP(ctrl) (its intended use)
All cosine methods use identical OE-axis removal. Run: sbatch .../method_comparison.sbatch
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, celltypist
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from data.load_joung import _lognorm
from recall_at_k import TARGETS
D = '/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
GT = {k.upper(): v for k, v in TARGETS.items()}


def score(scoredf, name):
    """scoredf: [TF × cell_types]; higher = better match. Score recall@k vs TARGETS keywords."""
    scoredf = scoredf.copy(); scoredf.index = scoredf.index.astype(str).str.upper()
    types = np.array([str(c) for c in scoredf.columns]); C = len(types); tl = np.array([t.lower() for t in types])
    r1 = r5 = r10 = 0; ranks = []; tops = []; n = 0
    for tf in [t for t in GT if t in scoredf.index]:
        acc = np.array([any(kw in tl[i] for kw in GT[tf]) for i in range(C)])
        if acc.sum() == 0: continue
        order = np.argsort(-scoredf.loc[tf].values); best = int(np.where(acc[order])[0].min()) + 1
        ranks.append(best); r1 += best <= 1; r5 += best <= 5; r10 += best <= 10; tops.append(types[order[0]]); n += 1
    return {'method': name, 'n_types': C, 'n_TFs': n, 'recall@1': r1/n, 'recall@5': r5/n,
            'recall@10': r10/n, 'median_rank': int(np.median(ranks)), 'distinct_top': len(set(tops))}


def cosmat(Ydf, sigdf, drift_pcs=4):
    shared = sigdf.index.intersection(Ydf.columns)
    X = sigdf.loc[shared].values.astype(np.float64); Dd = X - X.mean(1, keepdims=True)
    keep = Dd.var(1) > np.quantile(Dd.var(1), 0.5); Dd, sh = Dd[keep], shared[keep]
    Dz = (Dd - Dd.mean(0)) / (Dd.std(0) + 1e-8)
    Y = Ydf[sh].values.astype(np.float64); Yc = Y - Y.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False); k = min(drift_pcs, len(S))
    Yc = Yc - (U[:, :k] * S[:k]) @ Vt[:k]
    Yz = (Yc - Yc.mean(1, keepdims=True)) / (Yc.std(1, keepdims=True) + 1e-8)
    return pd.DataFrame((Yz @ Dz) / Yz.shape[1], index=Ydf.index, columns=sigdf.columns)


def main():
    a = ad.read_h5ad(JOUNG); a.var_names = [str(v).upper() for v in a.var_names]; a.X = _lognorm(a.X)
    tf = a.obs['TF_gene'].astype(str).values; ctrl = np.asarray(a.obs['is_control'].values, bool)
    base = np.asarray(a.X[ctrl].mean(0)).ravel(); rows, names = [], []
    for g in sorted(set(tf)):
        m = (tf == g) & (~ctrl)
        if m.sum() >= 50: rows.append(np.asarray(a.X[m].mean(0)).ravel() - base); names.append(g)
    Ydf = pd.DataFrame(np.vstack(rows), index=names, columns=a.var_names); Ydf = Ydf.loc[:, ~Ydf.columns.duplicated()]

    g = pd.read_csv(f'{D}/GSE156793_S2_Metadata_genes.txt.gz'); id2s = dict(zip(g.gene_id, g.gene_short_name.astype(str).str.upper()))
    S6 = pd.read_csv(f'{D}/GSE156793_S6_gene_expression_celltype.txt.gz', index_col=0)
    S6.index = S6.index.map(lambda e: id2s.get(e)); S6 = S6[S6.index.notna()]; S6 = S6[~S6.index.duplicated()]
    CL = pd.read_csv(f'{HERE}/cao_type_mean_logexpr.csv', index_col=0).T; CL.index = CL.index.astype(str).str.upper(); CL = CL[~CL.index.duplicated()]
    model = celltypist.models.Model.load(f'{HERE}/cao_celltypist_model.pkl')
    LR = pd.DataFrame(model.classifier.coef_.T, index=[str(x).upper() for x in model.classifier.features],
                      columns=list(model.classifier.classes_)); LR = LR.loc[~LR.index.duplicated()]

    # chance row: per GT TF with m acceptable of C types, E[recall@k]=1-(1-m/C)^k; median rank ~C/2
    Ctypes = [str(c).lower() for c in S6.columns]; C = len(Ctypes)
    ms = [sum(any(kw in c for kw in GT[g]) for c in Ctypes) for g in GT if g in Ydf.index]
    ch = lambda k: float(np.mean([1 - (1 - m / C) ** k for m in ms]))
    results = [{'method': 'chance (random)', 'n_types': C, 'n_TFs': len(ms),
                'recall@1': round(ch(1), 2), 'recall@5': round(ch(5), 2), 'recall@10': round(ch(10), 2),
                'median_rank': C // 2, 'distinct_top': '—'}]
    results.append(score(cosmat(Ydf, S6), 'cosine_S6 (shipped)'))
    results.append(score(cosmat(Ydf, CL), 'cosine_celllevel (proposed)'))
    results.append(score(cosmat(Ydf, LR), 'celltypist_LRsig'))

    # per-cell ΔP (CellTypist's intended use): predict on control + each GT TF's cells, ΔP over classes
    rng = np.random.default_rng(0); idx, grp = [], []
    ci = np.where(ctrl)[0]; sel = rng.choice(ci, 3000, replace=False); idx += list(sel); grp += ['__ctrl__']*len(sel)
    gt_present = [t for t in GT if t in set(tf)]
    for gtf in gt_present:
        gi = np.where((tf == gtf) & (~ctrl))[0]
        if len(gi) < 30: continue
        s = rng.choice(gi, min(200, len(gi)), replace=False); idx += list(s); grp += [gtf]*len(s)
    sub = a[np.array(idx)].copy(); grp = np.array(grp)
    pred = celltypist.annotate(sub, model=model, majority_voting=False)
    P = pred.probability_matrix; cmean = P.values[grp == '__ctrl__'].mean(0)
    dP = {gtf: P.values[grp == gtf].mean(0) - cmean for gtf in gt_present if (grp == gtf).any()}
    dPdf = pd.DataFrame(dP).T; dPdf.columns = P.columns
    results.append(score(dPdf, 'celltypist_percell ΔP'))

    tab = pd.DataFrame(results)[['method', 'n_types', 'n_TFs', 'recall@1', 'recall@5', 'recall@10', 'median_rank', 'distinct_top']]
    print('\n================  METHOD COMPARISON (38 ground-truth master TFs)  ================')
    print(tab.to_string(index=False))
    tab.to_csv(f'{HERE}/method_comparison.csv', index=False)
    print('\n(recall@k = a biologically-correct cell type is in the TF\'s top-k; median_rank of correct type out of n_types,')
    print(' chance ~n_types/2; distinct_top = # unique top-1 calls across the 38 TFs — low = does not differentiate TFs)')


if __name__ == '__main__':
    main()

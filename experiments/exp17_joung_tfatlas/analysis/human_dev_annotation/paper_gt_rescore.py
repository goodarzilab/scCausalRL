"""Re-score every TF->cell-type annotation method against the JOUNG PAPER's authoritative,
experimentally-validated ground truth (paper_ground_truth.GROUND_TRUTH) instead of my hand-curated
set. Same cosine machinery as method_comparison.py; two scoring pools:
  all      : rank a correct type among all reference cell types (comparable to prior numbers)
  reachable: rank only among the ~23 cell types this assay can actually produce (Table S4B) -
             the fair pool, since no method can be expected to rank unreachable types correctly.

Methods: cosine_S6 (paper aggregate sig), cosine_celllevel (matched-norm sig), celltypist_LRsig,
celltypist_percell (ΔP, its intended use).  Run: python paper_gt_rescore.py
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, celltypist
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from data.load_joung import _lognorm
from paper_ground_truth import GROUND_TRUTH, REACHABLE_KEYWORDS
D = '/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
GT = {tf.upper(): kws for tf, (_, kws) in GROUND_TRUTH.items()}


def cosmat(Ydf, sigdf, drift_pcs=4):
    """cosine(TF shift, cell-type signature) over shared genes, after OE-axis (drift) removal."""
    shared = sigdf.index.intersection(Ydf.columns)
    X = sigdf.loc[shared].values.astype(np.float64); Dd = X - X.mean(1, keepdims=True)
    keep = Dd.var(1) > np.quantile(Dd.var(1), 0.5); Dd, sh = Dd[keep], shared[keep]
    Dz = (Dd - Dd.mean(0)) / (Dd.std(0) + 1e-8)
    Y = Ydf[sh].values.astype(np.float64); Yc = Y - Y.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False); k = min(drift_pcs, len(S))
    Yc = Yc - (U[:, :k] * S[:k]) @ Vt[:k]
    Yz = (Yc - Yc.mean(1, keepdims=True)) / (Yc.std(1, keepdims=True) + 1e-8)
    return pd.DataFrame((Yz @ Dz) / Yz.shape[1], index=Ydf.index, columns=sigdf.columns)


def score(scoredf, name):
    """recall@k + median rank of a correct type, in two pools: all types vs reachable-only."""
    scoredf = scoredf.copy(); scoredf.index = scoredf.index.astype(str).str.upper()
    types = np.array([str(c) for c in scoredf.columns]); tl = np.array([t.lower() for t in types])
    reach = np.array([any(kw in t for kw in REACHABLE_KEYWORDS) for t in tl])
    out = {'method': name, 'n_types': len(types), 'n_reachable': int(reach.sum())}
    for pool, mask in [('all', np.ones(len(types), bool)), ('reachable', reach)]:
        r1 = r5 = r10 = 0; ranks = []; tops = []; n = 0
        sub_types = types[mask]; sub_tl = tl[mask]
        for tf in [t for t in GT if t in scoredf.index]:
            acc = np.array([any(kw in c for kw in GT[tf]) for c in sub_tl])
            if acc.sum() == 0: continue
            vals = scoredf.loc[tf].values[mask]
            order = np.argsort(-vals); best = int(np.where(acc[order])[0].min()) + 1
            ranks.append(best); r1 += best <= 1; r5 += best <= 5; r10 += best <= 10
            tops.append(sub_types[order[0]]); n += 1
        out[f'{pool}_nTF'] = n
        out[f'{pool}_r@1'] = round(r1 / n, 2); out[f'{pool}_r@5'] = round(r5 / n, 2)
        out[f'{pool}_r@10'] = round(r10 / n, 2); out[f'{pool}_medrank'] = int(np.median(ranks))
        out[f'{pool}_ntop'] = len(set(tops))
    return out


def main():
    a = ad.read_h5ad(JOUNG); a.var_names = [str(v).upper() for v in a.var_names]; a.X = _lognorm(a.X)
    tf = a.obs['TF_gene'].astype(str).values; ctrl = np.asarray(a.obs['is_control'].values, bool)
    base = np.asarray(a.X[ctrl].mean(0)).ravel(); rows, names = [], []
    for g in sorted(set(tf)):
        m = (tf == g) & (~ctrl)
        if m.sum() >= 50: rows.append(np.asarray(a.X[m].mean(0)).ravel() - base); names.append(g)
    Ydf = pd.DataFrame(np.vstack(rows), index=names, columns=a.var_names); Ydf = Ydf.loc[:, ~Ydf.columns.duplicated()]
    print(f'Joung shifts: {Ydf.shape}; validated GT TFs present: '
          f'{sorted([t for t in GT if t in set(str(x).upper() for x in names)])}', flush=True)

    g = pd.read_csv(f'{D}/GSE156793_S2_Metadata_genes.txt.gz'); id2s = dict(zip(g.gene_id, g.gene_short_name.astype(str).str.upper()))
    S6 = pd.read_csv(f'{D}/GSE156793_S6_gene_expression_celltype.txt.gz', index_col=0)
    S6.index = S6.index.map(lambda e: id2s.get(e)); S6 = S6[S6.index.notna()]; S6 = S6[~S6.index.duplicated()]
    CL = pd.read_csv(f'{HERE}/cao_type_mean_logexpr.csv', index_col=0).T; CL.index = CL.index.astype(str).str.upper(); CL = CL[~CL.index.duplicated()]
    model = celltypist.models.Model.load(f'{HERE}/cao_celltypist_model.pkl')
    LR = pd.DataFrame(model.classifier.coef_.T, index=[str(x).upper() for x in model.classifier.features],
                      columns=list(model.classifier.classes_)); LR = LR.loc[~LR.index.duplicated()]

    results = []
    # chance: per GT TF with m acceptable of C types, E[recall@k]=1-(1-m/C)^k (over each pool)
    for pool_name, cols in [('all', S6.columns), ('reachable',
                            [c for c in S6.columns if any(kw in str(c).lower() for kw in REACHABLE_KEYWORDS)])]:
        Ctl = [str(c).lower() for c in cols]; C = len(Ctl)
        ms = [sum(any(kw in c for kw in GT[t]) for c in Ctl) for t in GT
              if t in set(str(x).upper() for x in names) and sum(any(kw in c for kw in GT[t]) for c in Ctl) > 0]
        ch = lambda k: float(np.mean([1 - (1 - m / C) ** k for m in ms]))
        print(f'chance [{pool_name}, C={C}, nTF={len(ms)}]: r@1={ch(1):.2f} r@5={ch(5):.2f} '
              f'r@10={ch(10):.2f} medrank~{C // 2}', flush=True)

    results.append(score(cosmat(Ydf, S6), 'cosine_S6 (shipped)'))
    results.append(score(cosmat(Ydf, CL), 'cosine_celllevel (proposed)'))
    results.append(score(cosmat(Ydf, LR), 'celltypist_LRsig'))

    # per-cell ΔP (CellTypist intended use): predict on control + each GT TF's cells, ΔP over classes
    rng = np.random.default_rng(0); idx, grp = [], []
    ci = np.where(ctrl)[0]; sel = rng.choice(ci, min(3000, len(ci)), replace=False); idx += list(sel); grp += ['__ctrl__'] * len(sel)
    present = [t for t in GT if t in set(tf)]
    for gtf in present:
        gi = np.where((tf == gtf) & (~ctrl))[0]
        if len(gi) < 30: continue
        s = rng.choice(gi, min(200, len(gi)), replace=False); idx += list(s); grp += [gtf] * len(s)
    sub = a[np.array(idx)].copy(); grp = np.array(grp)
    pred = celltypist.annotate(sub, model=model, majority_voting=False)
    P = pred.probability_matrix; cmean = P.values[grp == '__ctrl__'].mean(0)
    dP = {gtf: P.values[grp == gtf].mean(0) - cmean for gtf in present if (grp == gtf).any()}
    dPdf = pd.DataFrame(dP).T; dPdf.columns = P.columns
    results.append(score(dPdf, 'celltypist_percell ΔP'))

    cols = ['method', 'n_types', 'n_reachable',
            'all_nTF', 'all_r@1', 'all_r@5', 'all_r@10', 'all_medrank', 'all_ntop',
            'reachable_nTF', 'reachable_r@1', 'reachable_r@5', 'reachable_r@10', 'reachable_medrank', 'reachable_ntop']
    tab = pd.DataFrame(results)[cols]
    print('\n========  PAPER-VALIDATED GROUND TRUTH (Joung Fig 3D/4; 18 TFs, 8 cell types)  ========')
    print(tab.to_string(index=False))
    tab.to_csv(f'{HERE}/paper_gt_rescore.csv', index=False)
    print(f'\nsaved -> paper_gt_rescore.csv  | recall@k = a paper-validated cell type in TF top-k; '
          f'medrank = rank of correct type; ntop = distinct top-1 calls (differentiation).')


if __name__ == '__main__':
    main()

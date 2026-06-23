"""Head-to-head: does the Cao-trained CellTypist help annotate the Joung TFs vs our shift-cosine?

(a) PER-CELL ΔP: predict on Joung control + TF cells with the trained model, ΔP = meanP(OE) −
    meanP(ctrl). Saturation check (does it collapse to one direction like the pretrained models?).
(b) SIGNATURE HEAD-TO-HEAD: run our shift + OE-axis-removal + permutation-FDR pipeline twice on
    the SAME 172 Organ_cell_lineage types — once with the trained model's DISCRIMINATIVE LR
    gene-signatures, once with the GENERATIVE mean-deviation signatures — and compare confident-
    TF counts (q<0.05 / q<0.10). Tests whether discriminative signatures beat mean-deviation
    targets for shift alignment.

Run (high-mem):  sbatch analysis/human_dev_annotation/celltypist_custom_compare.sbatch
"""
import os, sys, json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import celltypist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from data.load_joung import _lognorm
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
MODEL = f'{HERE}/cao_celltypist_model.pkl'; TYPEMEAN = f'{HERE}/cao_type_mean_logexpr.csv'


def shift_cosine_fdr(Dsig, Ydf, n_perm=100, drift_pcs=4):
    """Dsig: DataFrame [genes × types]; Ydf: [TF × genes] induced shifts. Returns per-TF
    specificity, BH q, top call — using cosine alignment after OE-axis removal + a pooled
    gene-permutation null (identical machinery to the S6 cosine annotation)."""
    shared = Dsig.index.intersection(Ydf.columns)
    X = Dsig.loc[shared].values.astype(np.float64)
    D = X - X.mean(1, keepdims=True)
    keep = D.var(1) > np.quantile(D.var(1), 0.5)
    D, sh = D[keep], shared[keep]
    Dz = (D - D.mean(0)) / (D.std(0) + 1e-8)
    Y = Ydf[sh].values.astype(np.float64); Yc = Y - Y.mean(0, keepdims=True)
    if drift_pcs > 0:
        U, S, Vt = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False)
        k = min(drift_pcs, len(S)); Yc = Yc - (U[:, :k] * S[:k]) @ Vt[:k]
    rng = np.random.default_rng(0); specs, tops, pool = [], [], []
    cols = list(Dsig.columns)
    for i in range(len(Yc)):
        y = Yc[i]; yz = (y - y.mean()) / (y.std() + 1e-8)
        cos = (Dz.T @ yz) / len(yz)
        specs.append(float(cos.max() - cos.mean())); tops.append(cols[int(cos.argmax())])
        idx = np.argsort(rng.random((n_perm, len(yz))), axis=1)
        nc = (Dz.T @ yz[idx].T) / len(yz); pool.append(nc.max(0) - nc.mean(0))
    specs = np.array(specs); pool = np.sort(np.concatenate(pool))
    pv = (1 + (len(pool) - np.searchsorted(pool, specs, 'left'))) / (len(pool) + 1)
    order = np.argsort(pv); n = len(pv)
    qs = np.minimum.accumulate((pv[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n); q[order] = np.clip(qs, 0, 1)
    return pd.DataFrame({'TF': Ydf.index, 'top': tops, 'spec': specs, 'q': q})


def main():
    a = ad.read_h5ad(JOUNG); a.var_names = [str(v).upper() for v in a.var_names]
    a.X = _lognorm(a.X)                                   # log1p CP10k
    tf = a.obs['TF_gene'].astype(str).values; ctrl = np.asarray(a.obs['is_control'].values, bool)
    genes = list(a.var_names); Xd = a.X
    mean_rows = lambda m: np.asarray(Xd[m].mean(0)).ravel()
    base = mean_rows(ctrl)
    rows, names = [], []
    for g in sorted(set(tf)):
        m = (tf == g) & (~ctrl)
        if m.sum() >= 50: rows.append(mean_rows(m) - base); names.append(g)
    Ydf = pd.DataFrame(np.vstack(rows), index=names, columns=genes)
    Ydf = Ydf.loc[:, ~Ydf.columns.duplicated()]
    print(f'Joung shifts: {Ydf.shape[0]} TFs × {Ydf.shape[1]} genes', flush=True)

    model = celltypist.models.Model.load(MODEL)
    feats = [str(x).upper() for x in model.classifier.features]
    classes = list(model.classifier.classes_)
    coef = model.classifier.coef_                         # [n_classes × n_features]
    D_lr = pd.DataFrame(coef.T, index=feats, columns=classes)        # discriminative signatures
    D_lr = D_lr.loc[~D_lr.index.duplicated()]
    tm = pd.read_csv(TYPEMEAN, index_col=0); tm.columns = [str(c).upper() for c in tm.columns]
    D_md = (tm.T).copy(); D_md = D_md.loc[~D_md.index.duplicated()]  # [genes × types] mean log-expr
    print(f'LR signatures: {D_lr.shape} | mean-deviation: {D_md.shape}', flush=True)

    print('\n=== (b) SIGNATURE HEAD-TO-HEAD (shift + OE-removal + permFDR) ===', flush=True)
    res = {}
    for name, Dsig in [('LR_discriminative', D_lr), ('mean_deviation', D_md)]:
        r = shift_cosine_fdr(Dsig, Ydf)
        res[name] = r
        print(f"  {name:18s}: q<0.05 → {int((r['q']<0.05).sum())} | q<0.10 → {int((r['q']<0.10).sum())} | "
              f"median spec {r['spec'].median():.3f}")
        r.sort_values('spec', ascending=False).to_csv(f'{HERE}/celltypist_{name}_annotation.csv', index=False)
    # top-10 of each for eyeballing
    for name in res:
        print(f"\n  top-8 ({name}):")
        print(res[name].sort_values('spec', ascending=False).head(8)[['TF', 'top', 'spec', 'q']].to_string(index=False))

    print('\n=== (a) PER-CELL ΔP saturation check (subsample) ===', flush=True)
    rng = np.random.default_rng(0); TFS = ['NRL', 'SCRT2', 'MSGN1', 'GRHL3', 'T', 'PDX1', 'PAX5', 'ATOH7', 'CDX2', 'MEIS2']
    idx, grp = [], []
    ci = np.where(ctrl)[0]; sel = rng.choice(ci, min(3000, len(ci)), replace=False); idx += list(sel); grp += ['control'] * len(sel)
    for g in TFS:
        gi = np.where((tf == g) & (~ctrl))[0]
        if len(gi) < 30: continue
        s = rng.choice(gi, min(300, len(gi)), replace=False); idx += list(s); grp += [g] * len(s)
    sub = a[np.array(idx)].copy(); grp = np.array(grp)
    pred = celltypist.annotate(sub, model=model, majority_voting=False)
    P = pred.probability_matrix; ctrl_mean = P.values[grp == 'control'].mean(0)
    for g in TFS:
        m = grp == g
        if m.sum() == 0: continue
        dP = P.values[m].mean(0) - ctrl_mean; top = P.columns[np.argsort(dP)[::-1][:3]]
        print(f"  {g:7s} ΔP→ " + ', '.join(f'{c} (+{dP[list(P.columns).index(c)]:.2f})' for c in top))
    print('\nsaved → celltypist_{LR_discriminative,mean_deviation}_annotation.csv')


if __name__ == '__main__':
    main()

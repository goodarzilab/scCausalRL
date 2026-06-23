"""Test CellTypist as an alternative to our shift-cosine annotation of the Joung TFs.

CellTypist = per-cell logistic-regression classifier on ABSOLUTE expression. Our problem is
annotating the TF-induced SHIFT (all Joung cells are hESC-derived → absolute profiles are
baseline-dominated). So we test it two ways and check whether it actually sees TF identity:

  (1) SATURATION CHECK: per-cell predicted-label distribution for control vs each TF's OE cells.
      If OE ≈ control (labels barely move), CellTypist is blind to the TF shift on this data.
  (2) ΔP SHIFT: mean class-probability over OE cells − over control cells → which fetal cell-type
      probability the TF shifts toward. Compare the top ΔP call to our cosine annotation.

Model: Developing_Human_Organs (broad fetal endoderm). NOTE: no single pretrained CellTypist
model spans all germ layers, so the label space is inherently partial for a pan-fate question.

Run (high-mem):  sbatch analysis/human_dev_annotation/try_celltypist.sbatch
"""
import os, sys, json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import celltypist
from celltypist import models

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # exp17 root
from data.load_joung import _lognorm                          # log1p CP10k (CellTypist's expected input)
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
MODEL = 'Developing_Human_Organs.pkl'
N_PER_TF, N_CTRL = 300, 3000
# mix of top-FDR + textbook-fate TFs to sanity-check directions
TFS = ['NRL', 'SCRT2', 'MSGN1', 'GRHL3', 'T', 'SOX13', 'TFAP2E', 'ATOH7', 'PDX1', 'PAX5',
       'FIGLA', 'MEIS2', 'LHX1', 'HOXA6', 'KLF5', 'TFAP2A', 'NHLH1', 'CDX2', 'FERD3L', 'HOXB2']


def main():
    models.download_models(model=MODEL, force_update=False)
    print('loading Joung...', flush=True)
    a = ad.read_h5ad(JOUNG)                              # X = lognorm CP10k (CellTypist's expected input)
    a.var_names = [str(v).upper() for v in a.var_names]
    tf = a.obs['TF_gene'].astype(str).values
    ctrl = np.asarray(a.obs['is_control'].values, bool)
    rng = np.random.default_rng(0)
    # subsample control + the chosen TFs
    keep_idx, keep_tf = [], []
    ci = np.where(ctrl)[0]; sel = rng.choice(ci, min(N_CTRL, len(ci)), replace=False)
    keep_idx += list(sel); keep_tf += ['control'] * len(sel)
    for g in TFS:
        gi = np.where((tf == g) & (~ctrl))[0]
        if len(gi) < 30: continue
        s = rng.choice(gi, min(N_PER_TF, len(gi)), replace=False)
        keep_idx += list(s); keep_tf += [g] * len(s)
    sub = a[np.array(keep_idx)].copy(); sub.obs['grp'] = keep_tf
    sub.X = _lognorm(sub.X)                              # raw counts → log1p CP10k (per-cell)
    print(f'subsample: {sub.n_obs} cells ({sub.obs.grp.value_counts().to_dict()})', flush=True)

    print('running CellTypist...', flush=True)
    pred = celltypist.annotate(sub, model=MODEL, majority_voting=False)
    lab = pred.predicted_labels['predicted_labels'].values
    P = pred.probability_matrix                          # [cells × celltypes] DataFrame
    grp = np.array(keep_tf)

    # (1) saturation: predicted-label distribution, control vs TFs
    print('\n=== (1) SATURATION — top predicted label per group (fraction) ===')
    for g in ['control'] + TFS:
        m = grp == g
        if m.sum() == 0: continue
        vc = pd.Series(lab[m]).value_counts(normalize=True)
        print(f"  {g:8s} n={m.sum():4d} | top: " + ', '.join(f'{k} {v:.0%}' for k, v in vc.head(3).items()))

    # (2) ΔP shift vs control, compare to cosine annotation
    ctrl_mean = P.values[grp == 'control'].mean(0)
    cos = pd.read_csv(os.path.join(HERE, 'joung_human_annotation.csv')).set_index('TF')
    print('\n=== (2) ΔP top fetal-organ shift (OE−ctrl) vs our cosine call ===')
    rows = []
    for g in TFS:
        m = grp == g
        if m.sum() == 0: continue
        dP = P.values[m].mean(0) - ctrl_mean
        top = P.columns[np.argsort(dP)[::-1][:3]]
        cos_call = cos.loc[g, 'top_celltype'] if g in cos.index else 'NA'
        cos_q = cos.loc[g, 'fdr_qval'] if g in cos.index else np.nan
        print(f"  {g:8s} ΔP→ {', '.join(f'{c} (+{dP[list(P.columns).index(c)]:.2f})' for c in top)}   | cosine: {cos_call} (q={cos_q:.2f})")
        rows.append({'TF': g, 'celltypist_top': top[0], 'celltypist_dP': float(dP.max()),
                     'cosine_top': cos_call, 'cosine_q': float(cos_q) if pd.notna(cos_q) else None})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'celltypist_compare.csv'), index=False)
    print('\nsaved → celltypist_compare.csv')


if __name__ == '__main__':
    main()

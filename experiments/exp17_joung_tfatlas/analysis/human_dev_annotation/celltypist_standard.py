"""Standard CellTypist usage (pretrained model + majority voting) on the Joung TF atlas —
the way people normally annotate cell types. Reports the majority-vote cell-type label per
TF population (and control), so we can see (a) what fetal cell types these hESC-OE cells get
called, and (b) whether the calls actually differ across TFs or saturate on one baseline label.

Run (high-mem):  sbatch analysis/human_dev_annotation/celltypist_standard.sbatch
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
import celltypist
from celltypist import models

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from data.load_joung import _lognorm
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
MODELS = ['Developing_Human_Organs.pkl', 'Pan_Fetal_Human.pkl']
N_PER_TF, N_CTRL = 400, 4000
TFS = ['NRL','SCRT2','MSGN1','GRHL3','T','SOX13','ATOH7','PDX1','PAX5','CDX2',
       'MEIS2','LHX1','HOXA6','TFAP2A','NHLH1','FERD3L','HOXB2','FIGLA','KLF5','TFAP2E']


def main():
    for m in MODELS: models.download_models(model=m, force_update=False)
    a = ad.read_h5ad(JOUNG); a.var_names = [str(v).upper() for v in a.var_names]
    tf = a.obs['TF_gene'].astype(str).values; ctrl = np.asarray(a.obs['is_control'].values, bool)
    rng = np.random.default_rng(0); idx, grp = [], []
    ci = np.where(ctrl)[0]; sel = rng.choice(ci, min(N_CTRL, len(ci)), replace=False)
    idx += list(sel); grp += ['control']*len(sel)
    for g in TFS:
        gi = np.where((tf==g)&(~ctrl))[0]
        if len(gi) < 30: continue
        s = rng.choice(gi, min(N_PER_TF, len(gi)), replace=False); idx += list(s); grp += [g]*len(s)
    sub = a[np.array(idx)].copy(); sub.obs['grp'] = grp; sub.X = _lognorm(sub.X)
    print(f'subsample {sub.n_obs} cells\n', flush=True)

    for MODEL in MODELS:
        print(f'\n################  MODEL = {MODEL}  (majority_voting=True)  ################', flush=True)
        pred = celltypist.annotate(sub, model=MODEL, majority_voting=True)
        mv = pred.predicted_labels['majority_voting'].values
        print(f"{'group':9s} {'majority-vote cell type (fraction of pop)':55s}")
        calls = {}
        for g in ['control']+TFS:
            mm = np.array(grp)==g
            if mm.sum()==0: continue
            vc = pd.Series(mv[mm]).value_counts(normalize=True)
            calls[g] = vc.index[0]
            print(f"  {g:8s} {vc.index[0]} ({vc.iloc[0]:.0%})" + (f"; {vc.index[1]} ({vc.iloc[1]:.0%})" if len(vc)>1 else ""))
        # how many DISTINCT dominant labels across the TFs? (saturation diagnostic)
        tf_calls = [calls[g] for g in TFS if g in calls]
        print(f"\n  distinct dominant labels across {len(tf_calls)} TFs: {len(set(tf_calls))}  "
              f"({'SATURATED → same label' if len(set(tf_calls))<=2 else 'some diversity'})")
        print(f"  control dominant label: {calls.get('control')}")


if __name__ == '__main__':
    main()

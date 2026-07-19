"""Enumerate the FIXED combo CV folds (deterministic, seed-independent) from the v2b combo universe.
Eligible pairs = both TFs are TRAIN singles at the FIXED single split (seed 42) AND >=20 cells.
  LOTO (leave-one-TF-out): one fold per distinct combo-TF; fold holds out ALL pairs containing it.
  LOPO (leave-one-pair-out): one fold per pair; fold holds out that single pair.
Writes configs/ablations/combo_folds.json = {tfs, all_pairs, loto:{tf:[[a,b]..]}, lopo:{name:[[a,b]]}}.
"""
import os, sys, json, numpy as np, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.combo_eval_v2b import parse_tfs, COMBO
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
de = dd['de_single']; gn = set(map(str, dd['gene_names']))
train_s = set(g for g in dd['pert_split']['train'] if g in de)
C = ad.read_h5ad(COMBO); tfset = C.obs['TF_set'].astype(str).values
from collections import Counter
cnt = Counter(tfset)
pairs = []
for ts in dict.fromkeys(tfset):
    tfs = [t for t in parse_tfs(ts) if t in gn]
    if len(tfs) != 2 or cnt[ts] < 20: continue
    if not all(t in train_s for t in tfs): continue     # eligible: both TFs TRAIN singles @ seed42
    pairs.append(tuple(sorted(tfs)))
pairs = sorted(set(pairs))
tfs = sorted(set(t for p in pairs for t in p))
loto = {t: [list(p) for p in pairs if t in p] for t in tfs}
lopo = {f"{p[0]}_{p[1]}": [list(p)] for p in pairs}
out = dict(single_split_seed=42, n_pairs=len(pairs), tfs=tfs,
           all_pairs=[list(p) for p in pairs], loto=loto, lopo=lopo)
os.makedirs('configs/ablations', exist_ok=True)
json.dump(out, open('configs/ablations/combo_folds.json', 'w'), indent=1)
print(f"ELIGIBLE pairs: {len(pairs)} | distinct TFs: {len(tfs)}")
print(f"TFs: {tfs}")
print(f"LOTO folds: {len(loto)} (held-out sizes: {sorted([len(v) for v in loto.values()], reverse=True)})")
print(f"LOPO folds: {len(lopo)} (each holds out 1 pair)")
print("wrote configs/ablations/combo_folds.json")

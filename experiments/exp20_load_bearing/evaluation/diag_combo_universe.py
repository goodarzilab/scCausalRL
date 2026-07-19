"""Characterize the Joung v2b COMBO universe to design a FIXED, principled train/test split.
Reports: how many pairs/triples are measurable (both TFs in v2b vocab + de) and ELIGIBLE (both TFs
TRAIN singles at seed 42, no leakage); the epistasis-magnitude distribution; constituent-overlap
cos(Δ_A,Δ_B); and the pairwise combo-effect similarity structure (are combos clustered → can we do
cluster-holdout?). This tells us what split is feasible and how to control train/test similarity.
"""
import os, sys, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.combo_eval_v2b import parse_tfs
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
COMBO_V2B = V + "/joung_combo_proc_v2b.h5ad"     # the file TRAINING used
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
train_s = set(g for g in dd['pert_split']['train'] if g in de)
test_s = set(g for g in dd['pert_split']['test'] if g in de)
drift = np.stack([np.asarray(de[g], np.float32) for g in train_s]).mean(0)

C = ad.read_h5ad(COMBO_V2B); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4)
cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0); driftm = drift[mask]
print(f"combo file vars={C.shape[1]}, intersection with v2b = {len(inter)} genes", flush=True)

n_pair = n_trip = 0; pairs = []
for ts in dict.fromkeys(tfset):
    tfs = [t for t in parse_tfs(ts) if t in gidx]
    mm = tfset == ts
    if mm.sum() < 20: continue
    if len(tfs) == 2: n_pair += 1
    elif len(tfs) >= 3: n_trip += 1; continue
    else: continue
    both_meas = all(t in de for t in tfs)
    both_train = all(t in train_s for t in tfs)
    touches_test = any(t in test_s for t in tfs)
    pairs.append(dict(ts=ts, tfs=tuple(sorted(tfs)), n=int(mm.sum()), both_meas=both_meas, both_train=both_train, touches_test=touches_test, mm=mm))
print(f"\nUNIVERSE (>=20 cells): {n_pair} pairs, {n_trip} triples", flush=True)
print(f"  pairs both-measurable (both TFs in de):        {sum(p['both_meas'] for p in pairs)}", flush=True)
print(f"  pairs ELIGIBLE (both TFs TRAIN singles @seed42): {sum(p['both_train'] for p in pairs)}", flush=True)
print(f"  pairs touching a HELD-OUT single (forced-out):   {sum(p['touches_test'] for p in pairs)}", flush=True)

elig = [p for p in pairs if p['both_train']]
print(f"\n=== structure of the {len(elig)} ELIGIBLE pairs ===", flush=True)
obs = {}; eps_mag = {}; cons_overlap = {}
for p in elig:
    of = Xk[p['mm']].mean(0) - ref; ob = of[col]                         # observed combo Δ (intersection)
    da = np.asarray(de[p['tfs'][0]], np.float32)[mask]; db = np.asarray(de[p['tfs'][1]], np.float32)[mask]
    eps = (ob - (da + db))                                               # observed epistasis
    dg = np.abs(ob - driftm) > 0.15
    eps_mag[p['tfs']] = float(np.abs(eps[dg]).mean()) if dg.sum() else 0.0
    ra, rb = da - driftm, db - driftm                                    # drift-removed single effects
    cons_overlap[p['tfs']] = float(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-9))
    obs[p['tfs']] = ob - driftm                                          # drift-removed combo effect
em = np.array(list(eps_mag.values())); co = np.array(list(cons_overlap.values()))
print(f"  epistasis |eps|/DEgene: min {em.min():.3f}  median {np.median(em):.3f}  max {em.max():.3f}", flush=True)
print(f"  constituent overlap cos(dA,dB): min {co.min():+.3f}  median {np.median(co):+.3f}  max {co.max():+.3f}", flush=True)
# pairwise combo-effect similarity (clustering potential)
keys = list(obs); M = np.stack([obs[k] for k in keys]); Mz = M - M.mean(1, keepdims=True); Mz = Mz / (np.linalg.norm(Mz, axis=1, keepdims=True) + 1e-9)
S = Mz @ Mz.T; off = S[np.triu_indices(len(keys), 1)]
print(f"  combo-effect pairwise cosine: mean {off.mean():+.3f}  median {np.median(off):+.3f}  frac>0.5 {float((off>0.5).mean()):.2f}  max {off.max():+.3f}", flush=True)
# rough cluster count via correlation-threshold connected components
import scipy.sparse.csgraph as csg
adj = (S > 0.5).astype(int); np.fill_diagonal(adj, 0)
ncomp, lab = csg.connected_components(sp.csr_matrix(adj), directed=False)
sizes = np.bincount(lab)
print(f"  connected components at cos>0.5: {ncomp} (sizes {sorted(sizes.tolist(), reverse=True)[:8]})", flush=True)
# how many distinct TFs participate in eligible pairs
tfs_in = set(t for p in elig for t in p['tfs'])
print(f"  distinct TFs across eligible pairs: {len(tfs_in)}  (each TF in ~{2*len(elig)/max(len(tfs_in),1):.1f} pairs avg)", flush=True)
print("\ndone", flush=True)

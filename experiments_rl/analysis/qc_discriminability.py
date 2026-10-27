"""Definitive test: is discrimination-at-chance a DATA floor or MODEL mode-collapse?
Split each high-quality TF's cells in half, compute two independent drift-removed Δ estimates,
and measure self-retrieval (does half-A of TF i nearest-match half-B of TF i among all TFs,
by cosine). High retrieval ⇒ perturbations ARE distinguishable in data (so the WM is failing);
chance retrieval ⇒ perts are genuinely alike at this depth (shared program dominates).
Also reports median pairwise cosine of observed Δ (how similar TF responses are)."""
import os, numpy as np, anndata as ad, scipy.sparse as sp, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
a=ad.read_h5ad('/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad',backed='r')
qc=pd.read_csv(f'{HERE}/qc_joung_perts.csv'); hq=set(qc[(qc.n_cells>=100)&(qc.n_DEG>=10)&(qc.ontarget_cohenD>0.2)].TF)
is_ctrl=(a.obs['is_control'].astype(str)=='True').values; tf=a.obs['TF_gene'].astype(str).str.upper().values
def dense(x): return (x.toarray() if sp.issparse(x) else np.asarray(x)).astype(np.float64)
rng=np.random.default_rng(0); CH=50000; g=a.shape[1]
# ctrl mean
cs=np.zeros(g); cr=np.where(is_ctrl)[0]
for i in range(0,len(cr),CH): cs+=dense(a.X[np.sort(cr[i:i+CH])]).sum(0)
ctrl_mean=cs/len(cr)
# accumulate two half-sums per HQ TF (deterministic split by cell index parity via hash)
from collections import defaultdict
sA=defaultdict(lambda:np.zeros(g)); sB=defaultdict(lambda:np.zeros(g)); nA=defaultdict(int); nB=defaultdict(int)
n=a.shape[0]
for i in range(0,n,CH):
    rows=np.arange(i,min(i+CH,n)); X=dense(a.X[rows]); tfs=tf[rows]; ic=is_ctrl[rows]
    for t in np.unique(tfs[~ic]):
        if t not in hq: continue
        m=np.where((tfs==t)&(~ic))[0]
        half=rng.random(len(m))<0.5
        sA[t]+=X[m[half]].sum(0); nA[t]+=int(half.sum())
        sB[t]+=X[m[~half]].sum(0); nB[t]+=int((~half).sum())
TFs=[t for t in sA if nA[t]>=20 and nB[t]>=20]
dA=np.stack([sA[t]/nA[t]-ctrl_mean for t in TFs]); dB=np.stack([sB[t]/nB[t]-ctrl_mean for t in TFs])
drift=dA.mean(0)
rA=dA-drift; rB=dB-drift                                  # drift-removed residuals
def norm(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
S=norm(rA)@norm(rB).T                                     # cosine A vs B
ret1=float(np.mean(S.argmax(1)==np.arange(len(TFs))))
# raw (no drift removal) retrieval for contrast
Sr=norm(dA)@norm(dB).T; ret_raw=float(np.mean(Sr.argmax(1)==np.arange(len(TFs))))
# pairwise similarity of observed responses
off=~np.eye(len(TFs),dtype=bool)
print(f"HQ TFs tested: {len(TFs)}  (chance retrieval = {1/len(TFs):.4f})")
print(f"DATA self-retrieval, drift-removed residual (halfA->halfB): {ret1:.3f}")
print(f"DATA self-retrieval, RAW delta:                            {ret_raw:.3f}")
print(f"median pairwise cosine of RAW Δ across TFs:       {np.median((norm(dA)@norm(dA).T)[off]):.3f}")
print(f"median pairwise cosine of drift-removed Δ:        {np.median((norm(rA)@norm(rA).T)[off]):.3f}")

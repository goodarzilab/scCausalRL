"""Norman (GSE133344, K562 CRISPRa) QC — same metrics as qc_joung for head-to-head."""
import os, sys, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0,'/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose')
from data.load_norman import load_norman
HERE=os.path.dirname(os.path.abspath(__file__)); DEG_D=0.5
nd=load_norman(path='/large_storage/goodarzilab/shervin/CausalBioRL/Norman/', n_hvg=8000)
a=nd['adata']; X=a.X; X=(X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float64)
genes=np.array([g.upper() for g in a.var_names]); g2i={g:i for i,g in enumerate(genes)}
ctrl=np.array(nd['control_idx']); single=nd['single_perts']
cm=X[ctrl].mean(0); cs=X[ctrl].std(0)+1e-6
rows=[]
for g,idx in single.items():
    idx=np.array(idx); mt=X[idx].mean(0); d=(mt-cm)/cs; ti=g2i.get(g.upper(),-1)
    dd=d.copy();
    if ti>=0: dd[ti]=0
    rows.append(dict(pert=g,n_cells=len(idx),in_vocab=(ti>=0),
        ontarget_cohenD=float(d[ti]) if ti>=0 else np.nan,
        ontarget_lfc=float(mt[ti]-cm[ti]) if ti>=0 else np.nan,
        n_DEG=int((np.abs(dd)>DEG_D).sum()), delta_norm=float(np.linalg.norm(mt-cm))))
df=pd.DataFrame(rows); df.to_csv(f'{HERE}/qc_norman_perts.csv',index=False)
iv=df[df.in_vocab]
print(f"NORMAN QC | genes(axis)={len(genes)} controls={len(ctrl)} single-perts={len(df)}")
print(f"  cells/pert: median={df.n_cells.median():.0f} mean={df.n_cells.mean():.0f} min={df.n_cells.min()} max={df.n_cells.max()}")
for t in (5,20,50,100,150): print(f"   >= {t} cells: {(df.n_cells>=t).sum()} ({100*(df.n_cells>=t).mean():.0f}%)")
print(f"  in-vocab TFs: {iv.shape[0]}/{len(df)} ; on-target CohenD median={iv.ontarget_cohenD.median():.2f} ; |D|>0.5: {100*(iv.ontarget_cohenD.abs()>0.5).mean():.0f}% (UP {100*(iv.ontarget_cohenD>0.5).mean():.0f}%)")
print(f"  on-target LFC median={iv.ontarget_lfc.median():.2f}")
print(f"  DEGs/pert: median={df.n_DEG.median():.0f} mean={df.n_DEG.mean():.0f} ; >=10 DEGs: {100*(df.n_DEG>=10).mean():.0f}%")
print(f"  ||delta|| median={df.delta_norm.median():.2f}")
print(f"  global expr mean={X.mean():.3f} var={X.var():.3f}")

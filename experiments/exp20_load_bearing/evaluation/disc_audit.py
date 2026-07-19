"""Audit disc_resid confounds + all-DEG Pearson, for the top model vs ESM-NN.
  disc_KOin   : drift-removed cosine discrimination INCLUDING the perturbed gene (current metric)
  disc_noKO   : SAME but the query's own perturbed gene EXCLUDED from every cosine (kills the KO-spike)
  pear_allDEG : Pearson(pred-drift, obs-drift) over ALL DE genes (|obs-drift|>0.1) per TF, KO-excluded
                (GEARS-style but all DEGs, not top-20) — the honest correctness metric
If disc_noKO ≈ chance while disc_KOin is high → disc was mostly the KO spike.
"""
import os, sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = VDIR + "/joung_singles_proc_v2b.h5ad"; EFF = VDIR + "/effective_tf_genes.txt"
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')
TOP = 'results/grid/v2b_retrfix_refresh_k20/checkpoints/phase1_epoch60.pt'
torch.set_grad_enabled(False)
dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
obs = np.stack([np.asarray(de[g], np.float32) for g in te]); DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
m, gnm, ck = load_shift_model(TOP, 'cpu'); p1 = ck.get('config', {}).get('phase1', {})
BETA = float(p1.get('pert_mutilate_beta', 1.0)); UMODE = p1.get('pert_mutilate_u_mode', 'shift')
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 400, replace=False)]]); _, _, eps = m.encoder(xc)
A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
pred = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [gidx[g]], A, zp, u_mode=UMODE, shift_beta=BETA)[0]).mean(0).cpu().numpy() - cr) for g in te])
z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
Etr = np.stack([esm.get(g, np.zeros(dim)) for g in tr]); Ete = np.stack([esm.get(g, np.zeros(dim)) for g in te])
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
pred_esm = DEtr[(En(Ete) @ En(Etr).T).argmax(1)]

def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
def disc_KOin(P, O):
    C = zr(P - drift) @ zr(O - drift).T; d = np.diag(C)[:, None]; T = len(P); return (d > C).sum(1) / (T - 1)
def disc_noKO(P, O):
    Pr, Or = P - drift, O - drift; T = len(P); out = np.zeros(T)
    for i, g in enumerate(te):
        keep = np.ones(P.shape[1], bool)
        if g in gidx: keep[gidx[g]] = False                         # drop THIS query's perturbed gene
        pq = Pr[i, keep]; pq = (pq - pq.mean()) / (np.linalg.norm(pq - pq.mean()) + 1e-9)
        Om = zr(Or[:, keep]); c = Om @ pq
        out[i] = (c[i] > c).sum() / (T - 1)
    return out
def pear_allDEG(P, O):
    Pr, Or = P - drift, O - drift; out = []
    for i, g in enumerate(te):
        deg = np.abs(Or[i]) > 0.15                                   # ALL DE genes (drift-removed |Δ|>0.1)
        if g in gidx: deg[gidx[g]] = False                          # exclude KO gene
        x, y = Pr[i, deg], Or[i, deg]
        out.append(np.corrcoef(x, y)[0, 1] if deg.sum() > 2 and x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
    return np.array(out), int(np.median([(np.abs(Or[i]) > 0.15).sum() for i in range(len(te))]))

for nm, P in [('top model', pred), ('ESM-NN', pred_esm)]:
    ki = disc_KOin(P, obs); nk = disc_noKO(P, obs); pd_, ndeg = pear_allDEG(P, obs)
    print(f"{nm:10} disc_KOin={ki.mean():.3f}  disc_noKO={nk.mean():.3f}  "
          f"pear_allDEG={np.nanmean(pd_):.3f}  (median {ndeg} DEGs/TF)", flush=True)

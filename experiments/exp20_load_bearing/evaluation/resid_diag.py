"""Diagnose WHY the model loses to copy/retrieval baselines on drift-removed Pnk/eff_spear but wins
disc_resid. For the top model and ESM-NN, over the 49 held-out TFs, report the drift-removed residual:
  ||pred_r||/||obs_r||  (magnitude ratio — is the model under-predicting the specific residual?)
  mean per-TF cos(pred_r, obs_r)  (direction)
  mean per-TF Pearson  (=Pnk_resid)
  cross-TF variance of pred_r (does the model predict ~the same residual for every TF = collapse?)
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
Etr /= (np.linalg.norm(Etr, 1, keepdims=True) + 1e-9); Ete /= (np.linalg.norm(Ete, 1, keepdims=True) + 1e-9)
pred_esm = DEtr[(Ete / (np.linalg.norm(Ete, axis=1, keepdims=True) + 1e-9) @ (Etr / (np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-9)).T).argmax(1)]

def rpt(name, P):
    Pr = P - drift; Or = obs - drift
    cos = np.array([np.dot(Pr[i], Or[i]) / (np.linalg.norm(Pr[i]) * np.linalg.norm(Or[i]) + 1e-9) for i in range(len(te))])
    pear = np.array([np.corrcoef(Pr[i], Or[i])[0, 1] for i in range(len(te))])
    ratio = np.linalg.norm(Pr, axis=1) / (np.linalg.norm(Or, axis=1) + 1e-9)
    # cross-TF collapse: how similar are the predicted residuals to EACH OTHER (mean pairwise cos)?
    Prn = Pr / (np.linalg.norm(Pr, axis=1, keepdims=True) + 1e-9); pw = Prn @ Prn.T
    collapse = (pw.sum() - len(te)) / (len(te) * (len(te) - 1))
    print(f"{name:10} ||pred_r||/||obs_r||={ratio.mean():.3f}  cos={cos.mean():.3f}  Pnk={np.nanmean(pear):.3f}  "
          f"cross-TF pred similarity={collapse:.3f} (1=all identical=collapse)", flush=True)
print(f"obs_r norm (mean) = {np.linalg.norm(obs - drift, axis=1).mean():.3f}")
rpt('top model', pred); rpt('ESM-NN', pred_esm)

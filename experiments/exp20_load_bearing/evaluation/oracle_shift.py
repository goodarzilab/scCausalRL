"""Localize the generalization failure: is it the ESM->shift ENCODER, or the do-op PATH?
Feed the model the IDEAL shift S = D+.(true residual) for each held-out TF (the shift the encoder
SHOULD produce) and run the actual do-op. Compare:
 (1) NORMAL  ShiftEncoder do-op                 -> baseline (~0)
 (2) ORACLE  D+.resid through the do-op solve    -> if HIGH: machinery fine, only encoder fails
 (3) DIRECT  z_prev + D+.resid decoded (no solve) -> pure decoder capacity ceiling
Run: python evaluation/oracle_shift.py
"""
import sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, '.')
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
obs = np.stack([np.asarray(de[g], np.float32) for g in te])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
m, gnm, c = load_shift_model('results/grid/v2b_retrfix_refresh_k20/checkpoints/phase1_epoch60.pt', 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 800, replace=False)]]); _, _, eps = m.encoder(xc)
A = m.dag.sample_A(1).squeeze(0); zprev = m.dag.scm_forward(eps, A); cr = m.decoder(zprev).mean(0).cpu().numpy()
D = m.decoder.effective_D().detach().cpu().numpy(); Dp = np.linalg.pinv(D)


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def suite(P):
    Pr, Or = P - drift, obs - drift; C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; T = len(P); disc = float(((d > C).sum(1) / (T - 1)).mean())
    pnk, adeg = [], []
    for i, g in enumerate(te):
        keep = np.ones(P.shape[1], bool); keep[gidx[g]] = False
        x, y = Pr[i, keep], Or[i, keep]; pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan)
        deg = keep & (np.abs(Or[i]) > 0.15); xd, yd = Pr[i, deg], Or[i, deg]
        adeg.append(np.corrcoef(xd, yd)[0, 1] if deg.sum() > 2 and xd.std() > 1e-9 else np.nan)
    return f"disc {disc:.3f}  Pnk {float(np.nanmean(pnk)):+.3f}  allDEG {float(np.nanmean(adeg)):+.3f}"


Pn = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zprev, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])
print("(1) NORMAL  ShiftEncoder do-op    :", suite(Pn), flush=True)
Po = []
for i, g in enumerate(te):
    Sor = torch.tensor((obs[i] - drift) @ Dp.T, dtype=torch.float32)
    u = (1.0 * Sor.abs()).clamp(max=1.0)
    zp, _, _ = m.dag.scm_intervene_u(eps, A, eps + Sor.unsqueeze(0), u)
    Po.append(m.decoder(zp).mean(0).cpu().numpy() - cr + drift)   # +drift: oracle targets the RESIDUAL, so add drift back (suite subtracts it)
print("(2) ORACLE  D+.resid thru do-op   :", suite(np.stack(Po)), flush=True)
Pd = np.stack([(m.decoder(zprev + torch.tensor((obs[i] - drift) @ Dp.T, dtype=torch.float32).unsqueeze(0)).mean(0).cpu().numpy() - cr + drift) for i in range(len(te))])
print("(3) DIRECT  z_prev+D+.resid decode:", suite(Pd), "  <- pure decoder capacity ceiling", flush=True)

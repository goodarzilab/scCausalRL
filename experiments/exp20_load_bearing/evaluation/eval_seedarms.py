"""Eval the split-seed robustness arms (deg2_s1, deg2_s7) on THEIR OWN held-out sets (split_seed 1/7),
+ ESM-kNN copy on the same split, so the copy-beating margin can be checked across different splits.
"""
import sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, '.')
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
ESM = 'data/joung_esm2_emb.npz'
torch.set_grad_enabled(False)
z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


for tag, sd in [('v2b_mmd1_deg2_s1', 1), ('v2b_mmd1_deg2_s7', 7)]:
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt", split_seed=sd)
    de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
    obs = np.stack([np.asarray(de[g], np.float32) for g in te]); DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
    rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 3000, replace=False)]])
    m, gnm, c = load_shift_model(f'results/grid/{tag}/checkpoints/phase1_epoch60.pt', 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
    _, _, eps = m.encoder(xc); A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    P = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])
    knn = DEtr[(En(np.stack([esm.get(g, np.zeros(dim)) for g in te])) @ En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])).T).argmax(1)]

    def suite(pred):
        Pr, Or = pred - drift, obs - drift; C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; T = len(pred)
        disc = float(((d > C).sum(1) / (T - 1)).mean()); pnk, adeg = [], []
        for i, g in enumerate(te):
            k = np.ones(P.shape[1], bool); k[gidx[g]] = False
            pnk.append(np.corrcoef(Pr[i, k], Or[i, k])[0, 1] if Pr[i, k].std() > 1e-9 else np.nan)
            dg = k & (np.abs(Or[i]) > 0.15); adeg.append(np.corrcoef(Pr[i, dg], Or[i, dg])[0, 1] if dg.sum() > 2 and Pr[i, dg].std() > 1e-9 else np.nan)
        return disc, float(np.nanmean(pnk)), float(np.nanmean(adeg))
    md, mp_, ma = suite(P); kd, kp, ka = suite(knn)
    print(f"{tag} (split_seed={sd}, n={len(te)}): model disc {md:.3f} Pnk {mp_:.3f} allDEG {ma:.3f}  |  ESM-kNN disc {kd:.3f} Pnk {kp:.3f} allDEG {ka:.3f}", flush=True)

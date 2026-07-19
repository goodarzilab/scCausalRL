"""DEEP diagnosis (v2b only): TRAIN vs HELD-OUT prediction, across models + baselines, on the honest
metrics. Decides overfit (train good / test bad) vs can't-fit (train ALSO bad = objective is wrong).
Per model, on a sampled 49 TRAIN TFs and the 49 TEST TFs:
  disc_resid        drift-removed cosine discrimination (KO-in)
  Pnk_resid         drift-removed KO-excl WHOLE-profile Pearson
  allDEG_pear       drift-removed KO-excl Pearson over DE genes (|obs-drift|>0.1)
  dir_cos           whole-profile drift-removed direction cosine
  mag_ratio         ||pred_r|| / ||obs_r||
Baselines (ESM-NN copy, additive/self) scored on TEST for reference.

Run: python evaluation/deep_diag.py
"""
import os, sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = VDIR + "/joung_singles_proc_v2b.h5ad"; EFF = VDIR + "/effective_tf_genes.txt"
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')
MODELS = [('top(retrfix)', 'v2b_retrfix_refresh_k20'), ('masked_go_l3', 'v2b_masked_go_l3'),
          ('learned(sparse)', 'v2b_sparse_con'), ('nlSCM', 'v2b_nlSCM_refresh_k20')]
torch.set_grad_enabled(False)
dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr_all = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
rng = np.random.default_rng(0); tr = list(np.array(tr_all)[rng.choice(len(tr_all), len(te), replace=False)])
drift = np.stack([np.asarray(de[g], np.float32) for g in tr_all]).mean(0)   # leak-free (all train)
DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr_all])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); xc_np = Xc[ctrl[rng.choice(len(ctrl), 400, replace=False)]]


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def suite(P, O, genes):
    Pr, Or = P - drift, O - drift
    C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; T = len(P)
    disc = float(((d > C).sum(1) / (T - 1)).mean())
    pnk, adeg, dcos = [], [], []
    for i, g in enumerate(genes):
        keep = np.ones(P.shape[1], bool)
        if g in gidx: keep[gidx[g]] = False
        x, y = Pr[i, keep], Or[i, keep]
        pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
        dcos.append(np.dot(Pr[i], Or[i]) / (np.linalg.norm(Pr[i]) * np.linalg.norm(Or[i]) + 1e-9))
        deg = keep & (np.abs(Or[i]) > 0.15)
        xd, yd = Pr[i, deg], Or[i, deg]
        adeg.append(np.corrcoef(xd, yd)[0, 1] if deg.sum() > 2 and xd.std() > 1e-9 and yd.std() > 1e-9 else np.nan)
    mag = float(np.mean(np.linalg.norm(Pr, axis=1) / (np.linalg.norm(Or, axis=1) + 1e-9)))
    return dict(disc=disc, Pnk=float(np.nanmean(pnk)), allDEG=float(np.nanmean(adeg)),
                cos=float(np.nanmean(dcos)), mag=mag)


def dopred(m, genes, gnm, umode, beta):
    g2 = {g: i for i, g in enumerate(gnm)}
    xc = torch.tensor(xc_np); _, _, eps = m.encoder(xc)
    A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    return np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode=umode, shift_beta=beta)[0]).mean(0).cpu().numpy() - cr) for g in genes])


print(f"{'model':16}{'split':6}{'disc':>7}{'Pnk':>8}{'allDEG':>8}{'cos':>7}{'mag':>7}")
for nm, tag in MODELS:
    ck = f"results/grid/{tag}/checkpoints/phase1_epoch60.pt"
    if not os.path.exists(ck): print(f"{nm}: no ckpt"); continue
    m, gnm, c = load_shift_model(ck, 'cpu'); p1 = c.get('config', {}).get('phase1', {})
    um = p1.get('pert_mutilate_u_mode', 'shift'); bt = float(p1.get('pert_mutilate_beta', 1.0))
    for split, genes in [('TRAIN', tr), ('TEST', te)]:
        obs = np.stack([np.asarray(de[g], np.float32) for g in genes]); s = suite(dopred(m, genes, gnm, um, bt), obs, genes)
        print(f"{nm:16}{split:6}{s['disc']:>7.3f}{s['Pnk']:>8.3f}{s['allDEG']:>8.3f}{s['cos']:>7.3f}{s['mag']:>7.2f}", flush=True)
# baselines on TEST
z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
Etr = En(np.stack([esm.get(g, np.zeros(dim)) for g in tr_all])); Ete = En(np.stack([esm.get(g, np.zeros(dim)) for g in te]))
obs_te = np.stack([np.asarray(de[g], np.float32) for g in te])
for nm, P in [('ESM-NN(copy)', DEtr[(Ete @ Etr.T).argmax(1)])]:
    s = suite(P, obs_te, te); print(f"{nm:16}{'TEST':6}{s['disc']:>7.3f}{s['Pnk']:>8.3f}{s['allDEG']:>8.3f}{s['cos']:>7.3f}{s['mag']:>7.2f}", flush=True)

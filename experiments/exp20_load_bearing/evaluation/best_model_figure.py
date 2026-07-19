"""Updated best-model figure (prior-off) on HELD-OUT TFs @4000 control cells.
BEST current predictive+load-bearing model = v2b_masked_tgt_delta_plus_mmd (masked_go + MMD, prior-off).
Left  : best model vs baselines on drift-removed disc / Pnk / allDEG (±1 SEM over 49 held-out TFs).
Right : A load-bearing — best model real vs SHUFFLED-A vs zero-A (Pnk, allDEG) = correctness gain of learned wiring.
Run: python evaluation/best_model_figure.py
"""
import os, sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
ESM = 'data/joung_esm2_emb.npz'
BEST = 'results/grid/v2b_mmd1_deg2/checkpoints/phase1_epoch60.pt'
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
obs = np.stack([np.asarray(de[g], np.float32) for g in te]); DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 4000, replace=False)]])


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def per_tf(P):                                   # returns disc(scalar), pnk[T], adeg[T]
    Pr, Or = P - drift, obs - drift
    C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; T = len(P); disc = ((d > C).sum(1) / (T - 1))
    pnk, adeg = [], []
    for i, g in enumerate(te):
        keep = np.ones(P.shape[1], bool); keep[gidx[g]] = False
        x, y = Pr[i, keep], Or[i, keep]; pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan)
        deg = keep & (np.abs(Or[i]) > 0.15); xd, yd = Pr[i, deg], Or[i, deg]
        adeg.append(np.corrcoef(xd, yd)[0, 1] if deg.sum() > 2 and xd.std() > 1e-9 else np.nan)
    return disc, np.array(pnk), np.array(adeg)


m, gnm, c = load_shift_model(BEST, 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}; _, _, eps = m.encoder(xc)


def dopred(A):
    zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    return np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])


Areal = m.dag.sample_A(1).squeeze(0)
pred_best = dopred(Areal)
# baselines
z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
pred_knn = DEtr[(En(np.stack([esm.get(g, np.zeros(dim)) for g in te])) @ En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])).T).argmax(1)]
pred_mean = np.repeat(drift[None], len(te), 0)
Xz = Xc[ctrl] - Xc[ctrl].mean(0); Xz = Xz / (np.linalg.norm(Xz, axis=0, keepdims=True) + 1e-9)
pred_corrblind = np.stack([Xz[:, gidx[g]] @ Xz for g in te]).astype(np.float32)
METHODS = [('best (MMD+DEG-MSE)', pred_best, '#1f9e89'), ('ESM-kNN copy', pred_knn, '#2f6db5'),
           ('corr-blind', pred_corrblind, '#a6611a'), ('perturb-mean', pred_mean, '#8a8a8a')]
# A-mode gain
def shuf(A, s):
    fl = A.flatten(); p = torch.randperm(fl.numel(), generator=torch.Generator().manual_seed(s))
    As = fl[p].reshape_as(A).clone(); As.fill_diagonal_(0.0); return m.dag._spectral_clamp(As.unsqueeze(0)).squeeze(0)
Amodes = {'real A': Areal, 'shuffled A': None, 'zero A': torch.zeros_like(Areal)}
again = {}
for nm, A in Amodes.items():
    if nm == 'shuffled A':
        pk = np.mean([per_tf(dopred(shuf(Areal, s)))[1] for s in range(2)], 0); ad = np.mean([per_tf(dopred(shuf(Areal, s)))[2] for s in range(2)], 0)
    else:
        _, pk, ad = per_tf(dopred(A))
    again[nm] = (np.nanmean(pk), np.nanmean(ad))

# ---- figure ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
METS = ['disc', 'Pnk', 'allDEG']; res = {}
for nm, P, col in METHODS:
    di, pk, ad = per_tf(P); res[nm] = {'disc': di, 'Pnk': pk, 'allDEG': ad}
x = np.arange(3); w = 0.2
for i, (nm, P, col) in enumerate(METHODS):
    means = [np.nanmean(res[nm][k]) for k in METS]
    sems = [np.nanstd(res[nm][k], ddof=1) / np.sqrt(np.isfinite(res[nm][k]).sum()) for k in METS]
    a1.bar(x + (i - 1.5) * w, means, w, yerr=sems, color=col, label=nm, alpha=0.92, error_kw=dict(elinewidth=1, capsize=3, ecolor='#333'))
for xi, ch in {0: 0.5, 1: 0.0, 2: 0.0}.items():
    a1.plot([xi - 0.45, xi + 0.45], [ch, ch], '--', lw=1.2, color='#c0392b')
a1.axhline(0, color='#444', lw=0.7); a1.set_xticks(x); a1.set_xticklabels(['disc_resid', 'Pnk_resid', 'allDEG'])
a1.set_ylabel('drift-removed value (±1 SEM, 49 held-out TFs)'); a1.legend(fontsize=9)
a1.set_title('Best prior-off model vs baselines (held-out singles)', fontsize=12)
xa = np.arange(2); cols = {'real A': '#1f9e89', 'shuffled A': '#a6611a', 'zero A': '#8a8a8a'}
for i, nm in enumerate(['real A', 'shuffled A', 'zero A']):
    a2.bar(xa + (i - 1) * 0.25, [again[nm][0], again[nm][1]], 0.25, color=cols[nm], label=nm, alpha=0.92)
a2.axhline(0, color='#444', lw=0.7); a2.set_xticks(xa); a2.set_xticklabels(['Pnk_resid', 'allDEG'])
a2.set_ylabel('drift-removed value (49 held-out TFs)'); a2.legend(fontsize=9)
a2.set_title('A is load-bearing: real wiring beats shuffled ≈ zero', fontsize=12)
fig.tight_layout(); out = 'results/figures/best_model_heldout.png'; fig.savefig(out, dpi=150, bbox_inches='tight')
print('best model:', {k: round(np.nanmean(res['best (masked_go+MMD)'][k]), 3) for k in METS})
print('A gain real-shuffled: Pnk', round(again['real A'][0] - again['shuffled A'][0], 3), 'allDEG', round(again['real A'][1] - again['shuffled A'][1], 3))
print('wrote', out, flush=True)

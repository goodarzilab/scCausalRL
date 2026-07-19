"""Updated held-out singles eval RESTRICTED to effective-OE perturbations (|self-LFC| >= 0.25, n=27),
vs the full n=49. Top model mmd1_deg2 + key lb9 arms + baselines. Error bars, n= labels.
Writes results/figures/lb9_ranking_lfc025.png and results/figures/best_model_lfc025.png.
Run: python evaluation/eval_lfc025.py
"""
import os, sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"; ESM = 'data/joung_esm2_emb.npz'
LFC = 0.25
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
obs = np.stack([np.asarray(de[g], np.float32) for g in te]); DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 3000, replace=False)]])
self_lfc = np.array([de[g][gidx[g]] for g in te]); STRONG = np.abs(self_lfc) >= LFC
print(f"effective-OE held-out (|self-LFC|>={LFC}): n={int(STRONG.sum())}/{len(te)}", flush=True)


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def per(P):
    Pr, Or = P - drift, obs - drift
    C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; T = len(P); disc = (d > C).sum(1) / (T - 1)
    pnk, adeg = [], []
    for i, g in enumerate(te):
        k = np.ones(len(gn), bool); k[gidx[g]] = False
        pnk.append(np.corrcoef(Pr[i, k], Or[i, k])[0, 1] if Pr[i, k].std() > 1e-9 else np.nan)
        dg = k & (np.abs(Or[i]) > 0.15); adeg.append(np.corrcoef(Pr[i, dg], Or[i, dg])[0, 1] if dg.sum() > 2 and Pr[i, dg].std() > 1e-9 else np.nan)
    return {'disc': disc, 'Pnk': np.array(pnk), 'allDEG': np.array(adeg)}


ARMS = [('mmd1_deg2 (top)', 'v2b_mmd1_deg2'), ('mmd2', 'v2b_mmd2'), ('mmd2_deg2', 'v2b_mmd2_deg2'),
        ('mmd1_retr_k20', 'v2b_mmd1_retr_k20'), ('nlSCM_mmd2_deg2', 'v2b_nlSCM_mmd2_deg2'),
        ('MMD=1 (base)', 'v2b_masked_tgt_delta_plus_mmd'), ('free357', 'v2b_free357_rf20_k10')]
R = {}
for nm, tag in ARMS:
    ck = f'results/grid/{tag}/checkpoints/phase1_epoch60.pt'
    if not os.path.exists(ck): continue
    m, gnm, c = load_shift_model(ck, 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}; _, _, eps = m.encoder(xc)
    A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    P = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])
    R[nm] = per(P); print(f"  {nm:18} n27 allDEG {np.nanmean(R[nm]['allDEG'][STRONG]):+.3f}  Pnk {np.nanmean(R[nm]['Pnk'][STRONG]):+.3f}", flush=True)
# baselines
z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
R['ESM-kNN copy'] = per(DEtr[(En(np.stack([esm.get(g, np.zeros(dim)) for g in te])) @ En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])).T).argmax(1)])
R['perturb-mean'] = per(np.repeat(drift[None], len(te), 0))
Xz = Xc[ctrl] - Xc[ctrl].mean(0); Xz = Xz / (np.linalg.norm(Xz, axis=0, keepdims=True) + 1e-9)
R['corr-blind'] = per(np.stack([Xz[:, gidx[g]] @ Xz for g in te]).astype(np.float32))
sem = lambda v: np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())

# ---- FIG 1: ranking on n=27 (allDEG + Pnk) ----
order = sorted([k for k in R if k not in ('perturb-mean', 'corr-blind')], key=lambda k: -np.nanmean(R[k]['allDEG'][STRONG]))
fig, ax = plt.subplots(figsize=(12, 6)); y = np.arange(len(order)); w = 0.4
ax.barh(y + w/2, [np.nanmean(R[k]['allDEG'][STRONG]) for k in order], w, xerr=[sem(R[k]['allDEG'][STRONG]) for k in order],
        color=['#1f9e89' if k == 'mmd1_deg2 (top)' else ('#2f6db5' if k == 'ESM-kNN copy' else '#9bb7d4') for k in order], label='allDEG', capsize=3)
ax.barh(y - w/2, [np.nanmean(R[k]['Pnk'][STRONG]) for k in order], w, xerr=[sem(R[k]['Pnk'][STRONG]) for k in order],
        color=['#eb6834' if k == 'mmd1_deg2 (top)' else '#f0b89a' for k in order], label='Pnk', capsize=3)
kn = np.nanmean(R['ESM-kNN copy']['allDEG'][STRONG]); ax.axvline(kn, color='#c0392b', ls='--', lw=1.2, label=f'ESM-kNN allDEG={kn:.2f}')
ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis(); ax.axvline(0, color='#444', lw=0.7)
ax.set_xlabel(f'drift-removed value  (±1 SEM, effective-OE held-out TFs, n={int(STRONG.sum())})'); ax.legend(fontsize=9)
ax.set_title(f'Held-out singles eval on EFFECTIVE-OE perturbations (|self-LFC|≥{LFC}, n={int(STRONG.sum())})', fontsize=12)
fig.tight_layout(); fig.savefig('results/figures/lb9_ranking_lfc025.png', dpi=150, bbox_inches='tight'); print('wrote lb9_ranking_lfc025.png', flush=True)

# ---- FIG 2: top model vs baselines, n=49 vs n=27 ----
METH = ['mmd1_deg2 (top)', 'ESM-kNN copy', 'corr-blind', 'perturb-mean']; COL = ['#1f9e89', '#2f6db5', '#a6611a', '#8a8a8a']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, mask, ttl in [(a1, np.ones(len(te), bool), f'ALL held-out (n={len(te)})'), (a2, STRONG, f'effective OE |self-LFC|≥{LFC} (n={int(STRONG.sum())})')]:
    x = np.arange(3); ww = 0.2
    for i, (nm, col) in enumerate(zip(METH, COL)):
        mu = [np.nanmean(R[nm][k][mask]) for k in ['disc', 'Pnk', 'allDEG']]; se = [sem(R[nm][k][mask]) for k in ['disc', 'Pnk', 'allDEG']]
        ax.bar(x + (i - 1.5) * ww, mu, ww, yerr=se, color=col, label=nm, alpha=0.92, error_kw=dict(elinewidth=1, capsize=3, ecolor='#333'))
    for xi, ch in {0: 0.5, 1: 0.0, 2: 0.0}.items(): ax.plot([xi - 0.45, xi + 0.45], [ch, ch], '--', lw=1.1, color='#c0392b')
    ax.axhline(0, color='#444', lw=0.7); ax.set_xticks(x); ax.set_xticklabels(['disc_resid', 'Pnk', 'allDEG']); ax.set_title(ttl, fontsize=12)
a1.set_ylabel('drift-removed value ±1 SEM'); a1.legend(fontsize=9)
fig.suptitle('Top model (MMD+DEG-MSE) vs baselines — filtering to effective OE widens the copy-beating margin', fontsize=12)
fig.tight_layout(); fig.savefig('results/figures/best_model_lfc025.png', dpi=150, bbox_inches='tight'); print('wrote best_model_lfc025.png', flush=True)

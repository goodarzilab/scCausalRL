"""Deck figures for the lb9 sweep (held-out singles, n=49 TFs, 2000 control cells, 3 A-samples avg):
  FIG A  lb9_ranking.png       — allDEG + Pnk per arm, ±1 SEM over 49 held-out TFs, sorted, ESM-kNN + chance refs
  FIG B  disc_vs_correctness.png — disc_resid does NOT track correctness (scatter across arms; winner low-disc/high-allDEG)
Also writes results/figures/lb9_ranking.csv (means+SEM) for the deck table.
Run: python evaluation/lb9_figures.py
"""
import os, sys, numpy as np, torch, scipy.sparse as sp, csv
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
obs = np.stack([np.asarray(de[g], np.float32) for g in te]); DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
NTE = len(te)


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def per_tf(P):
    Pr, Or = P - drift, obs - drift
    C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; disc = (d > C).sum(1) / (NTE - 1)
    pnk, adeg = [], []
    for i, g in enumerate(te):
        k = np.ones(P.shape[1], bool); k[gidx[g]] = False
        pnk.append(np.corrcoef(Pr[i, k], Or[i, k])[0, 1] if Pr[i, k].std() > 1e-9 else np.nan)
        dg = k & (np.abs(Or[i]) > 0.15)
        adeg.append(np.corrcoef(Pr[i, dg], Or[i, dg])[0, 1] if dg.sum() > 2 and Pr[i, dg].std() > 1e-9 else np.nan)
    return disc, np.array(pnk), np.array(adeg)


ARMS = [('masked_tgt_delta_plus_mmd', 'MMD=1 (base)'), ('mmd0p25', 'MMD=0.25'), ('mmd0p5', 'MMD=0.5'),
        ('mmd2', 'MMD=2'), ('mmd5', 'MMD=5'), ('mmd10', 'MMD=10'),
        ('mmd1_deg2', 'MMD=1 + DEG-MSE'), ('mmd2_deg2', 'MMD=2 + DEG-MSE'),
        ('mmd1_retr_k20', 'MMD=1 + retr(k20)'), ('mmd1_esmnoise', 'MMD=1 + ESM-noise'),
        ('mmd1_esmnoise_wd', 'MMD=1 + noise+wd'), ('nlSCM_mmd1', 'nlSCM + MMD=1'),
        ('nlSCM_mmd2_deg2', 'nlSCM + MMD=2 + DEG'), ('free357_rf20_k10', 'free decoder')]
R = {}
for tag, lab in ARMS:
    ck = f"results/grid/v2b_{tag}/checkpoints/phase1_epoch60.pt"
    if not os.path.exists(ck): print("skip", tag); continue
    m, gnm, c = load_shift_model(ck, 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
    rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 4000, replace=False)]]); _, _, eps = m.encoder(xc)
    A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    P = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])
    disc, pnk, adeg = per_tf(P)                                       # per-TF arrays; SEM over the 49 held-out TFs
    R[lab] = dict(disc=float(np.mean(disc)), Pnk=float(np.nanmean(pnk)), Pnk_sem=float(np.nanstd(pnk, ddof=1) / np.sqrt(np.isfinite(pnk).sum())),
                  allDEG=float(np.nanmean(adeg)), allDEG_sem=float(np.nanstd(adeg, ddof=1) / np.sqrt(np.isfinite(adeg).sum())))
    print(f"{lab:22} disc {R[lab]['disc']:.3f}  Pnk {R[lab]['Pnk']:.3f}  allDEG {R[lab]['allDEG']:.3f}", flush=True)
# ESM-kNN
z = np.load('data/joung_esm2_emb.npz', allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
di, pk, ad = per_tf(DEtr[(En(np.stack([esm.get(g, np.zeros(dim)) for g in te])) @ En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])).T).argmax(1)])
KNN = dict(disc=float(np.mean(di)), Pnk=float(np.nanmean(pk)), Pnk_sem=float(np.nanstd(pk, ddof=1) / np.sqrt(len(pk))),
           allDEG=float(np.nanmean(ad)), allDEG_sem=float(np.nanstd(ad, ddof=1) / np.sqrt(np.isfinite(ad).sum())))
print(f"{'ESM-kNN copy':22} disc {KNN['disc']:.3f}  Pnk {KNN['Pnk']:.3f}  allDEG {KNN['allDEG']:.3f}", flush=True)

with open('results/figures/lb9_ranking.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['arm', 'disc', 'Pnk', 'Pnk_sem', 'allDEG', 'allDEG_sem'])
    for lab in R: w.writerow([lab, f"{R[lab]['disc']:.3f}", f"{R[lab]['Pnk']:.3f}", f"{R[lab]['Pnk_sem']:.3f}", f"{R[lab]['allDEG']:.3f}", f"{R[lab]['allDEG_sem']:.3f}"])
    w.writerow(['ESM-kNN copy', f"{KNN['disc']:.3f}", f"{KNN['Pnk']:.3f}", f"{KNN['Pnk_sem']:.3f}", f"{KNN['allDEG']:.3f}", f"{KNN['allDEG_sem']:.3f}"])

TEAL, BLUE, GRAY, AMBER, RED = '#1f9e89', '#2f6db5', '#8a8a8a', '#d9821e', '#c0392b'
# ---- FIG A: ranking (allDEG + Pnk), sorted by allDEG desc ----
labs = sorted(R, key=lambda k: R[k]['allDEG'])
y = np.arange(len(labs))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
cols = [TEAL if ('DEG' in l or l == 'MMD=2') else (AMBER if 'retr' in l else BLUE) for l in labs]
a1.barh(y, [R[l]['allDEG'] for l in labs], xerr=[R[l]['allDEG_sem'] for l in labs], color=cols, alpha=0.9, error_kw=dict(elinewidth=1, capsize=2, ecolor='#333'))
a1.axvline(KNN['allDEG'], color=RED, ls='--', lw=1.5); a1.text(KNN['allDEG'], -0.9, f"ESM-kNN copy={KNN['allDEG']:.2f}", color=RED, fontsize=9, ha='center')
a1.axvline(0, color='#444', lw=0.7); a1.set_yticks(y); a1.set_yticklabels(labs, fontsize=10); a1.set_xlabel('allDEG (DE-gene Pearson, drift-removed) ±SEM')
a1.set_title('DE-gene prediction — held-out TFs (n=49)', fontsize=12); a1.grid(axis='x', alpha=0.15)
a2.barh(y, [R[l]['Pnk'] for l in labs], xerr=[R[l]['Pnk_sem'] for l in labs], color=cols, alpha=0.9, error_kw=dict(elinewidth=1, capsize=2, ecolor='#333'))
a2.axvline(KNN['Pnk'], color=RED, ls='--', lw=1.5); a2.text(KNN['Pnk'], -0.9, f"ESM-kNN={KNN['Pnk']:.2f}", color=RED, fontsize=9, ha='center')
a2.axvline(0, color='#444', lw=0.7); a2.set_xlabel('Pnk (whole-profile Pearson, drift-removed) ±SEM')
a2.set_title('Whole-profile prediction (n=49)', fontsize=12); a2.grid(axis='x', alpha=0.15)
fig.tight_layout(); fig.savefig('results/figures/lb9_ranking.png', dpi=150, bbox_inches='tight'); print('wrote lb9_ranking.png', flush=True)

# ---- FIG B: disc vs allDEG scatter (disc does not track correctness) ----
fig, ax = plt.subplots(figsize=(9, 6.5))
for lab in R:
    ax.scatter(R[lab]['disc'], R[lab]['allDEG'], s=70, color=BLUE, alpha=0.8, edgecolor='w', lw=0.6, zorder=3)
ax.scatter(KNN['disc'], KNN['allDEG'], s=130, color=RED, marker='*', zorder=4, label='ESM-kNN copy')
for lab in ['MMD=1 + DEG-MSE', 'MMD=10', 'MMD=1 (base)', 'MMD=2', 'free decoder']:
    if lab in R: ax.annotate(lab, (R[lab]['disc'], R[lab]['allDEG']), fontsize=9, xytext=(5, 4), textcoords='offset points')
ax.axhline(KNN['allDEG'], color=RED, ls='--', lw=1, alpha=0.6); ax.axvline(0.5, color=GRAY, ls=':', lw=1)
ax.text(0.503, ax.get_ylim()[0] + 0.02, 'disc chance', color=GRAY, fontsize=8)
ax.set_xlabel('disc_resid  (TF discrimination — the gameable metric)', fontsize=11)
ax.set_ylabel('allDEG  (actual DE-gene correctness)', fontsize=11)
ax.set_title('disc_resid does NOT track correctness\n(winner has LOW disc / HIGH allDEG; MMD=10 has HIGH disc / ~0 allDEG)', fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.15)
fig.tight_layout(); fig.savefig('results/figures/disc_vs_correctness.png', dpi=150, bbox_inches='tight'); print('wrote disc_vs_correctness.png', flush=True)

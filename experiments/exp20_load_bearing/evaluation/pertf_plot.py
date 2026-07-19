"""Per-held-out-TF Pnk and allDEG at 4000 control cells (stable), top model vs best-predictor vs
ESM-kNN. Shows whether the model is bimodal (some TFs good) or uniformly poor across all 49.
Run: python evaluation/pertf_plot.py
"""
import os, sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
obs = np.stack([np.asarray(de[g], np.float32) for g in te]); DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 4000, replace=False)]])


def per_tf(P):
    Pr, Or = P - drift, obs - drift; pnk, adeg = [], []
    for i, g in enumerate(te):
        keep = np.ones(P.shape[1], bool); keep[gidx[g]] = False
        x, y = Pr[i, keep], Or[i, keep]; pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan)
        deg = keep & (np.abs(Or[i]) > 0.15); xd, yd = Pr[i, deg], Or[i, deg]
        adeg.append(np.corrcoef(xd, yd)[0, 1] if deg.sum() > 2 and xd.std() > 1e-9 else np.nan)
    return np.array(pnk), np.array(adeg)


def dopred(tag):
    m, gnm, c = load_shift_model(f'results/grid/{tag}/checkpoints/phase1_epoch60.pt', 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
    _, _, eps = m.encoder(xc); A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    return np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])


z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
knn = DEtr[(En(np.stack([esm.get(g, np.zeros(dim)) for g in te])) @ En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])).T).argmax(1)]
SERIES = [('best (masked_go+MMD)', dopred('v2b_masked_tgt_delta_plus_mmd'), '#1f9e89'),
          ('retrfix (old top)', dopred('v2b_retrfix_refresh_k20'), '#2f6db5'),
          ('ESM-kNN copy', knn, '#e08214')]
data = {nm: per_tf(P) for nm, P, _ in SERIES}
for nm, _, _ in SERIES:
    pnk, adeg = data[nm]
    print(f"{nm:16} Pnk {np.nanmean(pnk):+.3f} | allDEG {np.nanmean(adeg):+.3f} | allDEG>0.1: {int(np.nansum(adeg>0.1))}/49 | allDEG<0: {int(np.nansum(adeg<0))}/49", flush=True)

order = np.argsort(-np.nan_to_num(data['ESM-kNN copy'][1], nan=-9))
xi = np.arange(len(te)); w = 0.27
fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
for mi, (metric, ylab) in enumerate([(1, 'allDEG Pearson (DE genes)'), (0, 'Pnk (whole profile)')]):
    for si, (nm, _, col) in enumerate(SERIES):
        vals = np.nan_to_num(data[nm][metric])[order]
        ax[mi].bar(xi + (si - 1) * w, vals, w, color=col, label=nm, alpha=0.9)
    ax[mi].axhline(0, color='#444', lw=0.8); ax[mi].set_ylabel(ylab + '\n(drift-removed)', fontsize=11)
    ax[mi].grid(axis='y', alpha=0.15)
ax[0].legend(fontsize=10, ncol=3, loc='upper right')
ax[0].set_title('Per held-out TF (n=49), 4000 control cells — sorted by ESM-kNN allDEG', fontsize=12)
ax[1].set_xlabel('held-out TF (sorted)'); ax[1].set_xticks(xi[::2])
ax[1].set_xticklabels([te[order[i]] for i in range(0, len(te), 2)], rotation=90, fontsize=6)
fig.tight_layout(); out = 'results/figures/pertf_heldout_4000.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); print('\nwrote', out, flush=True)

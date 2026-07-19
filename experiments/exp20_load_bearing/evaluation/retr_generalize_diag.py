"""WHY does the retrieval (a soft ESM-kNN) not generalize to held-out TFs / not even match ESM-kNN,
despite using the same mechanism + more info? For v2b_retrfix_refresh_k20:

A) Per-held-out-TF distribution of Pnk & allDEG (model vs ESM-kNN) -> bimodal or uniformly poor? PLOT.
B) Attention diagnostics per held-out TF: entropy, effective #neighbours (1/Σw²), max weight; does the
   top-attended train TF == the ESM-cosine nearest neighbour? (keys ARE projected ESM, so it SHOULD.)
C) Value bottleneck: values = D†·resid on the MASKED decoder. Ceiling = allDEG of D·(D†·resid) vs resid
   over TRAIN TFs (how much residual survives the decoder round-trip). Also: has the LEARNED retr_val
   drifted from a fresh D†·resid?
D) Decomposition on held-out: base-only / retrieval-only / full do-op / ESM-kNN copy / decode-retrieved-direct.

Run: python evaluation/retr_generalize_diag.py
"""
import os, sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
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
m, gnm, c = load_shift_model(TOP, 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
p1 = c.get('config', {}).get('phase1', {}); BETA = float(p1.get('pert_mutilate_beta', 1.0))
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 200, replace=False)]]); _, _, eps = m.encoder(xc)
A = m.dag.sample_A(1).squeeze(0); zp0 = m.dag.scm_forward(eps, A); cr = m.decoder(zp0).mean(0).cpu().numpy()


def per_tf(P, O):                                   # drift-removed per-TF Pnk (KO-excl) and allDEG
    Pr, Or = P - drift, O - drift; pnk, adeg = [], []
    for i, g in enumerate(te):
        keep = np.ones(P.shape[1], bool)
        if g in gidx: keep[gidx[g]] = False
        x, y = Pr[i, keep], Or[i, keep]
        pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
        deg = keep & (np.abs(Or[i]) > 0.15); xd, yd = Pr[i, deg], Or[i, deg]
        adeg.append(np.corrcoef(xd, yd)[0, 1] if deg.sum() > 2 and xd.std() > 1e-9 and yd.std() > 1e-9 else np.nan)
    return np.array(pnk), np.array(adeg)


def solve(S, u):
    zt = eps + S.unsqueeze(0); zpp, _, _ = m.dag.scm_intervene_u(eps, A, zt, u)
    return m.decoder(zpp).mean(0).cpu().numpy() - cr


def dopred(genes, mode):                            # mode: full / base / retr
    out = []
    for g in genes:
        gi = torch.tensor([g2[g]])
        Sb = m.compute_shift(gi, add_retrieval=False).sum(0)
        Sf = m.compute_shift(gi, add_retrieval=True).sum(0)
        Sr = Sf - Sb
        S = {'full': Sf, 'base': Sb, 'retr': Sr}[mode]
        out.append(solve(S, (BETA * S.abs()).clamp(max=1.0)))
    return np.stack(out)


# ---- ESM-kNN copy ----
z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}; dim = len(next(iter(esm.values())))
En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
Etr = En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])); Ete = En(np.stack([esm.get(g, np.zeros(dim)) for g in te]))
knn_idx = (Ete @ Etr.T).argmax(1); pred_knn = DEtr[knn_idx]
pred_full = dopred(te, 'full'); pred_base = dopred(te, 'base'); pred_retr = dopred(te, 'retr')

print("=== (A) held-out per-TF distribution (mean / median / #TFs allDEG>0.1 / #allDEG<0) ===")
for nm, P in [('full do-op', pred_full), ('base-only', pred_base), ('retr-only', pred_retr), ('ESM-kNN', pred_knn)]:
    pnk, adeg = per_tf(P, obs)
    print(f"  {nm:11} Pnk {np.nanmean(pnk):+.3f} (med {np.nanmedian(pnk):+.3f}) | "
          f"allDEG {np.nanmean(adeg):+.3f} (med {np.nanmedian(adeg):+.3f}) | "
          f">0.1: {int(np.nansum(adeg > 0.1))}/{len(te)} | <0: {int(np.nansum(adeg < 0))}/{len(te)}", flush=True)
pnk_f, adeg_f = per_tf(pred_full, obs); pnk_k, adeg_k = per_tf(pred_knn, obs)
r = np.corrcoef(adeg_f[np.isfinite(adeg_f) & np.isfinite(adeg_k)], adeg_k[np.isfinite(adeg_f) & np.isfinite(adeg_k)])[0, 1]
print(f"  per-TF corr(model allDEG, kNN allDEG) = {r:+.3f}  (are the same TFs easy for both?)", flush=True)

# ---- (B) attention diagnostics on held-out ----
print("\n=== (B) held-out attention (soft ESM-kNN) ===")
lab = m.pert_label[[g2[g] for g in te]]                     # [T, L]
q = m.retr_proj(lab); K = m.retr_proj(m.retr_key)
sc = (q @ K.t()) / (m.retr_proj.out_features ** 0.5 * m.retr_tau)
if m.retr_k < sc.shape[1]:
    kth = sc.topk(m.retr_k, dim=1).values[:, -1:]; sc = sc.masked_fill(sc < kth, float('-inf'))
attn = torch.softmax(sc, dim=1).numpy()                     # [T, N_tr]
ent = -(attn * np.log(attn + 1e-12)).sum(1); effn = 1.0 / (attn ** 2).sum(1); mx = attn.max(1)
retr_names = [gnm[int(j)] for j in m.retr_gene.numpy()]
top_att = attn.argmax(1)
# ESM-nn among the SAME memory bank (train+val), for apples-to-apples with attention
Emem = En(np.stack([esm.get(nm_, np.zeros(dim)) for nm_ in retr_names]))
esm_nn_mem = (Ete @ Emem.T).argmax(1)
match = float(np.mean([retr_names[top_att[i]] == retr_names[esm_nn_mem[i]] for i in range(len(te))]))
print(f"  retr_k={m.retr_k}  retr_tau={float(m.retr_tau):.3f}  retr_alpha={float(m.retr_alpha):.3f}  N_mem={len(retr_names)}")
print(f"  entropy mean {ent.mean():.2f} (max possible {np.log(m.retr_k):.2f}) | eff #neighbours {effn.mean():.1f} | max attn {mx.mean():.3f}")
print(f"  top-attended TF == ESM-nearest(mem): {match*100:.0f}% of held-out TFs", flush=True)
# hard-copy the top-attended memory TF's OBSERVED residual (what a hard kNN through THIS bank gives)
pred_hardmem = np.stack([np.asarray(de[retr_names[top_att[i]]], np.float32) for i in range(len(te))])
pnk_h, adeg_h = per_tf(pred_hardmem, obs)
print(f"  hard-copy top-attended mem residual: allDEG {np.nanmean(adeg_h):+.3f}  (vs soft retr-only {np.nanmean(adeg_f):+.3f})", flush=True)

# ---- (C) value bottleneck ----
print("\n=== (C) value bottleneck (D†·resid round-trip on masked decoder) ===")
D = m.decoder.effective_D().detach().cpu().numpy()          # [G, d]
Dp = np.linalg.pinv(D)                                       # [d, G]
tr_resid = DEtr - drift                                      # [Ntr, G]
proj = (tr_resid @ Dp.T) @ D.T                               # D·(D†·resid) projection onto col(D)
adeg_proj = []
for i, g in enumerate(tr):
    o = tr_resid[i]; deg = np.abs(o) > 0.15; deg[gidx[g]] = False
    x, y = proj[i, deg], o[deg]
    adeg_proj.append(np.corrcoef(x, y)[0, 1] if deg.sum() > 2 and x.std() > 1e-9 else np.nan)
evr = 1 - ((tr_resid - proj) ** 2).sum() / (tr_resid ** 2).sum()
print(f"  D·D† residual reconstruction on TRAIN: allDEG {np.nanmean(adeg_proj):+.3f} | frac energy kept {evr:.3f}")
print(f"  -> CEILING for what any retrieval value can carry (values live in col(D))", flush=True)
# has the learned retr_val drifted from a fresh D†·resid?
fresh = (tr_resid @ Dp.T).astype(np.float32)                 # what values would be if rebuilt now
# align memory rows (train subset) to tr order
mem_of = {retr_names[j]: j for j in range(len(retr_names))}
common = [g for g in tr if g in mem_of]
learned = m.retr_val.detach().cpu().numpy()[[mem_of[g] for g in common]]
freshc = np.stack([fresh[tr.index(g)] for g in common])
cos_lf = np.mean([np.dot(learned[i], freshc[i]) / (np.linalg.norm(learned[i]) * np.linalg.norm(freshc[i]) + 1e-9) for i in range(len(common))])
print(f"  cos(learned retr_val, fresh D†·resid) = {cos_lf:+.3f}  (1=faithful; low=drifted under loss)", flush=True)

# ---- plot: per-TF sorted allDEG (model vs kNN) + histogram ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
order = np.argsort(-np.nan_to_num(adeg_k, nan=-1))
xi = np.arange(len(te))
a1.bar(xi - 0.2, np.nan_to_num(adeg_k)[order], 0.4, color='#2a78d6', label='ESM-kNN copy', alpha=0.9)
a1.bar(xi + 0.2, np.nan_to_num(adeg_f)[order], 0.4, color='#1baf7a', label='full do-op (retrieval)', alpha=0.9)
a1.axhline(0, color='#444', lw=0.8); a1.set_xlabel('held-out TF (sorted by ESM-kNN allDEG)')
a1.set_ylabel('per-TF allDEG Pearson (drift-removed)'); a1.legend(fontsize=9)
a1.set_title('Per-held-out-TF DE-gene prediction: model vs ESM-kNN', fontsize=11)
bins = np.linspace(-0.6, 0.9, 26)
a2.hist(adeg_k[np.isfinite(adeg_k)], bins, color='#2a78d6', alpha=0.55, label='ESM-kNN')
a2.hist(adeg_f[np.isfinite(adeg_f)], bins, color='#1baf7a', alpha=0.55, label='full do-op')
a2.axvline(0, color='#444', lw=0.8); a2.set_xlabel('per-TF allDEG'); a2.set_ylabel('# held-out TFs')
a2.legend(fontsize=9); a2.set_title('Distribution (bimodal? uniformly low?)', fontsize=11)
fig.tight_layout(); out = 'results/figures/retr_generalize_diag.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); print('\nwrote', out, flush=True)

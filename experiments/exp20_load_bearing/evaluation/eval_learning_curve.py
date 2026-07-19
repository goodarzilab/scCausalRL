"""Combo data-efficiency LEARNING CURVE. Fixed 10-pair held-out test (lc_test.json); each model
trained the interaction head on N of the other 30 pairs (N in {2,4,8,16,30}, 3 draws each). Plots
do_mean allDEG + epistasis r_pair on the FIXED test vs N. N=0 = headless deg8 backbone (additive floor);
obs-additive & model-additive are constant references (backbone frozen). Writes results/figures/combo_learning_curve.png.
"""
import os, sys, json, glob, re, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, COMBO
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"; T15 = 0.15
torch.set_grad_enabled(False)
test10 = [tuple(p) for p in json.load(open('configs/ablations/lc_test.json'))['test10']]
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
train_s = [g for g in dd['pert_split']['train'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in train_s]).mean(0)
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0); xc_np = Xc[ctrl[rng.choice(len(ctrl), 2000, replace=False)]]
C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4); cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter]); driftm = drift[mask]
tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)
tf2ts = {tuple(sorted(t for t in parse_tfs(ts) if t in gidx)): ts for ts in dict.fromkeys(tfset) if len([t for t in parse_tfs(ts) if t in gidx]) == 2}


def predict(model, gnm, pairs):
    g2 = {g: i for i, g in enumerate(gnm)}; _, _, eps = model.encoder(torch.tensor(xc_np)); A = model.dag.sample_A(1).squeeze(0)
    zp = model.dag.scm_forward(eps, A); cr = model.decoder(zp).mean(0).cpu().numpy()
    sh = lambda gi: model.compute_shift(torch.tensor([gi])).sum(0)
    def solve(S, u):
        z, _, _ = model.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u); return model.decoder(z).mean(0).cpu().numpy() - cr
    out = {}
    for p in pairs:
        p = tuple(p)
        if p not in tf2ts or not all(t in g2 for t in p): continue
        mm = tfset == tf2ts[p]
        if mm.sum() < 20: continue
        ob = (Xk[mm].mean(0) - ref)[col]
        oa = np.zeros(len(gn), np.float32)
        for g in p: oa += np.asarray(de[g], np.float32)
        Sa, Sb = sh(g2[p[0]]), sh(g2[p[1]]); ua, ub = (Sa.abs()).clamp(max=1.0), (Sb.abs()).clamp(max=1.0); S = Sa + Sb
        si = model.compute_interaction(torch.tensor([g2[p[0]], g2[p[1]]])) if getattr(model, 'interaction_U', None) is not None else None
        if si is not None: S = S + si
        dm = solve(S, (ua + ub) / 2)[mask]; madd = (solve(Sa, ua) + solve(Sb, ub))[mask]
        out[p] = dict(obs=ob - driftm, obsadd=oa[mask] - driftm, dm=dm, madd=madd)
    return out


def metrics(rows):
    O = np.stack([r['obs'] for r in rows]); OA = np.stack([r['obsadd'] for r in rows]); DM = np.stack([r['dm'] for r in rows]); MA = np.stack([r['madd'] for r in rows]); N = len(rows); Od = O - O.mean(0)
    def a15(P):
        Pd = P - P.mean(0); return np.array([np.corrcoef(Pd[i, np.abs(Od[i]) > T15], Od[i, np.abs(Od[i]) > T15])[0, 1] if (np.abs(Od[i]) > T15).sum() > 2 and Pd[i, np.abs(Od[i]) > T15].std() > 1e-9 else np.nan for i in range(N)])
    dm, ma, oa = np.nanmean(a15(DM)), np.nanmean(a15(MA)), np.nanmean(a15(OA))
    dmask = np.abs(O) > T15; ns = np.array([(O - OA)[i][dmask[i]].mean() if dmask[i].sum() else 0. for i in range(N)]); nm = np.array([(DM - MA)[i][dmask[i]].mean() if dmask[i].sum() else 0. for i in range(N)])
    rp = float(np.corrcoef(ns, nm)[0, 1]) if np.std(ns) > 1e-9 and np.std(nm) > 1e-9 else np.nan
    return dm, ma, oa, rp


def latest(tag):
    cks = glob.glob(f'results/grid/{tag}/checkpoints/phase1_epoch*.pt'); return max(cks, key=lambda p: int(re.search(r'epoch(\d+)', p).group(1))) if cks else None


def sem(v): v = np.asarray([x for x in v if np.isfinite(x)], float); return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (v.mean() if len(v) else np.nan, 0.0)


# N=0 floor: headless deg8 backbone (do_mean = additive-composition, no learned interaction)
m0, g0, _ = load_shift_model(latest('v2b_mmd1_deg8'), 'cpu'); r0 = list(predict(m0, g0, test10).values())
dm0, ma_ref, oa_ref, rp0 = metrics(r0)
print(f"N=0 (headless deg8): do_mean {dm0:.3f}  r_pair {rp0:+.3f}", flush=True)
print(f"REFERENCES (constant): model-additive {ma_ref:.3f}   obs-additive {oa_ref:.3f}", flush=True)
curve = [(0, dm0, 0.0, rp0, 0.0)]
for N in [2, 4, 8, 16, 30]:
    dms, rps = [], []
    for d in range(1 if N == 30 else 3):
        ck = latest(f'v2b_lc_N{N:02d}_d{d}')
        if not ck: print(f"  MISSING N{N} d{d}", flush=True); continue
        m, gnm, _ = load_shift_model(ck, 'cpu'); rows = list(predict(m, gnm, test10).values())
        dm, _, _, rp = metrics(rows); dms.append(dm); rps.append(rp)
    (dm_m, dm_s), (rp_m, rp_s) = sem(dms), sem(rps)
    curve.append((N, dm_m, dm_s, rp_m, rp_s))
    print(f"N={N:2d} (draws={len(dms)}): do_mean {dm_m:.3f}±{dm_s:.3f}  r_pair {rp_m:+.3f}±{rp_s:.3f}", flush=True)

xs = [c[0] for c in curve]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5))
ax.errorbar(xs, [c[1] for c in curve], yerr=[c[2] for c in curve], marker='o', lw=2, capsize=3, color='#1f9e89', label='do_mean (head)')
ax.axhline(oa_ref, ls='--', color='#8a8a8a', label=f'obs-additive ({oa_ref:.2f})'); ax.axhline(ma_ref, ls=':', color='#c0392b', label=f'model-additive ({ma_ref:.2f})')
ax.set_xlabel('# combos in head training (N)'); ax.set_ylabel('do_mean allDEG on fixed 10-pair test'); ax.legend(fontsize=9); ax.set_title('Combo data-efficiency: allDEG vs #train combos', fontsize=11); ax.grid(alpha=0.2)
bx.errorbar(xs, [c[3] for c in curve], yerr=[c[4] for c in curve], marker='o', lw=2, capsize=3, color='#2f6db5'); bx.axhline(0, color='#444', lw=0.7)
bx.set_xlabel('# combos in head training (N)'); bx.set_ylabel('epistasis r_pair on fixed 10-pair test'); bx.set_title('Epistasis recovery vs #train combos', fontsize=11); bx.grid(alpha=0.2)
fig.tight_layout(); out = 'results/figures/combo_learning_curve.png'; fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out, flush=True)

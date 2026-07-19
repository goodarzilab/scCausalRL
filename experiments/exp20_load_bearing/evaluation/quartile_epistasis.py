"""Epistasis QUARTILES — FIXED metric. Bin the held-out Joung combos by traditional log-additive
epistasis |ε|, ε = Δ_AB − (Δ_A+Δ_B) drift-corrected. Per quartile, the metric is EPISTASIS-RECOVERY:
per-pair correlation between PREDICTED epistasis (do_mean − model_additive, drift-corrected) and
OBSERVED epistasis (obs − obs_additive, drift-corrected) over the pair's DE genes. This isolates the
non-additive component (subtracts the additive part from both sides) so it is NOT confounded by overall
response magnitude — unlike plain pearson(pred,obs), which rises with magnitude and misled the old plot.
 singles-trained (mmd1_deg2)   vs   combo-trained (deg2_Jcombo)   [obs-additive recovery ≡ 0 by construction].
Writes results/figures/quartile_epistasis.png.
"""
import os, sys, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, PROC, EFF, COMBO
from data.load_joung import load_joung
DEGT = 0.15
SING = 'results/grid/v2b_mmd1_deg2/checkpoints/phase1_epoch60.pt'
COMB = 'results/grid/v2b_mmd1_deg2_Jcombo_ho50/checkpoints/phase1_epoch60.pt'
torch.set_grad_enabled(False)
dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF); de = dd['de_single']
trS = [g for g in dd['pert_split']['train'] if g in de]; drift = np.stack([np.asarray(de[g], np.float32) for g in trS]).mean(0)
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0); xc_np = Xc[ctrl[rng.choice(len(ctrl), 800, replace=False)]]
C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4); cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)
held = {tuple(sorted(map(str, k))) for k in (torch.load(COMB, map_location='cpu', weights_only=False).get('config', {}).get('phase1', {}).get('combo_test_keys') or [])}


def evalm(ckpt, only_held):
    m, gn, ck = load_shift_model(ckpt, 'cpu'); gidx = {g: i for i, g in enumerate(gn)}; G = len(gn)
    BETA = float(ck.get('config', {}).get('phase1', {}).get('pert_mutilate_beta', 1.0))
    inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
    xc = torch.tensor(xc_np); _, _, eps = m.encoder(xc); A = m.dag.sample_A(1).squeeze(0)
    cr = m.decoder(m.dag.scm_forward(eps, A)).mean(0).cpu().numpy()

    def sh(gi): return m.compute_shift(torch.tensor([gi])).sum(0)

    def solve(S, u):
        zp, _, _ = m.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u); return m.decoder(zp).mean(0).cpu().numpy() - cr
    out = {}
    for ts in dict.fromkeys(tfset):
        tfs = [t for t in parse_tfs(ts) if t in gidx]; key = tuple(sorted(tfs))
        if len(tfs) != 2: continue
        if only_held and key not in held: continue
        mm = tfset == ts
        if mm.sum() < 20: continue
        of = Xk[mm].mean(0) - ref; obs = np.zeros(G, np.float32); obs[mask] = of[col]
        oa = np.zeros(G, np.float32)
        for g in tfs:
            if g in de: oa = oa + np.asarray(de[g], np.float32)
        Sa, Sb = sh(gidx[tfs[0]]), sh(gidx[tfs[1]]); ua, ub = (BETA * Sa.abs()).clamp(max=1.0), (BETA * Sb.abs()).clamp(max=1.0)
        pa, pb = solve(Sa, ua), solve(Sb, ub); dm = solve(Sa + Sb, (ua + ub) / 2)
        out[key] = dict(obs=obs[mask], obsadd=oa[mask], dm=dm[mask], madd=(pa + pb)[mask], mask=mask)
    return out


sng = evalm(SING, only_held=False); cmb = evalm(COMB, only_held=True)
keys = list(sng); driftm = drift[sng[keys[0]]['mask']]


def eps_recovery(d):
    """per-pair corr( predicted epistasis , observed epistasis ) over the pair's DE genes."""
    dg = np.abs(d['obs'] - driftm) > DEGT
    if dg.sum() < 3: return np.nan
    eps_obs = (d['obs'] - d['obsadd']) + driftm            # drift-corrected observed epistasis
    eps_mod = (d['dm'] - d['madd']) + driftm               # drift-corrected model epistasis (do_mean - additive)
    x, y = eps_mod[dg], eps_obs[dg]
    return np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan


# epistasis magnitude per pair (for quartile binning), from the singles-set observed data
epsmag = {}
for k in keys:
    dg = np.abs(sng[k]['obs'] - driftm) > DEGT
    eps = (sng[k]['obs'] - sng[k]['obsadd']) + driftm
    epsmag[k] = np.abs(eps[dg]).mean() if dg.sum() else 0.0
mags = np.array([epsmag[k] for k in keys]); qedges = np.quantile(mags, [0.25, 0.5, 0.75])
qof = lambda mg: int(np.searchsorted(qedges, mg))

rec_s = {q: [] for q in range(4)}; rec_c = {q: [] for q in range(4)}
for k in keys:
    q = qof(epsmag[k]); rec_s[q].append(eps_recovery(sng[k]))
    if k in cmb: rec_c[q].append(eps_recovery(cmb[k]))
mp = lambda L: (np.nanmean(L), np.nanstd(L, ddof=1) / np.sqrt(max(np.isfinite(L).sum(), 1))) if len([x for x in L if np.isfinite(x)]) else (np.nan, 0)

fig, ax = plt.subplots(figsize=(11, 6.3)); x = np.arange(4); w = 0.34
for i, (nm, rec, col) in enumerate([('singles-trained', rec_s, '#8a8a8a'), ('combo-trained (Jcombo)', rec_c, '#1f9e89')]):
    mu = [mp(rec[q])[0] for q in range(4)]; se = [mp(rec[q])[1] for q in range(4)]
    ax.bar(x + (i - 0.5) * w, mu, w, yerr=se, color=col, capsize=3, alpha=0.9, label=f'{nm} do_mean')
ax.axhline(0, color='#444', lw=0.8)
qn = [sum(qof(epsmag[k]) == q for k in keys) for q in range(4)]
cn = [sum(qof(epsmag[k]) == q for k in cmb) for q in range(4)]
qmean = [np.mean([epsmag[k] for k in keys if qof(epsmag[k]) == q]) for q in range(4)]
ax.set_xticks(x); ax.set_xticklabels([f"Q{q+1}\n|ε|≈{qmean[q]:.2f}\nsingles n={qn[q]} / combo n={cn[q]}" for q in range(4)])
ax.set_xlabel('epistasis quartile — traditional |ε| = |Δ_AB − (Δ_A+Δ_B)| (drift-corrected) →')
ax.set_ylabel('EPISTASIS RECOVERY  = corr(predicted ε, observed ε) over DE genes, ±SEM')
ax.legend(fontsize=11, loc='upper left')
ax.set_title('Epistasis recovery by quartile (magnitude-free): combo-trained predicts the epistatic\n'
             'deviation and does so MORE where there is more to predict; singles-trained ≈ 0. (obs-additive ≡ 0)', fontsize=11)
fig.tight_layout(); out = 'results/figures/quartile_epistasis.png'; fig.savefig(out, dpi=150, bbox_inches='tight')
print('EPISTASIS RECOVERY by quartile:')
for q in range(4): print(f"  Q{q+1} |eps|~{qmean[q]:.2f}: singles {mp(rec_s[q])[0]:+.3f} (n={qn[q]})  combo {mp(rec_c[q])[0]:+.3f} (n={cn[q]})", flush=True)
print('wrote', out, flush=True)

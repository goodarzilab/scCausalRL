"""Show 'cracks epistasis' directly: SAME held-out Joung combos, singles-trained (mmd1_deg2) vs
combo-trained (deg2_Jcombo_ho50, 50/50).
 LEFT  : per-pair OBSERVED net epistasis vs MODEL net epistasis (do_mean - additive). combo-trained
         tracks the diagonal (predicts which pairs are synergistic); singles-trained is flat (r~0).
 RIGHT : do_mean pearson binned by |observed epistasis| (2 bins). combo-trained stays ABOVE the
         obs-additive baseline even in the high-epistasis bin; singles-trained collapses to it.
Writes results/figures/cracks_epistasis.png.
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
gn0 = list(map(str, dd['gene_names'])); trS = [g for g in dd['pert_split']['train'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in trS]).mean(0)
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0); xc_np = Xc[ctrl[rng.choice(len(ctrl), 800, replace=False)]]
C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4); cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)
_tk = torch.load(COMB, map_location='cpu', weights_only=False).get('config', {}).get('phase1', {}).get('combo_test_keys') or []
HELD = {tuple(sorted(map(str, k))) for k in _tk}


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def eval_model(ckpt):
    m, gn, ck = load_shift_model(ckpt, 'cpu'); gidx = {g: i for i, g in enumerate(gn)}; G = len(gn)
    BETA = float(ck.get('config', {}).get('phase1', {}).get('pert_mutilate_beta', 1.0))
    inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
    xc = torch.tensor(xc_np); _, _, eps = m.encoder(xc); A = m.dag.sample_A(1).squeeze(0)
    zprev = m.dag.scm_forward(eps, A); cr = m.decoder(zprev).mean(0).cpu().numpy()

    def sh(gi): return m.compute_shift(torch.tensor([gi])).sum(0)

    def solve(S, u):
        zp, _, _ = m.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u); return m.decoder(zp).mean(0).cpu().numpy() - cr
    keys, O, OA, DM, MA = [], [], [], [], []
    for ts in dict.fromkeys(tfset):
        tfs = [t for t in parse_tfs(ts) if t in gidx]
        if len(tfs) != 2 or tuple(sorted(tfs)) not in HELD: continue
        mm = tfset == ts
        if mm.sum() < 20: continue
        of = Xk[mm].mean(0) - ref; obs = np.zeros(G, np.float32); obs[mask] = of[col]
        Sa, Sb = sh(gidx[tfs[0]]), sh(gidx[tfs[1]]); ua, ub = (BETA * Sa.abs()).clamp(max=1.0), (BETA * Sb.abs()).clamp(max=1.0)
        pa, pb = solve(Sa, ua), solve(Sb, ub)
        oa = np.zeros(G, np.float32)
        for g in tfs:
            if g in de: oa = oa + np.asarray(de[g], np.float32)
        keys.append(tuple(sorted(tfs))); O.append(obs[mask]); OA.append(oa[mask]); DM.append(solve(Sa + Sb, (ua + ub) / 2)[mask]); MA.append((pa + pb)[mask])
    O, OA, DM, MA = map(np.stack, (O, OA, DM, MA)); driftm = drift[mask]; Od = O - O.mean(0)
    def pear(P):
        Pd = P - P.mean(0); return np.array([np.corrcoef(Pd[i], Od[i])[0, 1] if Pd[i].std() > 1e-9 else np.nan for i in range(len(O))])
    EPSo = (O - OA) + driftm[None, :]; EPSm = DM - MA; dm_ = np.abs(O - driftm[None, :]) > DEGT
    net_o = np.array([EPSo[i][dm_[i]].mean() if dm_[i].sum() else 0 for i in range(len(O))])
    net_m = np.array([EPSm[i][dm_[i]].mean() if dm_[i].sum() else 0 for i in range(len(O))])
    return dict(keys=keys, pear_dm=pear(DM), pear_ma=pear(MA), pear_oa=pear(OA), net_o=net_o, net_m=net_m)


Sng = eval_model(SING); Cmb = eval_model(COMB)
# align to shared keys (both restrict to HELD, same order)
n = len(Cmb['keys']); print(f"[cracks] {n} shared held-out combos", flush=True)
r_s = np.corrcoef(Sng['net_o'], Sng['net_m'])[0, 1]; r_c = np.corrcoef(Cmb['net_o'], Cmb['net_m'])[0, 1]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
lim = max(np.abs(np.r_[Sng['net_o'], Sng['net_m'], Cmb['net_o'], Cmb['net_m']]).max() * 1.15, 0.05)
a1.plot([-lim, lim], [-lim, lim], ':', color='#bbb', lw=1); a1.axhline(0, color='#ddd', lw=0.7); a1.axvline(0, color='#ddd', lw=0.7)
a1.scatter(Sng['net_o'], Sng['net_m'], s=60, color='#8a8a8a', edgecolor='w', lw=0.5, label=f'singles-trained (r={r_s:+.2f})')
a1.scatter(Cmb['net_o'], Cmb['net_m'], s=60, color='#1f9e89', edgecolor='w', lw=0.5, label=f'combo-trained (r={r_c:+.2f})')
a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
a1.set_xlabel('OBSERVED net epistasis  (mean ε / DE gene)'); a1.set_ylabel("MODEL net epistasis  (do_mean − additive)")
a1.legend(fontsize=11); a1.set_title(f'Does the do-op predict epistasis? (same {n} held-out combos)', fontsize=12)
# RIGHT: do_mean pearson binned by |observed epistasis| (2 bins)
mag = np.abs(Cmb['net_o']); thr = np.median(mag); binmask = [mag <= thr, mag > thr]
xb = np.arange(2); w = 0.26
def mpb(v, mk): x = v[mk]; x = x[np.isfinite(x)]; return (np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
a2.bar(xb - w, [mpb(Sng['pear_dm'], b)[0] for b in binmask], w, yerr=[mpb(Sng['pear_dm'], b)[1] for b in binmask], color='#8a8a8a', capsize=3, label='singles-trained do_mean')
a2.bar(xb, [mpb(Cmb['pear_dm'], b)[0] for b in binmask], w, yerr=[mpb(Cmb['pear_dm'], b)[1] for b in binmask], color='#1f9e89', capsize=3, label='combo-trained do_mean')
a2.bar(xb + w, [mpb(Cmb['pear_oa'], b)[0] for b in binmask], w, color='#c0392b', alpha=0.6, capsize=3, label='obs-additive')
a2.axhline(0, color='#444', lw=0.7); a2.set_xticks(xb)
a2.set_xticklabels([f'low epistasis\n|ε|≤{thr:.2f} (n={int(binmask[0].sum())})', f'high epistasis\n|ε|>{thr:.2f} (n={int(binmask[1].sum())})'])
a2.set_ylabel('combo pearson (drift-removed) ±SEM'); a2.legend(fontsize=9)
a2.set_title('Combo-trained do_mean holds above additive in the HIGH-epistasis bin', fontsize=12)
fig.tight_layout(); out = 'results/figures/cracks_epistasis.png'; fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'singles r={r_s:+.3f}  combo r={r_c:+.3f}  | wrote {out}', flush=True)

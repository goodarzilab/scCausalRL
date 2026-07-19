"""Side-by-side epistasis: SAME architecture (MMD backbone), SAME dataset (Joung), SAME held-out combos —
singles-trained vs combo-trained (Joung combos, 50/50 split). Shows combo training makes do_mean beat
additive AND predict epistasis (obs-vs-model r). Held-out combos = the combo model's stashed test keys.
Run: python evaluation/epistasis_sidebyside.py
"""
import os, sys, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, zrow, PROC, EFF, COMBO
from data.load_joung import load_joung
DEGT = 0.15
SINGLES = 'results/grid/v2b_masked_tgt_delta_plus_mmd/checkpoints/phase1_epoch60.pt'
COMBOM = 'results/grid/v2b_mmd1_Jcombo_ho50/checkpoints/phase1_epoch60.pt'
torch.set_grad_enabled(False)
dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF); de = dd['de_single']
gn0 = list(map(str, dd['gene_names'])); trS = [g for g in dd['pert_split']['train'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in trS]).mean(0)
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
# held-out combo keys = combo model's stashed test keys
ckc = torch.load(COMBOM, map_location='cpu', weights_only=False)
tk = ckc.get('config', {}).get('phase1', {}).get('combo_test_keys') or []
heldout = {tuple(sorted(map(str, k))) for k in tk}
C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4); cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)


def eval_model(ck):
    m, gn, c = load_shift_model(ck, 'cpu'); g2 = {g: i for i, g in enumerate(gn)}; G = len(gn)
    gidx = {g: i for i, g in enumerate(gn)}
    inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
    rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 800, replace=False)]]); _, _, eps = m.encoder(xc)
    A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    def shift(gi): return m.compute_shift(torch.tensor([gi])).sum(0)
    def solve(S, u): return (m.decoder(m.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u)[0]).mean(0).cpu().numpy() - cr)
    def single(gi): Sg = shift(gi); return solve(Sg, Sg.abs().clamp(max=1.0)), Sg.abs().clamp(max=1.0)
    rows = []
    for ts in dict.fromkeys(tfset):
        tfs = [t for t in parse_tfs(ts) if t in gidx]
        if len(tfs) != 2 or tuple(sorted(tfs)) not in heldout: continue
        if (tfset == ts).sum() < 20: continue
        of = Xk[tfset == ts].mean(0) - ref; obs = np.zeros(G, np.float32); obs[mask] = of[col]
        (pa, ua), (pb, ub) = single(gidx[tfs[0]]), single(gidx[tfs[1]])
        S = shift(gidx[tfs[0]]) + shift(gidx[tfs[1]])
        oa = np.zeros(G, np.float32)
        for g in tfs:
            if g in de: oa = oa + np.asarray(de[g], np.float32)
        rows.append(dict(obs=obs[mask], obsadd=oa[mask], do_mean=solve(S, (ua + ub) / 2)[mask], modeladd=(pa + pb)[mask]))
    N = len(rows); driftm = drift[mask]
    O = np.stack([r['obs'] for r in rows]); Od = O - O.mean(0)
    def ppair(Praw):
        Pd = Praw - Praw.mean(0); pe, ad_ = [], []
        for i in range(N):
            pe.append(np.corrcoef(Pd[i], Od[i])[0, 1] if Pd[i].std() > 1e-9 and Od[i].std() > 1e-9 else np.nan)
            dg = np.abs(Od[i]) > DEGT
            ad_.append(np.corrcoef(Pd[i, dg], Od[i, dg])[0, 1] if dg.sum() > 2 and Pd[i, dg].std() > 1e-9 else np.nan)
        return np.array(pe), np.array(ad_)
    P = {k: (np.stack([r['obsadd'] for r in rows]) if k == 'obsadd' else np.stack([r[k] for r in rows])) for k in ('do_mean', 'modeladd', 'obsadd')}
    M = {k: ppair(P[k]) for k in P}
    EPS = (O - P['obsadd']) + driftm[None, :]; EPSm = P['do_mean'] - P['modeladd']
    dm = np.abs(O - driftm[None, :]) > DEGT
    ns = np.array([EPS[i][dm[i]].mean() if dm[i].sum() else 0 for i in range(N)]); nm = np.array([EPSm[i][dm[i]].mean() if dm[i].sum() else 0 for i in range(N)])
    r_pair = np.corrcoef(ns, nm)[0, 1]
    return N, M, r_pair


VARS = [('do_mean', '#eb6834', 'do-op (do_mean)'), ('modeladd', '#1f9e89', 'model-additive'), ('obsadd', '#8a8a8a', 'obs-additive')]
res = {}
for name, ck in [('singles', SINGLES), ('combo', COMBOM)]:
    N, M, r_pair = eval_model(ck); res[name] = (N, M, r_pair)
    dm_pe, dm_ad = float(np.nanmean(M['do_mean'][0])), float(np.nanmean(M['do_mean'][1]))
    ma_pe, ma_ad = float(np.nanmean(M['modeladd'][0])), float(np.nanmean(M['modeladd'][1]))
    print(f"{name:8} N={N} r_pair={r_pair:+.3f} | do_mean pear={dm_pe:.3f} allDEG={dm_ad:.3f} | model-add pear={ma_pe:.3f} allDEG={ma_ad:.3f}", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
titles = {'singles': 'Singles-trained (MMD, no combos)', 'combo': 'Combo-trained (MMD + Joung combos, 50/50)'}
METS = ['pearson', 'allDEG']
for ax, name in zip(axes, ['singles', 'combo']):
    N, M, r_pair = res[name]; x = np.arange(2); w = 0.25
    for i, (k, col, lab) in enumerate(VARS):
        mu = [np.nanmean(M[k][j]) for j in range(2)]; se = [np.nanstd(M[k][j], ddof=1) / np.sqrt(np.isfinite(M[k][j]).sum()) for j in range(2)]
        ax.bar(x + (i - 1) * w, mu, w, yerr=se, color=col, label=lab, alpha=0.92, error_kw=dict(elinewidth=1.1, capsize=3, ecolor='#333'))
    ax.axhline(0, color='#444', lw=0.7); ax.set_xticks(x); ax.set_xticklabels(['pearson\n(whole-profile)', 'allDEG\n(DE genes)'])
    ax.set_title(f"{titles[name]}   (n={N} held-out combos)\npredicts epistasis: r = {r_pair:+.2f}", fontsize=12)
    ax.grid(axis='y', alpha=0.15)
axes[0].set_ylabel('drift-removed value (±1 SEM)'); axes[0].legend(fontsize=10, loc='upper left')
fig.suptitle('Combo training: do-op beats additive & predicts epistasis (same held-out Joung combos)', fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig('results/figures/epistasis_singles_vs_combo.png', dpi=150, bbox_inches='tight')
print('wrote results/figures/epistasis_singles_vs_combo.png', flush=True)

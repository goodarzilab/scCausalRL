"""Frontier eval WITH combo-seed CIs. Three recipes spanning the singles<->combo frontier, each at
3 split seeds {1,7,42} (leakage-safe: fine-tune arms warm-start from seed-matched singles sources):
  frozen_r32   : freeze all-but-head (rank32)          — the 'both' / singles-safe endpoint
  partial_dch  : freeze encoder+shift_encoder, unfreeze decoder+DAG+rank16 head — frontier interior
  fromscratch  : deg2_Jcombo, no warm-start/freeze      — combo-max endpoint
Per arm: SINGLES (disc/Pnk/allDEG@0.15 on that SEED's held-out TFs) + COMBOS (do_mean disc/Pnk/allDEG
+ r_pair on that checkpoint's stashed held-out combos). Aggregated per recipe = mean±SEM across seeds.
Run: python evaluation/eval_frontier.py --device cpu
"""
import os, sys, glob, re, argparse, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, COMBO
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = V + "/joung_singles_proc_v2b.h5ad"; EFF = V + "/effective_tf_genes.txt"
NCTRL = 4000; T15 = 0.15
RECIPES = {
    'frozen_r32 deg2 (both)': ['v2b_fthead_r32', 'v2b_frontier_frozen_r32_s1', 'v2b_frontier_frozen_r32_s7'],
    'frozen_r32 deg5 (both)': ['v2b_deg5_frozen_r32', 'v2b_deg5_frozen_r32_s1', 'v2b_deg5_frozen_r32_s7'],
    'frozen_r32 deg3_t25 (both)': ['v2b_mmd1_deg3_t25_frozen_r32', 'v2b_deg3_t25_frozen_r32_s1', 'v2b_deg3_t25_frozen_r32_s7'],
    'frozen_r32 deg8 (both)': ['v2b_deg8_frozen_r32', 'v2b_deg8_frozen_r32_s1', 'v2b_deg8_frozen_r32_s7'],
    'partial_dch (interior)': ['v2b_frontier_partial_dch_s42', 'v2b_frontier_partial_dch_s1', 'v2b_frontier_partial_dch_s7'],
    'fromscratch (combo-max)': ['v2b_mmd1_deg2_Jcombo_ho50', 'v2b_frontier_fromscratch_s1', 'v2b_frontier_fromscratch_s7'],
}


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def msem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) if len(v) > 1 else (float(np.mean(v)) if len(v) else np.nan, 0.0)


_JCACHE = {}
def joung_for_seed(seed):
    if seed in _JCACHE: return _JCACHE[seed]
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF, split_seed=seed, train_frac=0.9, val_frac=0.0)
    de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
    obs = np.stack([np.asarray(de[g], np.float32) for g in te])
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(NCTRL, len(ctrl)), replace=False)]])
    _JCACHE[seed] = (de, gn, gidx, te, drift, obs, xc, tr)
    return _JCACHE[seed]


# ESM-kNN copy baseline (singles): predict each held-out TF with the residual of its nearest TRAIN
# TF by ESM cosine (model-free). Same metric suite as the model, on this seed's held-out split.
import os as _os
_ESM = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')
def esm_knn_singles(seed):
    de, gn, gidx, te, drift, obs, xc, tr = joung_for_seed(seed)
    z = np.load(_ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}
    dim = len(next(iter(esm.values()))); En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
    nn = (En(np.stack([esm.get(g, np.zeros(dim)) for g in te])) @ En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])).T).argmax(1)
    return singles_suite(DEtr[nn], te, gidx, drift, obs)   # (disc, Pnk, allDEG)


# combo data (seed-independent; held-out set comes from each checkpoint's combo_test_keys)
def load_combo():
    C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
    Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4)
    cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
    return Xk, cvar, C.obs['TF_set'].astype(str).values, Xk.mean(0)
Xk, cvar, tfset, ref = load_combo()


def singles_suite(P, te, gidx, drift, obs):
    Pr, Or = P - drift, obs - drift
    Cc = zr(Pr) @ zr(Or).T; dgc = np.diag(Cc)[:, None]; T = len(P)
    disc = float(((dgc > Cc).sum(1) / (T - 1)).mean())
    pnk, a15 = [], []
    for i, g in enumerate(te):
        keep = np.ones(P.shape[1], bool); keep[gidx[g]] = False
        x, y = Pr[i, keep], Or[i, keep]; pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan)
        d = keep & (np.abs(Or[i]) > T15); a15.append(np.corrcoef(Pr[i, d], Or[i, d])[0, 1] if d.sum() > 2 and Pr[i, d].std() > 1e-9 else np.nan)
    return disc, float(np.nanmean(pnk)), float(np.nanmean(a15))


def combo_suite(model, gnm, p1, de, gidx, drift, xc):
    g2 = {g: i for i, g in enumerate(gnm)}; G = len(gnm); BETA = float(p1.get('pert_mutilate_beta', 1.0))
    _tk = p1.get('combo_test_keys'); held = {tuple(sorted(map(str, k))) for k in _tk} if _tk else None
    inter = [g for g in gnm if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
    driftm = drift[mask]
    _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0); zp = model.dag.scm_forward(eps, A)
    cr = model.decoder(zp).mean(0).cpu().numpy()
    sh = lambda gi: model.compute_shift(torch.tensor([gi])).sum(0)
    def solve(S, u):
        z, _, _ = model.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u); return model.decoder(z).mean(0).cpu().numpy() - cr
    rows = []
    for ts in dict.fromkeys(tfset):
        tfs = [t for t in parse_tfs(ts) if t in g2]
        if len(tfs) != 2: continue
        if held is not None and tuple(sorted(tfs)) not in held: continue
        mm = tfset == ts
        if mm.sum() < 20: continue
        of = Xk[mm].mean(0) - ref; ob = np.zeros(G, np.float32); ob[mask] = of[col]
        oa = np.zeros(G, np.float32)
        for g in tfs:
            if g in de: oa += np.asarray(de[g], np.float32)
        Sa, Sb = sh(g2[tfs[0]]), sh(g2[tfs[1]]); ua, ub = (BETA * Sa.abs()).clamp(max=1.0), (BETA * Sb.abs()).clamp(max=1.0)
        S = Sa + Sb
        si = model.compute_interaction(torch.tensor([g2[tfs[0]], g2[tfs[1]]])) if getattr(model, 'interaction_U', None) is not None else None
        if si is not None: S = S + si
        pa, pb = solve(Sa, ua), solve(Sb, ub); dm = solve(S, (ua + ub) / 2)
        rows.append(dict(obs=ob[mask], obsadd=oa[mask], dm=dm[mask], madd=(pa + pb)[mask]))
    if len(rows) < 2: return None
    O = np.stack([r['obs'] for r in rows]); OA = np.stack([r['obsadd'] for r in rows])
    DM = np.stack([r['dm'] for r in rows]); MA = np.stack([r['madd'] for r in rows])
    Od = O - O.mean(0); N = len(rows)

    def _metrics(Praw):                       # disc/Pnk/allDEG for a prediction array vs O (across-pair drift-removed)
        Pd = Praw - Praw.mean(0)
        Pz, Oz = zr(Pd), zr(Od); Cm = Pz @ Oz.T; dgc = np.diag(Cm)
        disc = float(np.mean([(dgc[i] > np.delete(Cm[i], i)).mean() for i in range(N)]))
        pnk = float(np.nanmean([np.corrcoef(Pd[i], Od[i])[0, 1] if Pd[i].std() > 1e-9 and Od[i].std() > 1e-9 else np.nan for i in range(N)]))
        a = []
        for i in range(N):
            d = np.abs(Od[i]) > T15; a.append(np.corrcoef(Pd[i, d], Od[i, d])[0, 1] if d.sum() > 2 and Pd[i, d].std() > 1e-9 else np.nan)
        return disc, pnk, float(np.nanmean(a))
    d_dm, p_dm, a_dm = _metrics(DM)
    d_oa, p_oa, a_oa = _metrics(OA)
    d_ma, p_ma, a_ma = _metrics(MA)
    dmask = np.abs(O - driftm[None, :]) > T15
    eps_obs = (O - OA) + driftm[None, :]; eps_mod = DM - MA
    ns = np.array([eps_obs[i][dmask[i]].mean() if dmask[i].sum() else 0.0 for i in range(N)])
    nm = np.array([eps_mod[i][dmask[i]].mean() if dmask[i].sum() else 0.0 for i in range(N)])
    r_pair = float(np.corrcoef(ns, nm)[0, 1]) if np.std(ns) > 1e-9 and np.std(nm) > 1e-9 else np.nan
    return dict(n=N, disc=d_dm, Pnk=p_dm, a15=a_dm, r_pair=r_pair,
                oa_disc=d_oa, oa_Pnk=p_oa, oa_a15=a_oa, ma_disc=d_ma, ma_Pnk=p_ma, ma_a15=a_ma)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu'); ap.parse_args()
    torch.set_grad_enabled(False)
    print(f"{'arm':40}{'seed':>5}{'s_disc':>8}{'s_Pnk':>8}{'s_aDEG':>8} | {'cN':>3}{'c_disc':>8}{'c_Pnk':>8}{'c_aDEG':>8}{'r_pair':>9}", flush=True)
    KEYS = ('s_Pnk', 's_a15', 'c_Pnk', 'c_a15', 'r_pair', 'oa_Pnk', 'oa_a15', 'ma_Pnk', 'ma_a15')
    agg = {r: {k: [] for k in KEYS} for r in RECIPES}
    print(f"{'arm':40}{'seed':>5}{'s_Pnk':>8}{'s_aDEG':>8} | {'cN':>3}{'doMean_aDEG':>12}{'modAdd_aDEG':>12}{'obsAdd_aDEG':>12}{'r_pair':>9}", flush=True)
    for recipe, arms in RECIPES.items():
        for tag in arms:
            cks = glob.glob(f'results/grid/{tag}/checkpoints/phase1_epoch*.pt')
            if not cks: print(f"{tag:40}{'--':>5}  (no ckpt yet)", flush=True); continue
            ck = max(cks, key=lambda p: int(re.search(r'epoch(\d+)', p).group(1)))
            try:
                model, gnm, c = load_shift_model(ck, 'cpu'); cfg = c.get('config', {}) or {}; p1 = cfg.get('phase1', {}) or {}
                seed = int(cfg.get('data', {}).get('split_seed', 42))
                de, gn, gidx, te, drift, obs, xc, _tr = joung_for_seed(seed)
                um = p1.get('pert_mutilate_u_mode', 'shift'); bt = float(p1.get('pert_mutilate_beta', 1.0))
                _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0); zp = model.dag.scm_forward(eps, A); cr = model.decoder(zp).mean(0).cpu().numpy()
                g2 = {g: i for i, g in enumerate(gnm)}
                P = np.stack([(model.decoder(model.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode=um, shift_beta=bt)[0]).mean(0).cpu().numpy() - cr) for g in te])
                sd, sp_, sa = singles_suite(P, te, gidx, drift, obs)
                cm = combo_suite(model, gnm, p1, de, gidx, drift, xc)
                agg[recipe]['s_Pnk'].append(sp_); agg[recipe]['s_a15'].append(sa)
                if cm:
                    agg[recipe]['c_Pnk'].append(cm['Pnk']); agg[recipe]['c_a15'].append(cm['a15'])
                    agg[recipe]['r_pair'].append(cm['r_pair'])
                    agg[recipe]['oa_Pnk'].append(cm['oa_Pnk']); agg[recipe]['oa_a15'].append(cm['oa_a15'])
                    agg[recipe]['ma_Pnk'].append(cm['ma_Pnk']); agg[recipe]['ma_a15'].append(cm['ma_a15'])
                    cs = f"{cm['n']:>3}{cm['a15']:>12.3f}{cm['ma_a15']:>12.3f}{cm['oa_a15']:>12.3f}{cm['r_pair']:>+9.3f}"
                else:
                    cs = f"{'':>3}{'(no combos)':>45}"
                print(f"{tag:40}{seed:>5}{sp_:>8.3f}{sa:>8.3f} | {cs}", flush=True)
            except Exception as e:
                print(f"{tag:40}  FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
    # ESM-kNN singles baseline across the 3 seeds
    kp, ka = [], []
    for s in (42, 1, 7):
        try:
            _, p, a = esm_knn_singles(s); kp.append(p); ka.append(a)
        except Exception as e:
            print(f"ESM-kNN seed{s} FAIL: {e}", flush=True)
    ek_p, ek_a = msem(kp), msem(ka)
    print(f"\n{'='*118}\nPER-RECIPE (mean ± SEM across 3 seeds) — model do_mean vs model-additive vs obs-additive\n{'='*118}", flush=True)
    print(f"{'recipe':30}{'s_Pnk':>13}{'s_allDEG':>13} | {'doMean aDEG':>13}{'modAdd aDEG':>13}{'obsAdd aDEG':>13}{'r_pair':>14}", flush=True)
    print(f"{'BASELINE ESM-kNN (singles)':30}{ek_p[0]:>8.3f}±{ek_p[1]:<4.3f}{ek_a[0]:>8.3f}±{ek_a[1]:<4.3f} | {'(singles-only baseline)':>53}", flush=True)
    for r in RECIPES:
        sp_ = msem(agg[r]['s_Pnk']); sa = msem(agg[r]['s_a15'])
        ca = msem(agg[r]['c_a15']); ma = msem(agg[r]['ma_a15']); oa = msem(agg[r]['oa_a15']); rp = msem(agg[r]['r_pair'])
        print(f"{r:30}{sp_[0]:>8.3f}±{sp_[1]:<4.3f}{sa[0]:>8.3f}±{sa[1]:<4.3f} | {ca[0]:>8.3f}±{ca[1]:<4.3f}{ma[0]:>8.3f}±{ma[1]:<4.3f}{oa[0]:>8.3f}±{oa[1]:<4.3f}{rp[0]:>+9.3f}±{rp[1]:<4.3f}", flush=True)


if __name__ == '__main__':
    main()

"""MASTER v2b eval table: every v2b arm, SINGLES and COMBOS side by side, with SEM bars, at 4000
control cells (do_mean). Metrics (drift-removed, KO-excluded): disc_resid, Pnk, allDEG@0.15 (+ @0.10
for the DEG-threshold question). Combos use do_mean (S=Sa+Sb summed shift; u=per-single mean of
clip(beta|S_g|)) restricted to each arm's stashed held-out combos. Baselines: ESM-kNN copy (singles),
obs-additive & model-additive (combos). Dataset column J vs J+N from the checkpoint config.
SEM = std over eval units (49 held-out TFs for singles; held-out pairs for combos) / sqrt(n).
Run: python evaluation/eval_bigtable.py --device cpu [--limit N] [--only tagsubstr]
Writes results/figures/bigtable_v2b.csv
"""
import os, sys, glob, re, csv, argparse, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, COMBO
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = V + "/joung_singles_proc_v2b.h5ad"; EFF = V + "/effective_tf_genes.txt"
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')
NCTRL = 4000; T15, T10 = 0.15, 0.10


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) if len(v) > 1 else (float(np.mean(v)) if len(v) else np.nan, 0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--only', default='')
    a = ap.parse_args(); dev = a.device; torch.set_grad_enabled(False)

    # ---------- singles data ----------
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
    obs = np.stack([np.asarray(de[g], np.float32) for g in te])
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    ci = ctrl[rng.choice(len(ctrl), min(NCTRL, len(ctrl)), replace=False)]
    xc = torch.tensor(Xc[ci], device=dev)
    Or = obs - drift

    def singles_suite(P):                                   # P=[te,G] predicted Δ
        Pr = P - drift
        C = zr(Pr) @ zr(Or).T; dg = np.diag(C)[:, None]; T = len(P)
        disc_ps = ((dg > C).sum(1) / (T - 1)) if T > 1 else np.full(T, np.nan)
        pnk, a15, a10 = [], [], []
        for i, g in enumerate(te):
            keep = np.ones(P.shape[1], bool); keep[gidx[g]] = False
            x, y = Pr[i, keep], Or[i, keep]
            pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan)
            for thr, acc in ((T15, a15), (T10, a10)):
                d = keep & (np.abs(Or[i]) > thr); xd = Pr[i, d]
                acc.append(np.corrcoef(xd, Or[i, d])[0, 1] if d.sum() > 2 and xd.std() > 1e-9 else np.nan)
        return dict(disc=sem(disc_ps), Pnk=sem(pnk), a15=sem(a15), a10=sem(a10))

    # ---------- ESM-kNN singles baseline ----------
    z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}
    dim = len(next(iter(esm.values()))); En = lambda E: E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    DEtr = np.stack([np.asarray(de[g], np.float32) for g in tr])
    knn = (En(np.stack([esm.get(g, np.zeros(dim)) for g in te])) @ En(np.stack([esm.get(g, np.zeros(dim)) for g in tr])).T).argmax(1)
    pred_knn = DEtr[knn]

    # ---------- combo data ----------
    Cd = ad.read_h5ad(COMBO); Xk = Cd.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
    Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4)
    cvar = {g: i for i, g in enumerate(map(str, Cd.var_names))}
    inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
    tfset = Cd.obs['TF_set'].astype(str).values; ref = Xk.mean(0); driftm = drift[mask]
    G = len(gn)

    def combo_metrics(pred_mask, obs_mask):                # arrays [Npair, len(inter)] drift-removed upstream
        Od = obs_mask - obs_mask.mean(0); Pd = pred_mask - pred_mask.mean(0); N = len(Od)
        Pz, Oz = zr(Pd), zr(Od); Cm = Pz @ Oz.T; dgc = np.diag(Cm)
        disc = np.array([(dgc[i] > np.delete(Cm[i], i)).mean() for i in range(N)]) if N > 1 else np.full(N, np.nan)
        pnk = np.array([np.corrcoef(Pd[i], Od[i])[0, 1] if Pd[i].std() > 1e-9 and Od[i].std() > 1e-9 else np.nan for i in range(N)])
        a15 = []
        for i in range(N):
            d = np.abs(Od[i]) > T15
            a15.append(np.corrcoef(Pd[i, d], Od[i, d])[0, 1] if d.sum() > 2 and Pd[i, d].std() > 1e-9 else np.nan)
        return dict(disc=sem(disc), Pnk=sem(pnk), a15=sem(np.array(a15)))

    def combo_eval(model, gnm):
        g2 = {g: i for i, g in enumerate(gnm)}
        ck_p1 = combo_eval._p1
        BETA = float(ck_p1.get('pert_mutilate_beta', 1.0))
        _tk = ck_p1.get('combo_test_keys'); held = {tuple(sorted(map(str, k))) for k in _tk} if _tk else None
        _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0); z_prev = model.dag.scm_forward(eps, A)
        cr = model.decoder(z_prev).mean(0).cpu().numpy()

        def sh(gi): return model.compute_shift(torch.tensor([gi], device=dev)).sum(0)

        def solve(S, u):
            zp, _, _ = model.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u); return model.decoder(zp).mean(0).cpu().numpy() - cr
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
            si = model.compute_interaction(torch.tensor([g2[tfs[0]], g2[tfs[1]]], device=dev)) if getattr(model, 'interaction_U', None) is not None else None
            if si is not None: S = S + si
            pa, pb = solve(Sa, ua), solve(Sb, ub)
            dm = solve(S, (ua + ub) / 2)
            rows.append(dict(obs=ob[mask], obsadd=oa[mask], dm=dm[mask], madd=(pa + pb)[mask]))
        if len(rows) < 2: return None
        O = np.stack([r['obs'] for r in rows]); OA = np.stack([r['obsadd'] for r in rows])
        DM = np.stack([r['dm'] for r in rows]); MA = np.stack([r['madd'] for r in rows])
        m = combo_metrics(DM, O); m['n'] = len(rows)
        # epistasis r_pair (net-signed over DE genes, drift-corrected)
        Od = O - driftm[None, :]; dmask = np.abs(Od) > T15
        eps_obs = (O - OA) + driftm[None, :]; eps_mod = DM - MA
        ns = np.array([eps_obs[i][dmask[i]].mean() if dmask[i].sum() else 0.0 for i in range(len(rows))])
        nm = np.array([eps_mod[i][dmask[i]].mean() if dmask[i].sum() else 0.0 for i in range(len(rows))])
        m['r_pair'] = float(np.corrcoef(ns, nm)[0, 1]) if np.std(ns) > 1e-9 and np.std(nm) > 1e-9 else np.nan
        # store additive baselines (computed once vs this arm's held-out set)
        m['_obsadd'] = combo_metrics(OA, O); m['_modeladd'] = combo_metrics(MA, O)
        return m

    # ---------- arms ----------
    arms = sorted(glob.glob('results/grid/v2b_*/'))
    if a.only: arms = [x for x in arms if a.only in x]
    if a.limit: arms = arms[:a.limit]
    fields = ['arm', 'dataset', 'combo_trained', 's_disc', 's_Pnk', 's_a15', 's_a10',
              'c_n', 'c_disc', 'c_Pnk', 'c_a15', 'c_r_pair',
              'c_obsadd_Pnk', 'c_obsadd_a15', 'c_modeladd_Pnk', 'c_modeladd_a15']
    print(f"{'arm':40}{'data':5}{'cmb':4}{'s_disc':>8}{'s_Pnk':>13}{'s_a15':>13}{'s_a10':>13} | {'cN':>3}{'c_disc':>8}{'c_Pnk':>13}{'c_a15':>13}{'c_rpair':>9}", flush=True)
    def fmt(t): return f"{t[0]:.3f}±{t[1]:.3f}" if isinstance(t, tuple) else f"{t:.3f}"
    out_rows = []
    # baselines first
    bs = singles_suite(pred_knn)
    print(f"{'BASELINE ESM-kNN copy (singles)':40}{'J':5}{'-':4}{bs['disc'][0]:>8.3f}{fmt(bs['Pnk']):>13}{fmt(bs['a15']):>13}{fmt(bs['a10']):>13} |", flush=True)
    out_rows.append(dict(arm='BASELINE_ESM-kNN_copy', dataset='J', combo_trained='-',
                         s_disc=bs['disc'][0], s_Pnk=bs['Pnk'][0], s_a15=bs['a15'][0], s_a10=bs['a10'][0],
                         c_n='', c_disc='', c_Pnk='', c_a15='', c_r_pair=''))
    _printed_addbases = False
    for d in arms:
        tag = d.rstrip('/').split('/')[-1]
        cks = glob.glob(f'{d}checkpoints/phase1_epoch*.pt')
        if not cks: print(f"{tag:40} (no epoch ckpt)", flush=True); continue
        ck = max(cks, key=lambda p: int(re.search(r'epoch(\d+)', p).group(1)))
        try:
            model, gnm, c = load_shift_model(ck, dev)
            cfg = c.get('config', {}) or {}; p1 = cfg.get('phase1', {}) or {}
            ds = 'J+N' if (cfg.get('data', {}).get('dataset') == 'joung_norman' or 'JN' in tag) else 'J'
            cmb = 'Y' if p1.get('combo_test_keys') else ('H' if 'fthead' in tag or 'ixhead' in tag or 'combo' in tag else 'n')
            _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0); zp = model.dag.scm_forward(eps, A)
            um = p1.get('pert_mutilate_u_mode', 'shift'); bt = float(p1.get('pert_mutilate_beta', 1.0)); cr = model.decoder(zp).mean(0).cpu().numpy()
            P = np.stack([(model.decoder(model.intervene_shift_mutilate(eps, [gnm.index(g) if g in gnm else gidx[g]], A, zp, u_mode=um, shift_beta=bt)[0]).mean(0).cpu().numpy() - cr) for g in te])
            ss = singles_suite(P)
            combo_eval._p1 = p1
            cm = combo_eval(model, gnm)
            row = dict(arm=tag, dataset=ds, combo_trained=cmb,
                       s_disc=ss['disc'][0], s_Pnk=ss['Pnk'][0], s_a15=ss['a15'][0], s_a10=ss['a10'][0],
                       c_n=(cm['n'] if cm else ''), c_disc=(cm['disc'][0] if cm else ''),
                       c_Pnk=(cm['Pnk'][0] if cm else ''), c_a15=(cm['a15'][0] if cm else ''),
                       c_r_pair=(cm['r_pair'] if cm else ''),
                       c_obsadd_Pnk=(cm['_obsadd']['Pnk'][0] if cm else ''), c_obsadd_a15=(cm['_obsadd']['a15'][0] if cm else ''),
                       c_modeladd_Pnk=(cm['_modeladd']['Pnk'][0] if cm else ''), c_modeladd_a15=(cm['_modeladd']['a15'][0] if cm else ''))
            out_rows.append(row)
            cstr = (f"{cm['n']:>3}{cm['disc'][0]:>8.3f}{fmt(cm['Pnk']):>13}{fmt(cm['a15']):>13}{cm['r_pair']:>+9.3f}") if cm else f"{'':>3}{'(no combos)':>41}"
            print(f"{tag:40}{ds:5}{cmb:4}{ss['disc'][0]:>8.3f}{fmt(ss['Pnk']):>13}{fmt(ss['a15']):>13}{fmt(ss['a10']):>13} | {cstr}", flush=True)
            if cm and not _printed_addbases:
                oa, ma = cm['_obsadd'], cm['_modeladd']
                print(f"{'  (combo baselines, this held-out set)':40}{'':13}{'obs-add:':>34} {fmt(oa['Pnk']):>13}{fmt(oa['a15']):>13}", flush=True)
                print(f"{'':40}{'':13}{'model-add:':>34} {fmt(ma['Pnk']):>13}{fmt(ma['a15']):>13}", flush=True)
                _printed_addbases = True
        except Exception as e:
            print(f"{tag:40} FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
    with open('results/figures/bigtable_v2b.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in out_rows: w.writerow(r)
    print('\nwrote results/figures/bigtable_v2b.csv', flush=True)


if __name__ == '__main__':
    main()

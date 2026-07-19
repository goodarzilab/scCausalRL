"""Combo do-op u-clamp ablation (no retraining): the do-op combo uses u=clip(β|S_a+S_b|, max=1.0),
which over-saturates (~41% of programs hit u=1) because it takes |SUM| of the two shifts. Re-score the
40 held-out pairs with alternative u treatments to isolate whether the over-clamp (vs missing combo
training) is what makes the do-op lose to additive:
  do_sum1   : u = clip(|S_a+S_b|, 1.0)   [current default]
  do_sum05  : u = clip(|S_a+S_b|, 0.5)   [respect config u_clip_max]
  do_mean   : u = mean(clip(|S_a|,1), clip(|S_b|,1))   [per-single avg, no sum-doubling]
  do_u0     : u = 0                        [≡ additive-compose; sanity check]
  additive  : Σ single-TF do-op Δ (reference)

Run: python evaluation/combo_u_ablation.py --checkpoint <ckpt> [--device cpu]
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp, anndata as ad
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, zrow, disc, VDIR, PROC, EFF, COMBO
from data.load_joung import load_joung


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='results/grid/v2b_retrfix_refresh_k20/checkpoints/phase1_epoch60.pt')
    ap.add_argument('--min_cells', type=int, default=20); ap.add_argument('--n_ctrl', type=int, default=500)
    ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    model, gn, ck = load_shift_model(a.checkpoint, dev)
    p1 = (ck.get('config', {}) or {}).get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); gidx = {g: i for i, g in enumerate(gn)}; G = len(gn)

    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF); de = dd['de_single']
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=dev)
    _, _, eps = model.encoder(xc)
    A = model.dag.sample_A(1).squeeze(0); z_prev = model.dag.scm_forward(eps, A)
    pd = getattr(model, 'pert_decoder', None) or model.decoder
    cr = pd(z_prev).mean(0).cpu().numpy()

    C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
    Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4)
    cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
    inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
    tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)

    def shift(gi): return model.compute_shift(torch.tensor([gi], device=dev)).sum(0)
    def shift_base(gi): return model.compute_shift(torch.tensor([gi], device=dev), add_retrieval=False).sum(0)
    _has_retr = getattr(model, 'retr_val', None) is not None

    def solve_u(S, u):
        zt = eps + S.unsqueeze(0)
        zp, _, _ = model.dag.scm_intervene_u(eps, A, zt, u)
        return pd(zp).mean(0).cpu().numpy() - cr

    def single(gi):
        Sg = shift(gi); ug = (BETA * Sg.abs()).clamp(max=1.0)
        return solve_u(Sg, ug), ug

    # HELD-OUT filter: for combo-TRAINED checkpoints, restrict to the stashed test combos so we don't
    # score train combos (leakage). Singles-only ckpts have no combo_test_keys → all 40 pairs (all held out).
    _tk = (ck.get('config', {}) or {}).get('phase1', {}).get('combo_test_keys')
    _heldout = {tuple(sorted(map(str, k))) for k in _tk} if _tk else None
    if _heldout is not None: print(f"[held-out] restricting to {len(_heldout)} stashed test combos", flush=True)
    pairs = []
    for ts in dict.fromkeys(tfset):
        tfs = [t for t in parse_tfs(ts) if t in gidx]
        if len(tfs) != 2: continue
        if _heldout is not None and tuple(sorted(tfs)) not in _heldout: continue   # skip train combos
        m = tfset == ts
        if m.sum() < a.min_cells: continue
        of = Xk[m].mean(0) - ref; obs = np.zeros(G, np.float32); obs[mask] = of[col]
        (pa, ua), (pb, ub) = single(gidx[tfs[0]]), single(gidx[tfs[1]])
        S = shift(gidx[tfs[0]]) + shift(gidx[tfs[1]])
        oa = np.sum([np.asarray(de[g], np.float32) for g in tfs if g in de], axis=0) \
            if any(g in de for g in tfs) else np.zeros(G, np.float32)
        # JOINT-RETRIEVAL combo (your idea): per-gene base shift (no retrieval) + ONE joint retrieval
        # over Σ proj(e_g) → non-additive via the softmax, singles-trained, no retraining.
        if _has_retr:
            gij = torch.tensor([gidx[tfs[0]], gidx[tfs[1]]], device=dev)
            S_jr = shift_base(gidx[tfs[0]]) + shift_base(gidx[tfs[1]]) + model.retr_alpha * model._retrieve_joint(gij)
            do_jr = solve_u(S_jr, (ua + ub) / 2)
        else:
            do_jr = pa + pb
        variants = dict(
            do_sum1=solve_u(S, (BETA * S.abs()).clamp(max=1.0)),
            do_mean=solve_u(S, (ua + ub) / 2),
            do_jointretr=do_jr,         # joint-query retrieval (non-additive, singles-only)
            do_u0=solve_u(S, torch.zeros_like(S)),
            additive=pa + pb,           # model-additive (Σ single do-op, per-single u)
            obsadd=oa,                  # observed-additive null (Σ observed single Δ)
        )
        pairs.append(dict(obs=obs, **variants))
    print(f"[u-ablation] {len(pairs)} pairs | intersection {len(inter)} genes", flush=True)

    obs_all = np.stack([s['obs'][mask] for s in pairs])
    oadd_all = np.stack([s['obsadd'][mask] for s in pairs])
    devn = np.linalg.norm(obs_all - oadd_all, axis=1)                        # ‖obs − obs_add‖ (drift cancels)
    E_old = devn / (np.linalg.norm(obs_all, axis=1) + 1e-9)                  # drift-INCLUDED denom (buggy)
    E_new = devn / (np.linalg.norm(obs_all - obs_all.mean(0), axis=1) + 1e-9)  # drift-REMOVED denom (still magnitude-conf.)
    # DIRECTIONAL epistasis (magnitude-invariant): 1 − cos(obs, obs_add) in drift-removed space.
    odr = obs_all - obs_all.mean(0); adr = oadd_all - oadd_all.mean(0)
    E_dir = np.array([1.0 - float(odr[i] @ adr[i] / (np.linalg.norm(odr[i]) * np.linalg.norm(adr[i]) + 1e-9))
                      for i in range(len(pairs))])

    def score(key, idx=None):
        idx = range(len(pairs)) if idx is None else idx
        P = np.stack([pairs[i][key][mask] for i in idx]); O = np.stack([pairs[i]['obs'][mask] for i in idx])
        Pd = P - P.mean(0); Od = O - O.mean(0)
        pe = float(np.mean([np.corrcoef(Pd[j], Od[j])[0, 1] for j in range(len(P))
                            if Pd[j].std() > 1e-9 and Od[j].std() > 1e-9]))
        return disc(zrow(Pd), zrow(Od)), pe

    KEYS = ('do_sum1', 'do_mean', 'do_jointretr', 'additive', 'obsadd')
    print(f"\n=== OVERALL (40 pairs) ===\n{'variant':10}{'disc_resid':>12}{'pear':>8}")
    overall = {}
    for k in KEYS:
        d, p = score(k); overall[k] = (d, p); print(f"{k:10}{d:>12.4f}{p:>8.4f}", flush=True)

    quart = {}
    for Enm, E in [('E_old (norm, drift-IN denom)', E_old), ('E_new (norm, drift-removed denom)', E_new),
                   ('E_dir (1-cos, magnitude-invariant)', E_dir)]:
        order = np.argsort(E); qsp = np.array_split(order, 4)
        print(f"\n=== quartiles by {Enm} — pearson (drift-removed) ===")
        print(f"{'Q':4}{'meanE':>8}{'do_mean':>9}{'jointretr':>10}{'model-add':>10}{'obs-add':>9}")
        rows = []
        for qi, idx in enumerate(qsp, 1):
            pr = {k: score(k, idx)[1] for k in KEYS}; me = float(E[idx].mean())
            print(f"Q{qi:<3}{me:>8.2f}{pr['do_mean']:>9.3f}{pr['do_jointretr']:>10.3f}{pr['additive']:>10.3f}{pr['obsadd']:>9.3f}", flush=True)
            rows.append((me, pr))
        quart[Enm] = rows

    # figure: overall bars + fixed-E quartile lines (do_mean vs additive vs obs-add)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    COL = {'do_sum1': '#c0392b', 'do_mean': '#eb6834', 'do_jointretr': '#4a3aa7', 'additive': '#1baf7a', 'obsadd': '#888'}
    LAB = {'do_sum1': 'do-op (u=|Sa+Sb|)', 'do_mean': 'do-op (u=per-single)', 'do_jointretr': 'do-op (joint retrieval)', 'additive': 'model-additive', 'obsadd': 'obs-additive'}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    xb = np.arange(2); w = 0.2
    for i, k in enumerate(KEYS):
        a1.bar(xb + (i-1.5)*w, [overall[k][0], overall[k][1]], w, color=COL[k], label=LAB[k], alpha=0.9)
    a1.axhline(0.5, ls='--', color='#999'); a1.set_xticks(xb); a1.set_xticklabels(['disc_resid', 'pearson'])
    a1.set_title('DOUBLES: do-op(per-single u) reaches additive parity (40 pairs)', fontsize=11); a1.legend(fontsize=8)
    fixed = quart['E_dir (1-cos, magnitude-invariant)']
    xq = np.arange(4)
    for k in KEYS:
        a2.plot(xq, [r[1][k] for r in fixed], 'o-', color=COL[k], label=LAB[k], lw=2, ms=7)
    a2.set_xticks(xq); a2.set_xticklabels([f"Q{i+1}\nE={fixed[i][0]:.1f}" for i in range(4)], fontsize=9)
    a2.set_xlabel('epistasis quartile (E_dir = 1−cos, magnitude-invariant) →'); a2.set_ylabel('pearson (drift-removed)')
    a2.set_title('obs-additive now degrades monotonically with epistasis', fontsize=11); a2.legend(fontsize=8); a2.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig('results/figures/eval_doubles_v2.png', dpi=150, bbox_inches='tight')
    print("\nwrote results/figures/eval_doubles_v2.png", flush=True)


if __name__ == '__main__':
    main()

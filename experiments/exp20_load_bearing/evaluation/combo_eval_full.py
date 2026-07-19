"""Full combo (doubles) eval on the BEST prior-off model (masked_go+MMD), HELD-OUT pairs.
Fixes / adds vs eval_doubles_v2:
  - ERROR BARS (±1 SEM across the N held-out pairs) on the overall metric bars
  - allDEG metric added (drift-removed DE-gene Pearson per pair, KO/off-target)
  - do_mean = do-op with u = per-single mean of clip(beta|S_g|)  (shift S = Sa+Sb summed) — the recommended composer
  - TRADITIONAL log-additive epistasis  eps = Delta_AB - (Delta_A + Delta_B)  [= Y_AB-(Y_A+Y_B-Y0)],
    drift-corrected (Joung TF-OE share a differentiation drift that raw-additive double-counts):
      * per-gene eps distribution (synergy>0 / antagonism<0)
      * per-pair signed net epistasis classification
      * per-pair |eps| quartiles -> prediction Pearson (do_mean vs model-add vs obs-add), with SEM
Run: python evaluation/combo_eval_full.py [--checkpoint <ckpt>] [--device cpu]
"""
import os, sys, argparse, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, zrow, VDIR, PROC, EFF, COMBO
from data.load_joung import load_joung

BEST = 'results/grid/v2b_masked_tgt_delta_plus_mmd/checkpoints/phase1_epoch60.pt'
DEGT = 0.15                                                          # drift-removed |Δ| threshold for a "DE gene"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=BEST); ap.add_argument('--device', default='cpu')
    ap.add_argument('--n_ctrl', type=int, default=800); ap.add_argument('--min_cells', type=int, default=20)
    ap.add_argument('--tag', default='')                                                          # output-filename suffix
    a = ap.parse_args(); dev = torch.device(a.device); torch.set_grad_enabled(False)
    SUF = (f"_{a.tag}" if a.tag else "")
    model, gn, ck = load_shift_model(a.checkpoint, dev)
    p1 = (ck.get('config', {}) or {}).get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); gidx = {g: i for i, g in enumerate(gn)}; G = len(gn)

    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF); de = dd['de_single']
    trS = [g for g in dd['pert_split']['train'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in trS]).mean(0)                      # singles drift [G]
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=dev)
    _, _, eps = model.encoder(xc)
    A = model.dag.sample_A(1).squeeze(0); z_prev = model.dag.scm_forward(eps, A)
    cr = model.decoder(z_prev).mean(0).cpu().numpy()

    C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
    Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4)
    cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
    inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter])
    tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)
    driftm = drift[mask]                                                                          # drift on intersection

    def shift(gi): return model.compute_shift(torch.tensor([gi], device=dev)).sum(0)

    def solve_u(S, u):
        zp, _, _ = model.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u)
        return model.decoder(zp).mean(0).cpu().numpy() - cr

    def single(gi):
        Sg = shift(gi); ug = (BETA * Sg.abs()).clamp(max=1.0)
        return solve_u(Sg, ug), ug

    # HELD-OUT filter: for combo-TRAINED checkpoints restrict to the stashed test combos (no leakage);
    # singles-trained ckpts have no combo_test_keys -> all pairs are held out.
    _tk = p1.get('combo_test_keys')
    _heldout = {tuple(sorted(map(str, k))) for k in _tk} if _tk else None
    if _heldout is not None: print(f"[held-out] combo-trained ckpt: restricting to {len(_heldout)} stashed test combos", flush=True)
    rows = []
    for ts in dict.fromkeys(tfset):
        tfs = [t for t in parse_tfs(ts) if t in gidx]
        if len(tfs) != 2: continue
        if _heldout is not None and tuple(sorted(tfs)) not in _heldout: continue
        m = tfset == ts
        if m.sum() < a.min_cells: continue
        of = Xk[m].mean(0) - ref; obs = np.zeros(G, np.float32); obs[mask] = of[col]
        (pa, ua), (pb, ub) = single(gidx[tfs[0]]), single(gidx[tfs[1]])
        S = shift(gidx[tfs[0]]) + shift(gidx[tfs[1]])
        # Option-C interaction head (frozen-singles+head arm): the pair-specific latent shift S_int
        # is added PRE-SOLVE so the combo do-op reflects the trained head. Zero/None for models
        # without a head, so this is a no-op for every other arm (do_mean unchanged).
        if getattr(model, 'interaction_U', None) is not None:
            S_int = model.compute_interaction(torch.tensor([gidx[tfs[0]], gidx[tfs[1]]], device=dev))
            if S_int is not None:
                S = S + S_int
        oa = np.zeros(G, np.float32)
        for g in tfs:
            if g in de: oa = oa + np.asarray(de[g], np.float32)                                   # obs-additive Δ_A+Δ_B
        rows.append(dict(
            obs=obs[mask], obsadd=oa[mask],
            do_sum1=solve_u(S, (BETA * S.abs()).clamp(max=1.0))[mask],
            do_mean=solve_u(S, (ua + ub) / 2)[mask],
            modeladd=(pa + pb)[mask],
        ))
    N = len(rows); print(f"[combo] {N} held-out pairs | {len(inter)} intersection genes | model={os.path.basename(os.path.dirname(os.path.dirname(a.checkpoint)))}", flush=True)

    O = np.stack([r['obs'] for r in rows]); OAdd = np.stack([r['obsadd'] for r in rows])
    VARIANTS = {'do_sum1': 'do_sum1', 'do_mean': 'do_mean', 'modeladd': 'modeladd', 'obsadd': 'obsadd'}
    P = {k: (OAdd if k == 'obsadd' else np.stack([r[k] for r in rows])) for k in VARIANTS}

    # ---- per-pair metrics (drift-removed via across-pair mean; consistent with eval_doubles) ----
    Od = O - O.mean(0)
    def per_pair(Praw):
        Pd = Praw - Praw.mean(0)
        Pz, Oz = zrow(Pd), zrow(Od); Cm = Pz @ Oz.T; dgc = np.diag(Cm)
        disc = np.array([(dgc[i] > np.delete(Cm[i], i)).mean() for i in range(N)])
        pear = np.array([np.corrcoef(Pd[i], Od[i])[0, 1] if Pd[i].std() > 1e-9 and Od[i].std() > 1e-9 else np.nan for i in range(N)])
        adeg = []
        for i in range(N):
            deg = np.abs(Od[i]) > DEGT
            adeg.append(np.corrcoef(Pd[i, deg], Od[i, deg])[0, 1] if deg.sum() > 2 and Pd[i, deg].std() > 1e-9 else np.nan)
        return dict(disc=disc, pearson=pear, allDEG=np.array(adeg))
    M = {k: per_pair(P[k]) for k in VARIANTS}
    METS = ['disc', 'pearson', 'allDEG']
    print(f"{'variant':10}" + "".join(f"{m:>22}" for m in METS))
    for k in VARIANTS:
        print(f"{k:10}" + "".join(f"{np.nanmean(M[k][m]):>10.3f}±{np.nanstd(M[k][m], ddof=1)/np.sqrt(np.isfinite(M[k][m]).sum()):<11.3f}" for m in METS), flush=True)

    # ---- TRADITIONAL log-additive epistasis (drift-corrected) ----
    # raw additive double-counts drift on Joung; correct: eps_resid = resid_AB - (resid_A+resid_B),
    # resid_X = Δ_X - drift  ->  eps_resid = (O - OAdd) + drift  (drift added back once).
    EPS = (O - OAdd) + driftm[None, :]                                                            # [N, Gi] per-gene OBSERVED epistasis
    EPS_model = P['do_mean'] - P['modeladd']                                                       # [N, Gi] MODEL's own epistasis (do-op u-mutilation; drift cancels)
    Oresid = O - driftm[None, :]
    degmask_all = np.abs(Oresid) > DEGT
    # per-pair magnitude as MEAN |eps| per DE gene (same units as the per-gene histogram, ~0.1-0.5)
    eps_permag = np.array([np.abs(EPS[i][degmask_all[i]]).mean() if degmask_all[i].sum() > 0 else 0.0 for i in range(N)])
    net_signed = np.array([EPS[i][degmask_all[i]].mean() if degmask_all[i].sum() > 0 else 0.0 for i in range(N)])
    net_model = np.array([EPS_model[i][degmask_all[i]].mean() if degmask_all[i].sum() > 0 else 0.0 for i in range(N)])
    eps_de = EPS[degmask_all]                                                                     # per-(pair,DEgene) observed eps
    # can the model predict epistasis? per-pair and per-(pair,gene) correlation obs-vs-model
    r_pair = np.corrcoef(net_signed, net_model)[0, 1]
    og = np.concatenate([EPS[i][degmask_all[i]] for i in range(N)]); mg = np.concatenate([EPS_model[i][degmask_all[i]] for i in range(N)])
    r_gene = np.corrcoef(og, mg)[0, 1]
    print(f"\n[epistasis] mean|eps|/DEgene/pair={eps_permag.mean():.3f}  net-signed: {int((net_signed>0).sum())} synergistic / {int((net_signed<0).sum())} antagonistic"
          f"  | model-predicts-epistasis r_pair={r_pair:+.3f} r_gene={r_gene:+.3f}  (‖ε_model‖/‖ε_obs‖={np.linalg.norm(EPS_model)/np.linalg.norm(EPS):.2f})", flush=True)

    # ===================== FIGURE 1: doubles bars with error bars =====================
    COL = {'do_sum1': '#c0392b', 'do_mean': '#eb6834', 'modeladd': '#1f9e89', 'obsadd': '#8a8a8a'}
    LAB = {'do_sum1': 'do-op (u=|Sa+Sb|)', 'do_mean': 'do-op (u=per-single mean) ★', 'modeladd': 'model-additive', 'obsadd': 'obs-additive'}
    fig, ax = plt.subplots(figsize=(11, 6)); x = np.arange(len(METS)); w = 0.2
    for i, k in enumerate(VARIANTS):
        means = [np.nanmean(M[k][m]) for m in METS]
        sems = [np.nanstd(M[k][m], ddof=1) / np.sqrt(np.isfinite(M[k][m]).sum()) for m in METS]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=sems, color=COL[k], label=LAB[k], alpha=0.92,
               error_kw=dict(elinewidth=1.1, capsize=3, ecolor='#333'))
    for xi, ch in {0: 0.5, 1: 0.0, 2: 0.0}.items():
        ax.plot([xi - 0.45, xi + 0.45], [ch, ch], '--', lw=1.2, color='#c0392b')
    ax.axhline(0, color='#444', lw=0.7); ax.set_xticks(x); ax.set_xticklabels(['disc_resid', 'pearson (whole-profile)', 'allDEG (DE genes)'])
    ax.set_ylabel(f'drift-removed value  (±1 SEM over {N} held-out pairs)'); ax.legend(fontsize=9)
    ax.set_title(f'DOUBLES (held-out, best prior-off model): do_mean ties additive on all metrics', fontsize=12)
    fig.tight_layout(); fig.savefig(f'results/figures/eval_doubles_full{SUF}.png', dpi=150, bbox_inches='tight')
    print('wrote results/figures/eval_doubles_full.png', flush=True)

    # ===================== FIGURE 2: traditional epistasis =====================
    fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(18, 5.4))
    # (1) per-gene eps distribution
    b1.hist(np.clip(eps_de, -3, 3), bins=60, color='#4a3aa7', alpha=0.85)
    b1.axvline(0, color='#444', lw=1); b1.set_xlabel('per-gene epistasis  ε = Δ_AB − (Δ_A+Δ_B)  (drift-corrected, log2)')
    b1.set_ylabel('# (pair, DE-gene)'); b1.set_title('Epistasis distribution\n← antagonism (ε<0) | synergy (ε>0) →', fontsize=11)
    frac_syn = float((eps_de > 0.25).mean()); frac_ant = float((eps_de < -0.25).mean())
    b1.text(0.02, 0.95, f'|ε|>0.25:  {frac_syn*100:.0f}% synergy\n           {frac_ant*100:.0f}% antagonism', transform=b1.transAxes, va='top', fontsize=9)
    # (2) mean|eps|-per-DE-gene quartiles (SAME units as the histogram) -> prediction pearson, with SEM
    order = np.argsort(eps_permag); qs = np.array_split(order, 4)
    xq = np.arange(4); series = [('do_mean', '#eb6834'), ('modeladd', '#1f9e89'), ('obsadd', '#8a8a8a')]
    for k, c in series:
        mu = [np.nanmean(M[k]['pearson'][idx]) for idx in qs]
        se = [np.nanstd(M[k]['pearson'][idx], ddof=1) / np.sqrt(np.isfinite(M[k]['pearson'][idx]).sum()) for idx in qs]
        b2.errorbar(xq, mu, yerr=se, marker='o', lw=2, ms=7, capsize=3, color=c, label=LAB[k])
    b2.set_xticks(xq); b2.set_xticklabels([f"Q{i+1}\nmean|ε|={eps_permag[qs[i]].mean():.2f}" for i in range(4)])
    b2.set_xlabel('epistasis magnitude quartile (mean |ε| per DE gene, log2) →'); b2.set_ylabel('pearson (drift-removed) ±SEM')
    b2.set_title('Prediction degrades as true epistasis grows', fontsize=11); b2.legend(fontsize=9); b2.grid(alpha=0.15)
    # (3) CAN THE MODEL PREDICT EPISTASIS? observed vs model per-pair net epistasis
    lim = max(np.abs(net_signed).max(), np.abs(net_model).max()) * 1.1
    b3.axhline(0, color='#bbb', lw=0.7); b3.axvline(0, color='#bbb', lw=0.7)
    b3.scatter(net_signed, net_model, s=45, c='#4a3aa7', alpha=0.8, edgecolor='w', lw=0.5)
    b3.set_xlim(-lim, lim); b3.set_ylim(-lim, lim)
    b3.set_xlabel('OBSERVED net epistasis  (mean ε over DE genes)'); b3.set_ylabel("MODEL's net epistasis  (do_mean − additive)")
    b3.set_title(f'Can the model predict epistasis?  r_pair={r_pair:+.2f}, r_gene={r_gene:+.2f}\n‖ε_model‖/‖ε_obs‖={np.linalg.norm(EPS_model)/np.linalg.norm(EPS):.2f}', fontsize=11)
    fig.tight_layout(); fig.savefig(f'results/figures/epistasis_traditional{SUF}.png', dpi=150, bbox_inches='tight')
    print('wrote results/figures/epistasis_traditional.png', flush=True)


if __name__ == '__main__':
    main()

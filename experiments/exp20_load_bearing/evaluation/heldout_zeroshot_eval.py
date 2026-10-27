"""Held-out zero-shot evaluation of the Joung TF-atlas world model.

Hold out TFs (the seed-42 test split); predict each held-out TF's overexpression profile
ZERO-SHOT (the shift encoder generalises from the TF's control-PCA fingerprint — the model
never saw it perturbed); score predicted vs observed. EVERYTHING is on
Δ = mean(perturbed) − mean(hESC control), never raw expression (raw is dominated by the
shared basal hESC transcriptome and is uninformative).

Two metric families, saved for reuse (results/heldout_eval/<tag>/):

A. GENE-SPACE TRIO (Δ over the HVG axis)
   - pearson_delta      : mean over TFs of corr(Δ_pred, Δ_obs)               (generous floor)
   - de_direction       : sign agreement on genes that move (|Δ_obs|>thr)    (robust companion)
   - discrimination_cos : retrieval specificity — is Δ_pred_i closest (cos) to its OWN Δ_obs_i
                          among all held-out TFs (fraction-beaten + top-1). The load-bearing
                          metric a generic-drift predictor cannot fake.

B. IDENTITY-SPACE FIDELITY (developmental fate, via the Cao mouse NNLS pipeline)
   Deconvolve predicted & observed Δ against the early mouse cell-type panel (leak-free: the
   shared-OE nuisance axes are fit on TRAINING TFs only) → fate vector β over cell types.
   - fate_top1   : predicted dominant cell type == observed dominant cell type (headline)
   - fate_betacos: cosine of full β_pred vs β_obs (credits partial fate matches)
   - fate_entropy_r : corr of commitment strength H(β_pred) vs H(β_obs) across TFs

Run (base env):
  python evaluation/heldout_zeroshot_eval.py \
    --checkpoint results/grid/exp17_joung_tfatlas/checkpoints/phase1_epoch80.pt --tag epoch80
"""
import os, sys, json, argparse
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
ANN = os.path.join(ROOT, 'analysis', 'mouse_dev_annotation')


def _solve(LHS, rhs):
    return torch.linalg.solve(LHS, rhs.T).T


def zscore_rows(M):
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def gene_space(pred, obs, deg_thr=0.1):
    """pred, obs: [T, G] Δ matrices (same TF order). Returns per-TF + summary.

    Metrics (all averaged over TFs):
      pearson_delta    : corr(Δ_pred, Δ_obs) over genes                 (generous floor)
      de_direction     : sign agreement on movers (|Δ_obs|>thr)         (robust companion)
      effect_spearman  : Spearman rank-corr of Δ over genes             (effect-size ranking)
      deg_auroc        : recover DEGs (|Δ_obs|>thr) using |Δ_pred| as score (DEG detection)
      discrimination_cos / retrieval_top1 : cosine retrieval specificity (load-bearing)
    """
    T = pred.shape[0]
    pear, dirm, spear, auroc = [], [], [], []
    for i in range(T):
        p, o = pred[i], obs[i]
        if p.std() > 1e-8 and o.std() > 1e-8:
            pear.append(float(np.corrcoef(p, o)[0, 1]))
            rs = spearmanr(p, o).correlation
            if np.isfinite(rs):
                spear.append(float(rs))
        sig = np.abs(o) > deg_thr
        if sig.sum() >= 5:
            dirm.append(float((np.sign(p[sig]) == np.sign(o[sig])).mean()))
        is_deg = np.abs(o) > deg_thr                 # DEG-detection AUROC: |Δ_pred| ranks DEGs
        nd = int(is_deg.sum())
        if 0 < nd < len(is_deg) and np.abs(p).std() > 1e-12:
            auroc.append(float(roc_auc_score(is_deg.astype(int), np.abs(p))))
    # discrimination: cosine retrieval pred_i vs all obs_j
    Pz, Oz = zscore_rows(pred), zscore_rows(obs)
    C = Pz @ Oz.T                                   # [T, T] cos(pred_i, obs_j)
    diag = np.diag(C)[:, None]
    beaten = ((diag > C).sum(1) - 0) / (T - 1)      # frac of others each pred beats on its own target
    top1 = (C.argmax(1) == np.arange(T)).mean()
    _m = lambda a: float(np.mean(a)) if len(a) else float('nan')
    return (dict(pearson_delta=_m(pear), de_direction=_m(dirm),
                 effect_spearman=_m(spear), deg_auroc=_m(auroc),
                 discrimination_cos=float(beaten.mean()), retrieval_top1=float(top1), n=T),
            pear, dirm, beaten)


def fate_space(pred, obs, train_obs, gene_names, n_pc=4):
    """Deconvolve held-out pred & obs Δ to developmental fate β (cosine vs early mouse
    cell-type signatures), removing shared-OE nuisance axes fit on TRAINING TFs (leak-free)."""
    sys.path.insert(0, ANN)
    from nnls_annotate import load_reference        # MGI orthology + early-type restriction
    Xdf = load_reference(os.path.join(ANN, 'mouse_ref_Main_cell_type'), early_only=True)
    gidx = {g: i for i, g in enumerate(gene_names)}
    shared = [g for g in Xdf.index if g in gidx]
    cols = np.array([gidx[g] for g in shared])
    X = Xdf.loc[shared].values.astype(np.float64)
    D = X - X.mean(1, keepdims=True)
    gv = D.var(1); keep = gv > np.quantile(gv, 0.5)
    Dz = ((D - D.mean(0)) / (D.std(0) + 1e-8))[keep]      # [Gk, C]
    cell_types = list(Xdf.columns)
    sub = cols[keep]

    # nuisance (mean + top-PC) fit on TRAINING TFs' observed Δ over the same genes
    Tr = train_obs[:, sub].astype(np.float64); mu = Tr.mean(0)
    Trc = Tr - mu
    U, S, Vt = np.linalg.svd(Trc - Trc.mean(0), full_matrices=False)
    Vk = Vt[:min(n_pc, len(S))]

    def beta(M):
        Y = M[:, sub].astype(np.float64) - mu
        Y = Y - (Y @ Vk.T) @ Vk                          # strip shared nuisance axes
        Yz = zscore_rows(Y)
        cos = Yz @ Dz                                    # [T, C]
        cos = np.clip(cos, 0, None)
        s = cos.sum(1, keepdims=True)
        return np.divide(cos, s, out=np.zeros_like(cos), where=s > 0)

    bp, bo = beta(pred), beta(obs)
    ent = lambda B: np.array([-(b[b > 0] * np.log(b[b > 0])).sum() if b.sum() > 0 else 0 for b in B])
    top1 = (bp.argmax(1) == bo.argmax(1))
    bcos = np.array([float(np.dot(zscore_rows(bp[i:i+1])[0], zscore_rows(bo[i:i+1])[0])) for i in range(len(bp))])
    Hp, Ho = ent(bp), ent(bo)
    ent_r = float(np.corrcoef(Hp, Ho)[0, 1]) if len(Hp) > 2 else np.nan
    return (dict(fate_top1=float(top1.mean()), fate_betacos=float(np.nanmean(bcos)),
                 fate_entropy_r=ent_r, n_celltypes=len(cell_types)),
            bp, bo, top1, bcos, [cell_types[i] for i in bp.argmax(1)], [cell_types[i] for i in bo.argmax(1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'heldout_eval', args.tag); os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from evaluation.run_shift_eval_norman import load_shift_model
    from data.load_joung import load_joung
    model, gene_names, ck = load_shift_model(args.checkpoint, dev)
    d = model.latent_dim; n_genes = len(gene_names)
    p1 = ck.get('config', {}).get('phase1', {})
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); U_MAX = float(p1.get('pert_mutilate_u_max', 0.5))
    U_MODE = p1.get('pert_mutilate_u_mode', 'shift')

    dd = load_joung(n_hvg=None)
    gidx = {g: i for i, g in enumerate(gene_names)}
    de = dd['de_single']; Xc = dd['adata'].X
    import scipy.sparse as sp
    Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    test = [g for g in dd['pert_split']['test'] if g in de and g in gidx]
    train = [g for g in dd['pert_split']['train'] if g in de]
    print(f"[eval] held-out test TFs: {len(test)} | train TFs: {len(train)}", flush=True)

    rng = np.random.default_rng(0)
    ctrl = np.array(dd['control_idx']); sel = rng.choice(len(ctrl), min(args.n_ctrl_cells, len(ctrl)), replace=False)
    x_ctrl = torch.tensor(Xc[ctrl[sel]], dtype=torch.float32, device=dev)
    with torch.no_grad():
        _, _, eps_ctrl = model.encoder(x_ctrl)
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        LHS = torch.eye(d, device=dev, dtype=A.dtype) - A.T
        z_prev = torch.clamp(_solve(LHS, eps_ctrl), -20, 20)
        pert_dec = getattr(model, "pert_decoder", None) or model.decoder  # sep-head aware (no-op otherwise)
        ctrl_recon = pert_dec(z_prev).mean(0).cpu().numpy()

    # Two zero-shot predictors per held-out TF (both decode to Δ vs ctrl_recon):
    #   pred      = shift-native MUTILATION (headline): solve Σshift, mutilate ε with set-dep u
    #   pred_add  = PURE ADDITIVE (comparison): decode z_prev + z_S directly, no mutilation
    pred = np.zeros((len(test), n_genes), np.float32)
    pred_add = np.zeros((len(test), n_genes), np.float32)
    obs = np.stack([np.asarray(de[g], np.float32) for g in test])
    for k, g in enumerate(test):
        gi = gidx[g]
        with torch.no_grad():
            S = model.compute_shift(torch.tensor([gi], device=dev)).sum(0)
            z_S = torch.clamp(_solve(LHS, S.unsqueeze(0)), -20, 20)
            z_target = z_prev + z_S
            pred_add[k] = pert_dec(z_target).mean(0).cpu().numpy() - ctrl_recon   # additive baseline (KEEP legacy)
            u = (BETA * S.abs()).clamp(max=1.0) if U_MODE == 'shift' else \
                model.get_loadings()[torch.tensor([gi], device=dev)].sum(0).clamp(0, U_MAX)
            u = model.refine_u(u, S)                                  # nonlinear gate (no-op unless u_gate_mlp)
            if getattr(model, 'pert_intervene_mode', 'legacy') == 'structural':
                eps_tilde = eps_ctrl + S.unsqueeze(0)                # Fix 1: S into exogenous, single solve
            else:
                eps_tilde = model.mix_eps(eps_ctrl, z_target, u.unsqueeze(0))
            z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
            pred[k] = pert_dec(z_post).mean(0).cpu().numpy() - ctrl_recon
        if k % 50 == 0: print(f"  {k}/{len(test)}", flush=True)
    train_obs = np.stack([np.asarray(de[g], np.float32) for g in train])

    # RESIDUAL metrics: remove the shared across-train-TF drift from both pred and obs, then
    # re-score. This is the metric that matters for specificity (feature_probe.py) and the
    # objective the residual-trained models optimize. Drift estimated on train (leak-free).
    drift = train_obs.mean(0, keepdims=True)

    def score_all(P):
        g, pe, dm, bt = gene_space(P, obs)
        gr, _, _, _ = gene_space(P - drift, obs - drift)
        gr = {f'{kk}_resid': v for kk, v in gr.items() if kk != 'n'}
        try:                                              # fate-space needs nnls_annotate (not in exp20 fork)
            f, bp_, bo_, ftop1_, bcos_, pf_, of_ = fate_space(P, obs, train_obs, gene_names)
        except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
            print(f"[warn] fate_space skipped ({type(e).__name__}: {e}); gene-space metrics still saved", flush=True)
            f = {}; bp_ = bo_ = None; ftop1_ = bcos_ = np.array([]); pf_ = of_ = []
        return g, gr, f, pe, bt, (bp_, bo_, ftop1_, bcos_, pf_, of_)

    gs, gs_res, fs, pear, beaten, (bp, bo, ftop1, bcos, pred_fate, obs_fate) = score_all(pred)
    gs_a, gs_res_a, fs_a, pear_a, beaten_a, (bp_a, bo_a, ftop1_a, bcos_a, pf_a, of_a) = score_all(pred_add)
    add = {f'add_{kk}': v for kk, v in {**gs_a, **gs_res_a, **fs_a}.items() if kk != 'n'}

    summary = {'tag': args.tag, 'checkpoint': args.checkpoint, 'epoch': ck.get('epoch'),
               'intervention': 'mutilate (headline keys) + additive (add_* keys)',
               **gs, **gs_res, **fs, **add}
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)

    import pandas as pd
    nT = len(test)                                        # fate outputs may be empty if fate_space skipped
    def _fill(a, val=np.nan):
        a = np.asarray(a)
        return a if a.shape[0] == nT else np.full(nT, val)
    ftop1, bcos, ftop1_a, bcos_a = _fill(ftop1), _fill(bcos), _fill(ftop1_a), _fill(bcos_a)
    pred_fate = pred_fate if len(pred_fate) == nT else [None] * nT
    obs_fate = obs_fate if len(obs_fate) == nT else [None] * nT
    per = pd.DataFrame({'TF': test, 'pearson_delta': [pear[i] if i < len(pear) else np.nan for i in range(len(test))],
                        'discrimination_beaten': beaten, 'retrieval_correct': (np.arange(len(test)) == (zscore_rows(pred) @ zscore_rows(obs).T).argmax(1)),
                        'fate_top1_correct': ftop1, 'fate_betacos': bcos,
                        'pred_fate': pred_fate, 'obs_fate': obs_fate,
                        'add_pearson_delta': [pear_a[i] if i < len(pear_a) else np.nan for i in range(len(test))],
                        'add_discrimination_beaten': beaten_a,
                        'add_retrieval_correct': (np.arange(len(test)) == (zscore_rows(pred_add) @ zscore_rows(obs).T).argmax(1)),
                        'add_fate_top1_correct': ftop1_a, 'add_fate_betacos': bcos_a})
    per.to_csv(f"{out}/per_tf.csv", index=False)
    np.savez(f"{out}/profiles.npz", test=np.array(test), pred=pred, pred_add=pred_add, obs=obs,
             beta_pred=bp, beta_obs=bo, beta_pred_add=bp_a)
    print("\n===== HELD-OUT ZERO-SHOT EVAL (" + args.tag + ") =====")
    print("  [mutilate = shift-native headline | add_* = pure additive comparison]")
    for k, v in summary.items():
        if isinstance(v, float): print(f"  {k:28s} {v:.4f}")
    print(f"saved → {out}/")


if __name__ == '__main__':
    main()

"""Does the exp17 (singles-only-trained) world model understand EPISTASIS?

Bin the held-out combos+triples by how *non-additive* (epistatic) they are, then in each
quartile compare the model's **mutilation do-op** (shift-encoder, NOT pure additive shift —
mutilation does better across the board) against the **additive_obs oracle** (Σ OBSERVED
single-TF Δ). If the model captures epistasis, its do-op should close the gap to / beat
additive_obs *more* in the high-epistasis quartiles (where additivity fails).

Epistasis score per set (data-only, so quartiles are identical across models):
    E = ||Δ_obs − Δ_additive_obs|| / (||Δ_obs|| + eps)        (fraction of the combo
        response NOT explained by summing observed singles; higher = more epistatic)

Per quartile Q1(additive)…Q4(epistatic) and per model we report mean per-set:
    do_pearson  (mutilation do-op vs obs), addobs_pearson (additive_obs vs obs),
    gap = do − addobs, do_deg_auroc, addobs_deg_auroc, n, mean_E.

Run:  CKPTS=... sbatch scripts/epistasis_quartile.sbatch   (loops the 5 epoch-60 models)
"""
import os, sys, json, argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from evaluation.combo_triple_eval import parse_tfs, COMBO

MODELS = ['v2_orig', 'v2_esm', 'v2_esm_resid', 'v2_esm_resid_con', 'v2_esm_con']


def per_set_metrics(P, O, deg_thr=0.1):
    """per-set pearson over genes + DEG-AUROC (recover |Δ_obs|>thr with |Δ_pred|)."""
    pear = float(np.corrcoef(P, O)[0, 1]) if P.std() > 1e-8 and O.std() > 1e-8 else np.nan
    is_deg = np.abs(O) > deg_thr
    auroc = (float(roc_auc_score(is_deg.astype(int), np.abs(P)))
             if 0 < int(is_deg.sum()) < len(is_deg) and np.abs(P).std() > 1e-12 else np.nan)
    return pear, auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='epistasis_q')
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--min_cells', type=int, default=20)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'epistasis_quartile', args.tag); os.makedirs(out, exist_ok=True)
    import torch
    dev = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from evaluation.run_shift_eval_norman import load_shift_model
    from evaluation.heldout_zeroshot_eval import _solve
    from data.load_joung import load_joung, _lognorm
    import anndata as ad, scipy.sparse as sp, pandas as pd
    torch.set_grad_enabled(False)

    dd = load_joung(n_hvg=None)
    de = dd['de_single']; G = len(dd['gene_names'])
    Xc = dd['adata'].X; Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    ctrl = np.array(dd['control_idx']); ctrl_mean_combo = Xc[ctrl].mean(0)

    # ---- observed combos + additive_obs null + epistasis score (data-only) ----
    A2 = ad.read_h5ad(COMBO); A2.X = _lognorm(A2.X)
    Xk = A2.X.toarray() if sp.issparse(A2.X) else np.asarray(A2.X)
    tf_set = A2.obs['TF_set'].astype(str).values
    # gene index map from the FIRST model's gene_names (shared axis across the grid)
    _, gene_names0, _ = load_shift_model(f"{ROOT}/results/grid/v2_esm_con/checkpoints/phase1_epoch60.pt", dev)
    gidx = {g: i for i, g in enumerate(gene_names0)}

    sets = []
    for setname in sorted(set(tf_set)):
        tfs = parse_tfs(setname)
        gis = [gidx[t] for t in tfs if t in gidx]
        m = tf_set == setname
        if len(gis) < 2 or m.sum() < args.min_cells:
            continue
        obs = Xk[m].mean(0) - ctrl_mean_combo
        add_obs = np.sum([np.asarray(de[t], np.float32) for t in tfs if t in de], axis=0)
        resid = obs - add_obs
        E = float(np.linalg.norm(resid) / (np.linalg.norm(obs) + 1e-8))   # non-additive fraction
        add_pear = float(np.corrcoef(add_obs, obs)[0, 1]) if add_obs.std() > 1e-8 else np.nan
        sets.append({'set': '+'.join(tfs), 'k': len(gis), 'gis': gis, 'obs': obs,
                     'add_obs': add_obs, 'E': E, 'add_obs_pearson': add_pear})
    n = len(sets)
    E_arr = np.array([s['E'] for s in sets])
    # quartile edges by epistasis score
    q = np.quantile(E_arr, [0.25, 0.5, 0.75])
    qid = np.digitize(E_arr, q)                       # 0..3 = Q1..Q4
    for s, qq in zip(sets, qid):
        s['quartile'] = int(qq) + 1
    print(f"[epistasis] usable sets={n} (pairs={sum(s['k']==2 for s in sets)}, triples={sum(s['k']>=3 for s in sets)}); "
          f"E quartile edges={np.round(q,3)}", flush=True)

    # ---- per model: mutilation do-op per set ----
    summary = {'tag': args.tag, 'n_sets': n, 'E_quartile_edges': [float(x) for x in q], 'models': {}}
    per_rows = []
    for tag in MODELS:
        ckpt = f"{ROOT}/results/grid/{tag}/checkpoints/phase1_epoch60.pt"
        if not os.path.exists(ckpt):
            print(f"  skip {tag} (no ckpt)"); continue
        model, gene_names, ck = load_shift_model(ckpt, dev)
        d = model.latent_dim
        p1 = ck.get('config', {}).get('phase1', {})
        BETA = float(p1.get('pert_mutilate_beta', 1.0)); U_MAX = float(p1.get('pert_mutilate_u_max', 0.5))
        U_MODE = p1.get('pert_mutilate_u_mode', 'shift')
        rng = np.random.default_rng(0)
        sel = ctrl[rng.choice(len(ctrl), min(args.n_ctrl_cells, len(ctrl)), replace=False)]
        x_ctrl = torch.tensor(Xc[sel], dtype=torch.float32, device=dev)
        _, _, eps_ctrl = model.encoder(x_ctrl)
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        LHS = torch.eye(d, device=dev, dtype=A.dtype) - A.T
        z_prev = torch.clamp(_solve(LHS, eps_ctrl), -20, 20)
        ctrl_recon = model.decoder(z_prev).mean(0).cpu().numpy()

        def do_op(gis):
            S = torch.stack([model.compute_shift(torch.tensor([g], device=dev)).sum(0) for g in gis]).sum(0)
            z_S = torch.clamp(_solve(LHS, S.unsqueeze(0)), -20, 20)
            z_target = z_prev + z_S
            u = (BETA * S.abs()).clamp(max=1.0) if U_MODE == 'shift' else \
                model.get_loadings()[torch.tensor(gis, device=dev)].sum(0).clamp(0, U_MAX)
            eps_tilde = u.unsqueeze(0) * z_target + (1 - u).unsqueeze(0) * eps_ctrl
            z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
            return model.decoder(z_post).mean(0).cpu().numpy() - ctrl_recon

        # per-set do-op metrics
        for s in sets:
            do = do_op(s['gis'])
            do_p, do_a = per_set_metrics(do, s['obs'])
            ao_p, ao_a = per_set_metrics(s['add_obs'], s['obs'])
            per_rows.append({'model': tag, 'set': s['set'], 'k': s['k'], 'quartile': s['quartile'],
                             'E': s['E'], 'do_pearson': do_p, 'addobs_pearson': ao_p,
                             'do_deg_auroc': do_a, 'addobs_deg_auroc': ao_a, 'gap': do_p - ao_p})
        # aggregate by quartile
        mres = {}
        dfm = pd.DataFrame([r for r in per_rows if r['model'] == tag])
        for Q in [1, 2, 3, 4]:
            sub = dfm[dfm['quartile'] == Q]
            if len(sub) == 0: continue
            mres[f'Q{Q}'] = {
                'n': int(len(sub)), 'mean_E': float(sub['E'].mean()),
                'do_pearson': float(sub['do_pearson'].mean()),
                'addobs_pearson': float(sub['addobs_pearson'].mean()),
                'gap': float(sub['gap'].mean()),
                'do_deg_auroc': float(sub['do_deg_auroc'].mean()),
                'addobs_deg_auroc': float(sub['addobs_deg_auroc'].mean()),
                'do_wins_frac': float((sub['gap'] > 0).mean()),
            }
        mres['all'] = {'do_pearson': float(dfm['do_pearson'].mean()),
                       'addobs_pearson': float(dfm['addobs_pearson'].mean()),
                       'gap': float(dfm['gap'].mean())}
        summary['models'][tag] = mres
        print(f"  {tag}: " + " | ".join(
            f"Q{Q} do={mres.get(f'Q{Q}',{}).get('do_pearson',float('nan')):.3f}/add={mres.get(f'Q{Q}',{}).get('addobs_pearson',float('nan')):.3f}"
            for Q in [1, 2, 3, 4]), flush=True)

    json.dump(summary, open(f"{out}/summary.json", 'w'), indent=2)
    pd.DataFrame(per_rows).to_csv(f"{out}/per_set.csv", index=False)
    print(f"\nsaved → {out}/")


if __name__ == '__main__':
    main()

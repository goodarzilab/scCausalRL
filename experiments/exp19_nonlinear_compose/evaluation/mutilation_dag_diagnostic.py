"""DAG-health + mutilation (u) diagnostic for a shift-fed mutilation checkpoint.

Answers: is the DAG degenerate (weights→0, spectral→0)? does it recover CollecTRI
above the correlation floor? is the latent disentangling (ε independent, z carries
info, z≠ε)? AND — the mutilation-specific part — how much of the DAG actually gets
mutilated per perturbation (u distribution), or is u degenerate (≈0 → reduces to
additive; ≈1 → full pin)?

Usage:
  python evaluation/mutilation_dag_diagnostic.py --checkpoint <ckpt> --config <yaml> \
      --out <dir> --tag <label> [--max_cells 40000]
"""
import os, sys, argparse, json, warnings
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from evaluation.run_shift_eval_norman import load_shift_model


def offdiag_corr(M):
    c = np.corrcoef(M.T); d = c.shape[0]
    return float(np.abs(c[~np.eye(d, dtype=bool)]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='results/mut_diag'); ap.add_argument('--tag', default='model')
    ap.add_argument('--max_cells', type=int, default=40000); ap.add_argument('--device', default=None)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import yaml; config = yaml.safe_load(open(args.config))
    model, gene_names, ckpt = load_shift_model(args.checkpoint, device)
    n_genes = len(gene_names); d = model.latent_dim
    p1 = (ckpt.get('config', {}) or {}).get('phase1', {})
    u_mode = p1.get('pert_mutilate_u_mode', 'loading'); u_max = float(p1.get('pert_mutilate_u_max', 0.5)); beta = float(p1.get('pert_mutilate_beta', 1.0))
    sc = {'tag': args.tag, 'u_mode': u_mode, 'u_max': u_max, 'beta': beta}

    # ── 1. static DAG health ──
    with torch.no_grad():
        A = model.dag.get_signed_edges().cpu().numpy()
        W = model.dag.get_edge_weights().detach().cpu().numpy()
        D = model.decoder.D.detach().cpu().numpy()
        A_t = model.dag.get_signed_edges()
        GRN = model.decoder.project_dag_to_genes(A_t).detach().cpu().numpy(); np.fill_diagonal(GRN, 0)
    np.fill_diagonal(A, 0)
    sc['spectral_radius'] = float(np.max(np.abs(np.linalg.eigvals(A))))
    sc['edge_density'] = float((np.abs(A) > 1e-8).mean())
    sc['mean_abs_weight'] = float(np.abs(W).mean()); sc['max_abs_weight'] = float(np.abs(W).max())
    sc['frac_weights_near_zero'] = float((np.abs(W) < 1e-4).mean())
    sc['grn_nonzero_frac'] = float((np.abs(GRN) > 1e-8).mean())

    # ── 2. load cells (dataset from config) ──
    import scipy.sparse
    ds = config['data'].get('dataset', 'replogle')
    if ds == 'norman':
        from data.load_norman import load_norman
        dd = load_norman(path=config['data']['norman_path'], n_hvg=config['data']['n_hvg'],
                         force_include=list(gene_names))
    else:
        from data.load_replogle import load_replogle
        dd = load_replogle(path=config['data']['replogle_path'], n_hvg=config['data']['n_hvg'],
                           cell_line=config['data'].get('replogle_cell_line', 'k562'),
                           max_cells=min(args.max_cells, config['data'].get('replogle_max_cells', args.max_cells)),
                           split_seed=config['data'].get('split_seed', 42))
    ad = dd['adata']; X = (ad.X.toarray() if scipy.sparse.issparse(ad.X) else np.asarray(ad.X)).astype('float32')

    # ── 3. CollecTRI recovery vs correlation floor ──
    from utils.bio_priors import load_collectri
    col = load_collectri(); gup = [g.upper() for g in gene_names]; gidx = {g: i for i, g in enumerate(gup)}
    pairs = col[col['source'].isin(gidx) & col['target'].isin(gidx)].drop_duplicates(['source', 'target'])
    prior = np.zeros((n_genes, n_genes), bool)
    for s, t in zip(pairs['source'], pairs['target']):
        if s != t: prior[gidx[s], gidx[t]] = True
    offm = ~np.eye(n_genes, dtype=bool); y = prior[offm].astype(int)
    ctrl_idx = np.array(dd['control_idx']); Xc = X[ctrl_idx]
    corr = np.nan_to_num(np.abs(np.corrcoef(Xc.T))); np.fill_diagonal(corr, 0)
    sc['grn_collectri_auroc'] = float(roc_auc_score(y, np.abs(GRN)[offm]))
    sc['corr_collectri_auroc'] = float(roc_auc_score(y, corr[offm]))
    sc['collectri_pairs'] = int(len(pairs))

    # ── 4. latent disentanglement (ε vs z) ──
    rng = np.random.default_rng(0)
    pert_idx = np.array(sorted(set(range(ad.n_obs)) - set(ctrl_idx.tolist())))
    ne = int(min(4000, len(ctrl_idx), len(pert_idx)))
    sel = np.concatenate([rng.choice(ctrl_idx, ne, replace=False), rng.choice(pert_idx, ne, replace=False)])
    labels = np.array([0] * ne + [1] * ne)
    with torch.no_grad():
        mu, lv, eps = model.encoder(torch.tensor(X[sel], dtype=torch.float32, device=device))
        z = model.dag.scm_forward(eps, model.dag.sample_A(1).squeeze(0))
    eps_np, z_np = eps.cpu().numpy(), z.cpu().numpy()
    sc['eps_offdiag_corr'] = offdiag_corr(eps_np); sc['z_offdiag_corr'] = offdiag_corr(z_np)
    sc['z_vs_eps_rel_l2'] = float(np.mean(np.linalg.norm(z_np - eps_np, axis=1) / (np.linalg.norm(eps_np, axis=1) + 1e-8)))
    sc['silhouette_eps'] = float(silhouette_score(eps_np, labels)); sc['silhouette_z'] = float(silhouette_score(z_np, labels))
    sc['auroc_pertctrl_z'] = float(cross_val_score(LogisticRegression(max_iter=500), z_np, labels, cv=3, scoring='roc_auc').mean())

    # ── 5. MUTILATION u-analysis: how much of A is clamped per perturbation? ──
    pert_genes = [g for g in dd.get('pert_genes', list(dd.get('single_perts', {}).keys())) if g in gene_names]
    gi = torch.tensor([gene_names.index(g) for g in pert_genes], device=device)
    with torch.no_grad():
        S = model.compute_shift(gi)                                   # [P, d]
        if u_mode == 'shift':
            U = (beta * S.abs()).clamp(max=1.0).cpu().numpy()         # [P, d]
        else:
            L = model.get_loadings()[gi].clamp(0.0, u_max).cpu().numpy()
            U = L
    sc['n_pert_genes'] = int(len(pert_genes))
    sc['u_mean'] = float(U.mean()); sc['u_median'] = float(np.median(U))
    sc['u_frac_clamped_gt0.5'] = float((U > 0.5).mean())              # programs strongly cut
    sc['u_frac_active_gt0.1'] = float((U > 0.1).mean())
    sc['u_frac_untouched_lt0.01'] = float((U < 0.01).mean())          # programs ~ observational
    sc['u_progs_clamped_per_pert_median'] = float(np.median((U > 0.5).sum(1)))   # how many of d programs cut, typical pert
    sc['u_progs_active_per_pert_median'] = float(np.median((U > 0.1).sum(1)))
    sc['u_max_over_perts'] = float(U.max())
    # degeneracy verdict
    if sc['u_mean'] < 0.02:
        sc['u_verdict'] = 'DEGENERATE-LOW (≈additive: u≈0, no mutilation)'
    elif sc['u_frac_active_gt0.1'] > 0.9:
        sc['u_verdict'] = 'DEGENERATE-HIGH (≈full pin: most programs clamped)'
    else:
        sc['u_verdict'] = 'HEALTHY (sparse intermediate mutilation)'

    print(f"\n{'='*74}\nMUTILATION+DAG DIAGNOSTIC — {args.tag} (u_mode={u_mode}, beta={beta}, u_max={u_max})\n{'='*74}")
    print("[static DAG]")
    for k in ['spectral_radius', 'edge_density', 'mean_abs_weight', 'max_abs_weight', 'frac_weights_near_zero', 'grn_nonzero_frac']:
        print(f"  {k:32s} {sc[k]:.4f}")
    print("[CollecTRI]")
    print(f"  grn_collectri_auroc={sc['grn_collectri_auroc']:.4f}  corr_floor={sc['corr_collectri_auroc']:.4f}  (pairs={sc['collectri_pairs']})")
    print("[disentanglement]")
    for k in ['eps_offdiag_corr', 'z_offdiag_corr', 'z_vs_eps_rel_l2', 'silhouette_eps', 'silhouette_z', 'auroc_pertctrl_z']:
        print(f"  {k:32s} {sc[k]:.4f}")
    print(f"[mutilation u] over {sc['n_pert_genes']} pert genes")
    for k in ['u_mean', 'u_median', 'u_frac_clamped_gt0.5', 'u_frac_active_gt0.1', 'u_frac_untouched_lt0.01',
              'u_progs_clamped_per_pert_median', 'u_progs_active_per_pert_median', 'u_max_over_perts']:
        print(f"  {k:36s} {sc[k]:.4f}")
    print(f"  ⇒ u_verdict: {sc['u_verdict']}")
    json.dump(sc, open(os.path.join(args.out, f'mut_diag_{args.tag}.json'), 'w'), indent=2)
    np.save(os.path.join(args.out, f'u_matrix_{args.tag}.npy'), U)
    print(f"\nsaved → {args.out}/mut_diag_{args.tag}.json (+ u_matrix)")


if __name__ == '__main__':
    main()

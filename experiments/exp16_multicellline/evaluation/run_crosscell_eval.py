"""Per-cell-line held-out single-pert eval for the 4-cell-line exp16 model.

exp16 trained jointly on K562+RPE1+HepG2+Jurkat (all Replogle/Nadig CRISPRi), with the
cell line carried as the modality covariate. This evaluates, for EACH cell line, the
held-out single perturbations (the deterministic seed-42 test split — genes the model
never trained on, in any line), with the cell-line modality threaded into both the
control encoding and the shift. Same-modality CRISPRi → AUROC and Pearson-DEG are both
meaningful. The question: does joint 4-line training transfer per line, vs the K562-only
model's weak cross-cell transfer (exp13 RPE1 0.694 ≈ corr 0.679)?

Usage:
  python evaluation/run_crosscell_eval.py --checkpoint <ckpt> --output_dir <dir>
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.run_perturbation_eval import compute_observed_de, _safe_mean
from evaluation.run_norman_combo_eval import _score
from data.load_replogle_celline_aligned import load_replogle_celline_aligned
from data.load_norman import split_single_perts


def _solve(LHS, rhs):
    return torch.linalg.solve(LHS, rhs.T).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', default='results/crosscell')
    ap.add_argument('--max_cells', type=int, default=40000)
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--min_cells_per_pert', type=int, default=5)
    ap.add_argument('--max_perts', type=int, default=200, help="subsample held-out perts/line for speed")
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, gene_names, ckpt = load_shift_model(args.checkpoint, device)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names); d = model.latent_dim
    cfg = ckpt.get('config', {}) or {}
    celline_files = cfg['data']['celline_files']            # ordered dict → modality_id = index
    cl_to_mod = {cl: i for i, cl in enumerate(celline_files)}
    p1 = cfg.get('phase1', {})
    U_MODE = p1.get('pert_mutilate_u_mode', 'shift')
    U_MAX = float(p1.get('pert_mutilate_u_max', 0.5))
    BETA = float(p1.get('pert_mutilate_beta', 1.0))
    seed = int(cfg['data'].get('split_seed', 42))
    tf = float(cfg['data'].get('single_pert_train_frac', 0.8))
    vf = float(cfg['data'].get('single_pert_val_frac', 0.1))
    print(f"[crosscell] {len(celline_files)} lines {list(celline_files)}  u_mode={U_MODE} beta={BETA}")

    # load each line aligned to the model axis
    import scipy.sparse
    lines = {}
    union_genes = set()
    for cl, path in celline_files.items():
        dd = load_replogle_celline_aligned(path, target_gene_names=gene_names, cell_line=cl,
                                           max_cells=args.max_cells, min_cells_per_pert=args.min_cells_per_pert)
        X = dd['adata'].X
        X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
        de_single, _, _ = compute_observed_de(X, dd, gene_names)
        lines[cl] = dict(dd=dd, X=X, de=de_single)
        union_genes |= set(de_single.keys())

    # reproduce the training held-out split over the UNION of single-pert genes (seed 42)
    split = split_single_perts({g: [] for g in sorted(union_genes)}, train_frac=tf, val_frac=vf, seed=seed)
    test_genes = set(split['test'])
    print(f"  union single genes={len(union_genes)}  held-out(test)={len(test_genes)}")

    rng = np.random.default_rng(0)
    rows, per_pert = [], []
    for cl, L in lines.items():
        mod = cl_to_mod[cl]
        X, de = L['X'], L['de']
        ctrl_idx = np.array(L['dd']['control_idx'])
        sel = rng.choice(len(ctrl_idx), min(args.n_ctrl_cells, len(ctrl_idx)), replace=False)
        x_ctrl = torch.tensor(X[ctrl_idx[sel]], dtype=torch.float32, device=device)
        mod_ctrl = torch.full((x_ctrl.shape[0],), mod, dtype=torch.long, device=device)
        corr = np.nan_to_num(np.abs(np.corrcoef(X[ctrl_idx].T))); np.fill_diagonal(corr, 0.0)
        with torch.no_grad():
            _, _, eps_ctrl = model.encoder(x_ctrl, modality_id=mod_ctrl)
            A = model.dag.sample_A(n_samples=1).squeeze(0)
            LHS = torch.eye(d, device=device, dtype=A.dtype) - A.T
            z_prev = torch.clamp(_solve(LHS, eps_ctrl), -20, 20)
            ctrl_recon = model.decoder(z_prev).mean(0).cpu().numpy()

        keys = [g for g in de if g in test_genes]
        if args.max_perts and len(keys) > args.max_perts:
            keys = list(rng.choice(keys, args.max_perts, replace=False))
        m_au, c_au, m_pr = [], [], []
        for g in keys:
            gi = gene_to_idx[g]
            mask = np.ones(n_genes, bool); mask[gi] = False
            with torch.no_grad():
                S = model.compute_shift(torch.tensor([gi], device=device),
                                        modality_id=torch.tensor([mod], device=device)).sum(0)  # [d]
                z_S = torch.clamp(_solve(LHS, S.unsqueeze(0)), -20, 20)
                z_target = z_prev + z_S
                u = (BETA * S.abs()).clamp(max=1.0) if U_MODE == 'shift' else \
                    model.get_loadings()[torch.tensor([gi], device=device)].sum(0).clamp(0, U_MAX)
                eps_tilde = u.unsqueeze(0) * z_target + (1 - u).unsqueeze(0) * eps_ctrl
                z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
                pred = model.decoder(z_post).mean(0).cpu().numpy() - ctrl_recon
            sc = _score(pred, de[g]['lfc'], de[g]['effect'], de[g]['sig'], mask)
            cb = _score(corr[gi], de[g]['lfc'], de[g]['effect'], de[g]['sig'], mask)
            if 'auroc' in sc: m_au.append(sc['auroc']); c_au.append(cb.get('auroc', np.nan))
            if 'pearson_deg' in sc: m_pr.append(sc['pearson_deg'])
            per_pert.append({'cell_line': cl, 'pert': g, 'model_auroc': sc.get('auroc', np.nan),
                             'corr_auroc': cb.get('auroc', np.nan), 'model_pearson_deg': sc.get('pearson_deg', np.nan)})
        rows.append({'cell_line': cl, 'modality_id': mod, 'n_heldout': len(keys),
                     'model_auroc': _safe_mean(m_au), 'corr_auroc': _safe_mean(c_au),
                     'model_pearson_deg': _safe_mean(m_pr)})
        print(f"  {cl:7s} (mod {mod}): n={len(keys):4d}  model_AUROC={_safe_mean(m_au):.4f}  "
              f"corr_AUROC={_safe_mean(c_au):.4f}  model_pearson_deg={_safe_mean(m_pr):.4f}", flush=True)

    pd.DataFrame(per_pert).to_csv(os.path.join(args.output_dir, 'per_pert.csv'), index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    print("\n===== exp16 CROSS-CELL (held-out singles per line) =====")
    print(summ.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"saved → {args.output_dir}/")


if __name__ == '__main__':
    main()

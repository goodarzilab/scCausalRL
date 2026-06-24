"""Modality-AWARE eval for the exp15 combined (Replogle CRISPRi + Norman CRISPRa) model.

The combined model conditions the shift scale on modality (shift_scale_mod[CONTROL/
CRISPRa/CRISPRi]); the stock run_norman_combo_eval calls compute_shift WITHOUT a
modality, so it silently uses the global shift_scale instead of the Norman/CRISPRa
scale. This script fixes that: control cells are encoded with modality CONTROL(0), and
every perturbation shift uses the screen's modality (Norman→CRISPRa=1, Replogle→
CRISPRi=2). It evaluates singles AND (for Norman) combos with the SAME shift-native
mutilation do-operator the model was trained with — modality threaded in.

Predictors (vs observed DE, target genes masked):
  shift_add     — Σ single modality-aware shift predictions (zero-shot additive)
  shift_uscore  — modality-aware shift-native mutilation (the trained do-operator)
  additive_obs  — Σ observed single effects (combo only; observed-singles oracle)
  corr          — control gene-gene correlation baseline
Metrics: AUROC (direction-agnostic), Spearman(|effect|), Pearson-DEG.

Usage:
  python evaluation/run_modality_eval.py --checkpoint <ckpt> --config <cfg> \
      --pert_modality crispra --output_dir results/modality_eval_norman
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
from evaluation.run_perturbation_eval import compute_observed_de, _safe_mean, _extract
from evaluation.run_norman_combo_eval import _score
from data.load_norman_aligned import load_norman_aligned

MODALITY = {'control': 0, 'crispra': 1, 'crispri': 2}


def _solve(LHS, rhs):
    return torch.linalg.solve(LHS, rhs.T).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--pert_modality', default='crispra', choices=['crispra', 'crispri'])
    ap.add_argument('--norman_path', default=None)
    ap.add_argument('--output_dir', default='results/modality_eval')
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--min_cells_per_pert', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import yaml
    config = yaml.safe_load(open(args.config))
    norman_path = args.norman_path or config['data']['norman_path']
    model, gene_names, ckpt = load_shift_model(args.checkpoint, device)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names); d = model.latent_dim

    p1 = (ckpt.get('config', {}) or {}).get('phase1', {})
    U_MODE = p1.get('pert_mutilate_u_mode', 'shift')
    U_MAX = float(p1.get('pert_mutilate_u_max', 0.5))
    BETA = float(p1.get('pert_mutilate_beta', 1.0))
    mod_pert = MODALITY[args.pert_modality]
    print(f"[modality-eval] pert_modality={args.pert_modality}({mod_pert})  "
          f"u_mode={U_MODE}  u_max={U_MAX}  beta={BETA}")

    dd = load_norman_aligned(norman_path, target_gene_names=gene_names,
                             min_cells_per_pert=args.min_cells_per_pert)
    import scipy.sparse
    X = dd['adata'].X
    X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
    de_single, de_combo, _ = compute_observed_de(X, dd, gene_names)
    de_combo = {k: v for k, v in de_combo.items()
                if k[0] in de_single and k[1] in de_single}

    # ---- control encode (modality CONTROL) + eval DAG ----
    rng = np.random.default_rng(args.seed)
    ctrl_idx = np.array(dd['control_idx'])
    sel = rng.choice(len(ctrl_idx), min(args.n_ctrl_cells, len(ctrl_idx)), replace=False)
    x_ctrl_t = torch.tensor(X[ctrl_idx[sel]], dtype=torch.float32, device=device)
    mod_ctrl = torch.zeros(x_ctrl_t.shape[0], dtype=torch.long, device=device)  # CONTROL=0
    corr_ctrl = np.nan_to_num(np.abs(np.corrcoef(X[ctrl_idx].T))); np.fill_diagonal(corr_ctrl, 0.0)

    with torch.no_grad():
        _, _, eps_ctrl = model.encoder(x_ctrl_t, modality_id=mod_ctrl)
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        I_mat = torch.eye(d, device=device, dtype=A.dtype)
        LHS = I_mat - A.T
        z_prev = torch.clamp(_solve(LHS, eps_ctrl), -20, 20)
        ctrl_recon = model.decoder(z_prev).mean(0).cpu().numpy()

    def shift_for(genes):
        """Modality-aware summed shift S=[d] + per-gene latent z_S for `genes`."""
        gi = torch.tensor([gene_to_idx[g] for g in genes], device=device)
        mod = torch.full((len(genes),), mod_pert, dtype=torch.long, device=device)
        S_each = model.compute_shift(gi, modality_id=mod)        # [k, d]
        return S_each

    @torch.no_grad()
    def predict(genes):
        """(shift_add pred, shift_uscore pred) gene-space deltas for a gene set."""
        S_each = shift_for(genes)                                 # [k, d]
        z_S_each = torch.clamp(_solve(LHS, S_each), -20, 20)      # [k, d]
        # shift_add: sum of single-gene decoded deltas
        add = np.zeros(n_genes, np.float32)
        for j in range(len(genes)):
            xh = model.decoder(torch.clamp(z_prev + z_S_each[j:j+1], -20, 20)).mean(0).cpu().numpy()
            add += (xh - ctrl_recon)
        # shift_uscore: modality-aware shift-native mutilation (sum shift over the set)
        S = S_each.sum(0)                                         # [d]
        z_S = torch.clamp(_solve(LHS, S), -20, 20)
        z_target = z_prev + z_S.unsqueeze(0)
        if U_MODE == 'shift':
            u = (BETA * S.abs()).clamp(max=1.0)
        else:
            u = model.get_loadings()[torch.tensor([gene_to_idx[g] for g in genes], device=device)].sum(0).clamp(0, U_MAX)
        eps_tilde = u.unsqueeze(0) * z_target + (1 - u).unsqueeze(0) * eps_ctrl
        z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
        uscore = model.decoder(z_post).mean(0).cpu().numpy() - ctrl_recon
        return add, uscore

    # ---- singles ----
    srows = []
    for g, de in de_single.items():
        gi = gene_to_idx[g]
        mask = np.ones(n_genes, bool); mask[gi] = False
        add, usc = predict([g])
        r = {'pert': g, 'n_de': int(de['sig'][mask].sum())}
        for pname, pred in [('shift_add', add), ('shift_uscore', usc)]:
            for mk, mv in _score(pred, de['lfc'], de['effect'], de['sig'], mask).items():
                r[f'{pname}_{mk}'] = mv
        srows.append(r)
    sdf = pd.DataFrame(srows)
    sdf.to_csv(os.path.join(args.output_dir, 'singles_per_pert.csv'), index=False)

    # ---- combos (Norman) ----
    crows = []
    for (ga, gb), de in de_combo.items():
        gi_a, gi_b = gene_to_idx[ga], gene_to_idx[gb]
        mask = np.ones(n_genes, bool); mask[gi_a] = False; mask[gi_b] = False
        add, usc = predict([ga, gb])
        add_obs = de_single[ga]['lfc'] + de_single[gb]['lfc']
        corr_sig = corr_ctrl[gi_a] + corr_ctrl[gi_b]
        r = {'combo': f'{ga}+{gb}', 'n_de': int(de['sig'][mask].sum())}
        for pname, pred in [('shift_add', add), ('additive_obs', add_obs),
                            ('shift_uscore', usc), ('corr', corr_sig)]:
            for mk, mv in _score(pred, de['lfc'], de['effect'], de['sig'], mask).items():
                r[f'{pname}_{mk}'] = mv
        crows.append(r)
    cdf = pd.DataFrame(crows)
    cdf.to_csv(os.path.join(args.output_dir, 'combo_per_pert.csv'), index=False)

    # ---- summary ----
    def col(df, c):
        return float(df[c].mean()) if c in df.columns else float('nan')
    summary = {'pert_modality': args.pert_modality, 'n_singles': len(sdf), 'n_combos': len(cdf)}
    for m in ['shift_add', 'shift_uscore']:
        summary[f'single_{m}_auroc'] = col(sdf, f'{m}_auroc')
        summary[f'single_{m}_spearman'] = col(sdf, f'{m}_spearman')
    for m in ['shift_add', 'additive_obs', 'shift_uscore', 'corr']:
        summary[f'combo_{m}_auroc'] = col(cdf, f'{m}_auroc')
        summary[f'combo_{m}_spearman'] = col(cdf, f'{m}_spearman')
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)

    print(f"\n=== MODALITY-AWARE EVAL ({args.pert_modality}) ===")
    print(f"singles n={len(sdf)}: shift_add AUROC={summary['single_shift_add_auroc']:.4f}  "
          f"shift_uscore AUROC={summary['single_shift_uscore_auroc']:.4f}")
    if len(cdf):
        print(f"combos  n={len(cdf)}: shift_add={summary['combo_shift_add_auroc']:.4f}  "
              f"additive_obs={summary['combo_additive_obs_auroc']:.4f}  "
              f"shift_uscore(mutilation)={summary['combo_shift_uscore_auroc']:.4f}  "
              f"corr={summary['combo_corr_auroc']:.4f}")
    print(f"saved → {args.output_dir}/")


if __name__ == '__main__':
    main()

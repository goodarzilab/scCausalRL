"""
Shift-pathway perturbation evaluation for the shift-encoder experiments (exp2-5).

This is the eval counterpart to the `perturbation_set_loss_shift` training path.
Instead of the leaky `dz = D⁺·dx` intervention (handled by run_perturbation_eval.py),
it predicts a held-out perturbation via the LEARNED shift:

    S^q = c · ShiftEncoder(pert_label[q])             # pert_label baked into the ckpt
    z_post = (I − A^T)^{-1} (ε_ctrl + S^q)            # additive shift, no mutilation
    x̂^q   = decoder(z_post)

Key properties:
  • Zero-shot: pert_label[q] is the control-PCA embedding of gene q, present for
    held-out genes. NO observed dx enters the prediction (no leakage), unlike
    target_lfc/observed_full.
  • Penetrance selection (SCCVAE §2.2): optionally grid-search a scalar c on
    pseudo-bulk MSE (leaks the mean strength only). `--c_mode fixed` (c=1, strict,
    uses the learned shift_scale) is the honest default; `--c_mode search` reports
    the penetrance-selected number alongside.

Metrics per held-out perturbation (target gene masked from scoring):
  • downstream-DEG AUROC, Spearman (|effect| and signed), direction accuracy, Pearson-on-DEGs
  • distributional: RBF-MMD and energy distance of predicted vs observed cell SETS,
    with the control-vs-observed value as a floor.

Baselines (same held-out genes):
  • Correlation (|corr| from control cells) — DEG-detection comparator.
  • Co-expression-neighbour LOO: predict q's effect as the mean observed lfc of its
    k nearest TRAIN perturbations in the SAME control-PCA space the shift encoder uses.
    This is the fair zero-shot "prior-knowledge" comparator — the shift encoder must
    beat "average your co-expression neighbours" to be earning its keep.

Usage:
    python evaluation/run_shift_eval.py \
        --checkpoint results/phase1/checkpoints/phase1_epoch150.pt \
        --config configs/exp2_shift_linear.yaml \
        --output_dir results/phase1/shift_eval --c_mode both
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from models.causal_vae import CausalVAE
from utils.metrics import pearson_on_degs
from training.losses import mmd_rbf, distributional_loss
# Reuse the observed-DE computation, correlation baseline, and aggregation helpers.
from evaluation.run_perturbation_eval import (
    compute_observed_de, compute_correlation_matrix, evaluate_baseline,
    _safe_mean, _safe_sem, _extract,
)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL + DATA LOADING (shift-aware)
# ═══════════════════════════════════════════════════════════════════════════
def load_shift_model_and_data(config_path, checkpoint_path, device):
    """Load data + a shift-encoder checkpoint.

    The model is reconstructed from the checkpoint's OWN saved config
    (ckpt['config']['model']) so decoder_type / use_shift_encoder / shift_* match
    training exactly. The pert_label buffer (persistent) loads with the state dict,
    so the eval uses the same control-PCA labels baked at training time.
    """
    import yaml, scipy.sparse
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = config.get('data', {}).get('dataset', 'replogle').lower()
    if dataset == 'replogle':
        from data.load_replogle import load_replogle
        data_dict = load_replogle(
            path=config['data']['replogle_path'],
            n_hvg=config['data']['n_hvg'],
            cell_line=config['data'].get('replogle_cell_line', 'k562'),
            max_cells=config['data'].get('replogle_max_cells', 200000),
            train_frac=config['data'].get('single_pert_train_frac', 0.8),
            val_frac=config['data'].get('single_pert_val_frac', 0.1),
            split_seed=config['data'].get('split_seed', 42),
        )
    elif dataset == 'norman':
        from data.load_norman import load_norman
        data_dict = load_norman(path=config['data']['norman_path'],
                                n_hvg=config['data']['n_hvg'])
    else:
        raise ValueError(f"shift eval not wired for dataset={dataset!r}")

    adata = data_dict['adata']
    gene_names = data_dict['gene_names']
    n_genes = len(gene_names)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']
    # Prefer the checkpoint's saved model config; fall back to the eval config.
    cfg_m = (ckpt.get('config', {}) or {}).get('model', config['model'])
    n_batches = int(sd['encoder.batch_emb.weight'].shape[0]) if 'encoder.batch_emb.weight' in sd else 0
    n_modalities = int(sd['encoder.modality_emb.weight'].shape[0]) if 'encoder.modality_emb.weight' in sd else 0

    model = CausalVAE(
        n_genes=n_genes, latent_dim=cfg_m['latent_dim'],
        tau_init=cfg_m.get('tau_min', cfg_m.get('tau_init', 1.0)),
        n_dag_samples=cfg_m.get('n_dag_samples', 16),
        dagma_s=cfg_m.get('dagma_s', 1.0),
        n_modalities=n_modalities, n_batches=n_batches,
        decoder_type=cfg_m.get('decoder_type', 'linear'),
        decoder_hidden=int(cfg_m.get('decoder_hidden', 1024)),
        decoder_layers=int(cfg_m.get('decoder_layers', 2)),
        use_shift_encoder=bool(cfg_m.get('use_shift_encoder', True)),
        shift_label_dim=int(cfg_m.get('shift_label_dim', 50)),
        shift_hidden=int(cfg_m.get('shift_hidden', 1024)),
        shift_n_layers=int(cfg_m.get('shift_n_layers', 1)),
        shift_output_activation=cfg_m.get('shift_output_activation', 'none'),
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    if not model.use_shift_encoder:
        raise RuntimeError("checkpoint has no shift encoder — use run_perturbation_eval.py instead")
    print(f"Loaded shift model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
          f"decoder={cfg_m.get('decoder_type', 'linear')})")

    if 'gene_names' in ckpt and list(gene_names) != list(ckpt['gene_names']):
        n_mismatch = sum(1 for a, b in zip(gene_names, ckpt['gene_names']) if a != b)
        raise RuntimeError(
            f"gene_names do not match checkpoint ({n_mismatch} differ). The eval "
            f"config must match training (max_cells / n_hvg / split_seed).")

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)
    return config, model, data_dict, X, gene_names, ckpt


# ═══════════════════════════════════════════════════════════════════════════
# SHIFT-PATHWAY MODEL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
def _scm_solve(LHS, rhs):
    """Solve (I − A^T) zᵀ = rhsᵀ for z. LHS [d,d], rhs [N,d] → z [N,d]."""
    return torch.linalg.solve(LHS, rhs.T).T


def evaluate_single_perts_shift(model, X, data_dict, de_dict, gene_names, device,
                                n_ctrl_cells=500, c_mode='fixed',
                                c_grid=None, n_dist_cells=256, seed=42,
                                mutilate=False, mut_u_mode='loading',
                                mut_u_max=0.5, mut_beta=1.0):
    """Predict held-out single perturbations through the learned shift pathway.

    c_mode: 'fixed'  → c=1 (use the learned shift_scale; strict, no leak)
            'search' → grid-search c on pseudo-bulk MSE (penetrance selection)
    Returns list of per-perturbation metric dicts (includes chosen_c).
    """
    if c_grid is None:
        c_grid = np.round(np.arange(0.0, 3.01, 0.25), 3)
    control_idx = np.array(data_dict['control_idx'])
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    d = model.latent_dim
    n_genes = len(gene_names)

    rng = np.random.default_rng(seed)
    ctrl_sample = rng.choice(len(control_idx), min(n_ctrl_cells, len(control_idx)), replace=False)
    x_ctrl_t = torch.tensor(X[control_idx[ctrl_sample]], dtype=torch.float32, device=device)
    obs_ctrl_dist = torch.tensor(
        X[control_idx[rng.choice(len(control_idx), min(n_dist_cells, len(control_idx)), replace=False)]],
        dtype=torch.float32, device=device)

    with torch.no_grad():
        _, _, eps_ctrl = model.encoder(x_ctrl_t)                # [Nc, d]
        A_eval = model.dag.sample_A(n_samples=1).squeeze(0)     # deterministic single DAG
        I_mat = torch.eye(d, device=device, dtype=A_eval.dtype)
        LHS = I_mat - A_eval.T
        z_prev = torch.clamp(_scm_solve(LHS, eps_ctrl), -20, 20)
        x_hat_prev = model.decoder(z_prev)                      # [Nc, G]
        ctrl_recon_mean = x_hat_prev.mean(0).cpu().numpy()
        c_grid_t = torch.tensor(c_grid, dtype=torch.float32, device=device)

    results = []
    de_single_adata = data_dict['adata'].uns.get('de_single', {})

    for pert_gene, de_data in de_dict.items():
        if pert_gene not in gene_to_idx:
            continue
        gi = gene_to_idx[pert_gene]
        obs_lfc = de_data['lfc']
        obs_effect = de_data['effect']
        obs_sig = de_data['sig']
        obs_lfc_t = torch.tensor(obs_lfc, dtype=torch.float32, device=device)

        with torch.no_grad():
            chosen_c = 1.0
            if mutilate:
                # COHERENT mutilation forward (same operator the model was trained with):
                # shift defines the target, do-operator propagates it. No penetrance search.
                z_post, _u = model.intervene_shift_mutilate(
                    eps_ctrl, [gi], A_eval, z_prev,
                    u_mode=mut_u_mode, u_clip_max=mut_u_max, shift_beta=mut_beta)
                z_post = torch.clamp(z_post, -20, 20)                     # [Nc, d]
                x_hat_post = model.decoder(z_post)
                pred_mean = x_hat_post.mean(0).cpu().numpy()
            else:
                S = model.compute_shift(torch.tensor([gi], device=device))   # [1, d]
                z_S = torch.clamp(_scm_solve(LHS, S), -20, 20)               # [1, d]
                # ── penetrance c (closed-form latent superposition; decode per c) ──
                if c_mode == 'search':
                    z_post_c = z_prev.unsqueeze(0) + c_grid_t[:, None, None] * z_S.unsqueeze(0)  # [C, Nc, d]
                    C = z_post_c.shape[0]
                    x_hat_c = model.decoder(z_post_c.reshape(-1, d)).reshape(C, -1, n_genes)
                    pred_de_c = x_hat_c.mean(1) - torch.tensor(ctrl_recon_mean, device=device)  # [C, G]
                    mse_c = ((pred_de_c - obs_lfc_t[None, :]) ** 2).mean(1)                       # [C]
                    chosen_c = float(c_grid_t[int(torch.argmin(mse_c))].item())
                z_post = torch.clamp(z_prev + chosen_c * z_S, -20, 20)        # [Nc, d]
                x_hat_post = model.decoder(z_post)                            # [Nc, G]
                pred_mean = x_hat_post.mean(0).cpu().numpy()

        pred_de = pred_mean - ctrl_recon_mean

        # Mask the perturbed gene itself from downstream scoring.
        mask = np.ones(n_genes, dtype=bool); mask[gi] = False
        pred_abs = np.abs(pred_de[mask]); pred_signed = pred_de[mask]
        obs_eff_m = obs_effect[mask]; obs_lfc_m = obs_lfc[mask]; obs_sig_m = obs_sig[mask]

        r = {'perturbation': pert_gene, 'n_cells': de_data['n_cells'],
             'n_de': int(obs_sig_m.sum()), 'chosen_c': chosen_c}
        if len(pred_abs) > 10:
            r['spearman'] = spearmanr(pred_abs, obs_eff_m)[0]
            r['spearman_signed'] = spearmanr(pred_signed, obs_lfc_m)[0]
        n_sig = int(obs_sig_m.sum())
        if 3 <= n_sig < len(obs_sig_m):
            try:
                r['auroc'] = roc_auc_score(obs_sig_m.astype(int), pred_abs)
            except ValueError:
                pass
        if n_sig >= 3:
            r['direction_acc'] = float(
                (np.sign(pred_signed[obs_sig_m]) == np.sign(obs_lfc_m[obs_sig_m])).mean())
        r['pearson_deg'] = pearson_on_degs(pred_de, obs_lfc, threshold=0.05)
        r['mse_mean'] = float(np.mean((pred_de[mask] - obs_lfc_m) ** 2))

        # ── distributional metrics: predicted vs observed cell SETS ──
        idxs = data_dict['single_perts'].get(pert_gene, [])
        if len(idxs) >= 8:
            with torch.no_grad():
                obs_idx = rng.choice(len(idxs), min(n_dist_cells, len(idxs)), replace=False)
                obs_pert = torch.tensor(X[np.asarray(idxs)[obs_idx]], dtype=torch.float32, device=device)
                npred = min(n_dist_cells, x_hat_post.shape[0])
                pred_cells = x_hat_post[:npred]
                ctrl_cells = obs_ctrl_dist[:obs_pert.shape[0]]
                r['mmd_model'] = float(mmd_rbf(pred_cells, obs_pert).item())
                r['mmd_control'] = float(mmd_rbf(ctrl_cells, obs_pert).item())
                r['energy_model'] = float(distributional_loss(pred_cells, obs_pert, kind='energy').item())
                r['energy_control'] = float(distributional_loss(ctrl_cells, obs_pert, kind='energy').item())
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CO-EXPRESSION-NEIGHBOUR LOO BASELINE
# ═══════════════════════════════════════════════════════════════════════════
def coexp_neighbour_baseline(model, de_train, de_eval, gene_names, k=10):
    """Predict each eval gene's effect as the mean observed lfc of its k nearest
    TRAIN perturbations in the model's control-PCA label space (the same feature
    the shift encoder consumes). Fair zero-shot 'prior-knowledge' comparator."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    pert_label = model.pert_label.detach().cpu().numpy()          # [G, L]
    norm = np.linalg.norm(pert_label, axis=1, keepdims=True) + 1e-8
    unit = pert_label / norm

    train_genes = [g for g in de_train if g in gene_to_idx]
    if len(train_genes) < k + 1:
        return []
    train_gi = np.array([gene_to_idx[g] for g in train_genes])
    train_lfc = np.stack([de_train[g]['lfc'] for g in train_genes])  # [T, G]
    train_unit = unit[train_gi]                                       # [T, L]

    results = []
    for g, de in de_eval.items():
        if g not in gene_to_idx:
            continue
        gi = gene_to_idx[g]
        sims = train_unit @ unit[gi]                                  # cosine sim [T]
        order = np.argsort(-sims)
        # exclude exact self if the eval gene is also in the train list
        nbrs = [j for j in order if train_gi[j] != gi][:k]
        pred_de = train_lfc[nbrs].mean(0)

        obs_lfc = de['lfc']; obs_effect = de['effect']; obs_sig = de['sig']
        mask = np.ones(len(gene_names), dtype=bool); mask[gi] = False
        r = {'perturbation': g, 'n_de': int(obs_sig[mask].sum())}
        if mask.sum() > 10:
            r['spearman'] = spearmanr(np.abs(pred_de[mask]), obs_effect[mask])[0]
        n_sig = int(obs_sig[mask].sum())
        if 3 <= n_sig < int(mask.sum()):
            try:
                r['auroc'] = roc_auc_score(obs_sig[mask].astype(int), np.abs(pred_de[mask]))
            except ValueError:
                pass
        r['pearson_deg'] = pearson_on_degs(pred_de, obs_lfc, threshold=0.05)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def _print_table(title, rows):
    print(f"\n  --- {title} ---")
    print(f"  {'Method':>32s} {'AUROC':>8s} {'Spear':>8s} {'Pear-DEG':>9s} {'N':>5s}")
    for name, res in rows:
        print(f"  {name:>32s} {_safe_mean(_extract(res, 'auroc')):>8.4f} "
              f"{_safe_mean(_extract(res, 'spearman')):>8.4f} "
              f"{_safe_mean(_extract(res, 'pearson_deg')):>9.4f} {len(res):>5d}")


def main():
    ap = argparse.ArgumentParser(description="Shift-pathway perturbation eval")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output_dir', default='results/shift_eval')
    ap.add_argument('--device', default=None)
    ap.add_argument('--c_mode', default='both', choices=['fixed', 'search', 'both'],
                    help="'fixed' (c=1, strict), 'search' (penetrance grid), 'both'.")
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--n_dist_cells', type=int, default=256)
    ap.add_argument('--coexp_k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--wandb', action='store_true', help="log summary to W&B project CausalBioRL")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")

    config, model, data_dict, X, gene_names, ckpt = load_shift_model_and_data(
        args.config, args.checkpoint, device)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    X_ctrl = X[data_dict['control_idx']]

    # Detect mutilation-trained checkpoints → eval with the SAME (coherent) operator.
    _p1 = (ckpt.get('config', {}) or {}).get('phase1', {})
    MUT = _p1.get('pert_loss_mode', 'shift') == 'shift_mutilate'
    MUT_KW = dict(mutilate=MUT, mut_u_mode=_p1.get('pert_mutilate_u_mode', 'loading'),
                  mut_u_max=float(_p1.get('pert_mutilate_u_max', 0.5)),
                  mut_beta=float(_p1.get('pert_mutilate_beta', 1.0)))
    if MUT:
        print(f"[single-eval] mutilation-coherent: u_mode={MUT_KW['mut_u_mode']} "
              f"u_max={MUT_KW['mut_u_max']} beta={MUT_KW['mut_beta']}")

    # Prefer the split saved in the checkpoint; otherwise reconstruct it from the
    # data loader. The periodic phase1_epoch{N}.pt checkpoints only carry
    # epoch/model/config/gene_names (pert_split is saved only in phase1_best.pt),
    # but the split is DETERMINISTIC in split_seed, so loading the same config
    # reproduces the identical train/val/test gene split used during training.
    pert_split = ckpt.get('pert_split', None)
    if pert_split is None:
        pert_split = data_dict.get('pert_split', None)
        if pert_split is not None:
            print("  (pert_split not in checkpoint — reconstructed from data loader "
                  "via split_seed; identical to training split)")
    if pert_split is not None:
        pert_split = {k: set(v) for k, v in pert_split.items()}
        print(f"Split: {len(pert_split['train'])} train / {len(pert_split['test'])} test genes")
    else:
        raise RuntimeError(
            "no pert_split in checkpoint OR data_dict — cannot do held-out eval. "
            "Ensure the dataset loader produces a pert_split (split_seed set).")

    print("\nComputing observed DE...")
    de_single, de_combo, ctrl_mean = compute_observed_de(X, data_dict, gene_names)
    corr_matrix = compute_correlation_matrix(X_ctrl, gene_names)

    c_modes = ['fixed', 'search'] if args.c_mode == 'both' else [args.c_mode]
    de_train_full = {g: de for g, de in de_single.items() if g in pert_split['train']}

    summary = {}
    all_rows = {}
    for split_name, split_genes in [('train', pert_split['train']),
                                    ('test', pert_split['test'])]:
        de_split = {g: de for g, de in de_single.items() if g in split_genes}
        if not de_split:
            print(f"\n[{split_name}] no scorable genes — skipping")
            continue
        print(f"\n{'='*64}\nSPLIT={split_name.upper()}  ({len(de_split)} genes)\n{'='*64}")

        rows = []
        for cm in c_modes:
            model_res = evaluate_single_perts_shift(
                model, X, data_dict, de_split, gene_names, device,
                n_ctrl_cells=args.n_ctrl_cells, c_mode=cm,
                n_dist_cells=args.n_dist_cells, seed=args.seed, **MUT_KW)
            tag = f"Shift model (c={cm})"
            rows.append((tag, model_res))
            all_rows[f'{split_name}_model_{cm}'] = model_res
            summary[f'{split_name}_model_{cm}_auroc'] = _safe_mean(_extract(model_res, 'auroc'))
            summary[f'{split_name}_model_{cm}_pearson_deg'] = _safe_mean(_extract(model_res, 'pearson_deg'))
            summary[f'{split_name}_model_{cm}_spearman'] = _safe_mean(_extract(model_res, 'spearman'))
            summary[f'{split_name}_model_{cm}_mmd'] = _safe_mean(_extract(model_res, 'mmd_model'))
            summary[f'{split_name}_model_{cm}_mmd_ctrl_floor'] = _safe_mean(_extract(model_res, 'mmd_control'))
            summary[f'{split_name}_model_{cm}_energy'] = _safe_mean(_extract(model_res, 'energy_model'))
            summary[f'{split_name}_model_{cm}_energy_ctrl_floor'] = _safe_mean(_extract(model_res, 'energy_control'))

        corr_res = evaluate_baseline(corr_matrix, gene_to_idx, de_split, gene_names)
        coexp_res = coexp_neighbour_baseline(model, de_train_full, de_split, gene_names, k=args.coexp_k)
        rows.append(("Correlation (control)", corr_res))
        rows.append((f"Co-exp neighbour (k={args.coexp_k})", coexp_res))
        all_rows[f'{split_name}_corr'] = corr_res
        all_rows[f'{split_name}_coexp'] = coexp_res
        summary[f'{split_name}_corr_auroc'] = _safe_mean(_extract(corr_res, 'auroc'))
        summary[f'{split_name}_coexp_auroc'] = _safe_mean(_extract(coexp_res, 'auroc'))
        summary[f'{split_name}_coexp_pearson_deg'] = _safe_mean(_extract(coexp_res, 'pearson_deg'))

        _print_table(f"{split_name} single-pert", rows)
        # distributional summary line (model vs control floor)
        for cm in c_modes:
            mr = all_rows[f'{split_name}_model_{cm}']
            print(f"  [c={cm}] MMD model/ctrl-floor = "
                  f"{_safe_mean(_extract(mr,'mmd_model')):.4f} / {_safe_mean(_extract(mr,'mmd_control')):.4f}"
                  f"   Energy model/ctrl-floor = "
                  f"{_safe_mean(_extract(mr,'energy_model')):.4f} / {_safe_mean(_extract(mr,'energy_control')):.4f}")

    # ── Save ──
    for key, res in all_rows.items():
        if res:
            pd.DataFrame(res).to_csv(os.path.join(args.output_dir, f'{key}.csv'), index=False)
    summary['checkpoint'] = args.checkpoint
    summary['epoch'] = ckpt.get('epoch', None)
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    print(f"\nSaved to {args.output_dir}/")

    # Headline
    if 'test_model_fixed_auroc' in summary:
        print(f"\nHEADLINE (test, held-out):")
        print(f"  shift model AUROC (c=1)     : {summary.get('test_model_fixed_auroc', float('nan')):.4f}")
        if 'test_model_search_auroc' in summary:
            print(f"  shift model AUROC (c-search): {summary.get('test_model_search_auroc'):.4f}")
        print(f"  correlation baseline AUROC  : {summary.get('test_corr_auroc', float('nan')):.4f}")
        print(f"  co-exp neighbour AUROC      : {summary.get('test_coexp_auroc', float('nan')):.4f}")

    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=config.get('wandb', {}).get('project', 'CausalBioRL'),
                             name=f"shift_eval_{os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))}",
                             config={'checkpoint': args.checkpoint}, reinit=True)
            run.log({f'eval/{k}': v for k, v in summary.items() if isinstance(v, (int, float))})
            run.finish()
            print("Logged summary to W&B.")
        except Exception as e:
            print(f"W&B logging skipped ({e}).")


if __name__ == '__main__':
    main()

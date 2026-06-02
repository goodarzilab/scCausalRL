"""
External validation: evaluate Norman-trained CausalVAE on Replogle K562 perturbations.

This is a zero-shot generalization test:
- Same cell line (K562), different modality (CRISPRa -> CRISPRi)
- Gene axis aligned to Norman's training HVGs
- Only Replogle perturbations of genes in Norman's HVG set are evaluable

Usage:
    python evaluation/run_replogle_validation.py \
        --checkpoint checkpoints/phase1_best.pt \
        --replogle_path /large_storage/goodarzilab/causal_state/pretrain_data/replogle \
        --max_cells 50000
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from models.causal_vae import CausalVAE
from data.load_replogle_aligned import load_replogle_aligned

# Reuse existing eval functions
from evaluation.run_perturbation_eval import (
    compute_observed_de, compute_correlation_matrix, compute_grn_matrix,
    evaluate_baseline, evaluate_single_perts,
    plot_single_pert_comparison, plot_auroc_histogram, plot_per_pert_scatter,
    _safe_mean, _safe_sem, _extract,
)


def main():
    parser = argparse.ArgumentParser(description="Replogle External Validation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Norman-trained CausalVAE checkpoint')
    parser.add_argument('--replogle_path', type=str,
                        default='/large_storage/goodarzilab/causal_state/pretrain_data/replogle')
    parser.add_argument('--max_cells', type=int, default=50000,
                        help='Downsample Replogle for speed (default 50k)')
    parser.add_argument('--n_dag_samples', type=int, default=32,
                        help='(Unused — kept for API compat; eval uses single DAG)')
    parser.add_argument('--output_dir', type=str, default='results/replogle_validation')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load checkpoint + model ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading Norman-trained model")
    print("=" * 60)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    target_gene_names = ckpt['gene_names']
    n_genes = len(target_gene_names)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"  Norman HVGs: {n_genes}")

    cfg_m = config['model']
    model = CausalVAE(
        n_genes=n_genes, latent_dim=cfg_m['latent_dim'],
        tau_init=cfg_m['tau_min'], n_dag_samples=cfg_m['n_dag_samples'],
        dagma_s=cfg_m['dagma_s'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ── Step 2: Load Replogle aligned to Norman gene axis ───────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading Replogle (aligned to Norman HVGs)")
    print("=" * 60)
    data_dict = load_replogle_aligned(
        args.replogle_path, target_gene_names, max_cells=args.max_cells)

    gene_names = data_dict['gene_names']
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    X = data_dict['adata'].X  # [n_cells, n_genes] — aligned
    X_ctrl = X[data_dict['control_idx']]

    # ── Step 3: Compute observed DE on Replogle ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Computing observed DE on Replogle")
    print("=" * 60)
    de_single, _, ctrl_mean = compute_observed_de(X, data_dict, gene_names)
    # Drop any de_single entries without enough DEGs
    print(f"  Replogle perturbations with DE: {len(de_single)}")

    # ── Step 4: Baselines ───────────────────────────────────────────────
    # Two correlation baselines for thorough comparison:
    #   (1) Norman correlation: fair "extracted structure transfer" comparison.
    #       Both DAG and correlation derived from Norman, tested on Replogle.
    #   (2) Replogle correlation: stronger test — does training pay off vs.
    #       cheap test-time correlation that uses Replogle's own control cells?
    print("\n" + "=" * 60)
    print("STEP 4: Computing baselines")
    print("=" * 60)

    # (1) Norman correlation baseline
    print("  [1/3] Loading Norman to compute Norman-corr baseline...")
    from data.load_norman import load_norman
    norman_dict = load_norman(
        path=config['data']['norman_path'],
        n_hvg=config['data']['n_hvg'],
        train_frac=config['data'].get('single_pert_train_frac', 0.8),
        val_frac=config['data'].get('single_pert_val_frac', 0.1),
        split_seed=config['data'].get('split_seed', 42),
    )
    norman_X = norman_dict['adata'].X
    if hasattr(norman_X, 'toarray'):
        norman_X = norman_X.toarray()
    norman_X = np.asarray(norman_X, dtype=np.float32)
    norman_X_ctrl = norman_X[norman_dict['control_idx']]

    # Sanity: Norman gene order must match target_gene_names (checkpoint stores Norman's order)
    assert list(norman_dict['gene_names']) == list(target_gene_names), \
        "Norman gene order does not match checkpoint — re-train or check seed"

    corr_norman = compute_correlation_matrix(norman_X_ctrl, gene_names)
    del norman_X, norman_X_ctrl, norman_dict

    # (2) Replogle correlation baseline (uses Replogle control cells — "cheap test-time" baseline)
    print("  [2/3] Computing Replogle-corr baseline (test-time correlation)...")
    corr_replogle = compute_correlation_matrix(X_ctrl, gene_names)

    # (3) GRN from learned DAG (Norman-trained)
    print("  [3/3] Computing GRN baseline (D@A@D^T)...")
    grn_matrix = compute_grn_matrix(model, device)

    # ── Step 5: Evaluate ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Zero-shot model evaluation on Replogle")
    print("=" * 60)
    model_res = evaluate_single_perts(
        model, X, data_dict, de_single, gene_names, ctrl_mean,
        device, n_dag_samples=args.n_dag_samples, n_ctrl_cells=50)

    corr_norman_res = evaluate_baseline(corr_norman, gene_to_idx, de_single, gene_names)
    corr_replogle_res = evaluate_baseline(corr_replogle, gene_to_idx, de_single, gene_names)
    grn_res = evaluate_baseline(grn_matrix, gene_to_idx, de_single, gene_names)

    # ── Step 6: Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REPLOGLE EXTERNAL VALIDATION RESULTS")
    print("=" * 60)
    print(f"{'Method':>40s} {'AUROC':>10s} {'Spearman':>10s} {'N':>6s}")
    print(f"{'─' * 70}")
    for name, res in [
        ('Causal VAE (Norman-trained)',           model_res),
        ('Norman correlation (transfer baseline)', corr_norman_res),
        ('Replogle correlation (test-time baseline)', corr_replogle_res),
        ('Norman GRN (D@A@D^T)',                  grn_res),
    ]:
        aurocs = _extract(res, 'auroc')
        spears = _extract(res, 'spearman')
        print(f"{name:>40s} {_safe_mean(aurocs):>10.4f} {_safe_mean(spears):>10.4f} "
              f"{len(res):>6d}")

    print("\n  Norman corr  = both DAG and corr derived from Norman → fair transfer test")
    print("  Replogle corr = corr from test data's own controls → 'cheap test-time' baseline")

    # ── Step 7: Plots ───────────────────────────────────────────────────
    # Use Norman-corr as the primary baseline shown in the bar chart
    plot_single_pert_comparison(model_res, corr_norman_res, grn_res, args.output_dir,
                                 split_name='replogle_external')
    plot_per_pert_scatter(model_res, corr_norman_res, args.output_dir, 'replogle')
    plot_auroc_histogram(model_res, args.output_dir, 'replogle')

    # ── Step 8: Save CSVs ───────────────────────────────────────────────
    if model_res:
        pd.DataFrame(model_res).to_csv(
            os.path.join(args.output_dir, 'replogle_model.csv'), index=False)
    if corr_norman_res:
        pd.DataFrame(corr_norman_res).to_csv(
            os.path.join(args.output_dir, 'replogle_corr_norman.csv'), index=False)
    if corr_replogle_res:
        pd.DataFrame(corr_replogle_res).to_csv(
            os.path.join(args.output_dir, 'replogle_corr_replogle.csv'), index=False)

    summary = {
        'n_perturbations': len(model_res),
        'model_auroc': _safe_mean(_extract(model_res, 'auroc')),
        'model_spearman': _safe_mean(_extract(model_res, 'spearman')),
        'model_direction_acc': _safe_mean(_extract(model_res, 'direction_acc')),
        'corr_norman_auroc': _safe_mean(_extract(corr_norman_res, 'auroc')),
        'corr_norman_spearman': _safe_mean(_extract(corr_norman_res, 'spearman')),
        'corr_replogle_auroc': _safe_mean(_extract(corr_replogle_res, 'auroc')),
        'corr_replogle_spearman': _safe_mean(_extract(corr_replogle_res, 'spearman')),
        'grn_auroc': _safe_mean(_extract(grn_res, 'auroc')),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, 'summary.csv'), index=False)

    print(f"\nResults saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()

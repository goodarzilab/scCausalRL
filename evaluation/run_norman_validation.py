"""
External validation: evaluate Replogle-trained CausalVAE on Norman K562 perturbations.

This is the mirror of run_replogle_validation.py — same cross-modality generalization
test but in the opposite direction:
- Same cell line (K562), different modality (CRISPRi training → CRISPRa eval)
- Gene axis aligned to Replogle's HVGs (from checkpoint)
- Both single AND combo perturbations evaluated (combos are Norman's strength)

Three correlation baselines for fair comparison:
  1. Replogle correlation (transfer baseline) — from training data
  2. Norman correlation (test-time baseline) — uses Norman control cells

Usage:
    python evaluation/run_norman_validation.py \
        --checkpoint checkpoints/phase1_best.pt \
        --norman_path /large_storage/goodarzilab/shervin/scCausalRL/Norman/
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
from data.load_norman_aligned import load_norman_aligned

# Reuse existing eval functions
from evaluation.run_perturbation_eval import (
    compute_observed_de, compute_correlation_matrix, compute_grn_matrix,
    evaluate_baseline, evaluate_single_perts, evaluate_combo_perts,
    plot_combo_pert_comparison, plot_per_pert_scatter,
    _safe_mean, _safe_sem, _extract,
    COLORS, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
)


def plot_single_with_dual_corr(model_results, corr_replogle_results,
                                corr_norman_results, grn_results, out_dir,
                                split_name='norman_external'):
    """Bar chart with BOTH correlation baselines side-by-side.

    Replogle correlation = in-distribution (from training data) — fair transfer baseline
    Norman correlation = out-of-distribution (from test data's own controls) — cheap test-time baseline
    """
    methods = {}
    if model_results:
        methods['Causal VAE\n(Replogle-trained)'] = {
            'auroc': _extract(model_results, 'auroc'),
            'spearman': _extract(model_results, 'spearman'),
            'color': COLORS['model']}
    if corr_replogle_results:
        methods['Replogle corr\n(in-distribution)'] = {
            'auroc': _extract(corr_replogle_results, 'auroc'),
            'spearman': _extract(corr_replogle_results, 'spearman'),
            'color': COLORS['corr']}
    if corr_norman_results:
        methods['Norman corr\n(out-of-distribution)'] = {
            'auroc': _extract(corr_norman_results, 'auroc'),
            'spearman': _extract(corr_norman_results, 'spearman'),
            'color': '#666666'}  # darker grey to distinguish from in-dist corr
    if grn_results:
        methods['Learned GRN\n(D@A@D^T)'] = {
            'auroc': _extract(grn_results, 'auroc'),
            'spearman': _extract(grn_results, 'spearman'),
            'color': COLORS['grn']}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, ylabel, random_val, title in [
        (axes[0], 'auroc', 'AUROC', 0.5, 'DEG Detection (AUROC)'),
        (axes[1], 'spearman', 'Spearman ρ', 0.0, 'Effect Size Ranking (Spearman)'),
    ]:
        names, means, sems, colors = [], [], [], []
        for name, mdata in methods.items():
            vals = mdata.get(metric, [])
            if vals:
                names.append(name)
                means.append(_safe_mean(vals))
                sems.append(_safe_sem(vals))
                colors.append(mdata['color'])
        if not names:
            ax.set_visible(False)
            continue
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=sems, color=colors, edgecolor='black',
                      capsize=5, width=0.6, linewidth=1.2)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=TICK_SIZE, fontweight='bold')
        ax.axhline(random_val, color='red', ls='--', alpha=0.5, lw=1.5, label='Random')
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=TICK_SIZE - 1)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=LABEL_SIZE, fontweight='bold')
        ax.legend(fontsize=LEGEND_SIZE - 2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    n = len(model_results) if model_results else 0
    fig.suptitle(f'Single Perturbation Prediction — Norman External ({n} genes)\n',
                 fontsize=TITLE_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'single_pert_comparison_{split_name}.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_auroc_histogram_with_refs(model_results, corr_replogle_results,
                                    corr_norman_results, out_dir,
                                    pert_type='norman_single'):
    """Histogram of model AUROCs with both correlation baselines as reference lines."""
    aurocs = _extract(model_results, 'auroc')
    if len(aurocs) < 5:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(aurocs, bins=25, color=COLORS['model'], edgecolor='black', alpha=0.7,
            label=f'Model (n={len(aurocs)})')

    # Reference lines for both correlation baselines
    ax.axvline(0.5, color='red', ls='--', lw=1.5, alpha=0.7, label='Random (0.5)')

    model_mean = float(np.mean(aurocs))
    ax.axvline(model_mean, color='navy', ls='-', lw=2.5,
               label=f'Model mean = {model_mean:.3f}')

    if corr_replogle_results:
        c_in = _extract(corr_replogle_results, 'auroc')
        if c_in:
            mean_in = float(np.mean(c_in))
            ax.axvline(mean_in, color=COLORS['corr'], ls='-.', lw=2.5,
                       label=f'Replogle corr (in-dist) = {mean_in:.3f}')
    if corr_norman_results:
        c_out = _extract(corr_norman_results, 'auroc')
        if c_out:
            mean_out = float(np.mean(c_out))
            ax.axvline(mean_out, color='#666666', ls=':', lw=2.5,
                       label=f'Norman corr (out-of-dist) = {mean_out:.3f}')

    ax.set(xlabel='AUROC', ylabel='Count',
           title=f'Per-Perturbation AUROC distribution ({pert_type})')
    ax.legend(fontsize=LEGEND_SIZE - 1, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{pert_type}_auroc_hist.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Norman External Validation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Replogle-trained CausalVAE checkpoint')
    parser.add_argument('--norman_path', type=str,
                        default='/large_storage/goodarzilab/shervin/scCausalRL/Norman/')
    parser.add_argument('--n_dag_samples', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='results/norman_validation')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config override (else uses config stored in checkpoint).')
    parser.add_argument('--u_clip_max', type=float, default=None,
                        help='Override intervention.u_clip_max from config (for u_max sweep).')
    parser.add_argument('--combo_mode', type=str, default=None,
                        choices=[None, 'parallel', 'sequential'],
                        help="Combo intervention mode for u_score; defaults to config or 'parallel'.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load checkpoint + model ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading Replogle-trained model")
    print("=" * 60)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    if args.config is not None:
        import yaml
        with open(args.config) as f:
            override_cfg = yaml.safe_load(f)
        # Merge intervention block from override (so we can swap scheme/u_clip_max/combo_mode
        # without retraining or rewriting the checkpoint).
        if 'intervention' in override_cfg:
            config = dict(config)
            config['intervention'] = override_cfg['intervention']
            print(f"  Overrode intervention block from {args.config}: {config['intervention']}")
    target_gene_names = ckpt['gene_names']
    n_genes = len(target_gene_names)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"  Replogle HVGs: {n_genes}")
    print(f"  Dataset trained on: {config.get('data', {}).get('dataset', '?')}")

    cfg_m = config['model']
    sd = ckpt['model_state_dict']
    n_batches = int(sd['encoder.batch_emb.weight'].shape[0]) if 'encoder.batch_emb.weight' in sd else 0
    n_modalities = int(sd['encoder.modality_emb.weight'].shape[0]) if 'encoder.modality_emb.weight' in sd else 0
    model = CausalVAE(
        n_genes=n_genes, latent_dim=cfg_m['latent_dim'],
        tau_init=cfg_m['tau_min'], n_dag_samples=cfg_m['n_dag_samples'],
        dagma_s=cfg_m['dagma_s'],
        n_modalities=n_modalities, n_batches=n_batches,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    # ── Step 2: Load Norman aligned to Replogle gene axis ───────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading Norman (aligned to Replogle HVGs)")
    print("=" * 60)
    data_dict = load_norman_aligned(args.norman_path, target_gene_names)

    gene_names = data_dict['gene_names']
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    X = data_dict['adata'].X  # [n_cells, n_genes] — aligned
    X_ctrl = X[data_dict['control_idx']]

    # ── Step 3: Observed DE on Norman ───────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Computing observed DE on Norman")
    print("=" * 60)
    de_single, de_combo, ctrl_mean = compute_observed_de(X, data_dict, gene_names)
    print(f"  Single perturbations with DE: {len(de_single)}")
    print(f"  Combo perturbations with DE:  {len(de_combo)}")

    # ── Step 4: Baselines ───────────────────────────────────────────────
    # (1) Replogle correlation: training data — fair transfer baseline
    # (2) Norman correlation: test data — "cheap test-time" baseline
    # (3) GRN from learned DAG (Replogle-trained)
    print("\n" + "=" * 60)
    print("STEP 4: Computing baselines")
    print("=" * 60)

    print("  [1/3] Loading Replogle to compute Replogle-corr baseline...")
    from data.load_replogle import load_replogle
    replogle_dict = load_replogle(
        path=config['data']['replogle_path'],
        n_hvg=config['data']['n_hvg'],
        cell_line=config['data'].get('replogle_cell_line', 'k562'),
        max_cells=config['data'].get('replogle_max_cells', 200000),
        train_frac=config['data'].get('single_pert_train_frac', 0.8),
        val_frac=config['data'].get('single_pert_val_frac', 0.1),
        split_seed=config['data'].get('split_seed', 42),
    )
    replogle_X = replogle_dict['adata'].X
    if hasattr(replogle_X, 'toarray'):
        replogle_X = replogle_X.toarray()
    replogle_X = np.asarray(replogle_X, dtype=np.float32)
    replogle_X_ctrl = replogle_X[replogle_dict['control_idx']]

    assert list(replogle_dict['gene_names']) == list(target_gene_names), \
        "Replogle gene order does not match checkpoint — check seed/preprocessing"

    corr_replogle = compute_correlation_matrix(replogle_X_ctrl, gene_names)
    del replogle_X, replogle_X_ctrl, replogle_dict

    print("  [2/3] Computing Norman-corr baseline (test-time correlation)...")
    corr_norman = compute_correlation_matrix(X_ctrl, gene_names)

    print("  [3/3] Computing GRN baseline (D@A@D^T)...")
    grn_matrix = compute_grn_matrix(model, device)

    # ── Step 5: Single-pert evaluation ──────────────────────────────────
    intv_cfg = (config.get('intervention') or {})
    scheme = intv_cfg.get('scheme', 'threshold')
    u_clip_max = args.u_clip_max if args.u_clip_max is not None else intv_cfg.get('u_clip_max', 1.0)
    combo_mode = args.combo_mode if args.combo_mode is not None else intv_cfg.get('combo_mode', 'parallel')

    print(f"\n{'=' * 60}")
    print(f"EVALUATING SINGLE PERTS (Replogle-trained → Norman, {len(de_single)} genes)")
    print(f"  scheme={scheme}  u_clip_max={u_clip_max}  combo_mode={combo_mode}")
    print(f"{'=' * 60}")
    model_single = evaluate_single_perts(
        model, X, data_dict, de_single, gene_names, ctrl_mean,
        device, n_dag_samples=args.n_dag_samples, n_ctrl_cells=50,
        scheme=scheme, u_clip_max=u_clip_max)

    corr_replogle_single = evaluate_baseline(corr_replogle, gene_to_idx, de_single, gene_names)
    corr_norman_single = evaluate_baseline(corr_norman, gene_to_idx, de_single, gene_names)
    grn_single = evaluate_baseline(grn_matrix, gene_to_idx, de_single, gene_names)

    print(f"\n  --- Single Perturbation Results ---")
    print(f"  {'Method':>40s} {'AUROC':>10s} {'Spearman':>10s} {'N':>6s}")
    print(f"  {'─' * 70}")
    for name, res in [
        ('Causal VAE (Replogle-trained)',          model_single),
        ('Replogle correlation (transfer)',         corr_replogle_single),
        ('Norman correlation (test-time)',          corr_norman_single),
        ('Replogle GRN (D@A@D^T)',                  grn_single),
    ]:
        aurocs = _extract(res, 'auroc')
        spears = _extract(res, 'spearman')
        print(f"  {name:>40s} {_safe_mean(aurocs):>10.4f} {_safe_mean(spears):>10.4f} "
              f"{len(res):>6d}")

    # ── Step 6: Combo evaluation (Norman has combos, Replogle doesn't) ──
    print(f"\n{'=' * 60}")
    print(f"EVALUATING COMBO PERTS (Norman held-out combos, {len(de_combo)} combos)")
    print(f"{'=' * 60}")

    if len(de_combo) > 0:
        model_combo, additive_combo = evaluate_combo_perts(
            model, X, data_dict, de_single, de_combo, gene_names, ctrl_mean,
            device, n_dag_samples=args.n_dag_samples,
            scheme=scheme, u_clip_max=u_clip_max, combo_mode=combo_mode)

        # Correlation baselines for combos
        corr_replogle_combo = []
        corr_norman_combo = []
        for (ga, gb), de_data in de_combo.items():
            if ga not in gene_to_idx or gb not in gene_to_idx:
                continue
            obs_sig = de_data['sig']
            n_sig = int(obs_sig.sum())
            for corr_mat, dest in [(corr_replogle, corr_replogle_combo),
                                    (corr_norman, corr_norman_combo)]:
                scores = np.maximum(corr_mat[gene_to_idx[ga], :], corr_mat[gene_to_idx[gb], :])
                r = {'perturbation': f'{ga}+{gb}', 'n_de': n_sig}
                if 3 <= n_sig < len(obs_sig):
                    try:
                        r['auroc'] = roc_auc_score(obs_sig.astype(int), scores)
                    except ValueError:
                        pass
                if len(scores) > 10:
                    rho, _ = spearmanr(scores, de_data['effect'])
                    r['spearman'] = rho
                dest.append(r)

        print(f"\n  --- Combo Perturbation Results ---")
        print(f"  {'Method':>40s} {'AUROC':>10s} {'Spearman':>10s} {'Pearson':>10s} {'N':>5s}")
        for name, res in [
            ('Causal VAE (Replogle-trained)',        model_combo),
            ('Additive (Norman de_single)',          additive_combo),
            ('Replogle correlation',                 corr_replogle_combo),
            ('Norman correlation',                   corr_norman_combo),
        ]:
            aurocs = _extract(res, 'auroc')
            spears = _extract(res, 'spearman')
            pears = _extract(res, 'pearson_deg')
            print(f"  {name:>40s} {_safe_mean(aurocs):>10.4f} {_safe_mean(spears):>10.4f} "
                  f"{_safe_mean(pears):>10.4f} {len(res):>5d}")
    else:
        model_combo, additive_combo = [], []
        corr_replogle_combo, corr_norman_combo = [], []

    # ── Step 7: Plots ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("GENERATING PLOTS")
    print("=" * 60)
    # Bar chart with BOTH correlation baselines (in-dist Replogle + out-of-dist Norman)
    plot_single_with_dual_corr(model_single, corr_replogle_single, corr_norman_single,
                                grn_single, args.output_dir, split_name='norman_external')
    # Scatter vs in-distribution correlation (the fair transfer baseline)
    plot_per_pert_scatter(model_single, corr_replogle_single, args.output_dir, 'norman')
    # Histogram with BOTH correlation baselines as reference lines
    plot_auroc_histogram_with_refs(model_single, corr_replogle_single, corr_norman_single,
                                    args.output_dir, pert_type='norman_single')

    if model_combo:
        plot_combo_pert_comparison(model_combo, additive_combo, corr_replogle_combo,
                                    args.output_dir)
        # Combo histogram with both corr baselines
        plot_auroc_histogram_with_refs(model_combo, corr_replogle_combo, corr_norman_combo,
                                        args.output_dir, pert_type='norman_combo')

    # ── Step 8: CSVs ────────────────────────────────────────────────────
    if model_single:
        pd.DataFrame(model_single).to_csv(
            os.path.join(args.output_dir, 'norman_single_model.csv'), index=False)
    if corr_replogle_single:
        pd.DataFrame(corr_replogle_single).to_csv(
            os.path.join(args.output_dir, 'norman_single_corr_replogle.csv'), index=False)
    if corr_norman_single:
        pd.DataFrame(corr_norman_single).to_csv(
            os.path.join(args.output_dir, 'norman_single_corr_norman.csv'), index=False)
    if model_combo:
        pd.DataFrame(model_combo).to_csv(
            os.path.join(args.output_dir, 'norman_combo_model.csv'), index=False)
    if additive_combo:
        pd.DataFrame(additive_combo).to_csv(
            os.path.join(args.output_dir, 'norman_combo_additive.csv'), index=False)

    summary = {
        'n_single_perts': len(model_single),
        'n_combo_perts': len(model_combo),
        'single_model_auroc': _safe_mean(_extract(model_single, 'auroc')),
        'single_model_spearman': _safe_mean(_extract(model_single, 'spearman')),
        'single_corr_replogle_auroc': _safe_mean(_extract(corr_replogle_single, 'auroc')),
        'single_corr_norman_auroc': _safe_mean(_extract(corr_norman_single, 'auroc')),
        'single_grn_auroc': _safe_mean(_extract(grn_single, 'auroc')),
        'combo_model_auroc': _safe_mean(_extract(model_combo, 'auroc')),
        'combo_model_pearson': _safe_mean(_extract(model_combo, 'pearson_deg')),
        'combo_additive_auroc': _safe_mean(_extract(additive_combo, 'auroc')),
        'combo_additive_pearson': _safe_mean(_extract(additive_combo, 'pearson_deg')),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(args.output_dir, 'summary.csv'), index=False)

    print(f"\nResults saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()

"""
Perturbation Prediction Evaluation for scCausalRL.

Evaluates the Causal VAE's ability to predict perturbation effects, comparing
against correlation and GRN baselines. Adapted from MultiOmics grid search
analysis for single-modal (gene expression) perturbation prediction.

Usage:
    python evaluation/run_perturbation_eval.py --checkpoint checkpoints/phase1_best.pt
    python evaluation/run_perturbation_eval.py --checkpoint checkpoints/phase1_epoch150.pt --n_dag_samples 64
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, rankdata, norm
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from models.causal_vae import CausalVAE
from evaluation.eval_combinatorial import predict_combo
from utils.metrics import pearson_on_degs, direction_accuracy, cosine_similarity, mse_on_degs

# ── Plot style ──────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 16, 13, 11, 11
LINE_WIDTH = 2.0
COLORS = {'model': '#2171b5', 'corr': '#bdbdbd', 'grn': '#fd8d3c',
          'additive': '#74c476', 'random': '#f0f0f0'}

MIN_CELLS = 5
DE_PVAL_THRESHOLD = 0.05  # Mann-Whitney U p-value threshold for DEG definition


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_data_and_model(config_path, checkpoint_path, device):
    """Load training data + checkpoint. Dispatches on `config.data.dataset`.

    For 'replogle': loads Replogle with the same args used during training, so the
    HVG axis and pert_split align with the checkpoint. Combo evaluation will be a
    no-op (Replogle has no combos).

    For 'norman' (default): loads Norman.
    """
    import yaml, scipy.sparse
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset = config.get('data', {}).get('dataset', 'norman').lower()
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
    else:
        from data.load_norman import load_norman
        data_dict = load_norman(
            path=config['data']['norman_path'], n_hvg=config['data']['n_hvg'])

    adata = data_dict['adata']
    gene_names = data_dict['gene_names']
    n_genes = len(gene_names)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_m = config['model']
    sd = ckpt['model_state_dict']
    # Infer n_batches / n_modalities from the saved state dict so the constructor
    # produces matching parameter shapes — config doesn't carry these values.
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
    print(f"Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    # CRITICAL: data gene order must match the checkpoint's HVG axis, otherwise
    # the decoder/D rows are wired to different genes than the data we evaluate
    # on. This typically diverges when the eval config uses a different
    # max_cells than training (HVG selection is variance-based).
    if 'gene_names' in ckpt and list(gene_names) != list(ckpt['gene_names']):
        n_mismatch = sum(1 for a, b in zip(gene_names, ckpt['gene_names']) if a != b)
        raise RuntimeError(
            f"gene_names do not match checkpoint ({n_mismatch} positions differ). "
            f"This usually means the eval config has different max_cells / n_hvg / "
            f"split_seed than training. Ensure the eval config matches the training "
            f"config exactly. Checkpoint epoch: {ckpt.get('epoch', '?')}, "
            f"n_genes data={len(gene_names)} vs ckpt={len(ckpt['gene_names'])}."
        )

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)

    return config, model, data_dict, X, gene_names, ckpt


# ═══════════════════════════════════════════════════════════════════════════
# OBSERVED DIFFERENTIAL EXPRESSION
# ═══════════════════════════════════════════════════════════════════════════
def fast_mannwhitney_pvals(X_pert, X_ctrl):
    """Vectorized approximate Mann-Whitney U p-values (normal approx)."""
    n1, n2 = X_pert.shape[0], X_ctrl.shape[0]
    X_all = np.vstack([X_pert, X_ctrl])
    ranks = np.apply_along_axis(rankdata, 0, X_all)
    R1 = ranks[:n1, :].sum(axis=0)
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return 2 * norm.sf(np.abs((U1 - mu) / (sigma + 1e-10)))


def compute_observed_de(X, data_dict, gene_names):
    """Compute observed DE for all single + combo perturbations."""
    control_idx = data_dict['control_idx']
    single_perts = data_dict['single_perts']
    combo_perts = data_dict['combo_perts']

    X_ctrl = X[control_idx]
    ctrl_mean = X_ctrl.mean(axis=0)

    de_single, de_combo = {}, {}
    ctrl_std = X_ctrl.std(0) + 1e-8

    # Single perturbations
    for gene, idxs in single_perts.items():
        if len(idxs) < MIN_CELLS:
            continue
        X_pert = X[idxs]
        lfc = X_pert.mean(0) - ctrl_mean
        pvals = fast_mannwhitney_pvals(X_pert, X_ctrl)
        pert_std = X_pert.std(0) + 1e-8
        effect = np.abs(lfc) / np.sqrt((ctrl_std**2 + pert_std**2) / 2)  # pooled effect size
        de_single[gene] = {
            'lfc': lfc,
            'effect': effect,
            'pvals': pvals,
            'sig': pvals < DE_PVAL_THRESHOLD,
            'n_cells': len(idxs),
        }

    # Combo perturbations
    for (ga, gb), idxs in combo_perts.items():
        if len(idxs) < MIN_CELLS:
            continue
        X_pert = X[idxs]
        lfc = X_pert.mean(0) - ctrl_mean
        pvals = fast_mannwhitney_pvals(X_pert, X_ctrl)
        pert_std = X_pert.std(0) + 1e-8
        effect = np.abs(lfc) / np.sqrt((ctrl_std**2 + pert_std**2) / 2)
        de_combo[(ga, gb)] = {
            'lfc': lfc,
            'effect': effect,
            'pvals': pvals,
            'sig': pvals < DE_PVAL_THRESHOLD,
            'n_cells': len(idxs),
        }

    print(f"  Single perturbations with DE: {len(de_single)}")
    print(f"  Combo perturbations with DE:  {len(de_combo)}")
    return de_single, de_combo, ctrl_mean


# ═══════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════
def compute_correlation_matrix(X_ctrl, gene_names):
    """Compute |corr(gene_i, gene_j)| from control cells."""
    print("  Computing gene-gene correlation from control cells...")
    corr = np.abs(np.corrcoef(X_ctrl.T))
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)
    return corr


def compute_grn_matrix(model, device):
    """Compute gene-space GRN: |G_feat| = |D @ A_mean @ D^T|."""
    print("  Computing GRN from learned DAG (D @ A @ D^T)...")
    with torch.no_grad():
        A_mean = model.dag.get_edge_probs()
        G_feat = model.decoder.project_dag_to_genes(A_mean).cpu().numpy()
    G_abs = np.abs(G_feat)
    np.fill_diagonal(G_abs, 0)
    return G_abs


def evaluate_baseline(score_matrix, gene_to_idx, de_results, gene_names, exclude_self=True):
    """Evaluate a score matrix (corr or GRN) as a perturbation predictor.
    For each perturbed gene g, use score_matrix[g, :] to predict which genes change."""
    results = []
    for pert_gene, de_data in de_results.items():
        if pert_gene not in gene_to_idx:
            continue
        gi = gene_to_idx[pert_gene]
        scores = score_matrix[gi, :].copy()
        obs_effect = de_data['effect']
        obs_sig = de_data['sig']

        # Exclude self
        if exclude_self:
            mask = np.ones(len(gene_names), dtype=bool)
            mask[gi] = False
            scores = scores[mask]
            obs_effect = obs_effect[mask]
            obs_sig = obs_sig[mask]

        r = {'perturbation': pert_gene, 'n_de': int(obs_sig.sum())}

        # Spearman: rank correlation between predicted importance and actual effect
        if len(scores) > 10:
            rho, _ = spearmanr(scores, obs_effect)
            r['spearman'] = rho

        # AUROC: can the score distinguish DEGs from non-DEGs?
        n_sig = int(obs_sig.sum())
        if 3 <= n_sig < len(obs_sig):
            try:
                r['auroc'] = roc_auc_score(obs_sig.astype(int), scores)
            except ValueError:
                pass

        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MODEL-BASED PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_single_perts(model, X, data_dict, de_single, gene_names, ctrl_mean,
                          device, n_dag_samples=32, n_ctrl_cells=50,
                          scheme='threshold', u_clip_max=1.0):
    """
    Predict single perturbation effects.

    scheme='threshold' (legacy) — multi-column mutilate via |dε|>thr·max|dε|.
    scheme='u_score'             — pin program i with u_i = L[gene, i] (single-step,
                                    u_prev=0); see docs/architecture_u_score.md.
    """
    control_idx = np.array(data_dict['control_idx'])
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    rng = np.random.default_rng(42)
    ctrl_sample = rng.choice(len(control_idx), min(n_ctrl_cells, len(control_idx)), replace=False)
    x_ctrl_t = torch.tensor(X[control_idx[ctrl_sample]], dtype=torch.float32, device=device)

    # Encode control cells once
    with torch.no_grad():
        _, _, eps_ctrl = model.encoder(x_ctrl_t)
        # Reconstruct control for baseline
        A_single = model.dag.sample_A(n_samples=1).squeeze(0)
        z_ctrl = model.dag.scm_forward(eps_ctrl, A_single)
        x_hat_ctrl = model.decoder(z_ctrl).cpu().numpy()
    ctrl_recon_mean = x_hat_ctrl.mean(axis=0)

    results = []
    de_single_adata = data_dict['adata'].uns.get('de_single', {})

    for pert_gene, de_data in tqdm(de_single.items(), desc='Single perts'):
        if pert_gene not in gene_to_idx:
            continue
        gi = gene_to_idx[pert_gene]

        # Get dx (precomputed DE from full dataset)
        dx_np = de_single_adata.get(pert_gene, de_data['lfc'])
        dx_t = torch.tensor(np.asarray(dx_np, dtype=np.float32), device=device)

        with torch.no_grad():
            # Single deterministic DAG — no bootstrapping for evaluation
            A_eval = model.dag.sample_A(n_samples=1).squeeze(0)
            if scheme == 'u_score':
                # Single-step: u_prev=0 ⇒ u = L[gene, :]
                z_prev = model.dag.scm_forward(eps_ctrl, A_eval)
                d = model.latent_dim
                u_prev = torch.zeros(d, device=device)
                z_post, _u, _Atilde, _ztgt, _dz = model.intervene_gene_u(
                    eps_ctrl, dx_t, A_eval,
                    u_prev=u_prev, z_prev=z_prev,
                    gene_idx=gi, u_clip_max=u_clip_max,
                )
            else:
                z_post, _d_eps = model.intervene_gene(eps_ctrl, dx_t, A_eval, threshold_frac=0.1)
            x_hat_pert = model.decoder(z_post).cpu().numpy()
            pred_mean = x_hat_pert.mean(axis=0)  # avg over ctrl cells

        pred_de = pred_mean - ctrl_recon_mean
        obs_lfc = de_data['lfc']
        obs_effect = de_data['effect']
        obs_sig = de_data['sig']

        # Exclude perturbed gene itself
        mask = np.ones(len(gene_names), dtype=bool)
        mask[gi] = False

        pred_abs = np.abs(pred_de[mask])
        pred_signed = pred_de[mask]
        obs_eff_m = obs_effect[mask]
        obs_lfc_m = obs_lfc[mask]
        obs_sig_m = obs_sig[mask]

        r = {'perturbation': pert_gene, 'n_cells': de_data['n_cells'],
             'n_de': int(obs_sig_m.sum())}

        # Spearman: |predicted effect| vs |observed effect|
        if len(pred_abs) > 10:
            rho, _ = spearmanr(pred_abs, obs_eff_m)
            r['spearman'] = rho

        # Signed Spearman: predicted signed effect vs observed signed effect
        if len(pred_signed) > 10:
            rho_s, _ = spearmanr(pred_signed, obs_lfc_m)
            r['spearman_signed'] = rho_s

        # AUROC: detect DEGs
        n_sig = int(obs_sig_m.sum())
        if 3 <= n_sig < len(obs_sig_m):
            try:
                r['auroc'] = roc_auc_score(obs_sig_m.astype(int), pred_abs)
            except ValueError:
                pass

        # Direction accuracy on DEGs
        if n_sig >= 3:
            dir_agree = np.sign(pred_signed[obs_sig_m]) == np.sign(obs_lfc_m[obs_sig_m])
            r['direction_acc'] = float(dir_agree.mean())

        # Pearson on DEGs
        r['pearson_deg'] = pearson_on_degs(pred_de, obs_lfc, threshold=0.05)

        # Uncertainty: mean std across genes
        # No uncertainty for deterministic eval (bootstrapping is for RL only)

        results.append(r)

    return results


def evaluate_combo_perts(model, X, data_dict, de_single, de_combo, gene_names,
                         ctrl_mean, device, n_dag_samples=32, n_ctrl_cells=20,
                         scheme='threshold', u_clip_max=1.0,
                         combo_mode='parallel'):
    """Predict held-out combo perturbation effects.

    scheme='threshold' (legacy) | 'u_score' (decoder-loading).
    combo_mode='parallel' (default for u_score, AB ≡ BA by construction)
              | 'sequential' (legacy do(A) → recover ε → do(B)).
    """
    control_idx = np.array(data_dict['control_idx'])
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    de_single_adata = data_dict['adata'].uns.get('de_single', {})

    rng = np.random.default_rng(42)
    ctrl_sample = rng.choice(len(control_idx), min(n_ctrl_cells, len(control_idx)), replace=False)
    x_ctrl_t = torch.tensor(X[control_idx[ctrl_sample]], dtype=torch.float32, device=device)

    model_results, additive_results = [], []

    for (ga, gb), de_data in tqdm(de_combo.items(), desc='Combo perts'):
        if ga not in gene_to_idx or gb not in gene_to_idx:
            continue

        ga_idx, gb_idx = gene_to_idx[ga], gene_to_idx[gb]
        obs_lfc = de_data['lfc']
        obs_effect = de_data['effect']
        obs_sig = de_data['sig']

        # D† combo prediction
        dx_a = de_single_adata.get(ga, de_single.get(ga, {}).get('lfc'))
        dx_b = de_single_adata.get(gb, de_single.get(gb, {}).get('lfc'))
        dx_a_t = torch.tensor(np.asarray(dx_a, dtype=np.float32), device=device) if dx_a is not None else None
        dx_b_t = torch.tensor(np.asarray(dx_b, dtype=np.float32), device=device) if dx_b is not None else None

        with torch.no_grad():
            mean_AB, mean_BA, std_AB, std_BA = predict_combo(
                model, x_ctrl_t, ga_idx, gb_idx,
                dx_a=dx_a_t, dx_b=dx_b_t,
                n_samples=n_dag_samples, threshold_frac=0.1, device=device,
                scheme=scheme, u_clip_max=u_clip_max,
                combo_mode=combo_mode,
            )

        pred_mean = mean_AB.mean(axis=0)  # avg over ctrl cells
        pred_de = pred_mean - ctrl_mean

        r = {'combo': f'{ga}+{gb}', 'n_cells': de_data['n_cells'],
             'n_de': int(obs_sig.sum())}

        if len(pred_de) > 10:
            rho, _ = spearmanr(np.abs(pred_de), obs_effect)
            r['spearman'] = rho
            rho_s, _ = spearmanr(pred_de, obs_lfc)
            r['spearman_signed'] = rho_s

        n_sig = int(obs_sig.sum())
        if 3 <= n_sig < len(obs_sig):
            try:
                r['auroc'] = roc_auc_score(obs_sig.astype(int), np.abs(pred_de))
            except ValueError:
                pass

        if n_sig >= 3:
            dir_agree = np.sign(pred_de[obs_sig]) == np.sign(obs_lfc[obs_sig])
            r['direction_acc'] = float(dir_agree.mean())

        r['pearson_deg'] = pearson_on_degs(pred_de, obs_lfc, threshold=0.05)
        r['commutativity'] = float(np.mean((mean_AB.mean(0) - mean_BA.mean(0)) ** 2))

        model_results.append(r)

        # Additive baseline: ctrl + de_a + de_b
        eff_a = np.asarray(dx_a) if dx_a is not None else np.zeros(len(gene_names))
        eff_b = np.asarray(dx_b) if dx_b is not None else np.zeros(len(gene_names))
        add_pred = ctrl_mean + eff_a + eff_b
        add_de = add_pred - ctrl_mean  # = eff_a + eff_b

        ra = {'combo': f'{ga}+{gb}'}
        if len(add_de) > 10:
            rho, _ = spearmanr(np.abs(add_de), obs_effect)
            ra['spearman'] = rho
        if 3 <= n_sig < len(obs_sig):
            try:
                ra['auroc'] = roc_auc_score(obs_sig.astype(int), np.abs(add_de))
            except ValueError:
                pass
        ra['pearson_deg'] = pearson_on_degs(add_de, obs_lfc, threshold=0.05)
        if n_sig >= 3:
            dir_agree = np.sign(add_de[obs_sig]) == np.sign(obs_lfc[obs_sig])
            ra['direction_acc'] = float(dir_agree.mean())
        additive_results.append(ra)

    return model_results, additive_results


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════
def _safe_mean(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.mean(v) if v else 0.0

def _safe_sem(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0.0

def _extract(results, key):
    return [r[key] for r in results if key in r and not np.isnan(r.get(key, np.nan))]


def plot_single_pert_comparison(model_results, corr_results, grn_results, out_dir, split_name=''):
    """Bar chart: Model vs Correlation vs GRN for single perturbation prediction."""
    methods = {}
    if model_results:
        methods['Causal VAE\n(D† pipeline)'] = {
            'auroc': _extract(model_results, 'auroc'),
            'spearman': _extract(model_results, 'spearman'),
            'color': COLORS['model']}
    if corr_results:
        methods['Correlation\nBaseline'] = {
            'auroc': _extract(corr_results, 'auroc'),
            'spearman': _extract(corr_results, 'spearman'),
            'color': COLORS['corr']}
    if grn_results:
        methods['Learned GRN\n(D@A@D^T)'] = {
            'auroc': _extract(grn_results, 'auroc'),
            'spearman': _extract(grn_results, 'spearman'),
            'color': COLORS['grn']}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
                      capsize=5, width=0.55, linewidth=1.2)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=TICK_SIZE, fontweight='bold')
        ax.axhline(random_val, color='red', ls='--', alpha=0.5, lw=1.5, label='Random')
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=TICK_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=LABEL_SIZE, fontweight='bold')
        ax.legend(fontsize=LEGEND_SIZE-2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    n = len(model_results) if model_results else 0
    label = f' — {split_name}' if split_name else ''
    fig.suptitle(f'Single Perturbation Prediction ({n} genes{label})', fontsize=TITLE_SIZE, fontweight='bold')
    plt.tight_layout()
    suffix = f'_{split_name}' if split_name else ''
    plt.savefig(os.path.join(out_dir, f'single_pert_comparison{suffix}.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_combo_pert_comparison(model_results, additive_results, corr_results, out_dir):
    """Bar chart: Model vs Additive vs Correlation for combo prediction."""
    methods = {}
    if model_results:
        methods['Causal VAE\n(Sequential D†)'] = {
            'auroc': _extract(model_results, 'auroc'),
            'spearman': _extract(model_results, 'spearman'),
            'pearson_deg': _extract(model_results, 'pearson_deg'),
            'color': COLORS['model']}
    if additive_results:
        methods['Additive\nBaseline'] = {
            'auroc': _extract(additive_results, 'auroc'),
            'spearman': _extract(additive_results, 'spearman'),
            'pearson_deg': _extract(additive_results, 'pearson_deg'),
            'color': COLORS['additive']}
    if corr_results:
        methods['Correlation\nBaseline'] = {
            'auroc': _extract(corr_results, 'auroc'),
            'spearman': _extract(corr_results, 'spearman'),
            'color': COLORS['corr']}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, ylabel, random_val, title in [
        (axes[0], 'auroc', 'AUROC', 0.5, 'DEG Detection'),
        (axes[1], 'spearman', 'Spearman ρ', 0.0, 'Effect Ranking'),
        (axes[2], 'pearson_deg', 'Pearson r', 0.0, 'DE Correlation'),
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
                      capsize=5, width=0.55, linewidth=1.2)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=TICK_SIZE, fontweight='bold')
        ax.axhline(random_val, color='red', ls='--', alpha=0.5, lw=1.5)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=TICK_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=LABEL_SIZE, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    n = len(model_results) if model_results else 0
    fig.suptitle(f'Combo Perturbation Prediction — Held-Out ({n} combos)',
                 fontsize=TITLE_SIZE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'combo_pert_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_per_pert_scatter(model_results, corr_results, out_dir, pert_type='single'):
    """Scatter: model AUROC vs correlation AUROC per perturbation."""
    model_dict = {r['perturbation' if 'perturbation' in r else 'combo']: r
                  for r in model_results if 'auroc' in r}
    corr_dict = {r['perturbation']: r for r in corr_results if 'auroc' in r}

    common = set(model_dict.keys()) & set(corr_dict.keys())
    if len(common) < 5:
        return

    m_aurocs = [model_dict[k]['auroc'] for k in common]
    c_aurocs = [corr_dict[k]['auroc'] for k in common]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(c_aurocs, m_aurocs, s=25, alpha=0.6, color=COLORS['model'], edgecolor='k', lw=0.3)
    lims = [min(min(c_aurocs), min(m_aurocs)) - 0.05, max(max(c_aurocs), max(m_aurocs)) + 0.05]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.set(xlabel='Correlation Baseline AUROC', ylabel='Causal VAE AUROC',
           title=f'Per-Perturbation AUROC ({pert_type}, n={len(common)})')
    n_better = sum(1 for m, c in zip(m_aurocs, c_aurocs) if m > c)
    ax.text(0.05, 0.95, f'Model better: {n_better}/{len(common)}',
            transform=ax.transAxes, va='top', fontsize=TICK_SIZE)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{pert_type}_scatter_auroc.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_auroc_histogram(model_results, out_dir, pert_type='single'):
    """Histogram of per-perturbation AUROCs."""
    aurocs = _extract(model_results, 'auroc')
    if len(aurocs) < 5:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aurocs, bins=25, color=COLORS['model'], edgecolor='black', alpha=0.8)
    ax.axvline(0.5, color='red', ls='--', lw=1.5, label='Random (0.5)')
    ax.axvline(np.mean(aurocs), color='navy', ls='-', lw=2,
               label=f'Mean={np.mean(aurocs):.3f}')
    ax.set(xlabel='AUROC', ylabel='Count',
           title=f'Distribution of Per-Perturbation AUROC ({pert_type})')
    ax.legend(fontsize=LEGEND_SIZE)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{pert_type}_auroc_hist.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="scCausalRL Perturbation Evaluation")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase1_best.pt')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--n_dag_samples', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='results/perturbation_eval')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--u_clip_max', type=float, default=None,
                        help='Override intervention.u_clip_max from config (for u_max sweep).')
    parser.add_argument('--combo_mode', type=str, default=None,
                        choices=[None, 'parallel', 'sequential'],
                        help="Combo intervention mode for u_score; defaults to config value or 'parallel'.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Loading model and data")
    print("=" * 60)
    config, model, data_dict, X, gene_names, ckpt = load_data_and_model(
        args.config, args.checkpoint, device)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    X_ctrl = X[data_dict['control_idx']]

    # ── Step 2: Load split ────────────────────────────────────────────────
    pert_split = ckpt.get('pert_split', None)
    if pert_split is not None:
        # Convert lists back to sets (JSON serialization loses set type)
        pert_split = {k: set(v) for k, v in pert_split.items()}
        print(f"\nLoaded gene split from checkpoint: "
              f"{len(pert_split['train'])} train, {len(pert_split['val'])} val, "
              f"{len(pert_split['test'])} test")
    else:
        # No split in checkpoint — create one now (for older checkpoints)
        from data.load_norman import split_single_perts
        pert_split = split_single_perts(data_dict['single_perts'])
        print(f"\nNo split in checkpoint — created fresh: "
              f"{len(pert_split['train'])} train, {len(pert_split['val'])} val, "
              f"{len(pert_split['test'])} test")

    # ── Step 3: Observed DE ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Computing observed differential expression")
    print("=" * 60)
    de_single, de_combo, ctrl_mean = compute_observed_de(X, data_dict, gene_names)

    # ── Step 4: Baselines ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Computing baselines")
    print("=" * 60)
    corr_matrix = compute_correlation_matrix(X_ctrl, gene_names)
    grn_matrix = compute_grn_matrix(model, device)

    # ── Step 5: Evaluate by split ───────────────────────────────────────
    de_single_adata = data_dict['adata'].uns.get('de_single', {})

    all_single_results = {}
    for split_name, split_genes in [('train', pert_split['train']),
                                     ('test', pert_split['test'])]:
        de_split = {g: de for g, de in de_single.items() if g in split_genes}
        if not de_split:
            print(f"\n  [{split_name}] No genes with sufficient cells, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"EVALUATING SINGLE PERTS: {split_name.upper()} ({len(de_split)} genes)")
        print(f"{'=' * 60}")

        # Model predictions
        intv_cfg = config.get('intervention', {}) or {}
        scheme = intv_cfg.get('scheme', 'threshold')
        u_clip_max = args.u_clip_max if args.u_clip_max is not None else intv_cfg.get('u_clip_max', 1.0)
        model_res = evaluate_single_perts(
            model, X, data_dict, de_split, gene_names, ctrl_mean,
            device, n_dag_samples=args.n_dag_samples,
            scheme=scheme, u_clip_max=u_clip_max)

        # Baselines on same genes
        corr_res = evaluate_baseline(corr_matrix, gene_to_idx, de_split, gene_names)
        grn_res = evaluate_baseline(grn_matrix, gene_to_idx, de_split, gene_names)

        all_single_results[split_name] = {
            'model': model_res, 'corr': corr_res, 'grn': grn_res}

        print(f"\n  --- {split_name.upper()} Single Perturbation Results ---")
        print(f"  {'Method':>25s} {'AUROC':>10s} {'Spearman':>10s} {'N':>5s}")
        for name, res in [('Causal VAE (D†)', model_res),
                          ('Correlation', corr_res), ('GRN (D@A@D^T)', grn_res)]:
            aurocs = _extract(res, 'auroc')
            spears = _extract(res, 'spearman')
            print(f"  {name:>25s} {_safe_mean(aurocs):>10.4f} {_safe_mean(spears):>10.4f} {len(res):>5d}")

    # ── Step 6: Combo evaluation (both genes must be in train split) ────
    print(f"\n{'=' * 60}")
    print("EVALUATING COMBO PERTS (held-out, both genes in train split)")
    print("=" * 60)

    train_genes = pert_split['train']
    de_combo_valid = {k: v for k, v in de_combo.items()
                      if k[0] in train_genes and k[1] in train_genes}
    print(f"  Combos with both genes in train: {len(de_combo_valid)}/{len(de_combo)}")

    intv_cfg = config.get('intervention', {}) or {}
    scheme = intv_cfg.get('scheme', 'threshold')
    u_clip_max = args.u_clip_max if args.u_clip_max is not None else intv_cfg.get('u_clip_max', 1.0)
    combo_mode = args.combo_mode if args.combo_mode is not None else intv_cfg.get('combo_mode', 'parallel')
    print(f"  scheme={scheme}  u_clip_max={u_clip_max}  combo_mode={combo_mode}")
    model_combo, additive_combo = evaluate_combo_perts(
        model, X, data_dict, de_single, de_combo_valid, gene_names, ctrl_mean,
        device, n_dag_samples=args.n_dag_samples,
        scheme=scheme, u_clip_max=u_clip_max, combo_mode=combo_mode)

    # Correlation baseline for combos
    corr_combo = []
    for (ga, gb), de_data in de_combo_valid.items():
        if ga not in gene_to_idx or gb not in gene_to_idx:
            continue
        scores = np.maximum(corr_matrix[gene_to_idx[ga], :], corr_matrix[gene_to_idx[gb], :])
        obs_sig = de_data['sig']
        r = {'perturbation': f'{ga}+{gb}', 'n_de': int(obs_sig.sum())}
        n_sig = int(obs_sig.sum())
        if 3 <= n_sig < len(obs_sig):
            try:
                r['auroc'] = roc_auc_score(obs_sig.astype(int), scores)
            except ValueError:
                pass
        if len(scores) > 10:
            rho, _ = spearmanr(scores, de_data['effect'])
            r['spearman'] = rho
        corr_combo.append(r)

    print(f"\n  --- Combo Perturbation Results ---")
    print(f"  {'Method':>25s} {'AUROC':>10s} {'Spearman':>10s} {'Pearson':>10s} {'N':>5s}")
    for name, res in [('Causal VAE (D†)', model_combo),
                      ('Additive', additive_combo), ('Correlation', corr_combo)]:
        aurocs = _extract(res, 'auroc')
        spears = _extract(res, 'spearman')
        pears = _extract(res, 'pearson_deg')
        print(f"  {name:>25s} {_safe_mean(aurocs):>10.4f} {_safe_mean(spears):>10.4f} "
              f"{_safe_mean(pears):>10.4f} {len(res):>5d}")

    # ── Step 7: Epistasis stratification ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EPISTASIS STRATIFICATION")
    print("=" * 60)

    epistasis_scores = {}
    for (ga, gb), de_data in de_combo_valid.items():
        eff_a = np.asarray(de_single_adata.get(ga, np.zeros(n_genes)))
        eff_b = np.asarray(de_single_adata.get(gb, np.zeros(n_genes)))
        additive_pred = eff_a + eff_b
        observed = de_data['lfc']
        epistasis_scores[(ga, gb)] = float(np.mean((observed - additive_pred) ** 2))

    if model_combo and epistasis_scores:
        # Add epistasis to model results
        for r in model_combo:
            combo_key = tuple(r['combo'].split('+'))
            # Try both orderings
            epi = epistasis_scores.get(combo_key, epistasis_scores.get(combo_key[::-1], None))
            r['epistasis'] = epi

        for r in additive_combo:
            combo_key = tuple(r['combo'].split('+'))
            epi = epistasis_scores.get(combo_key, epistasis_scores.get(combo_key[::-1], None))
            r['epistasis'] = epi

        # Stratify into terciles
        epi_vals = sorted([r['epistasis'] for r in model_combo if r.get('epistasis') is not None])
        if len(epi_vals) >= 6:
            t1 = epi_vals[len(epi_vals) // 3]
            t2 = epi_vals[2 * len(epi_vals) // 3]

            print(f"  Epistasis tercile thresholds: low < {t1:.4f} < medium < {t2:.4f} < high")
            for label, lo, hi in [('Low', 0, t1), ('Medium', t1, t2), ('High', t2, float('inf'))]:
                m_sub = [r for r in model_combo
                         if r.get('epistasis') is not None and lo <= r['epistasis'] < hi]
                a_sub = [r for r in additive_combo
                         if r.get('epistasis') is not None and lo <= r['epistasis'] < hi]
                m_aur = _extract(m_sub, 'auroc')
                a_aur = _extract(a_sub, 'auroc')
                m_pear = _extract(m_sub, 'pearson_deg')
                a_pear = _extract(a_sub, 'pearson_deg')
                print(f"  {label:>8s} epistasis ({len(m_sub)} combos):")
                print(f"    Model AUROC={_safe_mean(m_aur):.4f}  Pearson={_safe_mean(m_pear):.4f}")
                print(f"    Additive AUROC={_safe_mean(a_aur):.4f}  Pearson={_safe_mean(a_pear):.4f}")

    # ── Step 8: Plots ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("GENERATING PLOTS")
    print("=" * 60)

    # Single pert plots — separate files for train vs test
    if 'test' in all_single_results:
        test_r = all_single_results['test']
        plot_single_pert_comparison(test_r['model'], test_r['corr'], test_r['grn'], args.output_dir, split_name='test')
        plot_per_pert_scatter(test_r['model'], test_r['corr'], args.output_dir, 'single_test')
        plot_auroc_histogram(test_r['model'], args.output_dir, 'single_test')

    if 'train' in all_single_results:
        train_r = all_single_results['train']
        plot_single_pert_comparison(train_r['model'], train_r['corr'], train_r['grn'], args.output_dir, split_name='train')

    # Combo plots
    plot_combo_pert_comparison(model_combo, additive_combo, corr_combo, args.output_dir)
    plot_auroc_histogram(model_combo, args.output_dir, 'combo')

    # Epistasis scatter: model vs additive pearson, colored by epistasis
    if model_combo and additive_combo:
        fig, ax = plt.subplots(figsize=(7, 6))
        m_pears = [r.get('pearson_deg', np.nan) for r in model_combo]
        a_pears = [r.get('pearson_deg', np.nan) for r in additive_combo]
        epis = [r.get('epistasis', 0) for r in model_combo]
        valid = [i for i in range(len(m_pears))
                 if not np.isnan(m_pears[i]) and not np.isnan(a_pears[i])]
        if len(valid) > 5:
            m_v = [m_pears[i] for i in valid]
            a_v = [a_pears[i] for i in valid]
            e_v = [epis[i] for i in valid]
            sc = ax.scatter(a_v, m_v, c=e_v, cmap='YlOrRd', s=30, alpha=0.7, edgecolor='k', lw=0.3)
            plt.colorbar(sc, ax=ax, label='Epistasis (MSE from additive)')
            lims = [min(min(a_v), min(m_v)) - 0.05, max(max(a_v), max(m_v)) + 0.05]
            ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
            ax.set(xlabel='Additive Baseline Pearson', ylabel='Causal VAE Pearson',
                   title='Combo: Model vs Additive (color=epistasis)')
            n_better = sum(1 for m, a in zip(m_v, a_v) if m > a)
            ax.text(0.05, 0.95, f'Model better: {n_better}/{len(valid)}',
                    transform=ax.transAxes, va='top', fontsize=TICK_SIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'combo_epistasis_scatter.png'),
                        dpi=200, bbox_inches='tight')
            plt.close()

    # ── Step 9: Save CSVs ──────────────────────────────────────────────
    for split_name, sdata in all_single_results.items():
        if sdata['model']:
            pd.DataFrame(sdata['model']).to_csv(
                os.path.join(args.output_dir, f'single_pert_model_{split_name}.csv'), index=False)
    if model_combo:
        pd.DataFrame(model_combo).to_csv(
            os.path.join(args.output_dir, 'combo_pert_model.csv'), index=False)
    if additive_combo:
        pd.DataFrame(additive_combo).to_csv(
            os.path.join(args.output_dir, 'combo_pert_additive.csv'), index=False)

    # Summary CSV
    summary = {}
    for split_name, sdata in all_single_results.items():
        summary[f'single_{split_name}_model_auroc'] = _safe_mean(_extract(sdata['model'], 'auroc'))
        summary[f'single_{split_name}_model_spearman'] = _safe_mean(_extract(sdata['model'], 'spearman'))
        summary[f'single_{split_name}_corr_auroc'] = _safe_mean(_extract(sdata['corr'], 'auroc'))
    summary['combo_model_auroc'] = _safe_mean(_extract(model_combo, 'auroc'))
    summary['combo_model_spearman'] = _safe_mean(_extract(model_combo, 'spearman'))
    summary['combo_additive_auroc'] = _safe_mean(_extract(additive_combo, 'auroc'))
    summary['combo_additive_pearson'] = _safe_mean(_extract(additive_combo, 'pearson_deg'))
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)

    print(f"\nResults saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()

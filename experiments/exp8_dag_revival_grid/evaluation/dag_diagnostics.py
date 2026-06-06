"""
Read-only DAG / Decoder / Mutilation diagnostics on an existing Phase 1 checkpoint.

Answers three questions WITHOUT retraining:
  1. How sparse/dense is the learned DAG?
  2. Is D^T D actually near identity (orthogonality working)?
  3. How many latent programs get mutilated for a typical perturbation,
     and is that number reasonable (proposal expects sparse, ~1-5 programs)?

Usage:
    python evaluation/dag_diagnostics.py \
        --checkpoint checkpoints/phase1_best.pt \
        --output_dir results/dag_diagnostics
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from models.causal_vae import CausalVAE


def load_model_and_data(checkpoint_path, device):
    """Load checkpoint + matching dataset for diagnostics."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    gene_names = ckpt['gene_names']
    n_genes = len(gene_names)
    pert_split = ckpt.get('pert_split', None)
    if pert_split is not None and isinstance(pert_split.get('train', None), list):
        pert_split = {k: set(v) for k, v in pert_split.items()}

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

    # Load matching dataset (for de_single)
    dataset_name = config['data'].get('dataset', 'norman')
    print(f"  Dataset: {dataset_name}, n_genes={n_genes}, latent_dim={cfg_m['latent_dim']}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")

    if dataset_name == 'norman':
        from data.load_norman import load_norman
        data_dict = load_norman(
            path=config['data']['norman_path'],
            n_hvg=config['data']['n_hvg'],
            train_frac=config['data'].get('single_pert_train_frac', 0.8),
            val_frac=config['data'].get('single_pert_val_frac', 0.1),
            split_seed=config['data'].get('split_seed', 42),
        )
    elif dataset_name == 'replogle':
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
    elif dataset_name == 'combined':
        from data.load_combined import load_combined
        data_dict = load_combined(
            norman_path=config['data']['norman_path'],
            replogle_path=config['data']['replogle_path'],
            n_hvg=config['data']['n_hvg'],
            replogle_max_cells=config['data'].get('replogle_max_cells', 100000),
            train_frac=config['data'].get('single_pert_train_frac', 0.8),
            val_frac=config['data'].get('single_pert_val_frac', 0.1),
            split_seed=config['data'].get('split_seed', 42),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Prefer pert_split saved in checkpoint (consistent with training)
    if pert_split is None:
        pert_split = data_dict.get('pert_split')

    return model, data_dict, pert_split, ckpt, dataset_name


# ═══════════════════════════════════════════════════════════════════════════
# 1) DAG / EDGE PROBABILITY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
def analyze_dag(model, out_dir):
    """Edge probability summary + histogram."""
    with torch.no_grad():
        edge_probs = model.dag.get_edge_probs().cpu().numpy()  # [d, d]
        theta = model.dag.theta.detach().cpu().numpy()
        tau = float(model.dag.tau.item())

    d = edge_probs.shape[0]
    n_edges_possible = d * (d - 1)  # diagonal is masked

    off_diag_mask = ~np.eye(d, dtype=bool)
    probs_off_diag = edge_probs[off_diag_mask]

    counts = {f'p > {t}': int((probs_off_diag > t).sum()) for t in [0.1, 0.3, 0.5, 0.7, 0.9]}

    summary = {
        'n_nodes': d,
        'tau': tau,
        'n_edges_possible': n_edges_possible,
        'mean_edge_prob': float(probs_off_diag.mean()),
        'median_edge_prob': float(np.median(probs_off_diag)),
        'max_edge_prob': float(probs_off_diag.max()),
        'std_edge_prob': float(probs_off_diag.std()),
        'theta_mean': float(theta.mean()),
        'theta_std': float(theta.std()),
        **counts,
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(probs_off_diag, bins=60, color='#2171b5', edgecolor='black', alpha=0.8)
    axes[0].axvline(0.5, color='red', ls='--', lw=1.5, label='p=0.5 (threshold)')
    axes[0].set(xlabel='Edge probability (off-diagonal)', ylabel='Count',
                title=f'Edge probabilities ({d} latent programs, tau={tau:.2f})')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    # Edge prob heatmap
    im = axes[1].imshow(edge_probs, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=axes[1], label='Edge probability')
    axes[1].set(title='Edge probability matrix A (i→j)',
                xlabel='target program j', ylabel='source program i')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'edge_prob_histogram.png'), dpi=200, bbox_inches='tight')
    plt.close()

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 2) ORTHOGONALITY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
def analyze_orthogonality(model, out_dir):
    """Compute D^T D / norm and report orthogonality metrics."""
    with torch.no_grad():
        D = model.decoder.D.detach().cpu()  # [n_genes, d]
        d = D.shape[1]
        n_genes = D.shape[0]
        DtD = (D.T @ D).numpy()  # [d, d]
        norm_sq = float((D ** 2).sum().item()) / d + 1e-8
        DtD_normalized = DtD / norm_sq  # should be ≈ I

        I = np.eye(d)
        diff = DtD_normalized - I
        ortho_loss = float((diff ** 2).sum())  # Frobenius norm squared

        off_diag_mask = ~np.eye(d, dtype=bool)
        off_diag_vals = DtD_normalized[off_diag_mask]
        diag_vals = np.diag(DtD_normalized)

        # Column norm uniformity (each column of D should have similar norm if orthogonal)
        col_norms = np.linalg.norm(D.numpy(), axis=0)

        # Decoder-sparsity diagnostics (added with lambda_D_sparse)
        D_np = D.numpy()
        D_abs = np.abs(D_np)
        D_l1 = float(D_abs.mean())                       # matches the penalty value
        # programs-per-gene: for each gene g, count programs with |D[g,i]|>0.01·max|D[g,:]|
        row_max = D_abs.max(axis=1, keepdims=True) + 1e-12
        programs_per_gene = (D_abs > 0.01 * row_max).sum(axis=1)
        # genes-per-program: symmetric
        col_max = D_abs.max(axis=0, keepdims=True) + 1e-12
        genes_per_program = (D_abs > 0.01 * col_max).sum(axis=0)

    summary = {
        'd_latent': d,
        'n_genes': D.shape[0],
        'ortho_loss_value': ortho_loss,
        'diag_mean': float(diag_vals.mean()),
        'diag_std': float(diag_vals.std()),
        'off_diag_mean_abs': float(np.abs(off_diag_vals).mean()),
        'off_diag_max_abs': float(np.abs(off_diag_vals).max()),
        'off_diag_std': float(off_diag_vals.std()),
        'col_norm_mean': float(col_norms.mean()),
        'col_norm_std': float(col_norms.std()),
        'col_norm_cv': float(col_norms.std() / (col_norms.mean() + 1e-8)),
        # Gene-level decoder-sparsity diagnostics
        'D_l1': D_l1,
        'mean_programs_per_gene': float(programs_per_gene.mean()),
        'median_programs_per_gene': float(np.median(programs_per_gene)),
        'mean_genes_per_program': float(genes_per_program.mean()),
        'median_genes_per_program': float(np.median(genes_per_program)),
    }

    # Plot heatmap of D^T D normalized
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(1.0, float(np.abs(DtD_normalized).max()))
    im = axes[0].imshow(DtD_normalized, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=axes[0])
    axes[0].set(title=f'D^T D normalized (should ≈ I)\northo loss = {ortho_loss:.4f}',
                xlabel='latent j', ylabel='latent i')

    axes[1].hist(off_diag_vals, bins=60, color='#fd8d3c', edgecolor='black', alpha=0.8,
                 label=f'Off-diagonal (target=0)\nmean(|·|)={summary["off_diag_mean_abs"]:.3f}')
    axes[1].hist(diag_vals, bins=20, color='#2171b5', edgecolor='black', alpha=0.7,
                 label=f'Diagonal (target=1)\nmean={summary["diag_mean"]:.3f}')
    axes[1].axvline(0, color='red', ls='--', lw=1, alpha=0.5)
    axes[1].axvline(1, color='red', ls='--', lw=1, alpha=0.5)
    axes[1].set(xlabel='Value', ylabel='Count', title='D^T D / norm value distribution')
    axes[1].legend()
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'decoder_DTD_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Separate sparsity histograms (programs/gene and genes/program)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(programs_per_gene, bins=min(40, d), color='#74c476',
                 edgecolor='black', alpha=0.85)
    axes[0].axvline(programs_per_gene.mean(), color='navy', ls='-', lw=2,
                    label=f'Mean={programs_per_gene.mean():.1f} / {d}')
    axes[0].axvline(d, color='red', ls='--', lw=1.5, label=f'Dense (d={d})')
    axes[0].set(xlabel='# programs with |D[g,i]| > 0.01·max',
                ylabel='# genes',
                title=f'programs_per_gene  (n_genes={n_genes})')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    axes[1].hist(genes_per_program, bins=min(40, n_genes // 10 if n_genes >= 200 else 20),
                 color='#fd8d3c', edgecolor='black', alpha=0.85)
    axes[1].axvline(genes_per_program.mean(), color='navy', ls='-', lw=2,
                    label=f'Mean={genes_per_program.mean():.0f}')
    axes[1].set(xlabel='# genes with |D[g,i]| > 0.01·max',
                ylabel='# programs', title=f'genes_per_program  (d={d})')
    axes[1].legend()
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'decoder_sparsity_hist.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 3) MUTILATION DIAGNOSTICS (PER-PERTURBATION)
# ═══════════════════════════════════════════════════════════════════════════
def analyze_mutilation(model, data_dict, pert_split, out_dir, n_dag_samples=8,
                       threshold_frac=0.1, device='cuda'):
    """For each test gene, compute mean_affected over N DAG samples + the threshold value."""
    de_single = data_dict['adata'].uns.get('de_single', {})
    if not de_single:
        print("  WARN: no de_single in data; skipping mutilation analysis")
        return {}

    # Choose gene set: test split if available, else all genes
    if pert_split is not None and len(pert_split.get('test', set())) > 0:
        # Handle namespaced keys: 'GENE__norman' vs plain 'GENE'
        # match keys whose un-namespaced gene name is in test split
        test_genes = pert_split['test']
        eval_keys = []
        for key in de_single.keys():
            gene_name = key.split('__')[0] if '__' in key else key
            if gene_name in test_genes:
                eval_keys.append(key)
        if not eval_keys:
            print(f"  WARN: no de_single keys match test split — using all {len(de_single)} genes")
            eval_keys = list(de_single.keys())
        else:
            print(f"  Test split contains {len(test_genes)} genes; {len(eval_keys)} have de_single")
    else:
        eval_keys = list(de_single.keys())
        print(f"  No test split; using all {len(eval_keys)} genes")

    d = model.latent_dim
    rows = []

    with torch.no_grad():
        # Sample N DAGs ONCE; reuse across all genes for comparability
        # Use train() mode for Gumbel diversity, then put back to eval
        model.dag.train()
        A_batch = model.dag.sample_A(n_samples=n_dag_samples)  # [N, d, d]
        model.dag.eval()
        I_mat = torch.eye(d, device=device)

        for key in eval_keys:
            dx_np = np.asarray(de_single[key], dtype=np.float32)
            dx_t = torch.tensor(dx_np, device=device)
            if dx_t.numel() != model.n_genes:
                # Combined dataset with different gene axis — skip
                continue

            dz = model.compute_dz_from_dx(dx_t)  # [d]
            dz_max_abs = float(dz.abs().max().item())
            dz_l1_over_l2 = float((dz.abs().sum() / (dz.norm(2) + 1e-8)).item())
            # Note: dz_l1_over_l2 ∈ [1, √d]; closer to 1 = sparser

            n_affected_per_sample = []
            threshold_values = []
            top_program_per_sample = []
            d_epsilon_max_abs = []
            d_epsilon_l1_over_l2 = []
            for n in range(n_dag_samples):
                A = A_batch[n]
                d_epsilon = (I_mat - A.T) @ dz  # [d]
                max_abs = float(d_epsilon.abs().max().item())
                threshold = threshold_frac * max_abs
                affected_mask = d_epsilon.abs() > threshold
                n_affected_per_sample.append(int(affected_mask.sum().item()))
                threshold_values.append(threshold)
                d_epsilon_max_abs.append(max_abs)
                d_epsilon_l1_over_l2.append(
                    float((d_epsilon.abs().sum() / (d_epsilon.norm(2) + 1e-8)).item())
                )
                top_program_per_sample.append(int(d_epsilon.abs().argmax().item()))

            n_aff_arr = np.array(n_affected_per_sample)
            top_counts = pd.Series(top_program_per_sample).value_counts().head(3).to_dict()
            top_3_str = ','.join([f'{k}({v})' for k, v in top_counts.items()])

            rows.append({
                'gene_key': key,
                'gene': key.split('__')[0] if '__' in key else key,
                'mean_affected': float(n_aff_arr.mean()),
                'std_affected': float(n_aff_arr.std()),
                'min_affected': int(n_aff_arr.min()),
                'max_affected': int(n_aff_arr.max()),
                'mean_threshold_value': float(np.mean(threshold_values)),
                'mean_max_abs_d_epsilon': float(np.mean(d_epsilon_max_abs)),
                'mean_d_epsilon_l1_over_l2': float(np.mean(d_epsilon_l1_over_l2)),
                'dz_max_abs': dz_max_abs,
                'dz_l1_over_l2': dz_l1_over_l2,
                'top_3_targets_with_counts': top_3_str,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'mutilation_per_pert.csv'), index=False)

    if len(df) > 0:
        # Histogram of mean_affected
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].hist(df['mean_affected'], bins=min(30, len(df)),
                     color='#fd8d3c', edgecolor='black', alpha=0.85)
        axes[0].axvline(d, color='red', ls='--', lw=1.5,
                        label=f'All programs (d={d})')
        axes[0].axvline(1, color='green', ls='--', lw=1.5,
                        label='Single program (ideal)')
        axes[0].axvline(df['mean_affected'].mean(), color='navy', ls='-', lw=2,
                        label=f'Mean={df["mean_affected"].mean():.1f}')
        axes[0].set(xlabel='mean # programs mutilated',
                    ylabel='# perturbations',
                    title=f'Mutilation count distribution (n={len(df)} test perts)')
        axes[0].legend()
        axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

        # Sparsity of dε (L1/L2 ratio): 1 = one-hot (sparse), √d = uniform (dense)
        sqrt_d = np.sqrt(d)
        axes[1].hist(df['mean_d_epsilon_l1_over_l2'], bins=min(30, len(df)),
                     color='#2171b5', edgecolor='black', alpha=0.85)
        axes[1].axvline(1.0, color='green', ls='--', lw=1.5,
                        label='one-hot (sparse, ideal)')
        axes[1].axvline(sqrt_d, color='red', ls='--', lw=1.5,
                        label=f'uniform (dense, √d={sqrt_d:.1f})')
        axes[1].axvline(df['mean_d_epsilon_l1_over_l2'].mean(), color='navy', ls='-', lw=2,
                        label=f'Mean={df["mean_d_epsilon_l1_over_l2"].mean():.2f}')
        axes[1].set(xlabel='||dε||₁ / ||dε||₂', ylabel='# perturbations',
                    title='dε sparsity (lower=sparser)')
        axes[1].legend()
        axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mutilation_n_affected_hist.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    if len(df) == 0:
        return {'n_perts': 0}

    summary = {
        'n_perts': len(df),
        'n_dag_samples': n_dag_samples,
        'threshold_frac': threshold_frac,
        'mean_n_affected': float(df['mean_affected'].mean()),
        'std_n_affected': float(df['mean_affected'].std()),
        'median_n_affected': float(df['mean_affected'].median()),
        'frac_perts_with_>10_affected': float((df['mean_affected'] > 10).mean()),
        'frac_perts_with_>30_affected': float((df['mean_affected'] > 30).mean()),
        'frac_perts_with_<=5_affected': float((df['mean_affected'] <= 5).mean()),
        'mean_threshold_value': float(df['mean_threshold_value'].mean()),
        'mean_max_abs_d_epsilon': float(df['mean_max_abs_d_epsilon'].mean()),
        'mean_d_epsilon_l1_over_l2': float(df['mean_d_epsilon_l1_over_l2'].mean()),
        'mean_dz_l1_over_l2': float(df['dz_l1_over_l2'].mean()),
    }
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 4) U-VECTOR (DECODER-LOADING) SPARSITY DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════
def analyze_u_sparsity(model, data_dict, pert_split, out_dir, device='cuda'):
    """
    For each test gene, compute u = L[g, :] under the u-score scheme and report
    sparsity. Single-gene perturbations should produce sparse u (few programs
    with appreciable mass).

    Reports:
      mean ||u||_0        — # programs with u_i > 0 (any mass)
      mean ||u||_0.5      — # programs with u_i > 0.5 (strongly intervened)
      mean ||u||_1        — total decoder-energy fraction allocated
      mean max(u)         — mass on the most-loaded program
      mean L1/L2 ratio    — 1 = one-hot, √d = uniform
    """
    de_single = data_dict['adata'].uns.get('de_single', {})
    gene_names = list(data_dict['adata'].var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    if not de_single:
        print("  WARN: no de_single; skipping u-vector analysis")
        return {}

    # Choose eval keys (test split if available)
    if pert_split is not None and len(pert_split.get('test', set())) > 0:
        test_genes = pert_split['test']
        eval_keys = []
        for key in de_single.keys():
            base = key.split('__')[0] if '__' in key else key
            if base in test_genes:
                eval_keys.append(key)
        if not eval_keys:
            eval_keys = list(de_single.keys())
    else:
        eval_keys = list(de_single.keys())

    d = model.latent_dim
    sqrt_d = float(np.sqrt(d))

    # Refresh L
    L, _T = model.compute_loadings()
    L_np = L.detach().cpu().numpy()  # [n_genes, d]

    rows = []
    for key in eval_keys:
        base = key.split('__')[0] if '__' in key else key
        if base not in gene_to_idx:
            continue
        gi = gene_to_idx[base]
        u = L_np[gi]  # [d]
        l1 = float(np.abs(u).sum())
        l2 = float(np.sqrt(np.square(u).sum()) + 1e-12)
        rows.append({
            'gene_key': key,
            'gene': base,
            'l0_any': int((u > 0).sum()),
            'l0_gt_0.1': int((u > 0.1).sum()),
            'l0_gt_0.5': int((u > 0.5).sum()),
            'l1': l1,
            'max_u': float(u.max()),
            'l1_over_l2': l1 / l2,
        })

    if not rows:
        print("  WARN: no eval keys for u-vector analysis")
        return {}

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'u_per_pert.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(df['l0_gt_0.5'], bins=min(30, len(df)),
                 color='#74c476', edgecolor='black', alpha=0.85)
    axes[0].axvline(df['l0_gt_0.5'].mean(), color='navy', ls='-', lw=2,
                    label=f'Mean={df["l0_gt_0.5"].mean():.1f}')
    axes[0].set(xlabel='# programs with u_i > 0.5',
                ylabel='# perturbations',
                title=f'u-vector strongly-intervened count (d={d})')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    axes[1].hist(df['l1_over_l2'], bins=min(30, len(df)),
                 color='#2171b5', edgecolor='black', alpha=0.85)
    axes[1].axvline(1.0, color='green', ls='--', lw=1.5, label='one-hot (sparse)')
    axes[1].axvline(sqrt_d, color='red', ls='--', lw=1.5, label=f'uniform (√d={sqrt_d:.1f})')
    axes[1].axvline(df['l1_over_l2'].mean(), color='navy', ls='-', lw=2,
                    label=f'Mean={df["l1_over_l2"].mean():.2f}')
    axes[1].set(xlabel='||u||₁ / ||u||₂', ylabel='# perturbations', title='u-vector sparsity')
    axes[1].legend()
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'u_sparsity_hist.png'), dpi=200, bbox_inches='tight')
    plt.close()

    summary = {
        'n_perts': len(df),
        'mean_l0_any': float(df['l0_any'].mean()),
        'mean_l0_gt_0.1': float(df['l0_gt_0.1'].mean()),
        'mean_l0_gt_0.5': float(df['l0_gt_0.5'].mean()),
        'mean_l1': float(df['l1'].mean()),
        'mean_max_u': float(df['max_u'].mean()),
        'mean_l1_over_l2': float(df['l1_over_l2'].mean()),
    }
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='DAG / mutilation diagnostics')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default='results/dag_diagnostics')
    parser.add_argument('--n_dag_samples', type=int, default=8)
    parser.add_argument('--threshold_frac', type=float, default=0.1)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    print("\n[1/3] Loading model + data...")
    model, data_dict, pert_split, ckpt, dataset_name = load_model_and_data(
        args.checkpoint, device)

    print("\n[2/3] Analyzing DAG structure + orthogonality...")
    dag_summary = analyze_dag(model, args.output_dir)
    ortho_summary = analyze_orthogonality(model, args.output_dir)

    print("\n[3/4] Analyzing mutilation per perturbation (threshold scheme)...")
    mut_summary = analyze_mutilation(model, data_dict, pert_split, args.output_dir,
                                      n_dag_samples=args.n_dag_samples,
                                      threshold_frac=args.threshold_frac,
                                      device=device)

    print("\n[4/4] Analyzing u-vector sparsity (u-score scheme)...")
    u_summary = analyze_u_sparsity(model, data_dict, pert_split, args.output_dir,
                                    device=device)

    # ── Summary ─────────────────────────────────────────────────────────
    d = dag_summary['n_nodes']
    print('\n' + '─' * 70)
    print(f' DAG / Mutilation Diagnostics — {os.path.basename(args.checkpoint)}')
    print('─' * 70)
    print(f' Dataset:                  {dataset_name}')
    print(f' Latent dim:               {d}')
    print(f' Epoch:                    {ckpt.get("epoch", "?")}')
    print('─' * 70)
    print(' DAG structure:')
    print(f'   tau:                    {dag_summary["tau"]:.4f}')
    print(f'   mean edge prob:         {dag_summary["mean_edge_prob"]:.4f}')
    print(f'   median edge prob:       {dag_summary["median_edge_prob"]:.4f}')
    print(f'   max edge prob:          {dag_summary["max_edge_prob"]:.4f}')
    print(f'   theta mean / std:       {dag_summary["theta_mean"]:.3f} / {dag_summary["theta_std"]:.3f}')
    print(f'   edges (p>0.1):          {dag_summary["p > 0.1"]} / {dag_summary["n_edges_possible"]}')
    print(f'   edges (p>0.3):          {dag_summary["p > 0.3"]}')
    print(f'   edges (p>0.5):          {dag_summary["p > 0.5"]}')
    print(f'   edges (p>0.7):          {dag_summary["p > 0.7"]}')
    print('─' * 70)
    print(' Orthogonality:')
    print(f'   ortho loss value:       {ortho_summary["ortho_loss_value"]:.4f}   (lower=better)')
    print(f'   diag mean (target=1):   {ortho_summary["diag_mean"]:.4f} ± {ortho_summary["diag_std"]:.4f}')
    print(f'   off-diag mean(|·|):     {ortho_summary["off_diag_mean_abs"]:.4f}   (target=0)')
    print(f'   off-diag max(|·|):      {ortho_summary["off_diag_max_abs"]:.4f}')
    print(f'   column-norm CV:         {ortho_summary["col_norm_cv"]:.4f}   (lower=more uniform)')
    print(f'   D_l1 (|D|.mean):        {ortho_summary["D_l1"]:.4f}')
    print(f'   mean programs/gene:     {ortho_summary["mean_programs_per_gene"]:.1f} / {d}   '
          f'(lower=sparser; biology target ~5–10)')
    print(f'   median programs/gene:   {ortho_summary["median_programs_per_gene"]:.0f}')
    print(f'   mean genes/program:     {ortho_summary["mean_genes_per_program"]:.0f}')
    print('─' * 70)
    if mut_summary.get('n_perts', 0) > 0:
        sqrt_d = np.sqrt(d)
        print(f' Mutilation (over {mut_summary["n_perts"]} test perts, N={mut_summary["n_dag_samples"]} DAG samples):')
        print(f'   threshold_frac:         {mut_summary["threshold_frac"]}')
        print(f'   mean n_affected:        {mut_summary["mean_n_affected"]:.1f} ± {mut_summary["std_n_affected"]:.1f}   (max = d = {d})')
        print(f'   median n_affected:      {mut_summary["median_n_affected"]:.1f}')
        print(f'   frac perts >10 affected: {mut_summary["frac_perts_with_>10_affected"]:.2%}')
        print(f'   frac perts >30 affected: {mut_summary["frac_perts_with_>30_affected"]:.2%}')
        print(f'   frac perts ≤5 affected:  {mut_summary["frac_perts_with_<=5_affected"]:.2%}')
        print(f'   mean max|dε|:           {mut_summary["mean_max_abs_d_epsilon"]:.4f}')
        print(f'   mean threshold value:   {mut_summary["mean_threshold_value"]:.4f}   (= frac · max|dε|)')
        print(f'   dε  L1/L2 (1=sparse, √d={sqrt_d:.1f}=dense): {mut_summary["mean_d_epsilon_l1_over_l2"]:.2f}')
        print(f'   dz  L1/L2 (1=sparse, √d={sqrt_d:.1f}=dense): {mut_summary["mean_dz_l1_over_l2"]:.2f}')
    print('─' * 70)
    print('\nInterpretation:')
    if mut_summary.get('mean_n_affected', 0) > d * 0.5:
        print(f'  ⚠️  Most programs mutilated per pert ({mut_summary["mean_n_affected"]:.0f}/{d}) — '
              f'DAG decomposition is degenerate (dε is dense, not sparse).')
    elif mut_summary.get('mean_n_affected', 0) > d * 0.25:
        print(f'  ⚠️  Many programs mutilated ({mut_summary["mean_n_affected"]:.0f}/{d}) — '
              f'DAG less localized than ideal.')
    elif mut_summary.get('mean_n_affected', 0) > 0:
        print(f'  ✓  Sparse mutilation ({mut_summary["mean_n_affected"]:.1f}/{d}) — '
              f'perturbations target a few latent programs as expected.')
    if u_summary.get('n_perts', 0) > 0:
        print('─' * 70)
        print(f' u-vector sparsity (over {u_summary["n_perts"]} test perts, single-gene u = L[g,:]):')
        print(f'   mean # programs (u>0):      {u_summary["mean_l0_any"]:.1f} / {d}')
        print(f'   mean # programs (u>0.1):    {u_summary["mean_l0_gt_0.1"]:.1f}')
        print(f'   mean # programs (u>0.5):    {u_summary["mean_l0_gt_0.5"]:.2f}')
        print(f'   mean ||u||₁ (Σ energy):     {u_summary["mean_l1"]:.3f}')
        print(f'   mean max(u):                {u_summary["mean_max_u"]:.3f}')
        sqrt_d = np.sqrt(d)
        print(f'   mean ||u||₁/||u||₂:         {u_summary["mean_l1_over_l2"]:.2f}   '
              f'(1=one-hot, √d={sqrt_d:.1f}=uniform)')
        print('─' * 70)

    if ortho_summary['off_diag_mean_abs'] > 0.3:
        print(f'  ⚠️  D is far from orthogonal (off-diag mean = {ortho_summary["off_diag_mean_abs"]:.3f}).')
    elif ortho_summary['off_diag_mean_abs'] > 0.1:
        print(f'  ◐  D is partially orthogonal (off-diag mean = {ortho_summary["off_diag_mean_abs"]:.3f}).')
    else:
        print(f'  ✓  D is nearly orthogonal (off-diag mean = {ortho_summary["off_diag_mean_abs"]:.3f}).')

    # Save summary
    combined = {
        'checkpoint': args.checkpoint,
        'dataset': dataset_name,
        **{f'dag/{k}': v for k, v in dag_summary.items()},
        **{f'ortho/{k}': v for k, v in ortho_summary.items()},
        **{f'mutilation/{k}': v for k, v in mut_summary.items()},
        **{f'u/{k}': v for k, v in u_summary.items()},
    }
    pd.DataFrame([combined]).to_csv(
        os.path.join(args.output_dir, 'diagnostics_summary.csv'), index=False)
    print(f'\nResults saved to {args.output_dir}/')


if __name__ == '__main__':
    main()

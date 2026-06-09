import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.causal_vae import CausalVAE
from utils.metrics import (
    mse_on_degs, pearson_correlation, cosine_similarity, 
    direction_accuracy, pearson_on_degs
)


def predict_combo(model, x_ctrl_tensor, gene_a_idx, gene_b_idx,
                  dx_a=None, dx_b=None,
                  pert_value=0.0, n_samples=16, threshold_frac=0.1, device='cpu',
                  scheme='threshold', u_clip_max=1.0,
                  combo_mode='parallel'):
    """
    Predict combination perturbation outcome.

    Three intervention schemes:
        scheme='threshold' (legacy) — do(A) via threshold multi-mutilate; ε recovered
                                       through the original (un-mutilated) graph between
                                       steps.  Always sequential.
        scheme='u_score' + combo_mode='parallel' (default) — single SCM solve with the
                                       SUMMED u and SUMMED dz of both perturbations.
                                       Order-invariant by construction (AB ≡ BA).
                                       Strict generalisation of the additive baseline;
                                       see plan §"The combo question".
        scheme='u_score' + combo_mode='sequential' — sequential do(A) → recover ε → do(B)
                                       (and vice versa for BA). Order-dependent.  Kept
                                       for ablation.

    Returns: (mean_pred_AB, mean_pred_BA, std_AB, std_BA)
    """
    model.eval()
    with torch.no_grad():
        mu, log_var, epsilon = model.encode(x_ctrl_tensor)
        # Deterministic DAG — use single hard-threshold A for eval
        A_single = model.dag.sample_A(n_samples=1).squeeze(0)  # [d, d]

        A = A_single

        if dx_a is not None and dx_b is not None and scheme == 'u_score' and combo_mode == 'parallel':
            # Parallel combo: single SCM solve with combined u + dz.
            # Commutativity is exact (AB === BA), so we return the same prediction twice.
            z0 = model.dag.scm_forward(epsilon, A)
            z_post, _u, _A_tilde = model.intervene_combo_u(
                epsilon,
                dx_list=[dx_a, dx_b],
                A=A,
                z_prev=z0,
                gene_indices=[gene_a_idx, gene_b_idx],
                u_clip_max=u_clip_max,
            )
            x_hat = model.decoder(z_post).cpu().numpy()
            x_hat_ab = x_hat
            x_hat_ba = x_hat
        elif dx_a is not None and dx_b is not None and scheme == 'u_score':
            # u-score SEQUENTIAL combo with persistence:
            #   step 1: u0 = 0,  z0 = SCM(ε)
            #   step 2: u1 carries forward and is added to L[gene_b, :]
            d = model.latent_dim
            B = epsilon.shape[0]
            u0 = torch.zeros(d, device=epsilon.device)
            z0 = model.dag.scm_forward(epsilon, A)
            # AB: do(A) → recover ε under u1 → do(B)
            z_after_a, u1_ab, _, _, _ = model.intervene_gene_u(
                epsilon, dx_a, A, u_prev=u0, z_prev=z0,
                gene_idx=gene_a_idx, u_clip_max=u_clip_max,
            )
            eps_after_a = model.recover_epsilon_u(z_after_a, A, u1_ab)
            z_after_ab, _u2_ab, _, _, _ = model.intervene_gene_u(
                eps_after_a, dx_b, A, u_prev=u1_ab, z_prev=z_after_a,
                gene_idx=gene_b_idx, u_clip_max=u_clip_max,
            )
            x_hat_ab = model.decoder(z_after_ab).cpu().numpy()

            # BA: do(B) → recover ε under u1 → do(A)
            z_after_b, u1_ba, _, _, _ = model.intervene_gene_u(
                epsilon, dx_b, A, u_prev=u0, z_prev=z0,
                gene_idx=gene_b_idx, u_clip_max=u_clip_max,
            )
            eps_after_b = model.recover_epsilon_u(z_after_b, A, u1_ba)
            z_after_ba, _u2_ba, _, _, _ = model.intervene_gene_u(
                eps_after_b, dx_a, A, u_prev=u1_ba, z_prev=z_after_b,
                gene_idx=gene_a_idx, u_clip_max=u_clip_max,
            )
            x_hat_ba = model.decoder(z_after_ba).cpu().numpy()
        elif dx_a is not None and dx_b is not None:
            # D† + multi-mutilate pipeline: do(A) → recover ε → do(B)
            z_after_a, _ = model.intervene_gene(epsilon, dx_a, A, threshold_frac)
            eps_after_a = model.recover_epsilon(z_after_a, A)
            z_after_ab, _ = model.intervene_gene(eps_after_a, dx_b, A, threshold_frac)
            x_hat_ab = model.decoder(z_after_ab).cpu().numpy()

            # Order BA: do(B) → recover ε → do(A)
            z_after_b, _ = model.intervene_gene(epsilon, dx_b, A, threshold_frac)
            eps_after_b = model.recover_epsilon(z_after_b, A)
            z_after_ba, _ = model.intervene_gene(eps_after_b, dx_a, A, threshold_frac)
            x_hat_ba = model.decoder(z_after_ba).cpu().numpy()
        else:
            # Legacy single-latent fallback
            latent_a = model._get_latent_for_gene(gene_a_idx)
            latent_b = model._get_latent_for_gene(gene_b_idx)

            z_after_a = model.dag.scm_intervene(epsilon, A, latent_a, pert_value)
            z_after_ab = model.dag.scm_intervene(z_after_a, A, latent_b, pert_value)
            x_hat_ab = model.decoder(z_after_ab).cpu().numpy()

            z_after_b = model.dag.scm_intervene(epsilon, A, latent_b, pert_value)
            z_after_ba = model.dag.scm_intervene(z_after_b, A, latent_a, pert_value)
            x_hat_ba = model.decoder(z_after_ba).cpu().numpy()

    return (
        x_hat_ab,   # [B, n_genes]
        x_hat_ba,   # [B, n_genes]
        np.zeros_like(x_hat_ab),  # no std for deterministic eval
        np.zeros_like(x_hat_ba),
    )


def additive_baseline(de_single, ctrl_mean, gene_a, gene_b):
    """Additive baseline: effect_A + effect_B + ctrl_mean."""
    effect_a = de_single.get(gene_a, np.zeros_like(ctrl_mean))
    effect_b = de_single.get(gene_b, np.zeros_like(ctrl_mean))
    return ctrl_mean + effect_a + effect_b


def eval_combinatorial(config, vae_path, data_dict=None, device=None, n_samples=32):
    """
    Evaluate combinatorial perturbation prediction.
    Train on singles, predict held-out combos.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating combinatorial prediction on {device}")
    
    # Load data
    if data_dict is None:
        from data.load_norman import load_norman
        data_dict = load_norman(
            path=config['data']['norman_path'],
            n_hvg=config['data']['n_hvg']
        )
    
    adata = data_dict['adata']
    gene_names = data_dict['gene_names']
    combo_perts = data_dict['combo_perts']
    control_idx = data_dict['control_idx']
    n_genes = len(gene_names)
    
    if len(combo_perts) == 0:
        print("No combo perturbations found. Skipping combinatorial evaluation.")
        return {}
    
    # Load model
    ckpt = torch.load(vae_path, map_location=device)
    cfg_model = config['model']
    model = CausalVAE(
        n_genes=n_genes,
        latent_dim=cfg_model['latent_dim'],
        tau_init=cfg_model['tau_min'],
        n_dag_samples=cfg_model['n_dag_samples'],
        dagma_s=cfg_model['dagma_s'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from {vae_path}")
    
    # Expression matrices
    import scipy.sparse
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)
    
    X_ctrl = X[control_idx]
    ctrl_mean = X_ctrl.mean(axis=0)
    de_single = adata.uns.get('de_single', {})
    de_combo = adata.uns.get('de_combo', {})
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Results storage
    results = {
        'mse_model_AB': [],
        'mse_model_BA': [],
        'mse_additive': [],
        'pearson_model_AB': [],
        'pearson_model_BA': [],
        'pearson_additive': [],
        'cosine_model_AB': [],
        'cosine_additive': [],
        'direction_acc_model': [],
        'direction_acc_additive': [],
        'commutativity_error': [],
        'combo_names': [],
    }
    
    print(f"\nEvaluating {len(combo_perts)} combo perturbations...")
    
    # Sample a representative control cell for predictions
    rng = np.random.default_rng(42)
    ctrl_sample_idx = rng.choice(len(control_idx), min(10, len(control_idx)), replace=False)
    X_ctrl_sample = X_ctrl[ctrl_sample_idx]
    x_ctrl_tensor = torch.tensor(X_ctrl_sample, dtype=torch.float32, device=device)
    
    for (gene_a, gene_b), pert_idxs in tqdm(combo_perts.items(), desc="Combo evaluation"):
        if gene_a not in gene_to_idx or gene_b not in gene_to_idx:
            continue
        if len(pert_idxs) == 0:
            continue
        
        gene_a_idx = gene_to_idx[gene_a]
        gene_b_idx = gene_to_idx[gene_b]
        
        # Real combo expression
        X_real_combo = X[pert_idxs].mean(axis=0)  # mean over perturbed cells
        
        # Model prediction — use D†-based targeting if de_single available
        dx_a_vec = de_single.get(gene_a, None)
        dx_b_vec = de_single.get(gene_b, None)
        dx_a_t = torch.tensor(dx_a_vec, dtype=torch.float32, device=device) if dx_a_vec is not None else None
        dx_b_t = torch.tensor(dx_b_vec, dtype=torch.float32, device=device) if dx_b_vec is not None else None
        intv_cfg = (config.get('intervention') or {}) if isinstance(config, dict) else {}
        mean_AB, mean_BA, std_AB, std_BA = predict_combo(
            model, x_ctrl_tensor, gene_a_idx, gene_b_idx,
            dx_a=dx_a_t, dx_b=dx_b_t,
            n_samples=n_samples, device=device,
            scheme=intv_cfg.get('scheme', 'threshold'),
            u_clip_max=intv_cfg.get('u_clip_max', 1.0),
        )
        pred_AB = mean_AB.mean(axis=0)  # average over control cells
        pred_BA = mean_BA.mean(axis=0)
        
        # Additive baseline
        pred_additive = additive_baseline(de_single, ctrl_mean, gene_a, gene_b)
        
        # Ground truth DE effect
        key = tuple(sorted([gene_a, gene_b]))
        true_de = de_combo.get(key, X_real_combo - ctrl_mean)
        
        # Metrics
        pred_de_AB = pred_AB - ctrl_mean
        pred_de_BA = pred_BA - ctrl_mean
        pred_de_add = pred_additive - ctrl_mean
        
        results['mse_model_AB'].append(mse_on_degs(pred_AB, X_real_combo, true_de))
        results['mse_model_BA'].append(mse_on_degs(pred_BA, X_real_combo, true_de))
        results['mse_additive'].append(mse_on_degs(pred_additive, X_real_combo, true_de))
        results['pearson_model_AB'].append(pearson_on_degs(pred_de_AB, true_de))
        results['pearson_model_BA'].append(pearson_on_degs(pred_de_BA, true_de))
        results['pearson_additive'].append(pearson_on_degs(pred_de_add, true_de))
        results['cosine_model_AB'].append(cosine_similarity(pred_de_AB, true_de))
        results['cosine_additive'].append(cosine_similarity(pred_de_add, true_de))
        results['direction_acc_model'].append(direction_accuracy(pred_de_AB, true_de))
        results['direction_acc_additive'].append(direction_accuracy(pred_de_add, true_de))
        results['commutativity_error'].append(np.mean((pred_AB - pred_BA) ** 2))
        results['combo_names'].append(f"{gene_a}+{gene_b}")
    
    # Aggregate
    summary = {}
    for k, v in results.items():
        if k != 'combo_names' and len(v) > 0:
            summary[f'{k}_mean'] = float(np.mean(v))
            summary[f'{k}_std'] = float(np.std(v))
    
    print("\n=== Combinatorial Evaluation Results ===")
    print(f"Number of combos evaluated: {len(results['combo_names'])}")
    print(f"\nModel (order A→B):")
    print(f"  MSE on DEGs:       {summary.get('mse_model_AB_mean', 0):.4f} ± {summary.get('mse_model_AB_std', 0):.4f}")
    print(f"  Pearson on DEGs:   {summary.get('pearson_model_AB_mean', 0):.4f} ± {summary.get('pearson_model_AB_std', 0):.4f}")
    print(f"  Cosine similarity: {summary.get('cosine_model_AB_mean', 0):.4f} ± {summary.get('cosine_model_AB_std', 0):.4f}")
    print(f"  Direction accuracy:{summary.get('direction_acc_model_mean', 0):.4f} ± {summary.get('direction_acc_model_std', 0):.4f}")
    print(f"\nAdditive baseline:")
    print(f"  MSE on DEGs:       {summary.get('mse_additive_mean', 0):.4f} ± {summary.get('mse_additive_std', 0):.4f}")
    print(f"  Pearson on DEGs:   {summary.get('pearson_additive_mean', 0):.4f} ± {summary.get('pearson_additive_std', 0):.4f}")
    print(f"\nCommutativity error (||pred_AB - pred_BA||²):")
    print(f"  {summary.get('commutativity_error_mean', 0):.6f} ± {summary.get('commutativity_error_std', 0):.6f}")
    
    summary['per_combo'] = results
    return summary

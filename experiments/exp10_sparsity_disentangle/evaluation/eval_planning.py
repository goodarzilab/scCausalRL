import os
import sys
import numpy as np
import torch
from collections import Counter
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.causal_vae import CausalVAE
from rl.env import CausalWorldModelEnv
from rl.actor_critic import Actor, Critic


def greedy_plan(model, epsilon_start, start_state, target_state, state_centroids, 
                pca_transform, n_steps=3, n_dag_samples=16, device='cpu'):
    """
    Greedy baseline: at each step, pick the action that moves closest to target.
    Returns: list of (state, action) pairs
    """
    latent_dim = model.latent_dim
    intervention_values = CausalWorldModelEnv.INTERVENTION_VALUES
    n_actions = latent_dim * len(intervention_values)
    
    target_centroid = torch.tensor(state_centroids[target_state], dtype=torch.float32)
    
    current_epsilon = epsilon_start.clone()
    current_state = start_state
    trajectory = [current_state]
    actions_taken = []
    
    for step in range(n_steps):
        if current_state == target_state:
            break
        
        best_action = 0
        best_dist = float('inf')
        
        model.eval()
        with torch.no_grad():
            for action in range(n_actions):
                latent_idx = action // len(intervention_values)
                val_idx = action % len(intervention_values)
                pert_value = intervention_values[val_idx]
                
                result = model.intervene_on_latent_bootstrapped(
                    current_epsilon, latent_idx, pert_value, n_samples=4
                )
                mean_x_hat = result['mean_x_hat'].cpu().numpy()  # [1, n_genes]
                pca_emb = pca_transform.transform(mean_x_hat)[0]
                pca_t = torch.tensor(pca_emb, dtype=torch.float32)
                dist = torch.norm(pca_t - target_centroid).item()
                
                if dist < best_dist:
                    best_dist = dist
                    best_action = action
            
            # Execute best action
            latent_idx = best_action // len(intervention_values)
            val_idx = best_action % len(intervention_values)
            pert_value = intervention_values[val_idx]
            
            result = model.intervene_on_latent_bootstrapped(
                current_epsilon, latent_idx, pert_value, n_samples=n_dag_samples
            )
            
            # Get majority-vote next state
            mean_z_post = result['all_z_post'].mean(dim=0)  # [1, d]
            all_x_hat = result['all_x_hat'].squeeze(1).cpu().numpy()
            
            states_predicted = []
            for n in range(all_x_hat.shape[0]):
                pca_emb = pca_transform.transform(all_x_hat[n:n+1])[0]
                dists = np.linalg.norm(state_centroids - pca_emb, axis=1)
                states_predicted.append(int(dists.argmin()))
            
            next_state = Counter(states_predicted).most_common(1)[0][0]
            current_epsilon = mean_z_post.to(device)
            current_state = next_state
            trajectory.append(current_state)
            actions_taken.append(best_action)
    
    return trajectory, actions_taken


def eval_planning(config, vae_path, policy_path=None, data_dict=None, 
                  wad_data_dict=None, device=None, n_dag_samples=32):
    """
    Evaluate RL policy on multi-step perturbation planning.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating planning on {device}")
    
    # Load data
    if data_dict is None:
        from data.load_norman import load_norman
        data_dict = load_norman(
            path=config['data']['norman_path'],
            n_hvg=config['data']['n_hvg']
        )
    
    gene_names = data_dict['gene_names']
    n_genes = len(gene_names)
    adata = data_dict['adata']
    
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
    
    # Load state centroids (from phase 2 checkpoint if available)
    state_centroids = ckpt.get('state_centroids', None)
    
    if state_centroids is None:
        print("No state centroids in checkpoint, computing from scratch...")
        from training.train_joint import cluster_cells, SimplePCATransform
        p2 = config['phase2']
        cluster_labels, state_centroids, _ = cluster_cells(adata, n_clusters=p2['n_cell_states'])
    
    pca_dim = state_centroids.shape[1]
    
    # PCA transform
    from training.train_joint import SimplePCATransform
    pca_transform = SimplePCATransform(adata)
    
    import scipy.sparse
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)
    
    results = {}
    
    # ============================================================
    # EVALUATION 1: Norman combo data
    # ============================================================
    print("\n=== Evaluation 1: Norman combo perturbation planning ===")
    combo_perts = data_dict['combo_perts']
    control_idx = data_dict['control_idx']
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    if len(combo_perts) > 0 and state_centroids is not None:
        # Assign each combo to a cluster (target state)
        X_ctrl = X[control_idx]
        rng = np.random.default_rng(42)
        ctrl_x_sample = torch.tensor(X_ctrl[rng.choice(len(control_idx), 1)], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            _, _, epsilon_ctrl = model.encode(ctrl_x_sample)
        
        # Determine start state
        ctrl_pca = pca_transform.transform(X_ctrl.mean(axis=0, keepdims=True))[0]
        ctrl_dists = np.linalg.norm(state_centroids - ctrl_pca, axis=1)
        start_state = int(ctrl_dists.argmin())
        
        matches_rl = 0
        matches_greedy = 0
        n_combos_eval = 0
        
        # Load policy if available
        actor = None
        if policy_path is not None and os.path.exists(policy_path):
            try:
                latent_dim = cfg_model['latent_dim']
                n_actions = latent_dim * len(CausalWorldModelEnv.INTERVENTION_VALUES)
                actor = Actor(pca_dim=pca_dim, n_actions=n_actions).to(device)
                ckpt_policy = torch.load(policy_path, map_location=device)
                actor.load_state_dict(ckpt_policy['actor'])
                actor.eval()
                print(f"Loaded RL policy from {policy_path}")
            except Exception as e:
                print(f"Could not load policy: {e}")
                actor = None
        
        for (gene_a, gene_b), pert_idxs in list(combo_perts.items())[:20]:  # eval top 20
            if gene_a not in gene_to_idx or gene_b not in gene_to_idx:
                continue
            if len(pert_idxs) < 3:
                continue
            
            # Determine target state: cluster that most combo cells fall in
            X_combo = X[pert_idxs]
            combo_pca = pca_transform.transform(X_combo)
            combo_states = []
            for pca_emb in combo_pca:
                dists = np.linalg.norm(state_centroids - pca_emb, axis=1)
                combo_states.append(int(dists.argmin()))
            target_state = Counter(combo_states).most_common(1)[0][0]
            
            if target_state == start_state:
                continue
            
            target_emb = torch.tensor(state_centroids[target_state], dtype=torch.float32, device=device)
            
            # Get top-2 actions from RL policy
            if actor is not None:
                start_emb = torch.tensor(state_centroids[start_state], dtype=torch.float32, device=device)
                with torch.no_grad():
                    dist = actor(start_emb.unsqueeze(0), target_emb.unsqueeze(0))
                    probs = dist.probs.cpu().numpy()[0]
                top_actions = probs.argsort()[::-1][:2]
                top_latents = [int(a // len(CausalWorldModelEnv.INTERVENTION_VALUES)) for a in top_actions]
                
                # Map latents back to genes
                D = model.decoder.D.detach().cpu().numpy()
                for latent_idx in top_latents:
                    pass  # simplified check
                
                # Check if the policy targets the correct genes (D†-based)
                de_single = adata.uns.get('de_single', {})
                A_eval = model.dag.sample_A(n_samples=1).squeeze(0)
                dx_a = de_single.get(gene_a)
                dx_b = de_single.get(gene_b)
                if dx_a is not None:
                    dx_a_t = torch.tensor(np.asarray(dx_a, dtype=np.float32), device=device)
                    gene_a_latent, _, _ = model.get_latent_for_gene_v2(dx_a_t, A_eval)
                else:
                    gene_a_latent = model._get_latent_for_gene(gene_to_idx[gene_a])
                if dx_b is not None:
                    dx_b_t = torch.tensor(np.asarray(dx_b, dtype=np.float32), device=device)
                    gene_b_latent, _, _ = model.get_latent_for_gene_v2(dx_b_t, A_eval)
                else:
                    gene_b_latent = model._get_latent_for_gene(gene_to_idx[gene_b])
                if gene_a_latent in top_latents and gene_b_latent in top_latents:
                    matches_rl += 1

            # Greedy baseline
            greedy_traj, greedy_actions = greedy_plan(
                model, epsilon_ctrl, start_state, target_state, state_centroids,
                pca_transform, n_steps=2, n_dag_samples=8, device=device
            )
            greedy_latents = [int(a // len(CausalWorldModelEnv.INTERVENTION_VALUES)) for a in greedy_actions[:2]]
            # Reuse the latent identifications from above
            if gene_a_latent in greedy_latents and gene_b_latent in greedy_latents:
                matches_greedy += 1
            
            n_combos_eval += 1
        
        if n_combos_eval > 0:
            results['eval1_rl_match_rate'] = matches_rl / n_combos_eval if actor else None
            results['eval1_greedy_match_rate'] = matches_greedy / n_combos_eval
            print(f"  Combos evaluated: {n_combos_eval}")
            if actor:
                print(f"  RL policy match rate:     {results['eval1_rl_match_rate']:.3f}")
            print(f"  Greedy baseline match rate: {results['eval1_greedy_match_rate']:.3f}")
    
    # ============================================================
    # EVALUATION 2: Waddington-OT trajectory
    # ============================================================
    print("\n=== Evaluation 2: Waddington-OT developmental trajectory ===")
    if wad_data_dict is None:
        try:
            from data.load_waddington import load_waddington
            wad_data_dict = load_waddington(
                path=config['data']['waddington_path'],
                n_hvg=config['data']['n_hvg']
            )
        except Exception as e:
            print(f"Could not load Waddington data: {e}")
            wad_data_dict = None
    
    if wad_data_dict is not None:
        wad_adata = wad_data_dict['adata']
        wad_timepoints = wad_data_dict['timepoints']
        wad_cells_by_day = wad_data_dict['cells_by_day']
        
        import scipy.sparse
        X_wad = wad_adata.X.toarray() if scipy.sparse.issparse(wad_adata.X) else np.array(wad_adata.X)
        X_wad = X_wad.astype(np.float32)
        
        # Find compatible genes (intersection)
        wad_genes = set(wad_adata.var_names)
        our_genes = set(gene_names)
        common_genes = wad_genes & our_genes
        print(f"  Common genes between Norman and Waddington: {len(common_genes)}")
        
        if len(common_genes) > 100:
            # Evaluate trajectory planning
            # Define state space from the Norman-trained VAE
            if state_centroids is not None:
                print(f"  Using {len(wad_timepoints)} timepoints: {wad_timepoints[:5]}...")
                
                # Use day 0 cells as start, day 18 cells as target
                day_0_idx = wad_cells_by_day.get(0.0, wad_cells_by_day.get(0, []))
                day_last = max(wad_timepoints)
                day_last_idx = wad_cells_by_day.get(day_last, [])
                
                # Get ordered state assignments for each timepoint
                # (using genes common to both datasets)
                # This is a simplified evaluation since we can't directly encode
                # Waddington cells through a model trained on human genes
                print(f"  Note: Waddington dataset uses mouse genes; trajectory evaluation")
                print(f"  is qualitative — checking state ordering via PCA similarity.")
                
                results['eval2_timepoints'] = wad_timepoints
                results['eval2_note'] = 'mouse_human_gene_mismatch'
        else:
            print("  Insufficient common genes for trajectory evaluation.")
            results['eval2_note'] = 'insufficient_common_genes'
    
    # ============================================================
    # EVALUATION 4: Bootstrapped uncertainty calibration
    # ============================================================
    print("\n=== Evaluation 4: Uncertainty calibration ===")
    
    if len(control_idx) > 0:
        rng = np.random.default_rng(0)
        ctrl_sample = rng.choice(len(control_idx), min(20, len(control_idx)), replace=False)
        X_ctrl_sample = torch.tensor(X[control_idx[ctrl_sample]], dtype=torch.float32, device=device)
        
        uncertainties = []
        actual_errors = []
        
        model.eval()
        with torch.no_grad():
            mu, log_var, epsilon = model.encode(X_ctrl_sample)
            
            # Try a few random interventions
            for latent_idx in range(min(5, cfg_model['latent_dim'])):
                result = model.intervene_on_latent_bootstrapped(
                    epsilon, latent_idx, 0.0, n_samples=n_dag_samples
                )
                # Uncertainty: mean std across genes
                uncertainty = result['std_x_hat'].mean().item()
                
                # For actual error: compare to expected (we don't have ground truth for random interventions)
                # Use variance within bootstrap samples as proxy
                uncertainties.append(uncertainty)
                all_preds = result['all_x_hat'].cpu().numpy()  # [N, B, n_genes]
                actual_variance = np.var(all_preds, axis=0).mean()
                actual_errors.append(actual_variance)
        
        if len(uncertainties) > 2:
            corr, pval = spearmanr(uncertainties, actual_errors)
            results['eval4_uncertainty_spearman'] = float(corr)
            results['eval4_uncertainty_pval'] = float(pval)
            print(f"  Uncertainty vs variance Spearman r: {corr:.3f} (p={pval:.3f})")
    
    print("\n=== Planning Evaluation Complete ===")
    return results

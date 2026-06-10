import os
import sys
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
import csv
import scipy.sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _init_wandb(config, run_name_key='phase2_run_name'):
    """Initialise wandb if enabled in config. Returns the run object or None."""
    wcfg = config.get('wandb', {})
    if not wcfg.get('enabled', False):
        return None
    try:
        import wandb
        run = wandb.init(
            project=wcfg.get('project', 'scCausalRL'),
            entity=wcfg.get('entity', None),
            name=wcfg.get(run_name_key, None),
            config={
                'data': config['data'],
                'model': config['model'],
                'phase2': config['phase2'],
            },
            reinit=True,
        )
        print(f"wandb run: {run.name}  ({run.url})")
        return run
    except ImportError:
        print("wandb not installed — skipping W&B logging.")
        return None
    except Exception as e:
        print(f"wandb init failed ({e}) — continuing without W&B.")
        return None


from models.causal_vae import CausalVAE
from rl.env import CausalWorldModelEnv
from rl.ppo import PPO
from training.losses import perturbation_consistency_loss_v2, compute_total_loss
from training.train_vae import make_interventional_dataset, sample_interventional_batch


def cluster_cells_leiden(adata, model, device, n_neighbors=15, resolution=1.0,
                         batch_size=512):
    """
    Cluster cells in latent z-space using Leiden community detection.

    1. Encode all cells → z via VAE + SCM
    2. Build kNN graph on z-space
    3. Leiden clustering
    4. Compute centroids in z-space

    Returns: (labels [n_cells], centroids [K, d], Z [n_cells, d])
    """
    import scanpy as sc

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float32)

    model.eval()
    z_all = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            mu, _, eps = model.encoder(batch)
            A = model.dag.sample_A(n_samples=1).squeeze(0)
            z = model.dag.scm_forward(eps, A)
            z_all.append(z.cpu().numpy())
    Z = np.vstack(z_all)  # [n_cells, d]
    Z = np.nan_to_num(Z, nan=0.0, posinf=20.0, neginf=-20.0)  # safety

    # Build kNN graph + Leiden in z-space
    adata_z = sc.AnnData(Z)
    sc.pp.neighbors(adata_z, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.leiden(adata_z, resolution=resolution)
    labels = adata_z.obs['leiden'].astype(int).values

    K = labels.max() + 1
    centroids = np.zeros((K, Z.shape[1]), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            centroids[k] = Z[mask].mean(axis=0)

    print(f"  Leiden clustering: {K} clusters (resolution={resolution})")
    print(f"  Cluster sizes: min={np.bincount(labels).min()}, "
          f"max={np.bincount(labels).max()}, mean={np.bincount(labels).mean():.0f}")

    return labels, centroids, Z


def train_joint(config, pretrained_vae_path, data_dict=None, device=None):
    """
    Phase 2: Joint CausalVAE + PPO co-training.

    Gene-level actions, z-space state representation, Leiden clustering.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained VAE
    checkpoint = torch.load(pretrained_vae_path, map_location=device, weights_only=False)
    cfg_model = config['model']

    # Load data
    if data_dict is None:
        dataset_name = config['data']['dataset']
        if dataset_name == 'norman':
            from data.load_norman import load_norman
            data_dict = load_norman(
                path=config['data']['norman_path'],
                n_hvg=config['data']['n_hvg']
            )
        else:
            from data.load_replogle import load_replogle
            data_dict = load_replogle(
                path=config['data']['replogle_path'],
                n_hvg=config['data']['n_hvg']
            )

    gene_names = data_dict['gene_names']
    n_genes = len(gene_names)
    adata = data_dict['adata']

    # Initialize model
    model = CausalVAE(
        n_genes=n_genes,
        latent_dim=cfg_model['latent_dim'],
        tau_init=cfg_model['tau_min'],
        n_dag_samples=cfg_model['n_dag_samples'],
        dagma_s=cfg_model['dagma_s'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained VAE from:", pretrained_vae_path)

    p2 = config['phase2']
    latent_dim = cfg_model['latent_dim']

    # Cluster cells in z-space using Leiden
    print("\nDefining cell state space via Leiden clustering in z-space...")
    cluster_labels, state_centroids, Z_all = cluster_cells_leiden(
        adata, model, device,
        n_neighbors=p2.get('leiden_n_neighbors', 15),
        resolution=p2.get('leiden_resolution', 1.0),
    )
    n_cell_states = state_centroids.shape[0]
    print(f"Cell states: {n_cell_states}, state_dim: {latent_dim}")

    # Training cells
    control_idx = data_dict['control_idx']
    single_perts = data_dict['single_perts']

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    X_ctrl = X[control_idx].astype(np.float32)

    # Precompute gene_max for overexpression actions
    gene_max = X.max(axis=0).astype(np.float32)

    # Interventional triples
    triples = make_interventional_dataset(data_dict)

    # Gene-level action space: n_genes × 2 (knockout + overexpress)
    n_actions = n_genes * p2.get('n_actions_per_gene', 2)

    ppo = PPO(
        state_dim=latent_dim,  # z-space state embeddings
        n_actions=n_actions,
        lr_actor=p2['policy_lr'],
        lr_critic=p2['critic_lr'],
        gamma=p2['gamma'],
        gae_lambda=p2['gae_lambda'],
        clip_epsilon=p2['clip_epsilon'],
        entropy_coeff=p2['entropy_coeff'],
        value_coeff=p2['value_coeff'],
        max_grad_norm=p2['max_grad_norm'],
        n_epochs_per_update=p2['n_epochs_per_update'],
        batch_size=p2['ppo_batch_size'],
        device=device,
    )

    intv_cfg = config.get('intervention', {}) or {}
    env = CausalWorldModelEnv(
        causal_vae=model,
        state_centroids=state_centroids,
        gene_names=gene_names,
        gene_max=gene_max,
        max_steps=p2['max_episode_steps'],
        n_dag_samples=cfg_model['n_dag_samples'],
        alpha=p2['alpha_pessimism'],
        beta_explore=p2['beta_explore'],
        gamma=p2['gamma'],
        lambda_cost=p2['lambda_cost'],
        threshold_frac=p2.get('pert_mutilate_threshold_frac',
                              config['phase1'].get('pert_mutilate_threshold_frac', 0.1)),
        device=device,
        intervention_scheme=intv_cfg.get('scheme', 'threshold'),
        u_clip_max=intv_cfg.get('u_clip_max', 1.0),
        u_accumulate=intv_cfg.get('u_accumulate', True),
        knockout_clamp=intv_cfg.get('knockout_clamp', True),
    )

    # VAE fine-tuning optimizer
    vae_optimizer = optim.Adam(model.parameters(), lr=p2['vae_lr'])

    # Track which genes the policy targets for policy-relevance upweighting
    gene_action_counts = defaultdict(int)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    log_path = 'logs/phase2_training.csv'
    rng = np.random.default_rng(42)
    wandb_run = _init_wandb(config, run_name_key='phase2_run_name')

    print(f"\nStarting Phase 2 joint training for {p2['n_iterations']} iterations...")
    print(f"  n_actions={n_actions} (gene-level: {n_genes} genes × {p2.get('n_actions_per_gene', 2)} dirs)")
    print(f"  n_cell_states={n_cell_states} (Leiden)")

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'episode_reward', 'episode_len',
                         'actor_loss', 'critic_loss', 'entropy',
                         'vae_total', 'vae_recon', 'vae_pert'])

    step_count = 0
    episode_rewards = []
    episode_lengths = []

    for iteration in range(1, p2['n_iterations'] + 1):
        model.eval()

        # Sample starting state from control cells
        ctrl_cell_idx = rng.integers(len(control_idx))
        ctrl_cell_x = torch.tensor(
            X_ctrl[ctrl_cell_idx:ctrl_cell_idx+1], dtype=torch.float32, device=device
        )

        with torch.no_grad():
            _, _, epsilon_start = model.encode(ctrl_cell_x)

        start_state = int(cluster_labels[control_idx[ctrl_cell_idx]])

        # Random target state (different from start)
        target_state = rng.integers(n_cell_states)
        while target_state == start_state:
            target_state = rng.integers(n_cell_states)

        env.current_epsilon = epsilon_start
        env.current_state = start_state
        env.set_target(target_state)
        env.step_count = 0

        # z-space state embeddings
        state_emb = env.get_state_embedding(start_state)
        target_emb = env.get_state_embedding(target_state)

        # Run episode
        episode_reward = 0.0
        episode_len = 0
        done = False
        current_state_emb = state_emb

        while not done:
            action, log_prob, value = ppo.select_action(current_state_emb, target_emb)

            # Track gene usage for policy relevance
            gene_used = action // len(CausalWorldModelEnv.INTERVENTION_DIRECTIONS)
            gene_action_counts[gene_used] += 1

            next_state, reward, done, info = env.step(action)

            ppo.store_transition(
                current_state_emb.cpu(), target_emb.cpu(),
                action, reward, done, log_prob, value,
            )

            next_state_emb = env.get_state_embedding(next_state)
            current_state_emb = next_state_emb

            episode_reward += reward
            episode_len += 1
            step_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_len)

        # PPO update
        ppo_stats = {}
        if step_count >= p2['n_steps_per_rollout']:
            ppo_stats = ppo.update(current_state_emb, target_emb)
            step_count = 0

        # VAE update with policy relevance
        vae_stats = {}
        if iteration % p2['vae_update_every'] == 0 and len(triples) > 0:
            model.train()

            # Policy relevance: upweight pert loss for frequently-targeted genes
            total_gene_actions = sum(gene_action_counts.values())
            pert_weight = 1.0
            pert_batch = sample_interventional_batch(triples, batch_size=64, rng=rng)
            if pert_batch is not None:
                x_ctrl_np, gene_idx_np, x_pert_np, dx_np = pert_batch
                x_ctrl_t = torch.tensor(x_ctrl_np, dtype=torch.float32, device=device)
                x_pert_t = torch.tensor(x_pert_np, dtype=torch.float32, device=device)
                dx_t = torch.tensor(dx_np, dtype=torch.float32, device=device)

                # Policy relevance weight for this gene
                if total_gene_actions > 0:
                    freq = gene_action_counts.get(gene_idx_np, 0) / total_gene_actions
                    pert_weight = 1.0 + (p2['policy_relevance_upweight'] - 1.0) * (freq * n_genes)

                sample_size = min(len(x_ctrl_t), 64)
                x_obs = x_ctrl_t[:sample_size]
                obs_out = model(x_obs)

                pert_loss_val, _ = perturbation_consistency_loss_v2(
                    x_ctrl_t, x_pert_t, dx_t, model,
                    threshold_frac=p2.get('pert_mutilate_threshold_frac',
                                          config['phase1'].get('pert_mutilate_threshold_frac', 0.1)),
                )
                pert_loss_val = pert_loss_val * pert_weight

                losses = compute_total_loss(
                    x=x_obs, x_hat=obs_out['x_hat'],
                    mu=obs_out['mu'], log_var=obs_out['log_var'],
                    A=obs_out['A'], model=model,
                    beta=1.0,
                    lambda_pert=p2.get('lambda_pert', 10.0),
                    lambda_sparse=0.01, lambda_spectral=10.0,
                    lambda_ortho=1.0, lambda_dagma=0.0, rho_dagma=1.0,
                    pert_loss=pert_loss_val, dag_active=True,
                )

                vae_optimizer.zero_grad()
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                vae_optimizer.step()

                vae_stats = {
                    'vae_total': losses['total'].item(),
                    'vae_recon': losses['recon'].item(),
                    'vae_pert': losses['pert'].item() if isinstance(losses['pert'], torch.Tensor) else losses['pert'],
                }

        # Re-cluster every recluster_every iterations
        if iteration % p2['recluster_every'] == 0:
            print(f"\n[Iter {iteration}] Re-clustering cells in z-space...")
            model.eval()
            cluster_labels, state_centroids, Z_all = cluster_cells_leiden(
                adata, model, device,
                n_neighbors=p2.get('leiden_n_neighbors', 15),
                resolution=p2.get('leiden_resolution', 1.0),
            )
            n_cell_states = state_centroids.shape[0]
            env.state_centroids = state_centroids
            env.centroids_tensor = torch.tensor(state_centroids, dtype=torch.float32, device=device)
            env.n_cell_states = n_cell_states

        # Logging and checkpointing
        if iteration % 100 == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            mean_len = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)

            print(
                f"Iter {iteration:6d} | "
                f"reward={mean_reward:.3f}, ep_len={mean_len:.1f}, "
                f"actor={ppo_stats.get('actor_loss', 0):.4f}, "
                f"critic={ppo_stats.get('critic_loss', 0):.4f}, "
                f"vae_pert={vae_stats.get('vae_pert', 0):.4f}"
            )

            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    iteration, mean_reward, mean_len,
                    ppo_stats.get('actor_loss', 0), ppo_stats.get('critic_loss', 0),
                    ppo_stats.get('entropy', 0),
                    vae_stats.get('vae_total', 0), vae_stats.get('vae_recon', 0),
                    vae_stats.get('vae_pert', 0),
                ])

            if wandb_run is not None:
                wandb_run.log({
                    'iteration': iteration,
                    'rl/mean_reward': mean_reward,
                    'rl/mean_ep_len': mean_len,
                    'rl/actor_loss': ppo_stats.get('actor_loss', 0.0),
                    'rl/critic_loss': ppo_stats.get('critic_loss', 0.0),
                    'rl/entropy': ppo_stats.get('entropy', 0.0),
                    'vae/total': vae_stats.get('vae_total', 0.0),
                    'vae/recon': vae_stats.get('vae_recon', 0.0),
                    'vae/pert': vae_stats.get('vae_pert', 0.0),
                }, step=iteration)

            ppo.save('checkpoints/phase2_policy_latest.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'vae_optimizer': vae_optimizer.state_dict(),
                'state_centroids': state_centroids,
                'cluster_labels': cluster_labels,
                'gene_names': gene_names,
                'config': config,
            }, 'checkpoints/phase2_vae_latest.pt')

    print("\nPhase 2 complete.")
    if wandb_run is not None:
        wandb_run.finish()
    ppo.save('checkpoints/phase2_policy_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_centroids': state_centroids,
        'gene_names': gene_names,
        'config': config,
    }, 'checkpoints/phase2_vae_final.pt')

    return model, ppo, env

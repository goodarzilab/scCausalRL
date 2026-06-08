import numpy as np
import torch
from collections import Counter
import math


class CausalWorldModelEnv:
    """
    Gym-like environment wrapping the trained CausalVAE as a simulator.

    State: discrete cell state (Leiden cluster), represented by z-space centroid
    Action: (gene_idx, direction) — gene-level perturbation via D† pipeline

    Transition (proposal Eq. 1):
        action → dx (gene-space) → dz = D†dx → dε = (I-A^T)dz
        → multi-mutilate A → solve z_{t+1} = (I-Ã^T)^{-1}(ε+dε)
        → recover ε_{t+1} = (I-A^T)z_{t+1}
    """

    INTERVENTION_DIRECTIONS = [-1, +1]  # knockout, overexpress

    def __init__(
        self,
        causal_vae,
        state_centroids,   # numpy array [K, d] — z-space centroids
        gene_names,        # list of gene names (for interpretable action output)
        gene_max,          # numpy array [n_genes] — max expression per gene (for overexpress)
        target_state=None,
        max_steps=3,
        n_dag_samples=16,
        alpha=1.0,
        beta_explore=0.1,
        gamma=0.99,
        lambda_cost=0.1,
        threshold_frac=0.1,
        device='cpu',
        intervention_scheme='threshold',
        u_clip_max=1.0,
        u_accumulate=True,
        knockout_clamp=True,
    ):
        self.causal_vae = causal_vae
        self.state_centroids = state_centroids  # [K, d] — z-space
        self.gene_names = gene_names
        self.gene_max = torch.tensor(gene_max, dtype=torch.float32, device=device)
        self.target_state = target_state
        self.max_steps = max_steps
        self.n_dag_samples = n_dag_samples
        self.alpha = alpha
        self.beta_explore = beta_explore
        self.gamma = gamma
        self.lambda_cost = lambda_cost
        self.threshold_frac = threshold_frac
        self.device = device

        # Intervention scheme: 'threshold' (legacy multi-column) or 'u_score'
        # (decoder-loading vector with episode-level persistence).
        self.intervention_scheme = intervention_scheme
        self.u_clip_max = float(u_clip_max)
        self.u_accumulate = bool(u_accumulate)
        self.knockout_clamp = bool(knockout_clamp)
        # Episode state for u-score path (also zero-initialised in reset()):
        self.current_u = torch.zeros(causal_vae.latent_dim, device=device)
        self.knockout_set = set()

        self.n_cell_states = state_centroids.shape[0]
        self.state_dim = state_centroids.shape[1]  # = latent_dim (z-space)
        self.latent_dim = causal_vae.latent_dim
        self.n_genes = causal_vae.n_genes
        self.n_actions = self.n_genes * len(self.INTERVENTION_DIRECTIONS)

        self.centroids_tensor = torch.tensor(
            state_centroids, dtype=torch.float32, device=device
        )

        self.current_epsilon = None  # [1, d]
        self.current_state = None    # int
        self.step_count = 0

    def _action_to_gene_and_direction(self, action):
        """Convert integer action to (gene_idx, delta).
        action: int in [0, n_genes*2)
        Returns: (gene_idx, delta) where delta ∈ {-1, +1}
        """
        n_dirs = len(self.INTERVENTION_DIRECTIONS)
        gene_idx = action // n_dirs
        dir_idx = action % n_dirs
        return gene_idx, self.INTERVENTION_DIRECTIONS[dir_idx]

    def _compute_dx(self, gene_idx, delta, x_current):
        """Compute gene-space perturbation effect dx.
        gene_idx: int — which gene to perturb
        delta: int — -1 (knockout) or +1 (overexpress)
        x_current: Tensor[1, n_genes] — current decoded expression
        Returns: dx Tensor[n_genes]
        """
        dx = torch.zeros(self.n_genes, device=self.device)
        if delta == -1:
            # Knockout: gene goes to 0 (in log1p space, 0 = no expression)
            dx[gene_idx] = -x_current[0, gene_idx]
        else:
            # Overexpress: gene goes to max observed value
            dx[gene_idx] = self.gene_max[gene_idx] - x_current[0, gene_idx]
        return dx

    def _assign_state_z(self, z):
        """Assign a latent z vector to nearest centroid.
        z: Tensor[d] or numpy array [d]
        Returns: int cluster index
        """
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
        dists = torch.norm(self.centroids_tensor - z.unsqueeze(0), dim=1)
        return dists.argmin().item()

    def _phi(self, z_centroid):
        """Quadratic potential: Φ(z) = -||z - z_target||² (proposal Eq. 2)."""
        if self.target_state is None:
            return 0.0
        target_centroid = self.state_centroids[self.target_state]
        return -float(np.sum((z_centroid - target_centroid) ** 2))

    def compute_reward(self, s_current, s_next_samples, s_target):
        """Compute bootstrapped pessimistic reward (proposal Eq. 3)."""
        N = len(s_next_samples)
        eps_val = 1e-8

        rewards = []
        for s_next in s_next_samples:
            r_goal = 1.0 if s_next == s_target else 0.0
            r_shape = (self.gamma * self._phi(self.state_centroids[s_next])
                       - self._phi(self.state_centroids[s_current]))
            r_cost = -self.lambda_cost
            rewards.append(r_goal + r_shape + r_cost)

        r_mean = np.mean(rewards)
        r_std = np.std(rewards) if N > 1 else 0.0
        r_penalized = r_mean - self.alpha * r_std

        # Exploration bonus: entropy of state distribution
        state_counts = Counter(s_next_samples)
        state_probs = [count / N for count in state_counts.values()]
        entropy = -sum(p * math.log(p + eps_val) for p in state_probs)
        r_explore = self.beta_explore * entropy

        return r_penalized + r_explore

    def get_state_embedding(self, state_idx):
        """Get z-space centroid embedding for a state index."""
        return torch.tensor(
            self.state_centroids[state_idx], dtype=torch.float32, device=self.device
        )

    def reset(self, epsilon_tensor=None, initial_state=None):
        self.step_count = 0
        if epsilon_tensor is not None:
            self.current_epsilon = epsilon_tensor.to(self.device)
        if initial_state is not None:
            self.current_state = initial_state
        # Reset u-score episode state (no-op for the threshold path)
        self.current_u = torch.zeros(self.latent_dim, device=self.device)
        self.knockout_set = set()
        return self.current_state

    def set_target(self, target_state):
        self.target_state = target_state

    def step(self, action):
        """
        Take a step: gene-level action → D† pipeline → multi-mutilate → solve SCM.

        action: int in [0, n_genes * 2)
        Returns: (next_state, reward, done, info)
        """
        gene_idx, delta = self._action_to_gene_and_direction(action)

        self.causal_vae.eval()
        with torch.no_grad():
            # Use training mode for DAG sampling to get Gumbel noise diversity
            # (eval mode hard-thresholds, giving identical samples = zero uncertainty)
            self.causal_vae.dag.train()
            A_batch = self.causal_vae.dag.sample_A(n_samples=self.n_dag_samples)
            self.causal_vae.dag.eval()

            # Decode current epsilon to get current expression (for dx computation).
            # Under u-score we use the mutilated graph (M·A^T) at the prior u so the
            # decoded x reflects what the model believes the cell looks like in the
            # persisted-intervention regime.
            if self.intervention_scheme == 'u_score':
                # Use first DAG sample for decoding x_current; uncertainty bootstrapped below.
                A0 = A_batch[0]
                one_minus_u = (1.0 - self.current_u).to(dtype=A0.dtype)
                LHS0 = torch.eye(self.latent_dim, device=self.device, dtype=A0.dtype) \
                       - one_minus_u.unsqueeze(1) * A0.T
                # Mix epsilon under prior u (no new intervention yet): eps_tilde = (1-u_prev)·eps_prev
                # (z_target is not defined without dx). Equivalently solve original SCM since u_prev
                # only matters for new dx mixing. Here we just want a sane x_current decoded.
                z_current = torch.linalg.solve(LHS0, self.current_epsilon.T).T
                z_current = torch.clamp(z_current, -20.0, 20.0)
            else:
                z_current = self.causal_vae.dag.scm_forward(self.current_epsilon, A_batch[0])
            x_current = self.causal_vae.decoder(z_current)  # [1, n_genes]

            # Persistent-knockout clamp: zero genes previously knocked out so that
            # dx for a new perturbation does not "see" them as still expressed.
            if self.intervention_scheme == 'u_score' and self.knockout_clamp and len(self.knockout_set) > 0:
                ko_idx = torch.tensor(sorted(self.knockout_set), dtype=torch.long, device=self.device)
                x_current = x_current.clone()
                x_current[:, ko_idx] = 0.0

            # Compute gene-space perturbation effect
            dx = self._compute_dx(gene_idx, delta, x_current)

            # Bootstrapped gene-level intervention
            all_z_post = []
            all_x_hat = []
            new_u = self.current_u                       # will be updated in u-score branch
            for n in range(self.n_dag_samples):
                A = A_batch[n]
                if self.intervention_scheme == 'u_score':
                    z_post, u_n, _Atilde, _ztgt, _dz = self.causal_vae.intervene_gene_u(
                        self.current_epsilon, dx, A,
                        u_prev=self.current_u, z_prev=z_current.squeeze(0),
                        gene_idx=gene_idx, u_clip_max=self.u_clip_max,
                    )
                    new_u = u_n                          # u is deterministic in A given L, z_prev, dx
                else:
                    z_post, _d_eps = self.causal_vae.intervene_gene(
                        self.current_epsilon, dx, A,
                        threshold_frac=self.threshold_frac,
                    )
                x_hat = self.causal_vae.decoder(z_post)
                all_z_post.append(z_post)
                all_x_hat.append(x_hat)

            all_z_post_t = torch.stack(all_z_post, dim=0)  # [N, 1, d]
            # Sanitize: replace any NaN/Inf from ill-conditioned SCM solves
            all_z_post_t = torch.nan_to_num(all_z_post_t, nan=0.0, posinf=20.0, neginf=-20.0)
            all_z_post_t = torch.clamp(all_z_post_t, min=-20.0, max=20.0)

        # Assign each z_post to nearest centroid in z-space
        s_next_samples = []
        for n in range(self.n_dag_samples):
            z_n = all_z_post_t[n].squeeze(0)  # [d]
            s_next = self._assign_state_z(z_n)
            s_next_samples.append(s_next)

        # Consensus: majority vote
        next_state = Counter(s_next_samples).most_common(1)[0][0]

        # Reward
        reward = self.compute_reward(
            self.current_state, s_next_samples, self.target_state
        )
        # Sanitize: clamp + handle NaN to prevent buffer corruption
        if not math.isfinite(reward):
            reward = 0.0
        reward = max(-10.0, min(10.0, reward))

        # Epsilon recovery for multi-step.
        # Threshold scheme:  ε_{t+1} = (I − A^T) z_{t+1}                 (original DAG)
        # u-score scheme:    ε_{t+1} = (I − diag(1-u) A^T) z_{t+1}       (mutilated DAG)
        # The mutilated form is used in u-score so the next step starts from a
        # noise vector consistent with the persisted intervention regime.
        mean_z_post = all_z_post_t.mean(dim=0)  # [1, d]
        mean_A = A_batch.mean(dim=0)            # [d, d] consensus A
        if self.intervention_scheme == 'u_score':
            self.current_epsilon = self.causal_vae.recover_epsilon_u(
                mean_z_post, mean_A, new_u
            )
            if self.u_accumulate:
                self.current_u = new_u
            if delta == -1:
                self.knockout_set.add(int(gene_idx))
        else:
            self.current_epsilon = self.causal_vae.recover_epsilon(
                mean_z_post, mean_A
            )
        self.current_state = next_state

        self.step_count += 1
        done = (self.step_count >= self.max_steps) or (next_state == self.target_state)

        info = {
            'gene_idx': gene_idx,
            'gene_name': self.gene_names[gene_idx] if gene_idx < len(self.gene_names) else f'gene_{gene_idx}',
            'direction': 'knockout' if delta == -1 else 'overexpress',
            's_next_samples': s_next_samples,
            'uncertainty': np.std([
                self.state_centroids[s] for s in s_next_samples
            ], axis=0).mean() if len(set(s_next_samples)) > 1 else 0.0,
        }

        return next_state, reward, done, info

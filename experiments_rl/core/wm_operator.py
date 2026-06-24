"""Uniform shift-native mutilation do-operator wrapper.

Every RL method (MPC, sequential PPO, set PPO, distillation) goes through this one
interface, so the world-model dynamics are identical by construction and the
cross-method value-add ladder is a fair comparison.

The operator:
  - samples an ensemble of N adjacency matrices from the Gumbel gate (bootstraps
    epistemic uncertainty for the pessimism term);
  - applies a SET of perturbations simultaneously via
    `model.intervene_shift_mutilate` (shifts are summed over the set; non-additivity
    comes from the set-dependent u and the mutilated solve z=(I-MA^T)^{-1}ε̃);
  - threads epsilon across steps via `recover_epsilon_u` (persisted intervention),
    so multi-step rollouts are genuine counterfactuals, not re-encodes.

All methods are @torch.no_grad — the world model is frozen; only the policy trains.
"""
import torch


class WorldModelOperator:
    def __init__(self, model, meta, n_dag_samples=16):
        self.model = model
        self.meta = meta
        self.device = meta['device']
        self.d = int(meta['latent_dim'])
        m = meta['mut']
        self.u_mode = m['u_mode']
        self.u_clip_max = m['u_clip_max']
        self.shift_beta = m['shift_beta']
        self.n_dag_samples = int(n_dag_samples)

    # ----- DAG ensemble ------------------------------------------------------
    @torch.no_grad()
    def sample_dags(self, n=None):
        """Sample n HARD-gate Gumbel adjacency matrices, centered on the eval-mode
        threshold. Returns [n, d, d].

        NOT the model's train-mode `sample_A`: that returns SOFT Concrete gates
        sigmoid((θ+G)/τ)∈(0,1) whose mean is dense and differs from the eval-mode
        hard A that the targets and the validated combo eval (0.899) use — running
        the rollout there is a different dynamical system. Instead we draw hard
        Gumbel-sigmoid samples gate=1{θ+G1−G2>0} (τ-invariant threshold; G1−G2 is
        Logistic noise), so the ensemble is DIVERSE on uncertain edges but CENTERED
        on the eval threshold for confident ones — a proper epistemic ensemble for
        the pessimism penalty. (Mean dynamics use mean_A; this ensemble is only the
        std/disagreement source.)"""
        n = int(n or self.n_dag_samples)
        dag = self.model.dag
        d = dag.d
        th = dag.theta.unsqueeze(0).expand(n, d, d)
        u1 = torch.empty_like(th).uniform_(1e-8, 1 - 1e-8)
        u2 = torch.empty_like(th).uniform_(1e-8, 1 - 1e-8)
        gumbel_diff = (-(-u1.log()).log()) - (-(-u2.log()).log())     # Logistic(0,1)
        gates_hard = ((th + gumbel_diff) > 0).float()                  # [n, d, d]
        A = gates_hard * dag.weights.unsqueeze(0) * dag.mask.unsqueeze(0)
        return A.detach()

    @torch.no_grad()
    def mean_A(self):
        """Deterministic hard-threshold adjacency (eval-mode gates). [d, d]."""
        was_training = self.model.dag.training
        self.model.dag.eval()
        try:
            A = self.model.dag.sample_A(n_samples=1).squeeze(0)
        finally:
            self.model.dag.train(was_training)
        return A.detach()

    # ----- dynamics ----------------------------------------------------------
    @torch.no_grad()
    def scm_forward(self, epsilon, A):
        """z = (I - A^T)^{-1} epsilon.  epsilon: [B, d], A: [d, d] -> [B, d]."""
        return self.model.dag.scm_forward(epsilon, A)

    @torch.no_grad()
    def apply(self, epsilon, gene_set, A):
        """Apply a SET of perturbations at one step under a single DAG.

        epsilon:  [B, d]
        gene_set: list[int] (length 1 = single, k = combo/set; [] = no-op)
        A:        [d, d]
        Returns (z_post [B, d], u [d], eps_next [B, d]).
        """
        if len(gene_set) == 0:
            z = self.scm_forward(epsilon, A)
            return z, torch.zeros(self.d, device=epsilon.device, dtype=epsilon.dtype), epsilon
        z_prev = self.scm_forward(epsilon, A)
        z_post, u = self.model.intervene_shift_mutilate(
            epsilon, list(gene_set), A, z_prev,
            u_mode=self.u_mode, u_clip_max=self.u_clip_max, shift_beta=self.shift_beta)
        eps_next = self.model.recover_epsilon_u(z_post, A, u)
        return z_post, u, eps_next

    @torch.no_grad()
    def apply_ensemble(self, eps_batch, gene_set, A_batch):
        """Apply the same gene_set under an ensemble of DAGs (one epsilon per DAG).

        eps_batch: [N, d] (per-DAG threaded noise)
        gene_set:  list[int]
        A_batch:   [N, d, d]
        Returns (z_post [N, d], u [N, d], eps_next [N, d]).
        """
        N = A_batch.shape[0]
        zs, us, es = [], [], []
        for n in range(N):
            z, u, e = self.apply(eps_batch[n:n + 1], gene_set, A_batch[n])
            zs.append(z.squeeze(0)); us.append(u); es.append(e.squeeze(0))
        return torch.stack(zs, 0), torch.stack(us, 0), torch.stack(es, 0)

    @torch.no_grad()
    def decode(self, z):
        """Latent -> gene-expression space (for data-grounded validation)."""
        return self.model.decoder(z)

    # ----- batched scoring over many candidate sets (large-pool MPC) ---------
    @torch.no_grad()
    def shift_for(self, gene_indices):
        """Per-gene latent shift S_g for a list/tensor of gene indices -> [K, d].
        Works zero-shot for ANY gene on the axis (pert_label defined for all)."""
        gi = torch.as_tensor(gene_indices, dtype=torch.long, device=self.device)
        return self.model.compute_shift(gi)                       # [K, d]

    @torch.no_grad()
    def apply_many(self, eps0, S_batch, A):
        """Vectorized shift-native mutilation for K candidate sets at once.

        eps0:    [d]   single control ε (shared start)
        S_batch: [K,d] summed shift per set (single = S_g; pair = S_a + S_b)
        A:       [d,d]
        Returns z_post [K, d]. Assumes u_mode='shift' (exp14): u = min(1, β|S|).
        Batches the K per-set mutilated solves (I − diag(1−u_k)Aᵀ) via one batched
        linalg.solve — this is what makes scoring thousands of candidates feasible.
        """
        d = self.d
        I = torch.eye(d, device=eps0.device, dtype=eps0.dtype)
        IA = I - A.T
        z_prev = torch.linalg.solve(IA, eps0)                      # [d]
        z_S = torch.linalg.solve(IA, S_batch.T).T                  # [K, d]
        z_target = z_prev.unsqueeze(0) + z_S                       # [K, d]
        u = (self.shift_beta * S_batch.abs()).clamp(max=1.0)       # [K, d]  (u_mode='shift')
        eps_tilde = u * z_target + (1.0 - u) * eps0.unsqueeze(0)    # [K, d]
        Mrow = (1.0 - u)                                           # [K, d]
        LHS = I.unsqueeze(0) - Mrow.unsqueeze(2) * A.T.unsqueeze(0)   # [K, d, d]
        z_post = torch.linalg.solve(LHS, eps_tilde.unsqueeze(2)).squeeze(2)  # [K, d]
        return torch.clamp(z_post, -20.0, 20.0)

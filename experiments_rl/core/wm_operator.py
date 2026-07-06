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
        # Gate-aware composition: exp19 ugate/epsmix models carry the non-additive
        # correction (refine_u / mix_eps) OUTSIDE intervene_shift_mutilate, so the plain
        # operator would silently drop their epistasis. Detect and route through the gate.
        # Norman / non-gated models have u_gate=eps_mix_net=None → identical to before.
        self.gate_aware = (
            hasattr(model, 'refine_u') and hasattr(model, 'mix_eps')
            and (getattr(model, 'u_gate', None) is not None
                 or getattr(model, 'eps_mix_net', None) is not None))
        if self.gate_aware:
            print("[wm_operator] gate-aware composition ON "
                  f"(u_gate={getattr(model, 'u_gate', None) is not None}, "
                  f"eps_mix={getattr(model, 'eps_mix_net', None) is not None})")

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
        if self.gate_aware:
            z_post, u = self._intervene_gated(epsilon, list(gene_set), A, z_prev)
        else:
            z_post, u = self.model.intervene_shift_mutilate(
                epsilon, list(gene_set), A, z_prev,
                u_mode=self.u_mode, u_clip_max=self.u_clip_max, shift_beta=self.shift_beta)
        eps_next = self.model.recover_epsilon_u(z_post, A, u)
        return z_post, u, eps_next

    @torch.no_grad()
    def _intervene_gated(self, epsilon, gene_set, A, z_prev):
        """ugate/epsmix-aware do-operator, mirroring epistasis_quartile_eval's do_op:
        S -> u (shift mode) -> refine_u (gate) -> mix_eps (eps correction) -> mutilated
        solve. epsilon [B,d], returns (z_post [B,d], u [d])."""
        gi = torch.as_tensor(gene_set, dtype=torch.long, device=epsilon.device).reshape(-1)
        S = self.model.compute_shift(gi).sum(dim=0)                      # [d] (gated in exp20)
        d = self.d
        if self.u_mode == 'shift':
            u = (self.shift_beta * S.abs()).clamp(max=1.0)              # [d]
        else:
            u = self.model.get_loadings()[gi].sum(dim=0).clamp(0.0, self.u_clip_max)
        u = self.model.refine_u(u, S)                                   # nonlinear gate (no-op in exp20 struct arms)
        if getattr(self.model, 'pert_intervene_mode', 'legacy') == 'structural':
            eps_tilde = epsilon + S.unsqueeze(0)                        # Fix 1: S into exogenous, single solve
        else:
            I = torch.eye(d, device=epsilon.device, dtype=epsilon.dtype)
            z_target = z_prev + torch.linalg.solve(I - A.T, S).unsqueeze(0)   # legacy pre-propagation
            eps_tilde = self.model.mix_eps(epsilon, z_target, u.unsqueeze(0))
        z_post, _, _ = self.model.dag.scm_intervene_u(epsilon, A, eps_tilde, u)
        return z_post, u

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
        u = (self.shift_beta * S_batch.abs()).clamp(max=1.0)       # [K, d]  (u_mode='shift')
        # Nonlinear SCM: mutilated Picard z = diag(1−u)⊙tanh(A^T z)+eps_tilde (structural only).
        if getattr(self.model.dag, 'scm_nonlinear', False) and \
           getattr(self.model, 'pert_intervene_mode', 'legacy') == 'structural':
            if self.gate_aware:
                u = self.model.refine_u(u, S_batch)
            eps_tilde = eps0.unsqueeze(0) + S_batch                # [K, d]
            omu = (1.0 - u)                                        # [K, d]
            z_post = eps_tilde
            for _ in range(getattr(self.model.dag, 'n_picard', 40)):
                z_post = omu * torch.tanh(z_post @ A) + eps_tilde  # (A^T z)_k = z_k @ A
            return torch.clamp(z_post, -20.0, 20.0)
        if getattr(self.model, 'pert_intervene_mode', 'legacy') == 'structural':
            # Fix 1: inject S into exogenous, single mutilated solve (no z_target pre-propagation)
            if self.gate_aware:
                u = self.model.refine_u(u, S_batch)
            eps_tilde = eps0.unsqueeze(0) + S_batch                # [K, d]
        else:
            IA = I - A.T
            z_prev = torch.linalg.solve(IA, eps0)                  # [d]
            z_target = z_prev.unsqueeze(0) + torch.linalg.solve(IA, S_batch.T).T   # [K, d]
            if self.gate_aware:
                u = self.model.refine_u(u, S_batch)                # [K, d] nonlinear gate
                eps_tilde = self.model.mix_eps(eps0.unsqueeze(0), z_target, u)   # [K, d]
            else:
                eps_tilde = u * z_target + (1.0 - u) * eps0.unsqueeze(0)    # [K, d]
        Mrow = (1.0 - u)                                           # [K, d]
        LHS = I.unsqueeze(0) - Mrow.unsqueeze(2) * A.T.unsqueeze(0)   # [K, d, d]
        z_post = torch.linalg.solve(LHS, eps_tilde.unsqueeze(2)).squeeze(2)  # [K, d]
        return torch.clamp(z_post, -20.0, 20.0)

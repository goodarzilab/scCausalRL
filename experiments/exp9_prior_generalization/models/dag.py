import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DAGModule(nn.Module):
    """
    Learnable DAG via factored Gumbel-Sigmoid GATES × real-valued WEIGHTS.

    The adjacency entry is now a product of two learnable parts:

        A[i, j]  =  gate[i, j]  *  weight[i, j]  *  mask[i, j]
        gate[i, j] = sigmoid((theta[i, j] + Gumbel) / tau)   ∈ [0, 1]
        weight[i, j] ∈ ℝ   (real-valued, Xavier-init)
        mask[i, j] = 0 if i == j else 1                      (no self-loops)

    This is the "two-parameter" parameterisation (see ports from MultiOmics
    `VAE_latent_dag_v2.py`).  Why factored:

      • `theta` controls *whether* an edge is active (gated by Gumbel-Sigmoid).
        Its gradient is filtered by `σ'(θ)/τ` and competes with Gumbel noise.
      • `weight` carries the edge's *signed magnitude*.  Its gradient is
        DIRECT — no Gumbel-sigmoid bottleneck — so the model can adjust edge
        strengths even when theta updates are weak.

    Empirically (CausalBioRL Phase F/G/H verifications) the single-parameter
    Gumbel-sigmoid faced an unsolvable trilemma: at θ near 0, Gumbel noise
    dominates; at θ near −3, σ'(θ) is saturated and gradients vanish.  The
    factored form sidesteps this — `weight` is learnable independently.

    A is now *signed* (entries can be negative, modelling inhibitory edges).
    Downstream SCM solve, mutilation, DAGMA constraint, spectral-stability
    penalty all work unchanged — they treat A as a generic real matrix.

    Parameters:
        d: int — number of latent programs (default 64)
        tau: float — Gumbel-Sigmoid temperature (anneal during training)
        dagma_s: float — DAGMA s parameter (kept for compatibility; DAGMA off in Phase F+)
    """

    def __init__(self, d=64, tau=1.0, dagma_s=1.0,
                 weight_init_scale=1.0, theta_init_scale=1.0):
        super().__init__()
        self.d = d
        self.dagma_s = dagma_s
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))

        # Gate logits.  Initialised near zero so the gate sigmoid sits around 0.5
        # — every edge is roughly "50% on" at init, and the weights determine the
        # actual signed strength.  `theta_init_scale` (default 1.0) multiplies the
        # 0.05 base scale (exp9 init probe).
        self.theta = nn.Parameter(torch.randn(d, d) * 0.05 * theta_init_scale)

        # Edge weights — real-valued, Xavier-initialised.
        # Xavier uniform: U(−√(6/(d+d)), +√(6/(d+d))) → std ≈ √(2/(2d)) = √(1/d).
        # For d=64 this gives std ≈ 0.125, range ≈ ±0.21.
        # `weight_init_scale` (default 1.0 ⇒ unchanged) multiplies the init so A
        # can start more influential — exp9 tests weight_init_scale=8 (~unit-scale
        # A entries) to see if a stronger init lets A survive the dynamics.
        self.weights = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.weights)
        if weight_init_scale != 1.0:
            self.weights.data.mul_(weight_init_scale)

        # Mask: no self-loops.  Registered as buffer so it moves with the module.
        self.register_buffer('mask', 1.0 - torch.eye(d, dtype=torch.float32))

    def set_tau(self, tau_val):
        self.tau.fill_(tau_val)

    def _sample_gates(self, n_samples=1):
        """Sample n_samples gate matrices via Gumbel-Sigmoid (training) or
        hard threshold (eval).  Returns Tensor[n_samples, d, d] in [0, 1]."""
        d = self.d
        theta_expanded = self.theta.unsqueeze(0).expand(n_samples, d, d)
        if self.training:
            U1 = torch.zeros_like(theta_expanded).uniform_().clamp(1e-8, 1 - 1e-8)
            U2 = torch.zeros_like(theta_expanded).uniform_().clamp(1e-8, 1 - 1e-8)
            G1 = -torch.log(-torch.log(U1))
            G2 = -torch.log(-torch.log(U2))
            gates = torch.sigmoid((theta_expanded + G1 - G2) / self.tau)
        else:
            probs = torch.sigmoid(theta_expanded / self.tau)
            gates = (probs > 0.5).float()
        return gates

    def sample_A(self, n_samples=1):
        """Sample n_samples adjacency matrices A = gate * weight * mask.

        Returns: Tensor[n_samples, d, d] — real-valued, can be negative.
        """
        gates = self._sample_gates(n_samples=n_samples)                 # [N, d, d] in [0, 1]
        # Broadcast: weights [d, d] · mask [d, d] · gates [N, d, d]
        A = gates * self.weights.unsqueeze(0) * self.mask.unsqueeze(0)  # [N, d, d]
        return A

    def get_edge_probs(self):
        """Return the per-edge gate probabilities sigmoid(theta / tau) * mask
        — the probability that each edge is "on".  Note this does NOT include
        the weight magnitudes; use `get_signed_edges` for the actual A values."""
        probs = torch.sigmoid(self.theta / self.tau)
        return probs * self.mask

    def get_signed_edges(self):
        """Return the deterministic signed edge matrix used at eval time:
        gate_hard * weight * mask.  Same convention as sample_A in eval mode."""
        probs = torch.sigmoid(self.theta / self.tau)
        gate_hard = (probs > 0.5).float()
        return gate_hard * self.weights * self.mask

    def get_edge_weights(self):
        """Return the raw learnable weights (without gate or mask)."""
        return self.weights

    def dagma_constraint(self, A):
        """
        DAGMA acyclicity constraint: h(A) = tr((I + s/d * A o A)^d) - d
        h(A) = 0 iff A is a DAG.

        A: Tensor[d, d] or Tensor[B, d, d]
        Returns: scalar (mean over batch if batched)
        """
        d = self.d
        s = self.dagma_s

        def _dagma_single(A_):
            # A_ shape: [d, d]
            M = torch.eye(d, device=A_.device, dtype=A_.dtype) + (s / d) * A_ * A_
            # Matrix power to d: use repeated squaring for large d
            try:
                # Use matrix_power for small d, eigendecomposition for large
                if d <= 64:
                    Md = torch.linalg.matrix_power(M, d)
                else:
                    # Eigendecomposition approach
                    eigvals, eigvecs = torch.linalg.eigh(M)
                    # Clamp negative eigenvalues (numerical issues)
                    eigvals = eigvals.clamp(min=1e-8)
                    eigvals_pow = eigvals ** d
                    Md = eigvecs @ torch.diag(eigvals_pow) @ eigvecs.T
                h = torch.trace(Md) - d
            except Exception:
                # Fallback: simple trace formula approximation
                h = torch.sum(A_ * A_)
            return h

        if A.dim() == 2:
            return _dagma_single(A)
        else:
            # Batched: average over batch
            h_vals = torch.stack([_dagma_single(A[i]) for i in range(A.shape[0])])
            return h_vals.mean()

    def mutilate(self, A, intervention_idx):
        """
        Apply do-operator: zero out column intervention_idx of A.
        This removes all incoming edges to node i.

        A: Tensor[d, d] or Tensor[B, d, d]
        intervention_idx: int
        Returns: Tensor same shape as A
        """
        A_tilde = A.clone()
        if A.dim() == 2:
            A_tilde[:, intervention_idx] = 0.0
        else:
            A_tilde[:, :, intervention_idx] = 0.0
        return A_tilde

    def scm_forward(self, epsilon, A):
        """
        Solve linear SCM: z = (I - A^T)^{-1} @ epsilon
        Uses torch.linalg.solve for numerical stability.

        epsilon: Tensor[B, d]
        A: Tensor[d, d]
        Returns: z Tensor[B, d]
        """
        d = self.d
        device = epsilon.device
        I = torch.eye(d, device=device, dtype=epsilon.dtype)
        # Solve (I - A^T) @ z^T = epsilon^T
        LHS = I - A.T  # [d, d]
        # torch.linalg.solve: solve LHS @ X = B, where B = epsilon.T
        z = torch.linalg.solve(LHS, epsilon.T).T  # [B, d]
        z = torch.clamp(z, min=-20.0, max=20.0)  # safety clamp if (I-A^T) ill-conditioned
        return z

    def scm_forward_batched(self, epsilon, A_batch):
        """
        Solve SCM for a batch of DAGs.

        epsilon: Tensor[B, d]
        A_batch: Tensor[N, d, d] -- N different DAG samples
        Returns: z Tensor[N, B, d]
        """
        d = self.d
        device = epsilon.device
        I = torch.eye(d, device=device, dtype=epsilon.dtype)
        results = []
        for n in range(A_batch.shape[0]):
            A = A_batch[n]
            LHS = I - A.T
            z = torch.linalg.solve(LHS, epsilon.T).T
            z = torch.clamp(z, min=-20.0, max=20.0)
            results.append(z)
        return torch.stack(results, dim=0)  # [N, B, d]

    def scm_intervene(self, epsilon, A, intervention_idx, intervention_value=0.0):
        """
        Apply do(z_i = v):
            1. Mutilate A (zero column intervention_idx)
            2. Set epsilon[:, intervention_idx] = intervention_value
            3. Solve the mutilated SCM

        epsilon: Tensor[B, d]
        A: Tensor[d, d]
        intervention_idx: int
        intervention_value: float
        Returns: z_post Tensor[B, d]
        """
        A_tilde = self.mutilate(A, intervention_idx)
        epsilon_tilde = epsilon.clone()
        epsilon_tilde[:, intervention_idx] = intervention_value
        z_post = self.scm_forward(epsilon_tilde, A_tilde)
        return z_post

    def scm_intervene_multi(self, epsilon, A, d_epsilon, threshold_frac=0.1):
        """
        Multi-program do-operator (proposal Eq. 1):
            1. Identify affected programs: |dε_i| > threshold_frac * max|dε|
            2. Mutilate A: zero columns for ALL affected programs
            3. Apply full dε shift: ε̃ = ε + dε
            4. Solve mutilated SCM: z = (I − Ã^T)^{-1} ε̃

        epsilon: Tensor[B, d] — current exogenous noise
        A: Tensor[d, d] — sampled adjacency matrix
        d_epsilon: Tensor[d] — exogenous perturbation vector (from D† pipeline)
        threshold_frac: float — fraction of max|dε| for mutilation threshold
        Returns: z_post Tensor[B, d]
        """
        threshold = threshold_frac * d_epsilon.abs().max()
        affected = d_epsilon.abs() > threshold

        # Mutilate: zero columns of A for all affected programs
        A_tilde = A.clone()
        A_tilde[:, affected] = 0.0

        # Apply full dε shift to epsilon
        epsilon_tilde = epsilon + d_epsilon.unsqueeze(0)  # [B, d]

        z_post = self.scm_forward(epsilon_tilde, A_tilde)
        return z_post

    def scm_intervene_u(self, epsilon, A, eps_tilde, u):
        """
        u-score (decoder-loading) do-operator.

        Given an intervention-strength vector u in [0, 1]^d:
            M           = diag(1 - u)
            A_tilde     = A · M           # column i of A scaled by (1 - u_i)
            (I - M A^T) z_post = eps_tilde

        Endpoint behaviour:
            u_i = 1 → row i of (I - M A^T) is e_i, so z_post[i] = eps_tilde[i]
                      (classical do-operator: z_i pinned, all parents detached).
            u_i = 0 → recovers original SCM at coordinate i.

        Conditioning never worse than original SCM (rows tend to e_i as u_i→1).

        epsilon:   Tensor[B, d] — present for API symmetry / future use; not used in the solve.
        A:         Tensor[d, d]
        eps_tilde: Tensor[B, d] — already-mixed effective exogenous noise
                                  (caller is responsible for u⊙z_target + (1-u)⊙eps_prev).
        u:         Tensor[d] in [0, 1].
        Returns:
            z_post  Tensor[B, d]
            A_tilde Tensor[d, d]    — A with column i scaled by (1 - u_i)
            M       Tensor[d, d]    — diag(1 - u)
        """
        d = self.d
        device = A.device
        dtype = eps_tilde.dtype
        I = torch.eye(d, device=device, dtype=dtype)

        one_minus_u = (1.0 - u).to(dtype=dtype)                       # [d]
        # A_tilde = A · diag(1 - u): scale column i by (1 - u_i)
        A_tilde = A * one_minus_u.unsqueeze(0)                        # [d, d]
        # LHS = I - M A^T = I - diag(1-u) · A^T
        LHS = I - one_minus_u.unsqueeze(1) * A.T                      # [d, d]
        z = torch.linalg.solve(LHS, eps_tilde.T).T                    # [B, d]
        z = torch.clamp(z, min=-20.0, max=20.0)
        M = torch.diag(one_minus_u)
        return z, A_tilde, M

    def spectral_radius_penalty(self, A):
        """
        Penalize spectral radius of A to ensure (I - A^T) is well-conditioned.
        Uses largest singular value of A (= sqrt of max eigenvalue of A^T A).
        loss = relu(max_singular_value - 0.95)

        A: Tensor[d, d]
        """
        try:
            # Eigenvalues of A^T @ A give squared singular values
            eigvals = torch.linalg.eigvalsh(A.T @ A)
            max_sv = eigvals.max().sqrt()  # largest singular value
        except Exception:
            # Fallback: use Frobenius norm proxy
            max_sv = torch.sum(A * A).sqrt() / (self.d ** 0.5)
        return F.relu(max_sv - 0.95)

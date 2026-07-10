import torch
import torch.nn as nn
import numpy as np
from .encoder import CausalEncoder, SENAEncoder
from .decoder import LinearDecoder, MLPDecoder
from .dag import DAGModule, _robust_solve
from .shift_encoder import ShiftEncoder


class CausalVAE(nn.Module):
    """
    Full causal VAE combining encoder, DAG module, and decoder.

    Forward pass (observational):
        x -> encoder -> (mu, log_var, epsilon) -> DAG.scm_forward(epsilon, A) -> z -> decoder -> x_hat

    Forward pass (interventional):
        x_ctrl -> encoder -> epsilon_ctrl
        do(z_i = v) via DAG.scm_intervene -> z_post
        decoder(z_post) -> x_hat_pert
    """

    def __init__(self, n_genes, latent_dim=64, tau_init=1.0, n_dag_samples=16,
                 dagma_s=1.0, n_modalities=0, n_batches=0, u_clip_max=1.0,
                 decoder_type='linear', decoder_hidden=1024, decoder_layers=2,
                 use_shift_encoder=False, shift_label_dim=50, shift_hidden=1024,
                 shift_n_layers=1, shift_output_activation='none',
                 shift_scale_init=1.0,
                 separate_pert_decoder=False, pert_decoder_type='linear',
                 pert_decoder_hidden=1024, pert_decoder_layers=2,
                 u_gate_mlp=False, nonlinear_eps_mix=False, gate_hidden=256, u_gate_pure=False,
                 dag_disabled=False,
                 encoder_type='mlp', sena_mask=None, sena_activation='tanh',
                 pert_intervene_mode='legacy', use_target_gate=False,
                 target_gate_tau=1.0, target_gate_bias_init=3.0,
                 frozen_decoder_weight=None, scm_nonlinear=False, n_picard=40,
                 scm_solve_ridge=0.0, decoder_mask=None, decoder_mask_freeze=False):
        super().__init__()
        # exp20 load-bearing-A fixes:
        #  pert_intervene_mode='structural' → single mutilated solve z_post=(I−diag(1−u)Aᵀ)⁻¹(ε+S),
        #    no z_target pre-propagation (Fix 1); 'legacy' reproduces exp19 exactly.
        #  use_target_gate → sparse per-TF target gate in the shift encoder (Fix 2).
        self.pert_intervene_mode = pert_intervene_mode
        self.use_target_gate = bool(use_target_gate)
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_dag_samples = n_dag_samples
        self.n_modalities = n_modalities
        self.n_batches = n_batches
        # u-score intervention strength ceiling.  At 1.0 the do-operator can fully
        # pin z_post[i] = z_target[i] (and zero column i of Ã), letting pert bypass A.
        # Phase G default 0.5 caps u so the SCM solve still requires A involvement.
        self.u_clip_max = float(u_clip_max)
        self.decoder_type = decoder_type
        self.use_shift_encoder = bool(use_shift_encoder)

        self.encoder_type = encoder_type
        if encoder_type == 'sena':
            # SENA-δ masked gene→pathway encoder. sena_mask is [n_genes, K] (member=1,
            # non-member=leak λ). Anchors latent programs to GO:BP processes → breaks the
            # encoder/decoder rotational symmetry that made A non-load-bearing.
            if sena_mask is None:
                raise ValueError("encoder_type='sena' requires sena_mask [n_genes, K]")
            self.encoder = SENAEncoder(input_dim=n_genes, latent_dim=latent_dim,
                                       mask=sena_mask, activation=sena_activation,
                                       n_modalities=n_modalities, n_batches=n_batches)
        else:
            self.encoder = CausalEncoder(input_dim=n_genes, latent_dim=latent_dim,
                                         n_modalities=n_modalities,
                                         n_batches=n_batches)
        self.dag = DAGModule(d=latent_dim, tau=tau_init, dagma_s=dagma_s, disabled=dag_disabled,
                             scm_nonlinear=scm_nonlinear, n_picard=n_picard)
        self.dag.solve_ridge = float(scm_solve_ridge)
        if decoder_type == 'mlp':
            self.decoder = MLPDecoder(latent_dim=latent_dim, output_dim=n_genes,
                                      hidden=decoder_hidden, n_layers=decoder_layers)
        else:
            self.decoder = LinearDecoder(latent_dim=latent_dim, output_dim=n_genes)

        # Optional SEPARATE perturbation decoder head. When set, the perturbation/residual
        # Δ prediction decodes z_post/z_prev through THIS head instead of the main decoder, so
        # the main linear `decoder` (which defines the GRN + carries the CollecTRI prior) is
        # not forced to also fit drift-removed residuals. It taps z_post (post-SCM), so A and
        # the encoder still receive interventional gradients — only `decoder` is decoupled.
        self.separate_pert_decoder = bool(separate_pert_decoder)
        if self.separate_pert_decoder:
            if pert_decoder_type == 'mlp':
                self.pert_decoder = MLPDecoder(latent_dim=latent_dim, output_dim=n_genes,
                                               hidden=pert_decoder_hidden, n_layers=pert_decoder_layers)
            else:
                self.pert_decoder = LinearDecoder(latent_dim=latent_dim, output_dim=n_genes)
        else:
            self.pert_decoder = None

        # exp20 Phase-2: FROZEN decoder ("SENA for the decoder"). Fixing D removes the
        # rotational freedom D_eff=D·(I−MAᵀ)⁻¹ that let a free decoder absorb the SCM operator,
        # so A becomes identified and MUST carry the latent→gene propagation. frozen_decoder_weight
        # is [n_genes, latent_dim] (GO:BP membership cols + PCA-coverage cols); requires_grad=False.
        if frozen_decoder_weight is not None:
            with torch.no_grad():
                self.decoder.D.copy_(frozen_decoder_weight.to(self.decoder.D.dtype))
            self.decoder.D.requires_grad_(False)
            if self.pert_decoder is not None:
                with torch.no_grad():
                    self.pert_decoder.D.copy_(frozen_decoder_weight.to(self.pert_decoder.D.dtype))
                self.pert_decoder.D.requires_grad_(False)

        # exp20 SENA-δ MASKED-LEARNABLE decoder: D stays learnable but every use is D⊙mask, where
        # mask = M + λ(1−M) (1 on annotated gene-pathway pairs, λ off). Unlike the frozen decoder it
        # keeps reconstruction capacity, but off-pathway gene-gene wiring is λ-suppressed so covariance
        # must route through the pathway latents + A. Each latent coordinate is a GO:BP pathway.
        if decoder_mask is not None and getattr(self.decoder, 'is_linear', False):
            dm = decoder_mask if torch.is_tensor(decoder_mask) else torch.as_tensor(decoder_mask)
            self.decoder.set_mask(dm, freeze=bool(decoder_mask_freeze))
            if self.pert_decoder is not None and getattr(self.pert_decoder, 'is_linear', False):
                self.pert_decoder.set_mask(dm, freeze=bool(decoder_mask_freeze))

        # ── Singles-trainable non-additivity for combo composition ───────────────────────
        # The ONLY non-additive channel under singles-only training that keeps the decoder
        # (GRN) linear. Both modules are initialised to ZERO output → the model STARTS exactly
        # at the proven linear-clamp baseline and learns a cross-program correction (stable,
        # unlike residual_target). A single TF's shift is already multi-program, so these get
        # real gradient from singles and yield non-additive combos as an emergent consequence.
        self.u_gate_mlp = bool(u_gate_mlp)
        self.u_gate_pure = bool(u_gate_pure)   # u = sigmoid(MLP(S)) directly, not β|S| + tanh(MLP(S))
        self.nonlinear_eps_mix = bool(nonlinear_eps_mix)
        if self.u_gate_mlp:
            self.u_gate = nn.Sequential(nn.Linear(latent_dim, gate_hidden), nn.ReLU(),
                                        nn.Linear(gate_hidden, latent_dim))
            nn.init.zeros_(self.u_gate[-1].weight); nn.init.zeros_(self.u_gate[-1].bias)
        else:
            self.u_gate = None
        if self.nonlinear_eps_mix:
            self.eps_mix_net = nn.Sequential(nn.Linear(latent_dim, gate_hidden), nn.ReLU(),
                                             nn.Linear(gate_hidden, latent_dim))
            nn.init.zeros_(self.eps_mix_net[-1].weight); nn.init.zeros_(self.eps_mix_net[-1].bias)
        else:
            self.eps_mix_net = None

        # ── Learned perturbation shift encoder (SCCVAE-style) ──────────────────
        # Replaces the leaky `dz = D⁺·dx` projection. `pert_label` is a per-gene
        # feature available for unseen genes (default: top-PCA of control
        # co-expression); the shift encoder maps it to a latent-program shift S^p
        # injected as z_post = (I − A^T)^{-1}(ε + c·S^p) (see compute_shift / the
        # shift set-loss). `shift_scale` is the learned penetrance scalar c.
        self.shift_label_dim = int(shift_label_dim)
        self.register_buffer('pert_label',
                             torch.zeros(n_genes, self.shift_label_dim),
                             persistent=True)
        self._pert_label_set = False
        if self.use_shift_encoder:
            self.shift_encoder = ShiftEncoder(
                label_dim=self.shift_label_dim, latent_dim=latent_dim,
                hidden=shift_hidden, n_layers=shift_n_layers,
                output_activation=shift_output_activation,
                use_target_gate=self.use_target_gate,
                gate_tau=float(target_gate_tau), gate_bias_init=float(target_gate_bias_init),
            )
            self.shift_scale = nn.Parameter(torch.tensor(float(shift_scale_init)))
        else:
            self.shift_encoder = None
            self.shift_scale = None

        # Cache for gene-to-latent mapping
        self._gene_to_latent_cache = None

        # Decoder loading matrix L[g, i] = D[g, i]^2 / sum_h D[h, i]^2 — fraction of
        # program i's decoder energy contributed by gene g. Refreshed when D changes.
        # Stored as a buffer so it ships with state_dict; recomputed each epoch in
        # training/train_vae.py and lazily inside compute_loadings() when accessed
        # before any refresh.
        self.register_buffer('_L', torch.zeros(n_genes, latent_dim), persistent=False)
        self.register_buffer('_T', torch.zeros(latent_dim), persistent=False)
        self._loadings_dirty = True

    def set_tau(self, tau_val):
        self.dag.set_tau(tau_val)

    # ──────────────────────────────────────────────────────────────────────
    # Learned perturbation shift (SCCVAE-style additive shift)
    # ──────────────────────────────────────────────────────────────────────
    def set_pert_labels(self, labels):
        """Install the per-gene perturbation-label matrix.

        labels: Tensor or array [n_genes, shift_label_dim]. Computed once from
        control cells (e.g. top-PCA of the transposed control expression matrix)
        in training/train_vae.py and copied into the `pert_label` buffer so it
        ships with the checkpoint.
        """
        labels = torch.as_tensor(labels, dtype=self.pert_label.dtype)
        if labels.shape != self.pert_label.shape:
            raise ValueError(
                f"pert_label shape {tuple(labels.shape)} != expected "
                f"{tuple(self.pert_label.shape)} (n_genes, shift_label_dim)")
        self.pert_label.copy_(labels.to(self.pert_label.device))
        self._pert_label_set = True

    def compute_shift(self, gene_idx):
        """Latent program shift S^p = c · ShiftEncoder(pert_label[gene_idx]).

        gene_idx: LongTensor[...] of perturbed gene indices.
        Returns:  Tensor[..., latent_dim] — the scaled additive latent shift.
        """
        if self.shift_encoder is None:
            raise RuntimeError("compute_shift called but use_shift_encoder=False")
        gi = torch.as_tensor(gene_idx, dtype=torch.long, device=self.pert_label.device)
        label = self.pert_label[gi]                       # [..., label_dim]
        shift = self.shift_encoder(label)                 # [..., d] raw magnitude head
        # Retrieval augmentation: blend in an ESM-similarity-weighted shift retrieved from a TRAIN-only
        # memory (self-excluded, top-k). Leakage-safe: DB = train TFs, query's own row masked.
        if getattr(self, 'retr_val', None) is not None:
            shift = shift + self.retr_alpha * self._retrieve(label, gi)
        # Fix 2: sparse per-TF target gate (ones when use_target_gate=False → identical to legacy).
        # Single choke point — every S-producer routes through compute_shift, so S is gated everywhere.
        gate = self.shift_encoder.sample_gate(label)      # [..., d] soft(train)/hard(eval)
        # Perturbation-blind (oracle hard-target) hook: if set, the shift can act ONLY at the true
        # intervention site — removes the dense-shift absorption channel so downstream effects must
        # flow through A. Off by default (getattr → None) → normal learned gate. Synthetic-eval only.
        og = getattr(self, 'oracle_gate', None)
        if og is not None:
            gate = og[gi]
        return self.shift_scale * (gate * shift)

    def install_retrieval(self, train_gene_idx, val_init, key_dim=128, k=10, tau=1.0):
        """Install the TRAIN-only retrieval memory. train_gene_idx: LongTensor[N_tr] gene indices of
        the train TFs (DB rows). val_init: [N_tr, d] latent value memory (init from D†·train residual).
        Keys = pert_label of those genes (ESM). Leakage-safe: only train TFs, self-masked at query."""
        import torch.nn as nn
        d = self.latent_dim
        self.register_buffer('retr_gene', torch.as_tensor(train_gene_idx, dtype=torch.long))
        self.register_buffer('retr_key', self.pert_label[self.retr_gene].detach().clone())  # [N_tr, label_dim]
        self.retr_val = nn.Parameter(torch.as_tensor(val_init, dtype=torch.float32))          # [N_tr, d]
        self.retr_proj = nn.Linear(self.retr_key.shape[1], key_dim, bias=False)
        self.retr_alpha = nn.Parameter(torch.tensor(0.5))
        self.retr_k = int(k); self.retr_tau = float(tau)
        self.to(self.retr_key.device)

    def _retrieve(self, label, gi):
        """ESM-similarity attention over the train memory, self-excluded + top-k. label[...,L], gi[...]
        -> retrieved shift [..., d]."""
        L = label.shape[-1]; d = self.latent_dim
        q = self.retr_proj(label.reshape(-1, L))                       # [M, key_dim]
        K = self.retr_proj(self.retr_key)                              # [N_tr, key_dim]
        scores = (q @ K.t()) / (self.retr_proj.out_features ** 0.5 * self.retr_tau)  # [M, N_tr]
        gflat = gi.reshape(-1)
        # self-mask: a query TF cannot attend to its OWN train-DB row (no self-leakage)
        selfmask = (gflat[:, None] == self.retr_gene[None, :])
        scores = scores.masked_fill(selfmask, float('-inf'))
        if self.retr_k < scores.shape[1]:                              # keep top-k neighbours only
            kth = scores.topk(self.retr_k, dim=1).values[:, -1:]
            scores = scores.masked_fill(scores < kth, float('-inf'))
        attn = torch.softmax(scores, dim=1)                            # [M, N_tr]
        retrieved = attn @ self.retr_val                               # [M, d]
        return retrieved.reshape(*label.shape[:-1], d)

    def compute_gate_prob(self, gene_idx):
        """Expected open-gate probability per program (for the L0 target-sparsity penalty).
        Returns ones when use_target_gate=False."""
        gi = torch.as_tensor(gene_idx, dtype=torch.long, device=self.pert_label.device)
        return self.shift_encoder.gate_prob(self.pert_label[gi])   # [..., d]

    def refine_u(self, u_base, S):
        """Cross-program nonlinear correction to the do-operator strength u.
        u_base, S: [..., d]. No-op (returns u_base) unless u_gate_mlp is set; the gate is
        zero-init so it starts at u_base and learns set-dependent (non-additive) gating."""
        if self.u_gate is None:
            return u_base
        if self.u_gate_pure:
            # u-score comes ENTIRELY from the gate: u = sigmoid(MLP(S)). Zero-init final layer →
            # u≡0.5 at start (non-degenerate), then learns the per-program clamp from S. Ignores β|S|.
            return torch.sigmoid(self.u_gate(S)).clamp(0.0, 1.0)
        return (u_base + torch.tanh(self.u_gate(S))).clamp(0.0, 1.0)

    def mix_eps(self, eps_ctrl, z_target, u):
        """Effective exogenous noise: ε̃ = ε_ctrl + u⊙(z_target−ε_ctrl) [+ nonlinear correction].
        All args broadcast on the last dim d. Starts at the linear interpolation (net zero-init);
        the correction couples programs → non-additive combos, trained on singles."""
        shift = u * (z_target - eps_ctrl)
        eps_tilde = eps_ctrl + shift
        if self.eps_mix_net is not None:
            eps_tilde = eps_tilde + torch.tanh(self.eps_mix_net(shift))
        return eps_tilde

    def intervene_shift_mutilate(self, epsilon, gene_indices, A, z_prev,
                                 u_mode='loading', u_clip_max=0.5, shift_beta=1.0):
        """Shift-fed mutilation do-operator (Tier-2), for single or combo perturbations.

            S         = Σ_k compute_shift(gene_k)                 (summed latent shift)
            z_S       = (I − A^T)^{-1} S ;  z_target = z_prev + z_S
            u         = shift_clamp_u(S, genes, mode)             (which programs to clamp)
            ε̃         = u ⊙ z_target + (1 − u) ⊙ ε
            z_post    = (I − M A^T)^{-1} ε̃,   M = diag(1 − u)      (mutilated solve)

        Reduces to the additive shift `z_prev + z_S` when u→1 on all driven programs,
        and to predict-control when u→0. Non-additivity for combos arises because u
        (and hence M) depends nonlinearly on the perturbed SET. Single-A version
        (eval / per-perturbation); training uses a batched form in the loss.

        epsilon:      Tensor[B, d]
        gene_indices: int | list[int] | LongTensor — the perturbed gene(s)
        A:            Tensor[d, d]
        z_prev:       Tensor[B, d]  (= scm_forward(epsilon, A))
        Returns: (z_post Tensor[B, d], u Tensor[d])
        """
        gi = torch.as_tensor(gene_indices, dtype=torch.long, device=epsilon.device).reshape(-1)
        S = self.compute_shift(gi).sum(dim=0)                        # [d] — summed (gated) shift
        d = self.latent_dim
        # per-program intervention strength u (SET semantics: sum over the genes)
        if u_mode == 'shift':
            u = (shift_beta * S.abs()).clamp(max=1.0)                # [d]  (S already summed)
        elif u_mode == 'loading':
            u = self.get_loadings()[gi].sum(dim=0).clamp(0.0, u_clip_max)  # [k,d]→[d]
        else:
            raise ValueError(f"unknown u_mode={u_mode!r}")
        if self.pert_intervene_mode == 'structural':
            # Fix 1: inject S into the exogenous noise; ONE mutilated solve propagates it
            # through A. No z_target pre-propagation → downstream (non-target) programs move
            # ONLY via A, so A=0 ⇒ no downstream effect (A becomes load-bearing by construction).
            eps_tilde = epsilon + S.unsqueeze(0)                     # [B, d]
        else:
            # legacy (exp19): pre-propagate S and inject the full target state (the leak).
            I = torch.eye(d, device=epsilon.device, dtype=epsilon.dtype)
            z_S = _robust_solve(I - A.T, S)                          # [d]
            z_target = z_prev + z_S.unsqueeze(0)                     # [B, d]
            eps_tilde = u.unsqueeze(0) * z_target + (1.0 - u).unsqueeze(0) * epsilon
        z_post, _, _ = self.dag.scm_intervene_u(epsilon, A, eps_tilde, u)
        return z_post, u

    def forward(self, x, bypass_scm=False, scm_alpha=None, modality_id=None,
                batch_id=None):
        """
        Observational forward pass.
        x: Tensor[B, n_genes]
        bypass_scm: if True, skip SCM solve and use z = epsilon directly.
                    Used during Phase A when DAG is not yet active.
        scm_alpha: float in [0, 1] or None. If provided, interpolate:
                   z = (1 - alpha) * epsilon + alpha * scm_forward(epsilon, A)
                   Used during DAG warm-in to smooth the transition.
                   None means full SCM (alpha=1) when not bypassing.
        modality_id: LongTensor[B] (optional, for multi-modality joint training)
        batch_id:    LongTensor[B] (optional, gem_group id for batch-effect conditioning)
        Returns: dict with x_hat, mu, log_var, epsilon, z, A
        """
        # Encode
        mu, log_var, epsilon = self.encoder(x, modality_id=modality_id,
                                            batch_id=batch_id)

        # Sample DAG
        A = self.dag.sample_A(n_samples=1).squeeze(0)  # [d, d]

        # SCM forward (or bypass, or interpolate)
        if bypass_scm:
            z = epsilon  # identity: no causal structure yet
        elif scm_alpha is not None and scm_alpha < 1.0:
            z_scm = self.dag.scm_forward(epsilon, A)  # [B, d]
            z = (1.0 - scm_alpha) * epsilon + scm_alpha * z_scm
        else:
            z = self.dag.scm_forward(epsilon, A)  # [B, d]

        # Decode
        x_hat = self.decoder(z)  # [B, n_genes]

        return {
            'x_hat': x_hat,
            'mu': mu,
            'log_var': log_var,
            'epsilon': epsilon,
            'z': z,
            'A': A,
        }

    def get_gene_to_latent_mapping(self):
        """
        For each gene g, find which latent program it most aligns with.
        latent_idx[g] = argmax_i |D[g, i]|
        Returns: (mapping array [n_genes], alignment strengths [n_genes])
        """
        D = self.decoder.effective_D().detach()  # [n_genes, d]
        D_abs = D.abs()
        latent_idx = D_abs.argmax(dim=1)  # [n_genes]
        strengths = D_abs.max(dim=1).values  # [n_genes]
        self._gene_to_latent_cache = latent_idx
        return latent_idx.cpu().numpy(), strengths.cpu().numpy()

    def _get_latent_for_gene(self, gene_idx):
        """Get the most aligned latent index for a given gene index (legacy argmax)."""
        D = self.decoder.effective_D().detach()  # [n_genes, d]
        latent_idx = D[gene_idx].abs().argmax().item()
        return latent_idx

    def compute_dz_from_dx(self, dx):
        """
        Project a gene-space change to latent space via pseudoinverse of D.

        Uses solve-based D†: dz = (D^T D + λI)^{-1} D^T dx

        dx: Tensor[n_genes]
        Returns: dz Tensor[d]
        """
        D = self.decoder.effective_D().detach()  # [n_genes, d]
        d = self.latent_dim
        DtD = D.T @ D + 1e-6 * torch.eye(d, device=D.device)  # [d, d]
        Dt_dx = D.T @ dx  # [d]
        return _robust_solve(DtD, Dt_dx)  # [d]

    def get_latent_for_gene_v2(self, dx, A):
        """
        Identify the primary intervention target latent using D† and DAG structure.

        1. dz = D† @ dx  (gene-space → latent-space)
        2. dε = (I - A^T) @ dz  (latent-space → exogenous-space via DAG)
        3. target = argmax|dε|  (which exogenous variable changed most)

        dx: Tensor[n_genes] — gene-space perturbation effect (e.g. de_single)
        A:  Tensor[d, d] — sampled adjacency matrix
        Returns: (latent_idx: int, d_epsilon: Tensor[d], shift: float)
        """
        dz = self.compute_dz_from_dx(dx)
        I_mat = torch.eye(self.latent_dim, device=A.device)
        d_epsilon = (I_mat - A.T) @ dz  # [d]
        latent_idx = d_epsilon.abs().argmax().item()
        return latent_idx, d_epsilon, d_epsilon[latent_idx].item()

    def intervene(self, x_ctrl, pert_gene_idx, pert_value=0.0, modality_id=None):
        """
        Single-sample intervention: do(gene pert_gene_idx = pert_value).

        x_ctrl: Tensor[B, n_genes]
        pert_gene_idx: int (index into gene expression vector)
        pert_value: float (0.0 = knockout)
        modality_id: LongTensor[B] (optional)
        Returns: dict with x_hat_pert, z_post, latent_idx, A_used
        """
        mu, log_var, epsilon = self.encoder(x_ctrl, modality_id=modality_id)
        A = self.dag.sample_A(n_samples=1).squeeze(0)  # [d, d]
        latent_idx = self._get_latent_for_gene(pert_gene_idx)
        z_post = self.dag.scm_intervene(epsilon, A, latent_idx, pert_value)
        x_hat_pert = self.decoder(z_post)

        return {
            'x_hat_pert': x_hat_pert,
            'z_post': z_post,
            'latent_idx': latent_idx,
            'A_used': A,
            'epsilon': epsilon,
        }

    def intervene_bootstrapped(self, x_ctrl, pert_gene_idx, pert_value=0.0,
                                n_samples=None, modality_id=None):
        """
        Bootstrapped intervention: sample N different DAGs.

        x_ctrl: Tensor[B, n_genes]
        modality_id: LongTensor[B] (optional)
        Returns: dict with all_x_hat [N, B, n_genes], mean_x_hat, std_x_hat
        """
        if n_samples is None:
            n_samples = self.n_dag_samples

        mu, log_var, epsilon = self.encoder(x_ctrl, modality_id=modality_id)
        latent_idx = self._get_latent_for_gene(pert_gene_idx)

        # Sample N DAGs
        A_batch = self.dag.sample_A(n_samples=n_samples)  # [N, d, d]

        all_x_hat = []
        for n in range(n_samples):
            A = A_batch[n]
            z_post = self.dag.scm_intervene(epsilon, A, latent_idx, pert_value)
            x_hat = self.decoder(z_post)
            all_x_hat.append(x_hat)

        all_x_hat = torch.stack(all_x_hat, dim=0)  # [N, B, n_genes]
        mean_x_hat = all_x_hat.mean(dim=0)
        std_x_hat = all_x_hat.std(dim=0)

        return {
            'all_x_hat': all_x_hat,
            'mean_x_hat': mean_x_hat,
            'std_x_hat': std_x_hat,
            'latent_idx': latent_idx,
            'epsilon': epsilon,
        }

    def intervene_on_latent_bootstrapped(self, epsilon, latent_idx, pert_value=0.0, n_samples=None):
        """
        Bootstrapped intervention directly on epsilon (used in RL env).

        epsilon: Tensor[B, d] -- exogenous noise (already encoded)
        latent_idx: int -- which latent program to intervene on
        Returns: dict with all_x_hat [N, B, n_genes], mean_x_hat, std_x_hat
        """
        if n_samples is None:
            n_samples = self.n_dag_samples

        A_batch = self.dag.sample_A(n_samples=n_samples)  # [N, d, d]

        all_x_hat = []
        all_z_post = []
        for n in range(n_samples):
            A = A_batch[n]
            z_post = self.dag.scm_intervene(epsilon, A, latent_idx, pert_value)
            x_hat = self.decoder(z_post)
            all_x_hat.append(x_hat)
            all_z_post.append(z_post)

        all_x_hat = torch.stack(all_x_hat, dim=0)  # [N, B, n_genes]
        all_z_post = torch.stack(all_z_post, dim=0)  # [N, B, d]
        mean_x_hat = all_x_hat.mean(dim=0)
        std_x_hat = all_x_hat.std(dim=0)

        return {
            'all_x_hat': all_x_hat,
            'all_z_post': all_z_post,
            'mean_x_hat': mean_x_hat,
            'std_x_hat': std_x_hat,
        }

    def multi_step_intervene(self, x_ctrl, pert_sequence, n_samples=None, modality_id=None):
        """
        Apply a sequence of perturbations.

        x_ctrl: Tensor[B, n_genes]
        pert_sequence: list of (gene_idx, pert_value) tuples
        n_samples: int (DAG bootstrap samples)
        modality_id: LongTensor[B] (optional)
        Returns: dict with:
            all_trajectories: [n_steps+1, N, B, n_genes]
            mean_trajectory: [n_steps+1, B, n_genes]
        """
        if n_samples is None:
            n_samples = self.n_dag_samples

        mu, log_var, epsilon = self.encoder(x_ctrl, modality_id=modality_id)
        A_batch = self.dag.sample_A(n_samples=n_samples)  # [N, d, d]

        # Initial decoded state
        all_trajectories = []
        x_hat_init_list = []
        for n in range(n_samples):
            z0 = self.dag.scm_forward(epsilon, A_batch[n])
            x_hat_init_list.append(self.decoder(z0))
        all_trajectories.append(torch.stack(x_hat_init_list, dim=0))  # [N, B, n_genes]

        # Current epsilon for each DAG sample -- track per-sample
        current_epsilons = [epsilon.clone() for _ in range(n_samples)]
        current_As = [A_batch[n] for n in range(n_samples)]

        for gene_idx, pert_value in pert_sequence:
            latent_idx = self._get_latent_for_gene(gene_idx)
            step_x_hats = []
            new_epsilons = []
            for n in range(n_samples):
                # Apply intervention
                A = current_As[n]
                eps = current_epsilons[n]
                z_post = self.dag.scm_intervene(eps, A, latent_idx, pert_value)
                x_hat = self.decoder(z_post)
                step_x_hats.append(x_hat)
                # For next step: treat z_post as new epsilon
                new_epsilons.append(z_post)
            all_trajectories.append(torch.stack(step_x_hats, dim=0))
            current_epsilons = new_epsilons

        all_traj = torch.stack(all_trajectories, dim=0)  # [n_steps+1, N, B, n_genes]
        mean_traj = all_traj.mean(dim=1)  # [n_steps+1, B, n_genes]

        return {
            'all_trajectories': all_traj,
            'mean_trajectory': mean_traj,
        }

    def intervene_gene(self, epsilon, dx, A, threshold_frac=0.1):
        """
        Gene-level intervention via D† pipeline (proposal Eq. 1).

        1. dz = D†·dx                    (gene-space → latent-space)
        2. dε = (I − A^T)·dz             (latent → exogenous)
        3. Multi-mutilate A where |dε| > threshold, apply full dε

        epsilon: Tensor[B, d] — current exogenous noise
        dx: Tensor[n_genes] — gene-space perturbation effect
        A: Tensor[d, d] — sampled adjacency matrix
        threshold_frac: float — fraction of max|dε| for mutilation
        Returns: (z_post [B, d], d_epsilon [d])
        """
        dz = self.compute_dz_from_dx(dx)  # [d]
        I_mat = torch.eye(self.latent_dim, device=A.device)
        d_epsilon = (I_mat - A.T) @ dz  # [d]
        z_post = self.dag.scm_intervene_multi(
            epsilon, A, d_epsilon, threshold_frac=threshold_frac
        )
        return z_post, d_epsilon

    def intervene_gene_bootstrapped(self, epsilon, dx, n_samples=None, threshold_frac=0.1):
        """
        Bootstrapped gene-level intervention: sample N DAGs, each producing
        a different prediction. Returns mean/std for uncertainty.

        epsilon: Tensor[B, d]
        dx: Tensor[n_genes] — gene-space perturbation effect
        Returns: dict with all_x_hat [N, B, n_genes], all_z_post [N, B, d],
                 mean_x_hat, std_x_hat, all_d_epsilon [N, d]
        """
        if n_samples is None:
            n_samples = self.n_dag_samples

        A_batch = self.dag.sample_A(n_samples=n_samples)  # [N, d, d]

        all_x_hat = []
        all_z_post = []
        all_d_epsilon = []
        for n in range(n_samples):
            A = A_batch[n]
            z_post, d_eps = self.intervene_gene(
                epsilon, dx, A, threshold_frac=threshold_frac
            )
            x_hat = self.decoder(z_post)
            all_x_hat.append(x_hat)
            all_z_post.append(z_post)
            all_d_epsilon.append(d_eps)

        all_x_hat = torch.stack(all_x_hat, dim=0)      # [N, B, n_genes]
        all_z_post = torch.stack(all_z_post, dim=0)     # [N, B, d]
        all_d_epsilon = torch.stack(all_d_epsilon, dim=0)  # [N, d]

        return {
            'all_x_hat': all_x_hat,
            'all_z_post': all_z_post,
            'mean_x_hat': all_x_hat.mean(dim=0),
            'std_x_hat': all_x_hat.std(dim=0),
            'all_d_epsilon': all_d_epsilon,
        }

    # ──────────────────────────────────────────────────────────────────────
    # u-score (decoder-loading) intervention scheme
    # ──────────────────────────────────────────────────────────────────────
    def compute_loadings(self, eps=1e-8):
        """
        Recompute the loading matrix L from the current decoder D.

        **Phase H — row-max normalization** (replaces the Phase B column-energy
        normalization `L[g,i] = D[g,i]^2 / Σ_h D[h,i]^2`, which collapsed to
        ≈ 1/n_genes per entry for dense orthogonal D, making u tiny and the
        do-operator barely engage A):

            L[g, i] = D[g, i]^2 / max_j D[g, j]^2

        Per-gene `max_i L[g, i] = 1.0` exactly (the strongest program for gene
        g gets full pinning). Other programs get fractional values in [0, 1].
        Combined with `u_clip_max = 0.5`, the strongest program is half-pinned,
        forcing the SCM solve to actually propagate the intervention through A.

        Trade-off vs Phase B: we lose the property `Σ_g L[g,i] = 1` (which had
        the "fraction of program i's energy from gene g" interpretation). The
        new property is "what fraction of each gene's *peak* loading does each
        program receive" — more directly aligned with how strongly the do-operator
        pins each program when we perturb that gene.

        T_i = Σ_h D[h,i]^2 (program-energy) is still computed and stored on
        `self._T` for diagnostics.

        Returns: (L Tensor[n_genes, d], T Tensor[d])
        """
        with torch.no_grad():
            D = self.decoder.effective_D().detach()                          # [n_genes, d]
            D_sq = D * D                                         # [n_genes, d]
            T = D_sq.sum(dim=0) + eps                            # [d] — program-energy (diagnostics)
            row_max = D_sq.max(dim=1, keepdim=True).values + eps  # [n_genes, 1] — per-gene peak
            L = D_sq / row_max                                   # [n_genes, d], max per row = 1.0
            # Match dtype/device of buffer slot
            self._L = L.to(device=self._L.device if self._L.numel() else L.device)
            self._T = T.to(device=self._T.device if self._T.numel() else T.device)
            self._loadings_dirty = False
        return self._L, self._T

    def get_loadings(self):
        """Return cached L (refresh lazily if it has not been computed yet)."""
        if self._loadings_dirty or self._L.numel() == 0 or float(self._T.abs().sum()) == 0.0:
            self.compute_loadings()
        return self._L

    def intervene_gene_u(self, epsilon, dx, A, u_prev, z_prev, gene_idx,
                        u_clip_max=1.0):
        """
        Single-step u-score intervention.

        Math (one step):
            dz       = D⁺ · dx                               (gene → latent)
            z_target = z_prev + dz
            u        = clip( u_prev + L[gene_idx, :],  0, u_clip_max )
            eps_tilde = u ⊙ z_target + (1 - u) ⊙ epsilon
            (z_post, A_tilde, M) = dag.scm_intervene_u(epsilon, A, eps_tilde, u)

        epsilon:   Tensor[B, d]                — exogenous noise at step t
        dx:        Tensor[n_genes]             — gene-space perturbation effect for this step
        A:         Tensor[d, d]                — sampled adjacency
        u_prev:    Tensor[d]                   — accumulated intervention strength so far
        z_prev:    Tensor[B, d] or Tensor[d]   — current latent (broadcast if [d])
        gene_idx:  int                         — gene being perturbed this step
        u_clip_max: float                      — clip ceiling for u (default 1.0)

        Returns:
            z_post   Tensor[B, d]
            u        Tensor[d]                  — updated intervention strength
            A_tilde  Tensor[d, d]               — column-scaled adjacency
            z_target Tensor[B, d]               — for diagnostics
            dz       Tensor[d]                  — for diagnostics
        """
        L = self.get_loadings()                                # [n_genes, d]
        dz = self.compute_dz_from_dx(dx)                       # [d]

        if z_prev.dim() == 1:
            z_target = z_prev.unsqueeze(0) + dz.unsqueeze(0)   # [1, d]
        else:
            z_target = z_prev + dz.unsqueeze(0)                # [B, d]

        u = (u_prev + L[gene_idx]).clamp(min=0.0, max=u_clip_max)  # [d]
        # Broadcast u over batch
        u_b = u.unsqueeze(0)                                   # [1, d]
        eps_tilde = u_b * z_target + (1.0 - u_b) * epsilon     # [B, d]

        z_post, A_tilde, _M = self.dag.scm_intervene_u(epsilon, A, eps_tilde, u)
        return z_post, u, A_tilde, z_target, dz

    def intervene_combo_u(self, epsilon, dx_list, A, z_prev, gene_indices,
                           u_clip_max=None):
        """
        Simultaneous (commutative) u-score do-operator for N co-perturbations.

        Math:
            u_combo  = clip( Σ_k L[gene_indices[k], :],  0, u_clip_max )
            dx_combo = Σ_k dx_list[k]                       (gene-space, additive by construction)
            dz_combo = D⁺ · dx_combo  =  Σ_k D⁺ · dx_list[k]  (linearity of D⁺)
            z_target = z_prev + dz_combo
            eps_tilde = u_combo ⊙ z_target + (1 - u_combo) ⊙ epsilon
            z_post = (I − M_combo · A^T)^{-1} eps_tilde      (single SCM solve)

        Order-independent by construction (commutativity error = exactly 0).
        Strict generalisation of the additive baseline: when A=0, M=I, and the
        SCM solve degenerates to identity, recovering ctrl + dz_combo (≈ additive
        in gene space).  When A has structure, the propagation through
        (I − MA^T)^{-1} adds learned regulatory epistasis on top.

        epsilon:       Tensor[B, d]                   — exogenous noise from encoder
        dx_list:       List[Tensor[n_genes]]          — per-gene gene-space shifts
                                                        (typically de_single[g])
        A:             Tensor[d, d]                   — sampled adjacency
        z_prev:        Tensor[B, d] or Tensor[d]      — un-intervened latent
                                                        (typically scm_forward(epsilon, A))
        gene_indices:  List[int] | LongTensor[N]      — genes being co-perturbed
        u_clip_max:    float or None                  — defaults to self.u_clip_max

        Returns:
            z_post     Tensor[B, d]
            u_combo    Tensor[d]                       — clipped per-program intervention strength
            A_tilde    Tensor[d, d]                    — column-scaled adjacency
        """
        if u_clip_max is None:
            u_clip_max = self.u_clip_max

        L = self.get_loadings()                              # [n_genes, d]
        gene_idx_t = torch.as_tensor(gene_indices, dtype=torch.long, device=L.device)
        u_combo = L[gene_idx_t].sum(dim=0).clamp(min=0.0, max=u_clip_max)  # [d]

        # dz_combo = D⁺ (Σ dx_k) = Σ D⁺ dx_k  (linearity of D⁺ — equivalent forms;
        # we sum dx first to minimise solves).
        dx_total = sum(dx_list)                              # Tensor[n_genes]
        dz_combo = self.compute_dz_from_dx(dx_total)         # [d]

        if z_prev.dim() == 1:
            z_target = z_prev.unsqueeze(0) + dz_combo.unsqueeze(0)
        else:
            z_target = z_prev + dz_combo.unsqueeze(0)

        u_b = u_combo.unsqueeze(0)                           # [1, d]
        eps_tilde = u_b * z_target + (1.0 - u_b) * epsilon   # [B, d]
        z_post, A_tilde, _M = self.dag.scm_intervene_u(epsilon, A, eps_tilde, u_combo)
        return z_post, u_combo, A_tilde

    def recover_epsilon_u(self, z, A, u):
        """
        Recover exogenous noise in the post-intervention (mutilated) regime:

            ε_{t+1} = (I − M A^T) z = z − diag(1-u) · A^T · z

        Used between RL/eval steps so the next step starts from a noise vector
        consistent with the persisted intervention. Reduces to recover_epsilon() at u=0.

        z: Tensor[B, d]
        A: Tensor[d, d]
        u: Tensor[d] in [0, 1]
        Returns: epsilon Tensor[B, d]
        """
        one_minus_u = (1.0 - u).to(dtype=z.dtype)             # [d]
        # (I - M A^T) z, batched as row-vectors: z − (z · A) · diag(1-u)
        # since for each row z_b: ((I - M A^T) z_b)_i = z_b[i] - (1-u_i) * (A^T z_b)_i
        # In matrix form on rows: z @ I − z @ (M A^T)^T = z − z @ A @ M
        eps = z - (z @ A) * one_minus_u.unsqueeze(0)           # [B, d]
        eps = torch.nan_to_num(eps, nan=0.0, posinf=20.0, neginf=-20.0)
        eps = torch.clamp(eps, min=-20.0, max=20.0)
        return eps

    def recover_epsilon(self, z, A):
        """
        Recover exogenous noise from endogenous variables:
        ε = (I − A^T) · z

        Used for multi-step interventions: after solving z_{t+1},
        recover ε_{t+1} for the next step.

        z: Tensor[B, d]
        A: Tensor[d, d]
        Returns: epsilon Tensor[B, d]
        """
        I_mat = torch.eye(self.latent_dim, device=A.device)
        eps = z @ (I_mat - A)  # [B, d] = [B, d] @ [d, d]
        # Clamp to prevent multi-step explosion + NaN guard
        eps = torch.nan_to_num(eps, nan=0.0, posinf=20.0, neginf=-20.0)
        eps = torch.clamp(eps, min=-20.0, max=20.0)
        return eps

    def encode(self, x, modality_id=None, batch_id=None):
        """Encode x to epsilon. Returns mu, log_var, epsilon.
        modality_id: LongTensor[B] (optional, for multi-modality models)
        batch_id:    LongTensor[B] (optional, gem_group id for batch-effect conditioning)
        """
        return self.encoder(x, modality_id=modality_id, batch_id=batch_id)

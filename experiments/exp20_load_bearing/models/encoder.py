import torch
import torch.nn as nn


class SENAEncoder(nn.Module):
    """SENA-δ encoder (de la Fuente et al., ICLR 2025): a masked gene→pathway layer
    that anchors latent programs to known GO:BP gene sets, breaking the rotational
    non-identifiability of the linear encoder→linear-decoder pair (so the latent DAG A
    becomes load-bearing & interpretable instead of absorbable into the decoder).

    Architecture (matches the paper):
        α   = act( x @ (W ⊙ M) + b )                      # pathway activities  [B, K]
        μ   = α @ Wμ ,   log_var = α @ Wσ                  # linear meta-pathway heads

    M is the gene×pathway membership mask with leak λ (1 for members, λ for non-members);
    stored as a persistent buffer so it reloads exactly. W is masked at every forward so
    non-member genes contribute only through the λ leak. Optional modality/batch shifts are
    added on the pathway-activity hidden (mirrors CausalEncoder), unused for singles runs.
    """

    def __init__(self, input_dim, latent_dim, mask, activation='tanh',
                 n_modalities=0, n_batches=0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        mask = torch.as_tensor(mask, dtype=torch.float32)
        assert mask.shape[0] == input_dim, f"mask rows {mask.shape[0]} != n_genes {input_dim}"
        self.n_pathways = int(mask.shape[1])
        self.register_buffer('M', mask, persistent=True)       # [n_genes, K]
        self.W = nn.Parameter(torch.empty(input_dim, self.n_pathways))
        nn.init.xavier_uniform_(self.W)
        self.bias = nn.Parameter(torch.zeros(self.n_pathways))
        self.activation = activation
        self._act = {'tanh': torch.tanh, 'relu': torch.relu,
                     'none': (lambda z: z)}[activation]
        self.n_modalities = n_modalities
        self.n_batches = n_batches
        self.modality_emb = nn.Embedding(n_modalities, self.n_pathways) if n_modalities > 0 else None
        self.batch_emb = nn.Embedding(n_batches, self.n_pathways) if n_batches > 0 else None
        self.mu_head = nn.Linear(self.n_pathways, latent_dim)
        self.log_var_head = nn.Linear(self.n_pathways, latent_dim)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def pathway_activity(self, x):
        """α = act(x @ (W⊙M) + b)  — the interpretable per-pathway activity [B, K]."""
        return self._act(x @ (self.W * self.M) + self.bias)

    def forward(self, x, modality_id=None, batch_id=None):
        alpha = self.pathway_activity(x)                        # [B, K]
        if self.modality_emb is not None:
            if modality_id is None:
                modality_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            alpha = alpha + self.modality_emb(modality_id)
        if self.batch_emb is not None:
            if batch_id is None:
                batch_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            alpha = alpha + self.batch_emb(batch_id)
        mu = torch.clamp(self.mu_head(alpha), min=-30.0, max=30.0)   # KL 0.5·mu² overflow guard (large d)
        log_var = torch.clamp(self.log_var_head(alpha), min=-10.0, max=10.0)
        epsilon = self.reparameterize(mu, log_var)
        return mu, log_var, epsilon


class CausalEncoder(nn.Module):
    """
    Maps gene expression (and optionally a modality token) to exogenous noise epsilon.

    Architecture:
        input (n_genes,) [+ modality_emb (16,)] -> Linear(d, 512) -> LayerNorm -> ReLU
                                                -> Linear(512, 256) -> LayerNorm -> ReLU
                                                -> Linear(256, 2*latent_dim) [mu, log_var]

    If `n_modalities > 0`, the encoder accepts an integer modality_id per cell and
    learns an embedding (default 16-dim) that is concatenated to the gene vector.
    Used for joint training across CRISPRa (Norman) + CRISPRi (Replogle) where the
    same gene perturbation produces opposite gene-space changes.
    """

    def __init__(self, input_dim, latent_dim=64, n_modalities=0, modality_dim=16,
                 n_batches=0, batch_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_modalities = n_modalities
        self.modality_dim = modality_dim
        self.n_batches = n_batches
        self.batch_dim = batch_dim

        if n_modalities > 0:
            self.modality_emb = nn.Embedding(n_modalities, modality_dim)
            mlp_in_dim = input_dim + modality_dim
        else:
            self.modality_emb = None
            mlp_in_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(mlp_in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        # Optional additive batch-effect shift, applied AFTER the MLP feature extraction.
        # Mirrors STATE's batch_encoder pattern: batch shift is independent of gene-level
        # feature extraction, so the same encoder weights work across gem_groups.
        if n_batches > 0:
            self.batch_emb = nn.Embedding(n_batches, 256)
        else:
            self.batch_emb = None

        self.mu_head = nn.Linear(256, latent_dim)
        self.log_var_head = nn.Linear(256, latent_dim)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(self, x, modality_id=None, batch_id=None):
        """
        x: Tensor[B, input_dim]
        modality_id: LongTensor[B] (optional; required if n_modalities > 0)
        batch_id:    LongTensor[B] (optional; required if n_batches > 0).  Per-cell
                     batch / gem_group index for additive batch-effect conditioning.
        Returns: mu, log_var, epsilon (each Tensor[B, latent_dim])
        """
        if self.modality_emb is not None:
            if modality_id is None:
                # Default to modality 0 (control) if not provided
                modality_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            m_emb = self.modality_emb(modality_id)  # [B, modality_dim]
            x_in = torch.cat([x, m_emb], dim=-1)
        else:
            x_in = x

        h = self.net(x_in)

        if self.batch_emb is not None:
            if batch_id is None:
                batch_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            h = h + self.batch_emb(batch_id)  # additive batch shift on the hidden

        mu = torch.clamp(self.mu_head(h), min=-30.0, max=30.0)   # KL 0.5·mu² overflow guard (large d)
        log_var = self.log_var_head(h)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        epsilon = self.reparameterize(mu, log_var)
        return mu, log_var, epsilon

import torch
import torch.nn as nn


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

        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        epsilon = self.reparameterize(mu, log_var)
        return mu, log_var, epsilon

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDecoder(nn.Module):
    """
    Linear decoder: x_hat = z @ D^T + b
    (equivalently x_hat = D @ z + b for single sample)

    D: nn.Parameter of shape (n_genes, d) -- decoder weight matrix
    b: nn.Parameter of shape (n_genes,) -- bias

    The linear structure allows: G_feat = D @ A @ D^T (gene-space GRN)
    """

    # Flag read by training/eval code to decide whether D-dependent regularizers
    # (orthogonality, decoder-L1, GRN prior via D·A·D^T, u-score loadings) apply.
    is_linear = True

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.D = nn.Parameter(torch.empty(output_dim, latent_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        self.mask_freeze = False      # True -> D fixed = mask exactly; else masked-LEARNABLE (W⊙mask)
        # NOTE: dec_mask is registered as a buffer only in set_mask() — models without a mask never
        # carry it, so their state_dict is unchanged and old checkpoints load strict-clean.

        nn.init.orthogonal_(self.D)

    def set_mask(self, mask, freeze=False):
        """Install the SENA gene->pathway λ-residual mask. `mask` is [n_genes, d] with 1.0 for
        annotated gene-pathway pairs and λ off-annotation (i.e. M + λ(1−M)); each latent = a pathway.
        masked-LEARNABLE (default): effective weight = D ⊙ mask, so annotated pairs learn at full
        strength and off-annotation pairs are λ-shrunk -> the decoder must route gene covariance
        through the pathway latents + A rather than wire arbitrary gene-gene edges.
        freeze=True: D := mask, requires_grad off (strict SENA; cripples reconstruction, not used)."""
        m = mask if torch.is_tensor(mask) else torch.as_tensor(mask, dtype=self.D.dtype)
        self.register_buffer('dec_mask', m.to(self.D.dtype))
        self.mask_freeze = bool(freeze)
        if self.mask_freeze:
            with torch.no_grad():
                self.D.copy_(self.dec_mask)
            self.D.requires_grad_(False)

    def effective_D(self):
        """Weight actually used in reconstruction / GRN readout / loadings."""
        m = getattr(self, 'dec_mask', None)
        if m is None:
            return self.D
        return m if self.mask_freeze else self.D * m

    def forward(self, z):
        """
        z: Tensor[B, latent_dim]
        Returns: x_hat Tensor[B, output_dim]
        """
        return z @ self.effective_D().T + self.b

    def orthogonality_loss(self):
        """
        L_ortho = ||D^T D / (||D||_F^2 / d + eps) - I||_F^2
        Encourages each latent program to map to a distinct set of genes.
        """
        d = self.latent_dim
        eps = 1e-8
        D = self.D  # [n_genes, d]
        DtD = D.T @ D  # [d, d]
        norm_sq = torch.sum(D ** 2) / d + eps
        DtD_normalized = DtD / norm_sq
        I = torch.eye(d, device=D.device, dtype=D.dtype)
        return torch.norm(DtD_normalized - I, p='fro') ** 2

    def project_dag_to_genes(self, A):
        """
        Project the latent DAG to gene space: G_feat = D @ A @ D^T
        G_feat[i,j] > 0 suggests gene i regulates gene j.

        A: Tensor[d, d]
        Returns: G_feat Tensor[n_genes, n_genes]
        """
        D = self.effective_D()
        return D @ A @ D.T


class MLPDecoder(nn.Module):
    """
    Nonlinear (MLP) decoder: x_hat = MLP(z).

    SCCVAE keeps the SCM linear-Gaussian in latent space but pairs it with a
    nonlinear decoder (2 hidden layers of 1024, LeakyReLU(0.2)). This lifts the
    rank-d linear bottleneck of `LinearDecoder` (where x_pert − x_ctrl is forced
    into colspace(D)) so the model can represent richer, higher-rank perturbation
    responses.

    There is no single decoder weight matrix `D`, so the LDVAE interpretability
    readouts (G_feat = D·A·D^T, column-orthogonality, decoder-L1, u-score
    loadings) do not apply. `is_linear = False` signals downstream code to skip
    those D-dependent regularizers; configs for this decoder set
    lambda_ortho / lambda_D_sparse / lambda_prior to 0.
    """

    is_linear = False

    def __init__(self, latent_dim, output_dim, hidden=1024, n_layers=2,
                 negative_slope=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers = []
        in_dim = latent_dim
        for _ in range(max(n_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LeakyReLU(negative_slope))
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, output_dim)

    def forward(self, z):
        """z: Tensor[..., latent_dim] → x_hat Tensor[..., output_dim].

        Accepts arbitrary leading dims (the set loss decodes [N,B,S,d] reshaped
        to [-1, d]); nn.Linear/Sequential broadcast over leading dims natively.
        """
        return self.out(self.net(z))

    def orthogonality_loss(self):
        """No-op: an MLP decoder has no orthogonal program basis to regularize."""
        return torch.zeros((), device=self.out.weight.device)

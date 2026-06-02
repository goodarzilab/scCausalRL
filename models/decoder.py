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

    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.D = nn.Parameter(torch.empty(output_dim, latent_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

        nn.init.orthogonal_(self.D)

    def forward(self, z):
        """
        z: Tensor[B, latent_dim]
        Returns: x_hat Tensor[B, output_dim]
        """
        return z @ self.D.T + self.b

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
        return self.D @ A @ self.D.T

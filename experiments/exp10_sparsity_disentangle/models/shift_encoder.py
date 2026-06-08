"""Learned perturbation shift encoder (SCCVAE-style).

Motivation
----------
The legacy intervention pathway lifts a gene-level effect to latent space via
`dz = D⁺·dx`, which (a) leaks the observed answer when `dx` is the full observed
differential expression, and (b) under a sparse/canonical input carries *no
learned perturbation knowledge* — it is purely a function of the decoder geometry
(`D⁺` of a one-hot), so the only thing telling the model "we perturbed gene g" is
gene g's decoder loading. That conflates "which programs g's expression lives on"
with "which programs perturbing g shifts" — two different objects.

The shift encoder instead *learns* a perturbation-specific latent shift `S^p` from
a gene-level FEATURE that exists for unseen genes (so it generalises zero-shot):

    S^p = ShiftEncoder(label_g)        label_g ∈ R^{label_dim}

`label_g` is the perturbation label for gene g. Following Liu/Zhang/Uhler (SCCVAE,
2025) we default to the top principal components of the *transposed* control
expression matrix — i.e. a gene embedding derived purely from control co-expression,
available for any gene in the panel including held-out perturbations.

The shift is injected additively into the structural equations (see
`CausalVAE.intervene_shift`):

    z_post = (I − A^T)^{-1} (ε + c · S^p)

where `c` is a (learned, or eval-time grid-searched) penetrance scalar. This is the
SCCVAE mechanism `U^p = A·U^p + Z + S^p` and replaces the `D⁺·dx` projection entirely.
"""
import torch
import torch.nn as nn


class ShiftEncoder(nn.Module):
    """MLP that maps a perturbation label to a latent-program shift vector.

    Args:
        label_dim:   dimensionality of the per-gene perturbation label.
        latent_dim:  d — number of latent programs (output shift dimension).
        hidden:      hidden width (SCCVAE uses 1024).
        n_layers:    number of hidden layers (>=1).
        output_activation: 'none' (raw real shift, default) or 'softmax'.
            SCCVAE softmaxes the shift so it is a distribution over programs and
            lets the signed penetrance scalar `c` carry direction/magnitude. We
            default to 'none' (a free real-valued shift) which is strictly more
            expressive; 'softmax' is provided for faithfulness/ablation.
        dropout:     dropout on hidden layers.
    """

    def __init__(self, label_dim, latent_dim, hidden=1024, n_layers=1,
                 output_activation='none', dropout=0.0):
        super().__init__()
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.output_activation = output_activation

        layers = []
        in_dim = label_dim
        for _ in range(max(n_layers, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, latent_dim)

    def forward(self, label):
        """label: Tensor[..., label_dim] → shift Tensor[..., latent_dim]."""
        h = self.net(label)
        s = self.out(h)
        if self.output_activation == 'softmax':
            s = torch.softmax(s, dim=-1)
        return s

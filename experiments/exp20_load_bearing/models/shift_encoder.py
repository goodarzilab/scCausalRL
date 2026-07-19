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
                 output_activation='none', dropout=0.0,
                 use_target_gate=False, gate_tau=1.0, gate_bias_init=3.0):
        super().__init__()
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.output_activation = output_activation

        # ── Generalization knob: ESM-noise augmentation (default 0.0 → exact no-op) ──
        # During TRAINING ONLY, additive Gaussian noise on the (ESM) label input before it
        # is mapped to a shift. Regularizes the ESM→shift map toward local smoothness (nearby
        # ESM embeddings → nearby shifts) to curb memorization of the train TFs. Set from
        # train_vae.py via p1['esm_noise_std']. Guarded by self.training in forward → true
        # no-op at eval (model.eval()) and when std==0.
        self.esm_noise_std = 0.0

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

        # ── Sparse TARGET gate (Fix 2, sparse-mechanism-shift) ────────────────
        # A SECOND head sharing the trunk that emits per-program gate logits over WHICH
        # latent programs this TF targets. It reads the same (ESM) label → zero-shot for
        # held-out TFs (no gene-id embedding). Applied in CausalVAE.compute_shift as
        # `shift = gate * raw`, so S is sparse in ALL callers. L0 (not L1-on-magnitude):
        # penalize gate OPEN-probability, leaving on-target magnitude free. Soft (Gumbel-
        # sigmoid) in train, hard (1{logit>0}) in eval — flips with self.training, matching
        # DAGModule._sample_gates. Dense-open init (zero weight, +bias) → starts at the
        # proven dense-shift baseline; the L0 term carves sparsity.
        self.use_target_gate = bool(use_target_gate)
        self.register_buffer('gate_tau', torch.tensor(float(gate_tau)))
        if self.use_target_gate:
            self.gate_out = nn.Linear(in_dim, latent_dim)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.constant_(self.gate_out.bias, float(gate_bias_init))
        else:
            self.gate_out = None

    def forward(self, label):
        """label: Tensor[..., label_dim] → RAW shift Tensor[..., latent_dim] (ungated)."""
        # ESM-noise augmentation (train-only, no-op when std==0): perturb the input embedding
        # so nearby ESM embeddings are forced to map to nearby shifts (local smoothness).
        if self.training and getattr(self, 'esm_noise_std', 0.0) > 0:
            label = label + torch.randn_like(label) * self.esm_noise_std
        h = self.net(label)
        s = self.out(h)
        if self.output_activation == 'softmax':
            s = torch.softmax(s, dim=-1)
        return s

    def _gate_logits(self, label):
        return self.gate_out(self.net(label))                       # [..., d]

    def sample_gate(self, label):
        """Per-program gate ∈ [0,1]. Soft Gumbel-sigmoid in train, hard 1{logit>0} in eval.
        Returns ones (no gating) when use_target_gate is False."""
        if not self.use_target_gate:
            return torch.ones(*label.shape[:-1], self.latent_dim,
                              device=label.device, dtype=label.dtype)
        logits = self._gate_logits(label)
        tau = float(self.gate_tau)
        if self.training:
            u1 = torch.rand_like(logits).clamp_(1e-8, 1 - 1e-8)
            u2 = torch.rand_like(logits).clamp_(1e-8, 1 - 1e-8)
            gumbel = (-torch.log(-torch.log(u1))) - (-torch.log(-torch.log(u2)))
            return torch.sigmoid((logits + gumbel) / tau)
        return (logits > 0).to(label.dtype)

    def gate_prob(self, label):
        """Deterministic expected open-gate probability sigmoid(logit/τ) — for the L0 penalty."""
        if not self.use_target_gate:
            return torch.ones(*label.shape[:-1], self.latent_dim,
                              device=label.device, dtype=label.dtype)
        return torch.sigmoid(self._gate_logits(label) / float(self.gate_tau))

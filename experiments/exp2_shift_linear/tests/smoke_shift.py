"""Synthetic CPU smoke test for the shift-encoder perturbation path.

Builds a tiny CausalVAE with the shift encoder, runs the observational
reconstruction + the shift set-loss (MSE + MMD [+ energy]) forward & backward on
random tensors, and asserts that gradients flow to every learnable component.

Usage:
    python tests/smoke_shift.py --decoder linear --energy 0
    python tests/smoke_shift.py --decoder mlp    --energy 1
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.causal_vae import CausalVAE
from training.losses import perturbation_set_loss_shift, compute_total_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoder', choices=['linear', 'mlp'], default='linear')
    ap.add_argument('--energy', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = 'cpu'
    G, d, label_dim = 200, 16, 8
    B, S = 2, 8

    model = CausalVAE(
        n_genes=G, latent_dim=d, tau_init=1.0, n_dag_samples=16, dagma_s=1.0,
        n_modalities=0, n_batches=4, u_clip_max=0.5,
        decoder_type=args.decoder, decoder_hidden=64, decoder_layers=2,
        use_shift_encoder=True, shift_label_dim=label_dim, shift_hidden=64,
        shift_n_layers=1, shift_output_activation='none',
    ).to(device)
    model.train()

    # Install random perturbation labels (stands in for the control-PCA embedding).
    model.set_pert_labels(torch.randn(G, label_dim))

    # Random synthetic data.
    x_ctrl_set = torch.randn(B, S, G)
    x_pert_set = x_ctrl_set + 0.5 * torch.randn(B, S, G)  # perturbed = ctrl + shift-ish
    gene_idx = torch.randint(0, G, (B,))
    batch_id = torch.randint(0, 4, (B,))

    # ── Observational reconstruction forward (recon + KL + DAG regularizers) ──
    x_obs = x_ctrl_set.reshape(B * S, G)
    batch_obs = batch_id.repeat_interleave(S)
    out = model(x_obs, bypass_scm=False, batch_id=batch_obs)

    # ── Shift perturbation set loss ──────────────────────────────────────────
    pert_loss, diag = perturbation_set_loss_shift(
        x_ctrl_set, x_pert_set, gene_idx=gene_idx, batch_id=batch_id,
        modality_id=None, model=model, n_dag_samples=1,
        mse_weight=1.0, mmd_weight=1.0, energy_weight=float(args.energy),
    )
    print('pert diag:', {k: round(v, 5) if isinstance(v, float) else v for k, v in diag.items()})

    losses = compute_total_loss(
        x=x_obs, x_hat=out['x_hat'], mu=out['mu'], log_var=out['log_var'],
        A=out['A'], model=model, beta=0.1,
        lambda_pert=10.0, lambda_dag=1.0, lambda_sparse=10.0,
        lambda_sparse_mode='product', lambda_spectral=0.1, lambda_ortho=1.0,
        lambda_prior=0.0, lambda_dagma=0.0, rho_dagma=0.0,
        lambda_D_sparse=(30.0 if args.decoder == 'linear' else 0.0),
        pert_loss=pert_loss, prior_loss=None, dag_active=True,
    )
    total = losses['total']
    print('total loss:', float(total.item()), '| recon:', float(losses['recon'].item()),
          '| pert:', float(losses['pert'].item()))

    total.backward()

    # ── Assert gradients reached every learnable component ───────────────────
    def gnorm(module):
        g = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
        return sum(g)

    checks = {
        'encoder': gnorm(model.encoder),
        'dag.theta': model.dag.theta.grad.norm().item() if model.dag.theta.grad is not None else 0.0,
        'shift_encoder': gnorm(model.shift_encoder),
        'shift_scale': model.shift_scale.grad.norm().item() if model.shift_scale.grad is not None else 0.0,
        'decoder': gnorm(model.decoder),
    }
    print('grad norms:', {k: round(v, 6) for k, v in checks.items()})
    for k, v in checks.items():
        assert v > 0, f"NO GRADIENT reached {k}"
    assert torch.isfinite(total), "loss is not finite"
    print(f"SMOKE OK  (decoder={args.decoder}, energy={args.energy})")


if __name__ == '__main__':
    main()

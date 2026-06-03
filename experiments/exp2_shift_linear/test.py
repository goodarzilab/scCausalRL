"""
Quick smoke tests for scCausalRL.
Verifies shapes, gradient flow, and core component logic without loading data.

Usage: python test.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    print("PyTorch not installed. Cannot run tests.")
    sys.exit(1)


def test_dag_module():
    print("--- DAGModule ---")
    from models.dag import DAGModule
    dag = DAGModule(d=16, tau=1.0, dagma_s=1.0)
    dag.train()

    # sample_A
    A_batch = dag.sample_A(n_samples=4)
    assert A_batch.shape == (4, 16, 16), f"Expected (4,16,16), got {A_batch.shape}"
    diag = A_batch.diagonal(dim1=-2, dim2=-1)
    assert diag.abs().max().item() < 1e-5, "Diagonal must be 0"
    print(f"  sample_A: {A_batch.shape} ✓")

    # DAGMA constraint
    h = dag.dagma_constraint(A_batch[0])
    assert h.requires_grad
    print(f"  dagma_constraint: {h.item():.4f}, requires_grad={h.requires_grad} ✓")

    # SCM forward
    eps = torch.randn(8, 16)
    z = dag.scm_forward(eps, A_batch[0])
    assert z.shape == (8, 16)
    print(f"  scm_forward: {z.shape} ✓")

    # SCM intervene
    z_post = dag.scm_intervene(eps, A_batch[0], intervention_idx=3, intervention_value=0.0)
    assert z_post.shape == (8, 16)
    print(f"  scm_intervene: {z_post.shape} ✓")

    # Spectral penalty
    sp = dag.spectral_radius_penalty(A_batch[0])
    assert isinstance(sp.item(), float)
    print(f"  spectral_radius_penalty: {sp.item():.4f} ✓")

    # Mutilate
    A_tilde = dag.mutilate(A_batch[0], 5)
    assert A_tilde[:, 5].abs().max().item() < 1e-6
    print(f"  mutilate: column 5 zeroed ✓")

    # Edge probs
    probs = dag.get_edge_probs()
    assert probs.shape == (16, 16)
    assert probs.diagonal().abs().max().item() < 1e-6
    print(f"  get_edge_probs: {probs.shape} ✓")


def test_encoder():
    print("--- CausalEncoder ---")
    from models.encoder import CausalEncoder
    enc = CausalEncoder(input_dim=200, latent_dim=16)
    enc.train()
    x = torch.randn(8, 200)
    mu, log_var, epsilon = enc(x)
    assert mu.shape == (8, 16)
    assert log_var.shape == (8, 16)
    assert epsilon.shape == (8, 16)
    print(f"  forward: mu={mu.shape}, log_var={log_var.shape}, epsilon={epsilon.shape} ✓")

    # Test eval mode (no reparameterization noise)
    enc.eval()
    with torch.no_grad():
        mu1, _, eps1 = enc(x)
        mu2, _, eps2 = enc(x)
    assert torch.allclose(eps1, eps2), "Eval mode should be deterministic"
    print(f"  eval mode deterministic ✓")


def test_decoder():
    print("--- LinearDecoder ---")
    from models.decoder import LinearDecoder
    dec = LinearDecoder(latent_dim=16, output_dim=200)
    z = torch.randn(8, 16)
    x_hat = dec(z)
    assert x_hat.shape == (8, 200)
    print(f"  forward: {x_hat.shape} ✓")

    ortho_loss = dec.orthogonality_loss()
    assert isinstance(ortho_loss.item(), float)
    print(f"  orthogonality_loss: {ortho_loss.item():.6f} ✓")

    A = torch.zeros(16, 16)
    G_feat = dec.project_dag_to_genes(A)
    assert G_feat.shape == (200, 200)
    print(f"  project_dag_to_genes: {G_feat.shape} ✓")


def test_causal_vae():
    print("--- CausalVAE ---")
    from models.causal_vae import CausalVAE
    model = CausalVAE(n_genes=200, latent_dim=16, tau_init=1.0, n_dag_samples=4)

    # Observational forward
    model.train()
    x = torch.randn(8, 200)
    out = model(x)
    assert out['x_hat'].shape == (8, 200)
    assert out['mu'].shape == (8, 16)
    assert out['A'].shape == (16, 16)
    print(f"  observational forward: x_hat={out['x_hat'].shape}, A={out['A'].shape} ✓")

    # Backward
    loss = out['x_hat'].sum()
    loss.backward()
    assert model.dag.theta.grad is not None
    print(f"  gradients flow through DAG ✓")

    # Intervention
    model.eval()
    with torch.no_grad():
        i_out = model.intervene(x, pert_gene_idx=10, pert_value=0.0)
    assert i_out['x_hat_pert'].shape == (8, 200)
    print(f"  intervene: {i_out['x_hat_pert'].shape} ✓")

    # Bootstrapped intervention
    with torch.no_grad():
        b_out = model.intervene_bootstrapped(x, pert_gene_idx=10, pert_value=0.0, n_samples=4)
    assert b_out['all_x_hat'].shape == (4, 8, 200)
    assert b_out['std_x_hat'].shape == (8, 200)
    print(f"  intervene_bootstrapped: all_x_hat={b_out['all_x_hat'].shape} ✓")

    # Multi-step
    with torch.no_grad():
        ms_out = model.multi_step_intervene(x, [(5, 0.0), (10, 0.0)], n_samples=4)
    assert ms_out['all_trajectories'].shape == (3, 4, 8, 200)
    print(f"  multi_step_intervene: all_trajectories={ms_out['all_trajectories'].shape} ✓")

    # Gene-to-latent mapping
    model.eval()
    latent_idx, strengths = model.get_gene_to_latent_mapping()
    assert latent_idx.shape == (200,)
    print(f"  gene_to_latent_mapping: {latent_idx.shape} ✓")


def test_losses():
    print("--- Loss Functions ---")
    from models.causal_vae import CausalVAE
    from training.losses import (
        reconstruction_loss, kl_divergence, perturbation_consistency_loss,
        sparsity_loss, compute_total_loss
    )

    model = CausalVAE(n_genes=100, latent_dim=8, tau_init=1.0, n_dag_samples=2)
    model.train()

    x = torch.randn(4, 100)
    out = model(x)

    l_recon = reconstruction_loss(x, out['x_hat'])
    assert l_recon.item() > 0
    print(f"  reconstruction_loss: {l_recon.item():.4f} ✓")

    l_kl = kl_divergence(out['mu'], out['log_var'])
    assert isinstance(l_kl.item(), float)
    print(f"  kl_divergence: {l_kl.item():.4f} ✓")

    l_sparse = sparsity_loss(out['A'])
    assert l_sparse.item() >= 0
    print(f"  sparsity_loss: {l_sparse.item():.4f} ✓")

    x_ctrl = torch.randn(4, 100)
    x_pert = torch.randn(4, 100)
    l_pert = perturbation_consistency_loss(x_ctrl, x_pert, 5, model)
    assert l_pert.item() > 0
    print(f"  perturbation_consistency_loss: {l_pert.item():.4f} ✓")

    losses = compute_total_loss(
        x, out['x_hat'], out['mu'], out['log_var'], out['A'], model,
        beta=1.0, dag_active=True, lambda_dagma=0.0, rho_dagma=1.0,
        pert_loss=l_pert
    )
    assert 'total' in losses
    losses['total'].backward()
    assert model.dag.theta.grad is not None
    print(f"  compute_total_loss: {losses['total'].item():.4f}, gradients OK ✓")


def test_rl_components():
    print("--- RL Components ---")
    from rl.actor_critic import Actor, Critic
    from rl.replay import RolloutBuffer

    actor = Actor(pca_dim=10, n_actions=48)
    critic = Critic(pca_dim=10)

    s = torch.randn(4, 10)
    t = torch.randn(4, 10)

    dist = actor(s, t)
    action, log_prob, entropy = actor.get_action(s, t)
    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    print(f"  Actor: action={action.shape}, log_prob={log_prob.shape} ✓")

    new_lp, new_ent = actor.evaluate_action(s, t, action)
    assert new_lp.shape == (4,)
    print(f"  Actor.evaluate_action ✓")

    value = critic(s, t)
    assert value.shape == (4,)
    print(f"  Critic: value={value.shape} ✓")

    # Rollout buffer
    buf = RolloutBuffer()
    for i in range(20):
        buf.add(
            torch.randn(10), torch.randn(10), i % 48,
            float(np.random.randn()), i == 19, -0.5, 0.5
        )
    buf.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert buf.returns is not None
    assert buf.advantages is not None
    batches = list(buf.get_batches(batch_size=8))
    assert len(batches) >= 2
    print(f"  RolloutBuffer: {len(buf)} transitions, {len(batches)} batches ✓")


def test_metrics():
    print("--- Metrics ---")
    from utils.metrics import (
        mse_on_degs, pearson_correlation, cosine_similarity, direction_accuracy
    )
    rng = np.random.default_rng(0)
    y_pred = rng.standard_normal(100)
    y_true = y_pred + rng.standard_normal(100) * 0.1
    de = rng.standard_normal(100)

    mse = mse_on_degs(y_pred, y_true, de)
    assert mse >= 0
    print(f"  mse_on_degs: {mse:.4f} ✓")

    r = pearson_correlation(y_pred, y_true)
    assert -1 <= r <= 1
    print(f"  pearson_correlation: {r:.4f} ✓")

    cs = cosine_similarity(y_pred, y_true)
    assert -1 <= cs <= 1
    print(f"  cosine_similarity: {cs:.4f} ✓")

    acc = direction_accuracy(y_pred, y_true)
    assert 0 <= acc <= 1
    print(f"  direction_accuracy: {acc:.4f} ✓")


def main():
    print("=" * 60)
    print("scCausalRL — Component Tests")
    print("=" * 60)
    print()

    tests = [
        ("DAGModule", test_dag_module),
        ("CausalEncoder", test_encoder),
        ("LinearDecoder", test_decoder),
        ("CausalVAE", test_causal_vae),
        ("Loss Functions", test_losses),
        ("RL Components", test_rl_components),
        ("Metrics", test_metrics),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            print(f"  PASS")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} test(s) failed.")
    print("=" * 60)
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

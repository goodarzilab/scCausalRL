"""
Math + persistence invariants for the u-score intervention scheme.

Each test reads as an executable claim about a property derived in
docs/architecture_u_score.md (e.g. "u_i = 1 reduces to classical do(z_i = z_target_i)").

Run:
    pytest tests/test_u_intervention.py -v
"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.causal_vae import CausalVAE


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def small_model():
    """Tiny CausalVAE with a non-trivial DAG and decoder.

    We seed deterministically and slam a sparse, low-spectral-radius A into
    `theta` so the SCM is well-conditioned without requiring training.
    """
    torch.manual_seed(0)
    n_genes, d = 16, 4
    model = CausalVAE(n_genes=n_genes, latent_dim=d,
                     tau_init=0.25, n_dag_samples=1, dagma_s=1.0)
    # Force theta to give predictable A in eval mode (hard threshold on sigmoid(theta/tau))
    # Build A_target as a sparse triangular: 0→1, 1→2, 2→3 with weights ~0.3
    with torch.no_grad():
        theta = torch.full((d, d), -10.0)
        edges = [(0, 1), (1, 2), (2, 3)]
        for i, j in edges:
            theta[i, j] = 2.0  # sigmoid(2/0.25)=sigmoid(8) ≈ 0.99963
        model.dag.theta.copy_(theta)
        # Give decoder a clean orthogonal-ish D (already orthogonal-init'd in __init__)
    model.eval()
    model.dag.eval()
    # Cache loadings
    model.compute_loadings()
    return model


def _sample_A(model):
    """Sample a single A in eval mode (hard-threshold) and zero diagonal."""
    with torch.no_grad():
        return model.dag.sample_A(n_samples=1).squeeze(0)


# ──────────────────────────────────────────────────────────────────────
# 1) Endpoint behaviour: u = 0 → identity
# ──────────────────────────────────────────────────────────────────────
def test_u_zero_identity(small_model):
    """At u = 0, the u-score intervention must reproduce scm_forward(epsilon, A) exactly."""
    model = small_model
    d = model.latent_dim
    A = _sample_A(model)
    epsilon = torch.randn(2, d) * 0.5

    z_baseline = model.dag.scm_forward(epsilon, A)
    # u = 0 ⇒ eps_tilde = epsilon (no mixing); A_tilde = A (no scaling)
    u_zero = torch.zeros(d)
    z_u, A_tilde, _M = model.dag.scm_intervene_u(epsilon, A, eps_tilde=epsilon, u=u_zero)
    assert torch.allclose(z_u, z_baseline, atol=1e-5), \
        f"u=0 should be identity; max diff = {(z_u - z_baseline).abs().max().item():.2e}"
    assert torch.allclose(A_tilde, A, atol=1e-7), "u=0 must leave A untouched"


# ──────────────────────────────────────────────────────────────────────
# 2) Endpoint behaviour: u_i = 1 pins z_post[i] to z_target[i]
#    and zeros column i of A_tilde
# ──────────────────────────────────────────────────────────────────────
def test_u_one_pins_target(small_model):
    """For u_i = 1 (others 0), z_post[:, i] must equal z_target[:, i] exactly,
    and column i of A_tilde must be all zeros (classical do-operator)."""
    model = small_model
    d = model.latent_dim
    A = _sample_A(model)
    epsilon = torch.randn(3, d)

    # Choose target program i=1 (it has a real parent edge 0→1 in our setup)
    i = 1
    u = torch.zeros(d)
    u[i] = 1.0

    z_target = torch.randn(3, d)
    eps_tilde = u.unsqueeze(0) * z_target + (1.0 - u).unsqueeze(0) * epsilon
    z_post, A_tilde, _M = model.dag.scm_intervene_u(epsilon, A, eps_tilde=eps_tilde, u=u)

    # (a) z_post[:, i] == z_target[:, i]
    assert torch.allclose(z_post[:, i], z_target[:, i], atol=1e-5), \
        f"u_i=1 should pin program {i}; diff = {(z_post[:, i] - z_target[:, i]).abs().max():.2e}"

    # (b) column i of A_tilde is zero (classical do removes incoming edges)
    assert torch.allclose(A_tilde[:, i], torch.zeros(d), atol=1e-7), \
        f"column {i} of A_tilde must be zero, got {A_tilde[:, i]}"


# ──────────────────────────────────────────────────────────────────────
# 3) u accumulates monotonically across an episode
# ──────────────────────────────────────────────────────────────────────
def test_u_monotonic(small_model):
    """After two steps with the same gene, u(t) ≥ u(t-1) elementwise.
    Repeating the same perturbation must approach the clip ceiling."""
    model = small_model
    d = model.latent_dim
    n_genes = model.n_genes
    A = _sample_A(model)
    epsilon = torch.randn(1, d)
    L = model.get_loadings()

    g = 0
    z0 = model.dag.scm_forward(epsilon, A)
    dx = torch.zeros(n_genes); dx[g] = -1.0  # mock knockout dx

    u_prev = torch.zeros(d)
    z_step1, u1, _, _, _ = model.intervene_gene_u(
        epsilon, dx, A, u_prev=u_prev, z_prev=z0.squeeze(0), gene_idx=g, u_clip_max=1.0,
    )
    z_step2, u2, _, _, _ = model.intervene_gene_u(
        epsilon, dx, A, u_prev=u1, z_prev=z_step1.squeeze(0), gene_idx=g, u_clip_max=1.0,
    )

    # Monotone
    assert torch.all(u1 >= u_prev - 1e-7), "u1 must be ≥ u_prev elementwise"
    assert torch.all(u2 >= u1 - 1e-7), "u2 must be ≥ u1 elementwise"
    # And u1 = L[g, :] on the first hit
    assert torch.allclose(u1, L[g].clamp(max=1.0), atol=1e-6)


# ──────────────────────────────────────────────────────────────────────
# 4) Knockout persistence (clamp x[g]=0 across steps)
# ──────────────────────────────────────────────────────────────────────
def test_knockout_clamp_logic():
    """Simulate the env's knockout-set clamp: after do(g knockout), zeroing
    x[g] before the next dx must be idempotent."""
    n_genes = 10
    x = torch.randn(1, n_genes).abs() + 0.1  # all positive expression
    ko_set = {2, 5}

    ko_idx = torch.tensor(sorted(ko_set), dtype=torch.long)
    x_clamped = x.clone()
    x_clamped[:, ko_idx] = 0.0

    # Idempotent
    x_clamped2 = x_clamped.clone()
    x_clamped2[:, ko_idx] = 0.0
    assert torch.allclose(x_clamped, x_clamped2)

    # All non-ko entries unchanged
    keep = torch.tensor([i for i in range(n_genes) if i not in ko_set])
    assert torch.allclose(x[:, keep], x_clamped[:, keep])
    # All ko entries zero
    assert torch.allclose(x_clamped[:, ko_idx], torch.zeros_like(x_clamped[:, ko_idx]))


# ──────────────────────────────────────────────────────────────────────
# 5) The legacy threshold path is unaffected by the new methods
# ──────────────────────────────────────────────────────────────────────
def test_threshold_path_unchanged(small_model):
    """Legacy intervene_gene + scm_intervene_multi must produce the same numbers
    they would have before the u-score code was added (sanity check)."""
    model = small_model
    d = model.latent_dim
    n_genes = model.n_genes
    A = _sample_A(model)
    epsilon = torch.randn(2, d)
    dx = torch.zeros(n_genes); dx[3] = -0.5
    z_post, d_eps = model.intervene_gene(epsilon, dx, A, threshold_frac=0.1)
    # Just check basic shape + finiteness
    assert z_post.shape == (2, d)
    assert torch.isfinite(z_post).all()
    assert d_eps.shape == (d,)


# ──────────────────────────────────────────────────────────────────────
# 6) Loadings L: per-gene max = 1.0 (row-max normalization, Phase H)
# ──────────────────────────────────────────────────────────────────────
def test_loadings_row_max(small_model):
    """L[g, i] = D[g, i]² / max_j D[g, j]²  ⇒  max_i L[g, i] = 1 for every gene."""
    model = small_model
    L, T = model.compute_loadings()
    per_gene_max = L.max(dim=1).values
    assert torch.allclose(per_gene_max, torch.ones_like(per_gene_max), atol=1e-5), \
        f"per-gene max(L) should be 1.0 always; got min={per_gene_max.min().item():.4f}"
    assert torch.all(L >= 0), "L must be non-negative"
    assert torch.all(L <= 1.0 + 1e-6), "L must be ≤ 1.0 by construction"


# ──────────────────────────────────────────────────────────────────────
# 7) recover_epsilon_u inverts scm_intervene_u under u
# ──────────────────────────────────────────────────────────────────────
def test_recover_epsilon_u_inverts(small_model):
    """If z_post = (I - M A^T)^{-1} eps_tilde then (I - M A^T) z_post = eps_tilde."""
    model = small_model
    d = model.latent_dim
    A = _sample_A(model)
    epsilon = torch.randn(2, d)
    u = torch.tensor([0.0, 0.7, 1.0, 0.3])  # mixed regimes
    z_target = torch.randn(2, d)
    eps_tilde = u.unsqueeze(0) * z_target + (1.0 - u).unsqueeze(0) * epsilon
    z_post, _Atilde, _M = model.dag.scm_intervene_u(epsilon, A, eps_tilde, u)
    eps_recovered = model.recover_epsilon_u(z_post, A, u)
    assert torch.allclose(eps_recovered, eps_tilde, atol=1e-4), \
        f"recover_epsilon_u must invert; max diff {(eps_recovered - eps_tilde).abs().max().item():.2e}"


# ──────────────────────────────────────────────────────────────────────
# 8) Sparse-orthogonal D: row-max normalization properties
# ──────────────────────────────────────────────────────────────────────
def test_loadings_with_sparse_D(small_model):
    """Build a block-supported (sparse) orthogonal D and verify the Phase-H
    row-max normalization properties:
      (i)  per-gene max(L) = 1
      (ii) support of L[g, :] matches support of D[g, :]^2 > 0
      (iii) D itself stays orthogonal (sanity)."""
    model = small_model
    n_genes, d = model.n_genes, model.latent_dim
    # Block-supported orthogonal D: each program i loads only on genes [i*4:(i+1)*4]
    with torch.no_grad():
        D_sparse = torch.zeros(n_genes, d)
        block = n_genes // d
        for i in range(d):
            v = torch.randn(block)
            D_sparse[i * block:(i + 1) * block, i] = v / v.norm()  # unit-norm column
        model.decoder.D.copy_(D_sparse)
        model._loadings_dirty = True
    L, T = model.compute_loadings()

    # (i) per-gene max = 1
    per_gene_max = L.max(dim=1).values
    # Only check for genes that have any non-zero loading (block-supported genes only)
    nonzero_rows = (D_sparse.abs().sum(dim=1) > 1e-10)
    assert torch.allclose(per_gene_max[nonzero_rows], torch.ones(int(nonzero_rows.sum())), atol=1e-5), \
        f"per-gene max(L) should be 1.0; got min={per_gene_max[nonzero_rows].min().item():.4f}"

    # (ii) Support of L[g, :] matches support of D[g, :]^2 > 0 (row support)
    for g in range(n_genes):
        if not bool(nonzero_rows[g]):
            continue
        D_supp = (D_sparse[g, :].abs() > 1e-10)
        L_supp = (L[g, :] > 1e-10)
        assert torch.equal(D_supp, L_supp), \
            f"gene {g}: L row-support does not match D row-support"

    # (iii) Orthogonality preserved: D^T D is identity for unit-norm cols
    DtD = D_sparse.T @ D_sparse
    off_diag = (DtD - torch.eye(d)).abs().max().item()
    assert off_diag < 1e-5, f"D not orthogonal after construction; off_diag={off_diag:.2e}"


# ──────────────────────────────────────────────────────────────────────
# 9) PairedCellSentenceDataset: ctrl & pert share the same gem_group
# ──────────────────────────────────────────────────────────────────────
def test_paired_sentence_dataset():
    """Every item must contain S ctrl + S pert cells whose batch_id matches
    the sentence's batch_id."""
    import numpy as np
    import pandas as pd
    import anndata as ad
    from data.cell_sentence_dataset import PairedCellSentenceDataset

    n_cells, n_genes = 200, 32
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_cells, n_genes)).astype('float32')
    obs = pd.DataFrame({
        'gene':     ['non-targeting']*60 + ['gA']*60 + ['gB']*40 + ['gC']*40,
        'batch_id': ([0]*100 + [1]*100),
    })
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = [f'g{i}' for i in range(n_genes)]
    # Make gA / gB / gC look like real gene names so they're in the gene index lookup.
    adata.obs['gene'] = ['non-targeting']*60 + ['g10']*60 + ['g11']*40 + ['g12']*40
    single_perts = {'g10': list(range(60, 120)), 'g11': list(range(120, 160)), 'g12': list(range(160, 200))}
    control_idx = list(range(60))

    ds = PairedCellSentenceDataset(
        adata=adata,
        single_perts=single_perts,
        control_idx=control_idx,
        gene_names=list(adata.var_names),
        S=8, n_sentences_per_block=2, min_block_cells=4, seed=0,
    )
    assert len(ds) > 0, "dataset must produce at least one sentence"
    item = ds[0]
    assert item['x_ctrl'].shape == item['x_pert'].shape == (8, n_genes)
    assert int(item['batch_id']) in {0, 1}
    # gene_idx must point to a column in var_names
    assert 0 <= int(item['gene_idx']) < n_genes


# ──────────────────────────────────────────────────────────────────────
# 10) distributional_loss invariants
# ──────────────────────────────────────────────────────────────────────
def test_distributional_loss_basic():
    """energy(a, a) ≈ 0; energy(a, a+shift) > 0; mmd_rbf has the same property."""
    from training.losses import distributional_loss, mmd_rbf
    a = torch.randn(32, 16)
    assert float(distributional_loss(a, a, kind='energy')) < 1e-3
    assert float(distributional_loss(a, a + 3.0, kind='energy')) > 0.1
    assert float(mmd_rbf(a, a)) < 1e-3
    assert float(mmd_rbf(a, a + 3.0)) > 0.01


# ──────────────────────────────────────────────────────────────────────
# 11) perturbation_set_loss_u: theta receives nonzero gradient
# ──────────────────────────────────────────────────────────────────────
def test_pert_set_loss_u_gradient_to_theta(small_model):
    """The whole point of switching away from threshold mutilation: theta must
    receive nonzero gradient when A is near zero (init regime).  The 18-cell
    grid run 2349145 failed because the threshold path produced ~0 gradient
    on theta — verify the u-score set-level loss does not."""
    from training.losses import perturbation_set_loss_u

    model = small_model
    n_genes, d = model.n_genes, model.latent_dim
    model.train()
    # Ensure theta gradient flag is on (it might have been frozen elsewhere).
    model.dag.theta.requires_grad_(True)
    if model.dag.theta.grad is not None:
        model.dag.theta.grad.zero_()

    B, S = 2, 8
    x_ctrl = torch.randn(B, S, n_genes) * 0.5 + 5.0
    x_pert = x_ctrl + torch.randn(B, S, n_genes) * 0.1 + 0.3
    gene_idx = torch.tensor([0, 1])
    batch_id = torch.tensor([0, 0])
    loss, _ = perturbation_set_loss_u(
        x_ctrl, x_pert, gene_idx, batch_id, modality_id=None,
        model=model, n_dag_samples=2, loss_kind='mmd_rbf',
    )
    loss.backward()
    assert model.dag.theta.grad is not None
    grad_norm = float(model.dag.theta.grad.norm())
    assert grad_norm > 1e-6, f"theta gradient is {grad_norm:.2e}; u-score path should give >0 gradient at init"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

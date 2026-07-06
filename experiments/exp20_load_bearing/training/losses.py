import torch
import torch.nn.functional as F
from models.dag import _robust_solve  # singular-safe (I - A^T) solve; see dag.py

# Optional dependency — geomloss provides the Sinkhorn / energy / Cramer OT losses
# used by STATE. If unavailable, perturbation_set_loss_u falls back to the in-house
# RBF-MMD implementation (mmd_rbf below).
try:
    from geomloss import SamplesLoss
    _ENERGY_LOSS = SamplesLoss(loss="energy", blur=0.05)
except Exception:
    SamplesLoss = None
    _ENERGY_LOSS = None


def reconstruction_loss(x, x_hat):
    """MSE reconstruction loss."""
    return F.mse_loss(x_hat, x, reduction='mean')


def kl_divergence(mu, log_var):
    """Standard VAE KL divergence."""
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def dagma_augmented_lagrangian(h_val, lambda_dagma, rho):
    """Augmented Lagrangian term for DAGMA acyclicity constraint."""
    return lambda_dagma * h_val + (rho / 2.0) * h_val ** 2


def sparsity_loss(A, model=None, mode='product'):
    """L1 sparsity penalty on the DAG.

    mode='product' (default, legacy): mean(|gates * weights|) — penalises the
        sampled edge magnitudes. Pulls confident-edge WEIGHTS toward zero in
        proportion to the gate probability, which can collapse magnitudes on
        topologically-confident edges.
    mode='gates': mean(sigmoid(theta / tau)) — penalises edge EXISTENCE
        (deterministic gate probabilities) without touching weights. Lets the
        surviving topology have its natural magnitude.
    """
    if mode == 'product':
        return A.abs().mean()
    if mode == 'gates':
        if model is None:
            raise ValueError("sparsity_loss(mode='gates') requires `model=` to read dag.theta")
        return torch.sigmoid(model.dag.theta / model.dag.tau).mean()
    raise ValueError(f"Unknown sparsity mode: {mode!r}. Use 'product' or 'gates'.")


def spectral_stability_loss(A, dag_module):
    """Penalize spectral radius to prevent near-singular SCM."""
    return dag_module.spectral_radius_penalty(A)


def orthogonality_loss(decoder):
    """Decoder orthogonality regularization."""
    return decoder.orthogonality_loss()


# ──────────────────────────────────────────────────────────────────────
# Set-level distributional losses (energy distance + RBF-MMD fallback)
# ──────────────────────────────────────────────────────────────────────
def _median_heuristic(x, y):
    """Median pairwise L2 distance over (x ∪ y) — the standard MMD bandwidth pick."""
    with torch.no_grad():
        combined = torch.cat([x, y], dim=0)
        D = torch.cdist(combined, combined, p=2)
        # exclude the trivial 0 self-distances on the diagonal
        n = D.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=D.device)
        med = D[mask].median()
    return med.clamp(min=1e-3)


def mmd_rbf(x, y, sigma=None):
    """Standard biased Gaussian-kernel MMD² between two sample sets.

    x, y : Tensor[N, d] and Tensor[M, d]
    sigma: float or None (None → median heuristic).
    """
    if sigma is None:
        sigma = _median_heuristic(x, y)
    xx = torch.exp(-torch.cdist(x, x).pow(2) / (2 * sigma ** 2))
    yy = torch.exp(-torch.cdist(y, y).pow(2) / (2 * sigma ** 2))
    xy = torch.exp(-torch.cdist(x, y).pow(2) / (2 * sigma ** 2))
    n, m = x.shape[0], y.shape[0]
    return xx.sum() / (n * n) + yy.sum() / (m * m) - 2 * xy.sum() / (n * m)


def distributional_loss(x_hat, x, kind="energy"):
    """Compare two empirical cell distributions (sets of points in R^G).

    kind = "energy"   → geomloss SamplesLoss(loss="energy", blur=0.05)
                        (Cramer / OT energy distance, bandwidth-free, STATE default)
    kind = "mmd_rbf" → Gaussian-kernel MMD² with median-heuristic bandwidth
    kind = "mse"     → fallback per-cell MSE (only sensible if cells are paired
                        index-to-index — kept for direct comparison with the legacy path)
    """
    if kind == "energy":
        if _ENERGY_LOSS is None:
            # geomloss unavailable — fall back automatically rather than crash.
            return mmd_rbf(x_hat, x)
        return _ENERGY_LOSS(x_hat, x)
    if kind == "mmd_rbf":
        return mmd_rbf(x_hat, x)
    if kind == "mse":
        return F.mse_loss(x_hat, x, reduction='mean')
    raise ValueError(f"Unknown distributional_loss kind: {kind!r}")


def decoder_sparsity_loss(decoder):
    """Mean-normalised L1 on all entries of D — gene-level decoder sparsity.

    Pushes each entry D[g, i] toward zero so that each gene loads on only a few
    programs (and each program is defined by only a few genes). Orthogonality
    on D's columns is scale-invariant (via column-norm normalisation), so this
    penalty composes cleanly with it — the equilibrium is sparse-but-non-zero
    orthogonal columns (sparse factor analysis).
    """
    return decoder.D.abs().mean()


def perturbation_consistency_loss(
    x_ctrl, x_pert, pert_gene_idx, model, n_dag_samples=8
):
    """
    Interventional consistency loss with multi-DAG averaging and data-driven
    intervention value.

    For a (x_ctrl, perturbation_gene, x_pert) triple:
        1. Encode x_ctrl -> epsilon_ctrl, x_pert -> epsilon_pert
        2. Derive intervention value from mean(epsilon_pert[:, latent_idx])
        3. Sample multiple DAGs and average the loss:
           a. Apply do(latent_idx = pert_value) on epsilon_ctrl
           b. Decode -> x_hat_pert
           c. MSE(x_hat_pert, x_pert)

    x_ctrl: Tensor[B, n_genes]
    x_pert: Tensor[B, n_genes]
    pert_gene_idx: int (gene index)
    model: CausalVAE
    n_dag_samples: int (number of DAG samples to average over)
    """
    mu_ctrl, _, epsilon_ctrl = model.encoder(x_ctrl)
    mu_pert, _, epsilon_pert = model.encoder(x_pert)

    latent_idx = model._get_latent_for_gene(pert_gene_idx)

    # Data-driven intervention value: mean of perturbed cells' latent
    # .detach() prevents gradients through the target value
    pert_value = epsilon_pert[:, latent_idx].mean().detach()

    # Sample multiple DAGs and average the perturbation loss
    A_batch = model.dag.sample_A(n_samples=n_dag_samples)
    losses = []
    for n in range(n_dag_samples):
        z_post = model.dag.scm_intervene(epsilon_ctrl, A_batch[n], latent_idx, pert_value)
        x_hat_pert = model.decoder(z_post)
        losses.append(F.mse_loss(x_hat_pert, x_pert, reduction='mean'))
    return torch.stack(losses).mean()


def perturbation_set_loss_u(
    x_ctrl_set,        # Tensor[B, S, n_genes] — paired control cells per sentence
    x_pert_set,        # Tensor[B, S, n_genes] — paired perturbed cells per sentence
    gene_idx,          # LongTensor[B]         — perturbed gene index
    batch_id,          # LongTensor[B]         — gem_group id (passed to encoder)
    modality_id,       # LongTensor[B] or None — modality id (Norman/Replogle/...)
    model,
    n_dag_samples=1,
    loss_kind="mse",
    match_delta=True,
    input_mode='observed_full',
    pert_direction_batch=None,   # LongTensor[B] +1 (CRISPRa) / -1 (CRISPRi)
    canonical_magnitude=2.0,
    aux_prop_weight=0.0,
):
    """u-score set-level perturbation consistency loss — VECTORISED.

    Per sentence b:
      1. Encode the S control cells under their gem_group/modality → eps_ctrl_b [S, d].
      2. dx = mean(x_pert[b]) − mean(x_ctrl[b])      — per-sentence, plate-matched
      3. dz = D⁺ dx                                  — gene → latent
      4. u  = L[gene_idx[b], :]                       — decoder-loading strength (single-step)

    Then jointly over (N DAG samples, B sentences):
      5. z_prev   = (I − A_n^T)^{-1}  eps_ctrl       [N, B, S, d]
      6. z_target = z_prev + dz                       [N, B, S, d]
      7. eps_tilde = u ⊙ z_target + (1−u) ⊙ eps_ctrl  [N, B, S, d]
      8. z_post   = (I − M_b A_n^T)^{-1} eps_tilde    [N, B, S, d]      (M_b = diag(1−u_b))
      9. x_hat    = decoder(z_post)                   [N, B, S, n_genes]
      10. loss   = distributional_loss(x_hat[n,b], x_pert[b], kind=loss_kind), averaged.

    `loss_kind="mse"` (default) is the per-cell MSE between the predicted and observed
    sentence sets.  Sets are random samples from the same conditional distribution, so the
    expected MSE has a non-zero floor equal to Var(pred)+Var(obs); the gradient direction is
    exactly the mean-shift term ‖μ_pred − μ_obs‖² — which is what the pert loss should push.
    `loss_kind="energy"` (geomloss) and `"mmd_rbf"` remain available for set-level
    distribution matching, at the cost of ~3× wall-time per step.

    `match_delta=True` (Phase H, default) compares the *predicted perturbation effect*
        delta_pred = mean(decode(z_post)) - mean(decode(z_prev))         (over S cells per sentence)
    against the *observed perturbation effect*
        delta_obs  = mean(x_pert) - mean(x_ctrl)                          (= dx_all[b])
    via MSE.  The decoder bias `b` cancels in the difference, and so does any "trivial"
    encoder/decoder path that fits x_pert ≈ x_ctrl.  The ONLY mechanism that can produce
    a non-zero delta_pred is the SCM solve through A, so A is structurally required to
    fit the loss.  This is the over-parameterisation fix from Phase H — see plan §H3a.

    `match_delta=False` reverts to the absolute matching (decode(z_post) vs x_pert), which
    was the Phase F/G behaviour. Kept for ablation.

    For training we set N=1 (the bootstrap is for RL/eval uncertainty, not training gradient
    quality).  N>1 is still supported and gets the full vectorised path.

    Returns: (loss: Tensor, diagnostics: dict).
    """
    B, S, G = x_ctrl_set.shape
    device = x_ctrl_set.device
    d = model.latent_dim

    # Encode all B * S control cells in one shot.
    x_ctrl_flat = x_ctrl_set.reshape(B * S, G)
    batch_flat = batch_id.repeat_interleave(S)
    mod_flat = modality_id.repeat_interleave(S) if modality_id is not None else None
    _, _, eps_ctrl_flat = model.encoder(
        x_ctrl_flat, modality_id=mod_flat, batch_id=batch_flat
    )
    eps_ctrl_set = eps_ctrl_flat.reshape(B, S, d)                  # [B, S, d]

    # u per sentence (one row per gene from the loading matrix).  Clamped to
    # model.u_clip_max (Phase G default 0.5) so the do-operator cannot fully
    # pin z_post and bypass the DAG — see Phase G diagnosis in the plan file.
    L = model.get_loadings()                                       # [n_genes, d]
    u_clip_max = float(getattr(model, 'u_clip_max', 1.0))
    u_all = L[gene_idx.long()].clamp(max=u_clip_max)                # [B, d]

    # dx_all is the per-sentence FULL observed delta — used both as the target
    # for match_delta (everywhere) AND as the source of the leaky 'observed_full'
    # input for the dz path. The Task-A fix builds a sparse dx_input here that
    # ONLY has a value at the perturbed gene index.
    dx_all = (x_pert_set.mean(1) - x_ctrl_set.mean(1)).detach()    # [B, G]

    from models.dx_input import build_dx_input_batch
    dx_input = build_dx_input_batch(
        input_mode=input_mode,
        gene_idx_batch=gene_idx,
        dx_full_batch=dx_all,
        n_genes=G,
        pert_direction_batch=pert_direction_batch,
        canonical_magnitude=canonical_magnitude,
    )                                                              # [B, G]

    # dz per sentence via batched D⁺.
    # D†: dz = (D^T D + λI)^{-1} D^T dx.  Batched form: dz [B, d] from dx [B, G].
    D = model.decoder.D.detach()                                   # [G, d]
    DtD = D.T @ D + 1e-6 * torch.eye(d, device=device, dtype=D.dtype)  # [d, d]
    Dt_dx_T = D.T @ dx_input.T                                     # [d, B]   (sparse if Task-A)
    dz_all = _robust_solve(DtD, Dt_dx_T).T                    # [B, d]

    # Sample N DAGs (in eval this is N=8 for uncertainty; in training default N=1).
    A_batch = model.dag.sample_A(n_samples=n_dag_samples)           # [N, d, d]
    I_mat = torch.eye(d, device=device, dtype=A_batch.dtype)

    # ── Vectorised solve for z_prev = (I − A_n^T)^{-1} eps_ctrl ──────────────
    # LHS_prev [N, d, d]; RHS = eps_ctrl_set with shape [B, S, d] reshaped to
    # broadcast across N.  The solve has batch dim N: solve N systems each of size d×d.
    LHS_prev = I_mat.unsqueeze(0) - A_batch.transpose(-1, -2)       # [N, d, d]
    # RHS [N, d, B*S]: stack eps_ctrl as columns for each N
    rhs_prev = eps_ctrl_set.reshape(B * S, d).T                     # [d, B*S]
    rhs_prev = rhs_prev.unsqueeze(0).expand(n_dag_samples, d, B * S).contiguous()
    z_prev_flat = _robust_solve(LHS_prev, rhs_prev)             # [N, d, B*S]
    z_prev = z_prev_flat.transpose(-1, -2).reshape(n_dag_samples, B, S, d)
    z_prev = torch.clamp(z_prev, min=-20.0, max=20.0)

    # ── eps_tilde, M, LHS_post for the do-step ──────────────────────────────
    one_minus_u = 1.0 - u_all                                       # [B, d]
    u_bcast = u_all[None, :, None, :]                               # [1, B, 1, d]
    z_target = z_prev + dz_all[None, :, None, :]                    # [N, B, S, d]
    eps_tilde = u_bcast * z_target + (1.0 - u_bcast) * eps_ctrl_set[None, :, :, :]
    #          [N, B, S, d]                                                 [1, B, S, d]

    # LHS_post[n, b] = I − diag(1−u_b) @ A_n^T,  shape [d, d]
    # Build as: I − (1−u_b)[i] · A_n^T[i, j]
    A_T = A_batch.transpose(-1, -2)                                 # [N, d, d]
    LHS_post = (
        I_mat[None, None, :, :]
        - one_minus_u[None, :, :, None] * A_T[:, None, :, :]
    )                                                               # [N, B, d, d]

    rhs_post = eps_tilde.transpose(-1, -2).contiguous()             # [N, B, d, S]
    z_post = _robust_solve(LHS_post, rhs_post).transpose(-1, -2)
    z_post = torch.clamp(z_post, min=-20.0, max=20.0)               # [N, B, S, d]

    # Decode all N*B*S cells in one shot.
    x_hat_post = model.decoder(z_post.reshape(-1, d)).reshape(n_dag_samples, B, S, G)

    # Always decode z_prev too — needed for match_delta and (when active) the
    # aux propagation loss. Cheap relative to the SCM solve.
    x_hat_prev = model.decoder(z_prev.reshape(-1, d)).reshape(n_dag_samples, B, S, G)

    # ── Loss ────────────────────────────────────────────────────────────────
    if match_delta:
        # Phase H — delta-matching.  Decode z_prev too so the bias `b` (and the
        # encoder/decoder "trivial recon path") cancels in the difference.
        # delta_pred[n, b] = mean over S of (decode(z_post[n,b,:]) - decode(z_prev[n,b,:]))
        delta_pred = (x_hat_post - x_hat_prev).mean(dim=2)          # [N, B, G]
        # dx_all is the per-sentence observed delta — exactly mean(x_pert) - mean(x_ctrl).
        # MSE between predicted and observed per-sentence mean shifts.
        if loss_kind == "mse":
            loss = (delta_pred - dx_all[None, :, :]).pow(2).mean()
        elif loss_kind in ("energy", "mmd_rbf"):
            # Distributional loss over the per-sentence delta as a "distribution" doesn't
            # make sense (each sentence gives one delta, not a set).  Fall back to MSE.
            loss = (delta_pred - dx_all[None, :, :]).pow(2).mean()
        else:
            raise ValueError(f"Unknown loss_kind: {loss_kind!r}")
    else:
        # Phase F/G behaviour — match the absolute predicted perturbed cells.
        # Kept for ablation; over-parameterised (encoder/decoder can fit without A).
        if loss_kind == "mse":
            loss = (x_hat_post - x_pert_set[None, :, :, :]).pow(2).mean()
        else:
            per = []
            for n in range(n_dag_samples):
                for b in range(B):
                    per.append(distributional_loss(x_hat_post[n, b], x_pert_set[b], kind=loss_kind))
            loss = torch.stack(per).mean()

    # ── V2 auxiliary propagation-only loss ─────────────────────────────────
    # When aux_prop_weight > 0, ALSO run a forward pass with SPARSE dx_input
    # (target_lfc, one scalar at the target gene) and add a downstream-only
    # MSE term: matches delta_pred_sparse vs dx_all on every gene EXCEPT the
    # target. Keeps the existing (possibly leaky) loss intact while forcing
    # the SCM + decoder to learn how to propagate single-gene perturbations.
    aux_loss_val = 0.0
    aux_dz_norm = 0.0
    if aux_prop_weight > 0:
        # Build sparse dx via target_lfc (Option A magnitude).
        dx_sparse = build_dx_input_batch(
            input_mode='target_lfc',
            gene_idx_batch=gene_idx,
            dx_full_batch=dx_all,
            n_genes=G,
            pert_direction_batch=pert_direction_batch,
            canonical_magnitude=canonical_magnitude,
        )                                                          # [B, G]
        Dt_dx_sparse_T = D.T @ dx_sparse.T                          # [d, B]
        dz_sparse = _robust_solve(DtD, Dt_dx_sparse_T).T       # [B, d]

        z_target_sparse = z_prev + dz_sparse[None, :, None, :]      # [N, B, S, d]
        eps_tilde_sparse = (
            u_bcast * z_target_sparse
            + (1.0 - u_bcast) * eps_ctrl_set[None, :, :, :]
        )
        rhs_post_sparse = eps_tilde_sparse.transpose(-1, -2).contiguous()
        z_post_sparse = _robust_solve(LHS_post, rhs_post_sparse).transpose(-1, -2)
        z_post_sparse = torch.clamp(z_post_sparse, min=-20.0, max=20.0)
        x_hat_post_sparse = model.decoder(
            z_post_sparse.reshape(-1, d)
        ).reshape(n_dag_samples, B, S, G)

        delta_pred_sparse = (x_hat_post_sparse - x_hat_prev).mean(dim=2)  # [N, B, G]

        # Mask out the target gene per batch element.
        downstream_mask = torch.ones((B, G), device=device, dtype=delta_pred_sparse.dtype)
        batch_arange_aux = torch.arange(B, device=device)
        downstream_mask[batch_arange_aux, gene_idx.long()] = 0.0

        aux_loss = (
            ((delta_pred_sparse - dx_all[None, :, :]) ** 2)
            * downstream_mask[None, :, :]
        ).sum() / (downstream_mask.sum() * n_dag_samples + 1e-8)
        loss = loss + aux_prop_weight * aux_loss
        aux_loss_val = float(aux_loss.item())
        aux_dz_norm = float(dz_sparse.norm(dim=1).mean().item())

    with torch.no_grad():
        diagnostics = {
            'loss': float(loss.item()),
            'kind': loss_kind,
            'match_delta': bool(match_delta),
            'input_mode': input_mode,
            'aux_prop_weight': float(aux_prop_weight),
            'aux_loss': aux_loss_val,
            'mean_dz_norm': float(dz_all.norm(dim=1).mean().item()),
            'mean_aux_dz_norm': aux_dz_norm,
            'mean_u_l1': float(u_all.sum(dim=1).mean().item()),
            'n_sentences': B,
            'n_dag_samples': n_dag_samples,
            'cell_sentence_S': S,
        }
    return loss, diagnostics


def perturbation_set_loss_shift(
    x_ctrl_set,        # Tensor[B, S, n_genes] — paired control cells per sentence
    x_pert_set,        # Tensor[B, S, n_genes] — paired perturbed cells per sentence
    gene_idx,          # LongTensor[B]         — perturbed gene index
    batch_id,          # LongTensor[B]         — gem_group id (passed to encoder)
    modality_id,       # LongTensor[B] or None — modality id (Norman/Replogle/...)
    model,
    n_dag_samples=1,
    mse_weight=1.0,
    mmd_weight=1.0,
    energy_weight=0.0,
    cos_weight=0.0,
    shift_l1_weight=0.0,
):
    """Shift-encoder set-level perturbation loss (SCCVAE-style additive shift).

    Replaces the leaky `dz = D⁺·dx` pathway of `perturbation_set_loss_u` with a
    *learned* perturbation shift `S^p = c · ShiftEncoder(pert_label[gene])`,
    injected additively into the SCM (no u-score mutilation):

        z_prev = (I − A^T)^{-1} ε_ctrl
        z_post = (I − A^T)^{-1} (ε_ctrl + S^p)        (same LHS — pure additive shift)
        x_hat  = decoder(z_post)

    The perturbation label exists for unseen genes, so this generalises zero-shot
    and contains NO observed-answer leakage. The loss is a weighted sum of:

      • mse_weight    · MSE(delta_pred, delta_obs)         — mean-shift (bias-cancelled
                          via decode(z_post) − decode(z_prev)); the SCCVAE MSE-on-mean term.
      • mmd_weight    · MMD(X̂^p_set, X^p_set)               — set-level distribution match.
      • energy_weight · energy(X̂^p_set, X^p_set)            — set-level energy distance (STATE).

    Works for both the linear decoder (interpretable D) and the MLP decoder; the
    shift path never references `D`.

    Returns: (loss: Tensor, diagnostics: dict).
    """
    B, S, G = x_ctrl_set.shape
    device = x_ctrl_set.device
    d = model.latent_dim
    N = n_dag_samples

    # Encode all B*S control cells under their gem_group / modality.
    x_ctrl_flat = x_ctrl_set.reshape(B * S, G)
    batch_flat = batch_id.repeat_interleave(S)
    mod_flat = modality_id.repeat_interleave(S) if modality_id is not None else None
    _, _, eps_ctrl_flat = model.encoder(
        x_ctrl_flat, modality_id=mod_flat, batch_id=batch_flat
    )
    eps_ctrl_set = eps_ctrl_flat.reshape(B, S, d)                  # [B, S, d]

    # Learned per-sentence latent shift S^p (= c · ShiftEncoder(label[gene])).
    S_shift = model.compute_shift(gene_idx)                        # [B, d]

    # Observed per-sentence mean shift (target for the MSE term). Detached so the
    # MSE pushes the prediction toward the observation, not vice-versa.
    dx_all = (x_pert_set.mean(1) - x_ctrl_set.mean(1)).detach()    # [B, G]

    # Sample N DAGs. z_prev and z_post share LHS = (I − A^T): the shift is purely
    # additive in exogenous space, so both solves use the same factorisation.
    A_batch = model.dag.sample_A(n_samples=N)                      # [N, d, d]
    I_mat = torch.eye(d, device=device, dtype=A_batch.dtype)
    LHS = I_mat.unsqueeze(0) - A_batch.transpose(-1, -2)           # [N, d, d]

    # z_prev = (I − A^T)^{-1} ε_ctrl
    rhs_prev = eps_ctrl_set.reshape(B * S, d).T                    # [d, B*S]
    rhs_prev = rhs_prev.unsqueeze(0).expand(N, d, B * S).contiguous()
    z_prev = _robust_solve(LHS, rhs_prev).transpose(-1, -2).reshape(N, B, S, d)
    z_prev = torch.clamp(z_prev, min=-20.0, max=20.0)

    # z_post = (I − A^T)^{-1} (ε_ctrl + S^p)   (shift broadcast over the S cells)
    eps_post_set = eps_ctrl_set + S_shift.unsqueeze(1)             # [B, S, d]
    rhs_post = eps_post_set.reshape(B * S, d).T                    # [d, B*S]
    rhs_post = rhs_post.unsqueeze(0).expand(N, d, B * S).contiguous()
    z_post = _robust_solve(LHS, rhs_post).transpose(-1, -2).reshape(N, B, S, d)
    z_post = torch.clamp(z_post, min=-20.0, max=20.0)

    # Decode (works for linear and MLP decoders; broadcasts over leading dims).
    x_hat_post = model.decoder(z_post.reshape(-1, d)).reshape(N, B, S, G)
    x_hat_prev = model.decoder(z_prev.reshape(-1, d)).reshape(N, B, S, G)

    # ── MSE on the bias-cancelled mean shift ────────────────────────────────
    delta_pred = (x_hat_post - x_hat_prev).mean(dim=2)            # [N, B, G]
    mse_term = (delta_pred - dx_all[None, :, :]).pow(2).mean()

    # ── Cosine-direction term on the mean shift ─────────────────────────────
    # 1 − cos(Δ̂, Δ_obs): matches the DIRECTION of the perturbation effect, and
    # is scale-invariant (magnitude is left to MMD + the penetrance scalar).
    cos_term = torch.zeros((), device=device)
    if cos_weight > 0:
        cos = torch.nn.functional.cosine_similarity(
            delta_pred, dx_all[None, :, :], dim=-1)               # [N, B]
        cos_term = (1.0 - cos).mean()

    # ── Set-level distributional terms ──────────────────────────────────────
    mmd_term = torch.zeros((), device=device)
    energy_term = torch.zeros((), device=device)
    if mmd_weight > 0 or energy_weight > 0:
        mmd_vals, energy_vals = [], []
        for n in range(N):
            for b in range(B):
                pred_set = x_hat_post[n, b]                       # [S, G]
                obs_set = x_pert_set[b]                           # [S, G]
                if mmd_weight > 0:
                    mmd_vals.append(distributional_loss(pred_set, obs_set, kind="mmd_rbf"))
                if energy_weight > 0:
                    energy_vals.append(distributional_loss(pred_set, obs_set, kind="energy"))
        if mmd_vals:
            mmd_term = torch.stack(mmd_vals).mean()
        if energy_vals:
            energy_term = torch.stack(energy_vals).mean()

    # ── Shift-L1 penalty (force the perturbation through A, not the free shift) ──
    # An L1 penalty on the learned shift S^p makes the cheap "bypass A" path
    # (z_post ≈ ε + S^p) costly, so to fit the downstream DE the model must route
    # a sparse shift THROUGH the SCM A. Direct fix for the A-collapse / bypass.
    shift_l1_term = S_shift.abs().mean()

    loss = (mse_weight * mse_term + cos_weight * cos_term
            + mmd_weight * mmd_term + energy_weight * energy_term
            + shift_l1_weight * shift_l1_term)

    with torch.no_grad():
        diagnostics = {
            'loss': float(loss.item()),
            'mse_term': float(mse_term.item()),
            'cos_term': float(cos_term.item()) if cos_weight > 0 else 0.0,
            'mmd_term': float(mmd_term.item()) if mmd_weight > 0 else 0.0,
            'energy_term': float(energy_term.item()) if energy_weight > 0 else 0.0,
            'shift_l1_term': float(shift_l1_term.item()),
            'shift_l1_weight': float(shift_l1_weight),
            'mse_weight': float(mse_weight),
            'cos_weight': float(cos_weight),
            'mmd_weight': float(mmd_weight),
            'energy_weight': float(energy_weight),
            'mean_shift_norm': float(S_shift.norm(dim=1).mean().item()),
            'shift_scale': float(model.shift_scale.detach().item()) if model.shift_scale is not None else 0.0,
            'n_sentences': B,
            'n_dag_samples': N,
            'cell_sentence_S': S,
        }
    return loss, diagnostics


def perturbation_set_loss_shift_mutilate(
    x_ctrl_set,        # Tensor[B, S, n_genes]
    x_pert_set,        # Tensor[B, S, n_genes]
    gene_idx,          # LongTensor[B]
    batch_id,          # LongTensor[B]
    modality_id,       # LongTensor[B] or None
    model,
    n_dag_samples=1,
    mse_weight=1.0,
    mmd_weight=1.0,
    energy_weight=0.0,
    u_mode='loading',     # 'loading' (clip Σ L[gene]) | 'shift' (min(1, β|S|))
    u_clip_max=0.5,
    shift_beta=1.0,
    shift_l1_weight=0.0,
    contrastive_weight=0.0,   # InfoNCE across the batch's TFs (attacks discrimination collapse)
    contrastive_temp=0.2,
    drift=None,               # [G] shared across-TF mean Δ (the differentiation program)
    residual_target=False,    # train the shift to predict Δ − drift (the TF-specific residual)
    fate_labels=None,         # [n_genes] int fate id (-1 = unannotated): identifiability reg
    ident_weight=0.0,         # supervised-contrastive-by-fate on the shift (structure latent by fate)
    ident_temp=0.2,
    dag_util_weight=0.0,      # DAG-utilization: penalize when A=0 predicts the Δ as well as A
    dag_util_margin=0.0,      # require the A-path to beat the no-A path by this MSE margin
    target_l0_weight=0.0,     # exp20 Fix 2: L0 penalty on the sparse per-TF target gate (open-prob)
    target_whiten=None,       # [G] per-gene std of the (drift-removed) target Δ: MSE on whitened
                              # residual so global MAGNITUDE stops being the easy win (mode-collapse fix)
):
    """Shift-FED MUTILATION set loss (Tier-2): the shift defines the latent target,
    the do-operator (column-mutilated SCM) propagates it.

        S         = c · ShiftEncoder(pert_label[gene])           (learned shift)
        z_prev    = (I − A^T)^{-1} ε_ctrl ;  z_S = (I − A^T)^{-1} S
        z_target  = z_prev + z_S
        u         = clip(Σ L[gene], u_max)  OR  min(1, β|S|)     (which programs clamped)
        ε̃         = u ⊙ z_target + (1 − u) ⊙ ε_ctrl
        z_post    = (I − M A^T)^{-1} ε̃,   M = diag(1 − u)        (per-sentence solve)

    Unlike `perturbation_set_loss_shift` (additive, shared LHS), here M depends on
    the perturbed gene per sentence, so the LHS is built per-sentence and solved
    batched [N, B, d, d]. For singles u clamps the perturbed gene's programs; the
    combo non-additivity (set-dependent M) emerges at eval. Loss = MSE(mean shift)
    + MMD(sets) [+ energy].
    """
    B, S, G = x_ctrl_set.shape
    device = x_ctrl_set.device
    d = model.latent_dim
    N = n_dag_samples

    x_ctrl_flat = x_ctrl_set.reshape(B * S, G)
    batch_flat = batch_id.repeat_interleave(S)
    mod_flat = modality_id.repeat_interleave(S) if modality_id is not None else None
    _, _, eps_ctrl_flat = model.encoder(x_ctrl_flat, modality_id=mod_flat, batch_id=batch_flat)
    eps_ctrl_set = eps_ctrl_flat.reshape(B, S, d)                       # [B, S, d]

    # gene_idx is [B, K] padded with -1 (K=2: combo=[ga,gb], single=[g,-1]).
    # Summed shift over the valid genes: S = Σ_k compute_shift(gene_k).
    if gene_idx.dim() == 1:
        gene_idx = gene_idx.unsqueeze(-1)                              # [B] -> [B, 1]
    gmask = (gene_idx >= 0).float()                                    # [B, K]
    gi_c = gene_idx.clamp(min=0)
    S_per = model.compute_shift(gi_c)                                  # [B, K, d]
    S_shift = (S_per * gmask.unsqueeze(-1)).sum(1)                     # [B, d]  (1 gene single, 2 combo)
    dx_all = (x_pert_set.mean(1) - x_ctrl_set.mean(1)).detach()        # [B, G]
    if residual_target and drift is not None:
        dx_all = dx_all - drift.view(1, -1)                            # train on TF-specific residual

    # per-sentence intervention strength u  [B, d] (single OR summed-over-combo)
    if u_mode == 'shift':
        u = (shift_beta * S_shift.abs()).clamp(max=1.0)               # [B, d]
    elif u_mode == 'loading':
        L_g = model.get_loadings()[gi_c]                              # [B, K, d]
        u = (L_g * gmask.unsqueeze(-1)).sum(1).clamp(0.0, u_clip_max) # [B, d]  (Σ loadings over combo)
    else:
        raise ValueError(f"unknown u_mode={u_mode!r}")
    u = model.refine_u(u, S_shift)                                    # cross-program nonlinear gate (no-op unless u_gate_mlp)
    one_minus_u = (1.0 - u)                                            # [B, d]

    A_batch = model.dag.sample_A(n_samples=N)                          # [N, d, d]
    I_mat = torch.eye(d, device=device, dtype=A_batch.dtype)
    LHS_obs = I_mat.unsqueeze(0) - A_batch.transpose(-1, -2)           # [N, d, d]

    # z_prev (control) under the UNMUTILATED graph — the Δ baseline (x_hat_prev below)
    rhs_prev = eps_ctrl_set.reshape(B * S, d).T.unsqueeze(0).expand(N, d, B * S).contiguous()
    z_prev = torch.clamp(_robust_solve(LHS_obs, rhs_prev).transpose(-1, -2).reshape(N, B, S, d), -20, 20)
    u_b = u.view(1, B, 1, d)
    if getattr(model, 'pert_intervene_mode', 'legacy') == 'structural':
        # Fix 1: inject S into the exogenous noise; the SINGLE mutilated solve below propagates
        # it through A only. No z_S/z_target pre-propagation → non-target programs move ONLY via
        # A, so A=0 ⇒ no downstream effect (A load-bearing by construction).
        eps_tilde = eps_ctrl_set.unsqueeze(0) + S_shift.view(1, B, 1, d)   # [N, B, S, d]
    else:
        # legacy (exp19): pre-propagate S and inject the full target state (the leak).
        z_S = _robust_solve(LHS_obs, S_shift.T.unsqueeze(0).expand(N, d, B).contiguous())
        z_S = z_S.transpose(-1, -2).reshape(N, B, d)                      # [N, B, d]
        z_target = z_prev + z_S.unsqueeze(2)                             # [N, B, S, d]
        eps_tilde = model.mix_eps(eps_ctrl_set.unsqueeze(0), z_target, u_b)

    # per-(N,B) mutilated LHS = I − diag(1−u_b) A_n^T   [N, B, d, d]
    A_T = A_batch.transpose(-1, -2)                                    # [N, d, d]
    LHS_mut = (I_mat.view(1, 1, d, d)
               - A_T.view(N, 1, d, d) * one_minus_u.view(1, B, d, 1))  # [N, B, d, d]
    rhs_mut = eps_tilde.transpose(-1, -2)                              # [N, B, d, S]
    z_post = _robust_solve(LHS_mut, rhs_mut).transpose(-1, -2)    # [N, B, S, d]
    z_post = torch.clamp(z_post, -20, 20)

    # decode through the SEPARATE perturbation head if present (decouples the GRN decoder
    # from the residual target); it taps z_post so A still gets interventional gradients.
    pert_dec = getattr(model, 'pert_decoder', None) or model.decoder
    x_hat_post = pert_dec(z_post.reshape(-1, d)).reshape(N, B, S, G)
    x_hat_prev = pert_dec(z_prev.reshape(-1, d)).reshape(N, B, S, G)

    delta_pred = (x_hat_post - x_hat_prev).mean(dim=2)                 # [N, B, G]
    if target_whiten is not None:
        # whiten per gene: divide pred+target by the target's per-gene std → the loss rewards
        # getting the DIRECTION of the residual right, not its overall magnitude (kills the
        # shift_scale shortcut that collapses all TFs to one scaled shared program).
        w = target_whiten.view(1, 1, -1).clamp_min(1e-3)
        mse_term = ((delta_pred - dx_all[None, :, :]) / w).pow(2).mean()
    else:
        mse_term = (delta_pred - dx_all[None, :, :]).pow(2).mean()

    # DAG-UTILIZATION: predict the SAME Δ with A=0 (the decoder+shift bypass). With A=0 the
    # SCM is identity: z_prev0=ε, z_target0=ε+S, ε̃0=mix_eps(...), z_post0=ε̃0 (no solves).
    # Penalize relu(mse_with_A − mse_without_A + margin) so the A-path MUST beat the bypass →
    # forces the prediction to flow THROUGH the latent DAG instead of around it via the decoder.
    dag_util_term = torch.zeros((), device=device)
    if dag_util_weight > 0:
        z_prev0 = eps_ctrl_set                                          # [B, S, d]
        if getattr(model, 'pert_intervene_mode', 'legacy') == 'structural':
            eps_tilde0 = eps_ctrl_set + S_shift.unsqueeze(1)           # A=0 identity: z_post0 = ε + S
        else:
            z_target0 = eps_ctrl_set + S_shift.unsqueeze(1)            # [B, S, d]
            eps_tilde0 = model.mix_eps(eps_ctrl_set, z_target0, u.unsqueeze(1))  # [B, S, d]
        dp0 = (pert_dec(eps_tilde0.reshape(-1, d)).reshape(B, S, G)
               - pert_dec(z_prev0.reshape(-1, d)).reshape(B, S, G)).mean(dim=1)   # [B, G]
        mse_term0 = (dp0 - dx_all).pow(2).mean()
        dag_util_term = torch.relu(mse_term - mse_term0 + dag_util_margin)

    # InfoNCE contrastive: each TF's predicted Δ must retrieve its OWN observed Δ over the
    # batch's other TFs — directly trains the discrimination the baseline collapses on.
    contrastive_term = torch.zeros((), device=device)
    if contrastive_weight > 0 and B >= 2:
        pred_c = delta_pred.mean(0)                                    # [B, G]
        # drift removal: subtract the across-TF (batch) mean so InfoNCE targets the
        # TF-SPECIFIC residual, not the shared differentiation drift (feature_probe.py
        # showed the held-out signal lives ENTIRELY in the drift-removed residual).
        pred_c = pred_c - pred_c.mean(0, keepdim=True)
        dx_res = dx_all - dx_all.mean(0, keepdim=True)
        def _cn(M):
            M = M - M.mean(1, keepdim=True)
            return M / (M.norm(dim=1, keepdim=True) + 1e-8)
        pz, oz = _cn(pred_c), _cn(dx_res)                            # [B, G]
        sim = (pz @ oz.T) / contrastive_temp                          # [B, B]  pred_i · obs_j
        key = gi_c[:, 0]                                              # TF key per sentence
        same = (key.unsqueeze(0) == key.unsqueeze(1))                # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(same & ~eye, float('-inf'))            # same-TF off-diag not a negative
        labels = torch.arange(B, device=device)
        contrastive_term = 0.5 * (torch.nn.functional.cross_entropy(sim, labels)
                                  + torch.nn.functional.cross_entropy(sim.T, labels))

    mmd_term = torch.zeros((), device=device)
    energy_term = torch.zeros((), device=device)
    if mmd_weight > 0 or energy_weight > 0:
        mmd_vals, energy_vals = [], []
        for n in range(N):
            for b in range(B):
                pred_set = x_hat_post[n, b]; obs_set = x_pert_set[b]
                if mmd_weight > 0:
                    mmd_vals.append(distributional_loss(pred_set, obs_set, kind="mmd_rbf"))
                if energy_weight > 0:
                    energy_vals.append(distributional_loss(pred_set, obs_set, kind="energy"))
        if mmd_vals: mmd_term = torch.stack(mmd_vals).mean()
        if energy_vals: energy_term = torch.stack(energy_vals).mean()

    # IDENTIFIABILITY regularizer: supervised contrastive on the shift by ANNOTATED FATE —
    # TFs sharing a developmental fate (Joung NNLS annotation) pull together, different
    # fates push apart. Structures the latent along the identifiable (fate) axis.
    ident_term = torch.zeros((), device=device)
    if ident_weight > 0 and fate_labels is not None and B >= 2:
        fa = fate_labels[gi_c[:, 0]]                                   # [B] fate id per TF (-1=none)
        z = S_shift - S_shift.mean(1, keepdim=True)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
        sim = (z @ z.T) / ident_temp                                  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float('-inf'))
        valid = fa >= 0
        pos = (fa.unsqueeze(0) == fa.unsqueeze(1)) & ~eye & valid.unsqueeze(0) & valid.unsqueeze(1)
        anchors = valid & (pos.sum(1) > 0)                            # anchors with ≥1 same-fate positive
        if anchors.any():
            logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)    # masked log-softmax
            # torch.where (not *pos): the -inf diagonal × 0 would be NaN
            sup = torch.where(pos, logp, torch.zeros_like(logp)).sum(1) / pos.float().sum(1).clamp(min=1)
            ident_term = -sup[anchors].mean()                         # SupCon

    shift_l1_term = S_shift.abs().mean()

    # exp20 Fix 2: L0 target-sparsity — penalize the EXPECTED number of OPEN target gates
    # (not the shift magnitude). Scale-free in the raw shift, so on-target magnitude is free;
    # closes gates the task loss doesn't pay for → sparse target support that A must propagate.
    target_l0_term = torch.zeros((), device=device)
    if target_l0_weight > 0 and getattr(model, 'use_target_gate', False):
        gp = model.compute_gate_prob(gi_c)                            # [B, K, d] open-prob
        target_l0_term = (gp * gmask.unsqueeze(-1)).sum(1).mean()     # sum over perturbed genes, mean over TFs×programs

    loss = (mse_weight * mse_term + mmd_weight * mmd_term
            + energy_weight * energy_term + shift_l1_weight * shift_l1_term
            + contrastive_weight * contrastive_term + ident_weight * ident_term
            + dag_util_weight * dag_util_term + target_l0_weight * target_l0_term)

    with torch.no_grad():
        diagnostics = {
            'loss': float(loss.item()), 'mse_term': float(mse_term.item()),
            'mmd_term': float(mmd_term.item()) if mmd_weight > 0 else 0.0,
            'energy_term': float(energy_term.item()) if energy_weight > 0 else 0.0,
            'contrastive_term': float(contrastive_term.item()) if contrastive_weight > 0 else 0.0,
            'ident_term': float(ident_term.item()) if ident_weight > 0 else 0.0,
            'dag_util_term': float(dag_util_term.item()) if dag_util_weight > 0 else 0.0,
            'target_l0_term': float(target_l0_term.item()) if target_l0_weight > 0 else 0.0,
            'gate_open_frac': (float(model.compute_gate_prob(gi_c).mean().item())
                               if getattr(model, 'use_target_gate', False) else 1.0),
            'shift_l1_term': float(shift_l1_term.item()),
            'u_mode': u_mode, 'u_clip_max': float(u_clip_max), 'shift_beta': float(shift_beta),
            'mean_u': float(u.mean().item()), 'max_u': float(u.max().item()),
            'mean_shift_norm': float(S_shift.norm(dim=1).mean().item()),
            'shift_scale': float(model.shift_scale.detach().item()) if model.shift_scale is not None else 0.0,
            'n_sentences': B, 'n_dag_samples': N, 'cell_sentence_S': S,
        }
    return loss, diagnostics


def perturbation_consistency_loss_v2(
    x_ctrl, x_pert, dx, model, n_dag_samples=8, threshold_frac=0.1,
    ctrl_modality_id=None, pert_modality_id=None,
):
    """
    Perturbation consistency loss matching proposal Eq. 4:

      L_pert = (1/N) Σ_n || D(I − Ã_n^T)^{-1}(ε_ctrl + dε) + b − x_pert ||²

    Pipeline per DAG sample:
      1. dz = D†·dx                          (gene → latent projection)
      2. dε = (I − A_n^T)·dz                 (latent → exogenous via DAG)
      3. ε̃ = ε_ctrl + dε                     (apply FULL dε vector)
      4. Ã = mutilate A where |dε_i| > thr   (multi-column mutilation)
      5. z_post = (I − Ã^T)^{-1} · ε̃         (solve mutilated SCM)
      6. x̂ = D·z_post + b                    (decode)
      7. MSE(x̂, x_pert)

    Multi-column mutilation keeps A in the gradient path (no DAG cancellation)
    while preserving multi-program perturbation effects.

    For multi-modality models (e.g., joint Norman+Replogle), `ctrl_modality_id`
    is the per-cell modality of the CONTROL cells (encodes control state) and
    `pert_modality_id` is the perturbation's modality (CRISPRa or CRISPRi).
    The loss encodes ε_ctrl using the perturbed modality so the encoder learns
    modality-conditioned exogenous noise distributions consistent with the
    target perturbed cells.

    x_ctrl: Tensor[B, n_genes] — control cells
    x_pert: Tensor[B, n_genes] — perturbed cells
    dx: Tensor[n_genes] — precomputed gene-space perturbation effect (de_single)
    ctrl_modality_id: LongTensor[B] or None — per-control-cell modality
    pert_modality_id: LongTensor[B] or None — per-perturbed-cell modality (matches pert dataset)
    model: CausalVAE
    n_dag_samples: int
    threshold_frac: float — fraction of max|dε| for multi-column mutilation
    Returns: (loss: Tensor, diagnostics: dict)
    """
    d = model.latent_dim
    device = x_ctrl.device

    # Step 1: Project gene-space change to latent space via D†
    dz = model.compute_dz_from_dx(dx)  # [d]

    # Step 2: Encode control cells under the PERTURBED modality
    # (Why: we want ε_ctrl + dε to land in the perturbed-modality distribution, so
    # the decoder reconstructs x_pert correctly. Using pert_modality for ε_ctrl
    # means "starting from a control cell as if it were in the pert modality".)
    enc_modality = pert_modality_id if pert_modality_id is not None else ctrl_modality_id
    _, _, epsilon_ctrl = model.encoder(x_ctrl, modality_id=enc_modality)  # [B, d]

    # Step 3: Multi-DAG intervention and loss
    I_mat = torch.eye(d, device=device)
    A_batch = model.dag.sample_A(n_samples=n_dag_samples)  # [N, d, d]

    mse_losses = []
    n_affected_list = []

    for n in range(n_dag_samples):
        A = A_batch[n]  # [d, d]

        # Compute dε = (I − A^T)·dz for this DAG (detach for threshold, not for gradient)
        d_epsilon = (I_mat - A.T) @ dz  # [d] — differentiable through A

        # Multi-column mutilation + full dε shift (DAG in gradient path)
        z_post = model.dag.scm_intervene_multi(
            epsilon_ctrl, A, d_epsilon, threshold_frac=threshold_frac
        )
        x_hat_pert = model.decoder(z_post)

        mse_losses.append(F.mse_loss(x_hat_pert, x_pert, reduction='mean'))

        # Diagnostics (detached)
        with torch.no_grad():
            threshold = threshold_frac * d_epsilon.abs().max()
            n_affected_list.append(int((d_epsilon.abs() > threshold).sum().item()))

    loss = torch.stack(mse_losses).mean()
    diagnostics = {
        'mse': loss.item(),
        'n_affected_programs': n_affected_list,
        'mean_affected': sum(n_affected_list) / len(n_affected_list),
    }
    return loss, diagnostics


# def biological_prior_loss(model, gene_names, collectri_net=None):
#     """
#     Soft biological prior loss using CollecTRI TF-target network.

#     Encourages the learned GRN (G_feat = D @ A @ D^T) to be consistent
#     with known TF-target interactions.

#     model: CausalVAE
#     gene_names: list of gene names in the expression matrix
#     collectri_net: pandas DataFrame with columns 'source' and 'target'
#     """
#     if collectri_net is None or len(collectri_net) == 0:
#         return torch.tensor(0.0, device=model.decoder.D.device)

#     # Get mean edge probability matrix
#     A_mean = model.dag.get_edge_probs()  # [d, d]
#     G_feat = model.decoder.project_dag_to_genes(A_mean)  # [n_genes, n_genes]

#     gene_to_idx = {g: i for i, g in enumerate(gene_names)}

#     prior_loss = torch.tensor(0.0, device=G_feat.device)
#     n_pairs = 0

#     for _, row in collectri_net.iterrows():
#         tf = str(row['source'])
#         target = str(row['target'])
#         if tf in gene_to_idx and target in gene_to_idx:
#             tf_idx = gene_to_idx[tf]
#             tgt_idx = gene_to_idx[target]
#             # Encourage G_feat[tf_idx, tgt_idx] > 0
#             prior_loss = prior_loss - torch.log(torch.sigmoid(G_feat[tf_idx, tgt_idx]) + 1e-8)
#             n_pairs += 1

#     if n_pairs > 0:
#         prior_loss = prior_loss / n_pairs

#     return prior_loss

def biological_prior_loss(model, gene_names, collectri_net=None, margin=0.1):
    """
    Soft biological prior loss using CollecTRI TF-target network.

    Encourages the learned GRN (G_feat = D @ A @ D^T) to be consistent
    with known TF-target interactions by using a Margin Ranking Loss.
    Known interactions are pushed to have higher scores than random background edges.

    model: CausalVAE
    gene_names: list of gene names in the expression matrix
    collectri_net: pandas DataFrame with columns 'source' and 'target'
    margin: the margin for torch.nn.MarginRankingLoss
    """
    device = model.decoder.D.device
    
    if collectri_net is None or len(collectri_net) == 0:
        return torch.tensor(0.0, device=device)

    # Get mean edge probability matrix and project to genes
    A_mean = model.dag.get_edge_probs()  # [d, d]
    G_feat = model.decoder.project_dag_to_genes(A_mean)  # [n_genes, n_genes]

    n_genes = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # 1. Build a boolean mask for the known priors
    mask_grn = torch.zeros((n_genes, n_genes), dtype=torch.bool, device=device)
    
    for _, row in collectri_net.iterrows():
        tf = str(row['source'])
        target = str(row['target'])
        if tf in gene_to_idx and target in gene_to_idx:
            tf_idx = gene_to_idx[tf]
            tgt_idx = gene_to_idx[target]
            mask_grn[tf_idx, tgt_idx] = True

    # 2. Flatten for vectorized operations (matching your prior code)
    proj_grn_flat = G_feat.flatten()
    idx_g = mask_grn.flatten()
    
    loss_prior = torch.tensor(0.0, device=device)
    criterion = torch.nn.MarginRankingLoss(margin=margin)

    # 3. Calculate margin ranking loss between signal and sampled background
    if idx_g.any():
        signal_scores = proj_grn_flat[idx_g]
        bg_indices = torch.nonzero(~idx_g, as_tuple=True)[0]
        
        if len(bg_indices) > 0:
            # Sample background indices to match the number of signal edges
            num_samples = min(len(signal_scores), len(bg_indices))
            rand_bg_idx = bg_indices[torch.randperm(len(bg_indices), device=device)[:num_samples]]
            
            background_scores = proj_grn_flat[rand_bg_idx]
            
            # Target is 1 because we want signal_scores > background_scores by at least `margin`
            target = torch.ones(num_samples, device=device)
            loss_prior = criterion(signal_scores[:num_samples], background_scores, target)

    return loss_prior


def compute_total_loss(
    x, x_hat, mu, log_var, A, model,
    beta=1.0,
    lambda_pert=10.0,
    lambda_dag=0.0,
    lambda_sparse=0.01,
    lambda_sparse_mode='product',
    lambda_spectral=10.0,
    lambda_ortho=1.0,
    lambda_prior=0.1,
    lambda_dagma=0.0,
    rho_dagma=1.0,
    lambda_D_sparse=0.0,
    pert_loss=None,
    prior_loss=None,
    dag_active=True,
    lambda_recon=1.0,         # exp20: down-weight the per-cell recon (the trivial path that
                              #        lets encoder+decoder fit x WITHOUT A). <1 pushes load onto SCM.
    residual_recon=False,     # exp20: reconstruct (x − drift) instead of raw x (near-no-op for
                              #        a linear decoder unless paired w/ separate/frozen pert decoder)
    drift=None,               # [G] shared across-TF mean Δ (for residual_recon)
    kl_free_bits=0.0,         # exp20: guarantee ≥ this many nats/dim of KL (guards posterior collapse
                              #        when lambda_recon is small)
):
    """
    Compute combined loss for Phase 1 training.

    Returns dict of all loss components and total loss.
    """
    losses = {}

    # Reconstruction (optionally on the drift-removed target, optionally down-weighted)
    recon_target = x
    if residual_recon and drift is not None:
        recon_target = x - drift.view(1, -1)
    l_recon = reconstruction_loss(recon_target, x_hat)
    losses['recon'] = l_recon

    # KL divergence (free-bits floor to keep the encoder informative when recon is down-weighted)
    l_kl = kl_divergence(mu, log_var)
    if kl_free_bits > 0:
        l_kl = torch.clamp(l_kl, min=float(kl_free_bits))
    losses['kl'] = l_kl

    # Couple KL to the recon weight so a small lambda_recon can't let KL crush an unregularized
    # encoder (ε→0 → z_prev collapses → pert loss starved).
    beta_eff = beta * lambda_recon if lambda_recon < 1.0 else beta
    total = lambda_recon * l_recon + beta_eff * l_kl

    # Orthogonality — always active (decoder regularization, not DAG-dependent)
    l_ortho = orthogonality_loss(model.decoder)
    losses['ortho'] = l_ortho
    total = total + lambda_ortho * l_ortho

    if dag_active:
        # DAGMA constraint
        h_val = model.dag.dagma_constraint(A)
        l_dag = dagma_augmented_lagrangian(h_val, lambda_dagma, rho_dagma)
        losses['dag'] = h_val
        losses['dag_lagrangian'] = l_dag
        total = total + l_dag

        # Sparsity (mode: 'product' = legacy L1 on sampled gates*weights;
        # 'gates' = L1 on deterministic edge probabilities sigmoid(θ/τ))
        l_sparse = sparsity_loss(A, model=model, mode=lambda_sparse_mode)
        losses['sparse'] = l_sparse
        total = total + lambda_sparse * l_sparse

        # Spectral stability
        l_spectral = spectral_stability_loss(A, model.dag)
        losses['spectral'] = l_spectral
        total = total + lambda_spectral * l_spectral

        # Gene-level decoder sparsity (L1 on entries of D). Only applies to the
        # linear decoder (the MLP decoder has no single D matrix).
        if lambda_D_sparse > 0 and getattr(model.decoder, 'is_linear', True):
            l_d_sparse = decoder_sparsity_loss(model.decoder)
            losses['d_sparse'] = l_d_sparse
            total = total + lambda_D_sparse * l_d_sparse
        else:
            losses['d_sparse'] = torch.tensor(0.0)
    else:
        losses['dag'] = torch.tensor(0.0)
        losses['dag_lagrangian'] = torch.tensor(0.0)
        losses['sparse'] = torch.tensor(0.0)
        losses['spectral'] = torch.tensor(0.0)
        losses['d_sparse'] = torch.tensor(0.0)

    # Perturbation consistency (clamp to prevent spike-driven gradient explosions)
    if pert_loss is not None:
        losses['pert'] = pert_loss  # log unclamped value
        pert_clamped = torch.clamp(pert_loss, max=50.0)
        total = total + lambda_pert * pert_clamped
    else:
        losses['pert'] = torch.tensor(0.0)

    # Biological prior
    if prior_loss is not None and lambda_prior > 0:
        losses['prior'] = prior_loss
        total = total + lambda_prior * prior_loss
    else:
        losses['prior'] = torch.tensor(0.0)

    losses['total'] = total
    return losses

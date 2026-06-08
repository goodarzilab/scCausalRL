"""Shared dx-input builders for eval and training.

The Causal VAE's intervention pipeline takes a length-n_genes `dx` vector and
projects it through D⁺ to obtain dz. Historically, both eval and training fed
the model the FULL OBSERVED differential-expression vector — which contains
the answer for every gene. To switch to legitimate perturbation-prediction
tasks, we build sparse `dx` vectors here:

  - input_mode = 'observed_full' : legacy leaky behaviour. Returns dx untouched.
  - input_mode = 'target_lfc'    : Option A — only the target gene gets a value
                                   equal to its OBSERVED lfc; everything else 0.
                                   Leaks ONE scalar; target is masked from scoring.
  - input_mode = 'canonical'     : Option B — strict zero-shot. The target gene
                                   gets `pert_direction * canonical_magnitude`;
                                   everything else 0. No observation leak.
"""
import torch


def build_dx_input_single(
    input_mode: str,
    target_idx,
    lfc_observed,
    n_genes: int,
    pert_direction: int = -1,
    canonical_magnitude: float = 2.0,
    device='cpu',
):
    """Build a length-n_genes dx vector for ONE perturbation.

    Args:
        input_mode: 'observed_full' | 'target_lfc' | 'canonical'
        target_idx: int (single pert) OR list[int] (combo)
        lfc_observed: length-n_genes observed lfc (only used for 'observed_full'
                      and 'target_lfc'). Can be numpy or torch.
        n_genes: number of genes (dx length).
        pert_direction: −1 for CRISPRi (knockdown), +1 for CRISPRa (activation).
                        Only used for 'canonical'.
        canonical_magnitude: magnitude for 'canonical' mode.
        device: torch device for the returned tensor.

    Returns:
        torch.Tensor [n_genes]
    """
    if input_mode == 'observed_full':
        if torch.is_tensor(lfc_observed):
            return lfc_observed.to(device=device, dtype=torch.float32)
        import numpy as np
        return torch.tensor(np.asarray(lfc_observed, dtype='float32'), device=device)

    dx = torch.zeros(n_genes, dtype=torch.float32, device=device)
    targets = [target_idx] if isinstance(target_idx, int) else list(target_idx)

    if input_mode == 'target_lfc':
        if torch.is_tensor(lfc_observed):
            lfc_np = lfc_observed.detach().cpu().numpy()
        else:
            import numpy as np
            lfc_np = np.asarray(lfc_observed)
        for t in targets:
            dx[t] = float(lfc_np[t])
        return dx

    if input_mode == 'canonical':
        val = float(pert_direction) * float(canonical_magnitude)
        for t in targets:
            dx[t] = val
        return dx

    raise ValueError(f'Unknown input_mode: {input_mode!r}. '
                     f"Expected 'observed_full', 'target_lfc', or 'canonical'.")


def build_dx_input_batch(
    input_mode: str,
    gene_idx_batch,
    dx_full_batch,
    n_genes: int,
    pert_direction_batch=None,
    canonical_magnitude: float = 2.0,
):
    """Vectorised batch version for the training loss.

    Args:
        input_mode: 'observed_full' | 'target_lfc' | 'canonical'.
        gene_idx_batch: LongTensor [B] of perturbed gene indices.
        dx_full_batch: Tensor [B, n_genes] — observed lfc per sentence.
                       For 'observed_full', returned as-is. For 'target_lfc',
                       only the target column is copied. For 'canonical',
                       unused.
        n_genes: G.
        pert_direction_batch: LongTensor [B] of +1 (CRISPRa) / -1 (CRISPRi),
                              or None → all -1 (default CRISPRi).
        canonical_magnitude: scalar multiplier for 'canonical' mode.

    Returns:
        Tensor [B, n_genes] dx_input.
    """
    if input_mode == 'observed_full':
        return dx_full_batch

    B = gene_idx_batch.shape[0]
    device = dx_full_batch.device
    out = torch.zeros((B, n_genes), dtype=dx_full_batch.dtype, device=device)
    batch_arange = torch.arange(B, device=device)
    gi = gene_idx_batch.long()

    if input_mode == 'target_lfc':
        out[batch_arange, gi] = dx_full_batch[batch_arange, gi]
        return out

    if input_mode == 'canonical':
        if pert_direction_batch is None:
            direction = torch.full((B,), -1.0, dtype=dx_full_batch.dtype, device=device)
        else:
            direction = pert_direction_batch.to(dtype=dx_full_batch.dtype, device=device)
        out[batch_arange, gi] = direction * float(canonical_magnitude)
        return out

    raise ValueError(f'Unknown input_mode: {input_mode!r}')

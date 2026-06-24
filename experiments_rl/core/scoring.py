"""Gene-space scoring — the falsifiable objective the planners optimize and are scored on.

The latent-distance objective is gameable (a planner can reach a target z via gene
sets that decode to the WRONG expression), so the planner instead optimizes the match
between its REACHED expression state (decode the do-operator output) and the target
perturbation's OBSERVED differential expression. Signed Pearson on the target's DEGs
(target genes masked) — higher is better; the same quantity is the ladder's primary
metric, so objective and evaluation are aligned and not gameable by latent geometry.
"""
import numpy as np
import torch


def torch_match(pred_rows, o_t, sel_idx):
    """Signed Pearson per row of pred_rows[:, sel_idx] vs o_t (all torch, on-device).
    pred_rows [K, n_genes]; o_t [S]; sel_idx [S]. Returns [K]."""
    p = pred_rows[:, sel_idx]
    p = p - p.mean(1, keepdim=True)
    o = o_t - o_t.mean()
    den = p.norm(dim=1) * o.norm() + 1e-8
    return (p * o).sum(1) / den


def observed_de(data, target_key):
    """Return (observed lfc [n_genes], DEG-eval mask [n_genes]) for a single/combo.
    The mask excludes the perturbed gene(s) themselves."""
    de = data.de_combo[target_key] if isinstance(target_key, tuple) else data.de_single[target_key]
    keys = target_key if isinstance(target_key, tuple) else (target_key,)
    mask = np.ones(len(data.gene_names), bool)
    for g in keys:
        mask[data.gene_to_idx[g]] = False
    return np.asarray(de['lfc'], dtype=np.float32), mask


def de_match(pred_de, lfc, mask, thresh=0.05):
    """Signed Pearson between predicted and observed DE on the target's DEGs
    (|lfc|>thresh) among non-target genes. Returns 0.0 if too few DEGs."""
    sel = mask & (np.abs(lfc) > thresh)
    if int(sel.sum()) < 3:
        return 0.0
    p = pred_de[sel]; o = lfc[sel]
    ps, os_ = p.std(), o.std()
    if ps < 1e-8 or os_ < 1e-8:
        return 0.0
    return float(np.corrcoef(p, o)[0, 1])

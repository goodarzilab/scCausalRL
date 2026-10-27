"""Per-gene perturbation descriptors for the shift encoder.

The shift encoder predicts a TF's latent effect from a per-gene feature `pert_label[g]`.
The baseline feature ('pca') is the control-cell co-expression PCA — which is INFORMATION-
FREE for TFs that are silent at baseline (the discrimination-collapse diagnosis: predicted
shifts for different held-out TFs had pairwise cosine 0.84). These descriptors add TF-IDENTITY
signal that generalises zero-shot:

  - 'pca'      : top-K PCA of the transposed control matrix (g's baseline co-expression).
  - 'regulon'  : top-K PCA of g's CollecTRI target-membership vector (g-as-TF → its targets);
                 a TF-identity descriptor known for any TF in CollecTRI, incl. held-out ones.
  - 'esm'      : reduced ESM2 protein embedding of TF g (sequence→function); loaded from a
                 precomputed npz {gene: emb} (build_esm_embeddings.py).

Descriptor string concatenates blocks, e.g. 'pca+regulon' → [control-PCA | regulon-PCA].
Each block is z-scored per component. Total width must equal the model's shift_label_dim.
"""
import os
import numpy as np
import torch


def _zscore(M):
    M = M - M.mean(0, keepdims=True)
    return M / (M.std(0, keepdims=True) + 1e-6)


def _pca_embed(M, q):
    """Top-q PCA embedding of rows of M [G, F] → [G, q], z-scored, padded if rank<q."""
    Mt = torch.as_tensor(M, dtype=torch.float32)
    qq = min(q, Mt.shape[0], Mt.shape[1])
    _, _, V = torch.pca_lowrank(Mt, q=qq, center=True)
    scores = (Mt - Mt.mean(0, keepdim=True)) @ V                  # [G, qq]
    scores = _zscore(scores.numpy())
    if qq < q:
        scores = np.concatenate([scores, np.zeros((scores.shape[0], q - qq), np.float32)], 1)
    return scores.astype(np.float32)


def _control_pca(X, control_idx, pca_dim, max_ctrl, seed):
    rng = np.random.default_rng(seed)
    sub = rng.choice(np.asarray(control_idx), min(max_ctrl, len(control_idx)), replace=False)
    M = np.asarray(X[sub]).T                                      # [G, n_ctrl]  (genes = rows)
    return _pca_embed(M, pca_dim)


def _regulon_pca(gene_names, regulon_dim):
    """Top-PCA of each gene's CollecTRI target-membership vector (g as TF → targets)."""
    from utils.bio_priors import load_collectri
    net = load_collectri(organism='human')
    gidx = {g.upper(): i for i, g in enumerate(gene_names)}
    G = len(gene_names)
    R = np.zeros((G, G), np.float32)                              # row g = g's targets
    if net is not None and len(net):
        for src, tgt in zip(net['source'].values, net['target'].values):
            si, ti = gidx.get(str(src).upper()), gidx.get(str(tgt).upper())
            if si is not None and ti is not None:
                R[si, ti] = 1.0
    n_tf = int((R.sum(1) > 0).sum())
    print(f"[descriptor] regulon: {n_tf}/{G} genes have CollecTRI targets")
    return _pca_embed(R, regulon_dim)


def _esm_embed(gene_names, esm_dim, esm_path):
    """Reduced ESM2 embedding per gene from a precomputed npz {gene->emb}. Missing→0."""
    if not os.path.exists(esm_path):
        raise FileNotFoundError(f"ESM embeddings not found at {esm_path}; run build_esm_embeddings.py first")
    z = np.load(esm_path, allow_pickle=True)
    emb = {str(g): v for g, v in zip(z['genes'], z['emb'])}
    dim0 = len(next(iter(emb.values())))
    M = np.stack([emb.get(g, np.zeros(dim0, np.float32)) for g in gene_names]).astype(np.float32)
    cov = int((np.abs(M).sum(1) > 0).sum())
    print(f"[descriptor] esm: {cov}/{len(gene_names)} genes have ESM2 embeddings (raw dim {dim0})")
    return _pca_embed(M, esm_dim)                                 # reduce + z-score to esm_dim


def build_pert_descriptor(X, control_idx, gene_names, m, config, descriptor='pca'):
    """Build the [n_genes, total_dim] descriptor for the given descriptor string."""
    seed = int(config['data'].get('split_seed', 42))
    pca_dim = int(m.get('shift_pca_dim', 50))
    reg_dim = int(m.get('shift_regulon_dim', 50))
    esm_dim = int(m.get('shift_esm_dim', 50))
    esm_path = m.get('shift_esm_path', os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'joung_esm2_emb.npz'))
    blocks = []
    for part in descriptor.split('+'):
        part = part.strip()
        if part == 'pca':
            blocks.append(_control_pca(X, control_idx, pca_dim,
                                       int(m.get('shift_pca_max_ctrl', 2000)), seed))
        elif part == 'regulon':
            blocks.append(_regulon_pca(gene_names, reg_dim))
        elif part == 'esm':
            blocks.append(_esm_embed(gene_names, esm_dim, esm_path))
        else:
            raise ValueError(f"unknown descriptor block {part!r}")
    return np.concatenate(blocks, axis=1).astype(np.float32)

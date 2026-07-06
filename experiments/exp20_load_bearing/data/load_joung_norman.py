"""Joung (hESC TF-OE) + Norman (K562 CRISPRa) multi-context loader for exp20.

Shared gene axis = Joung's (n_hvg=None); Norman is aligned onto it (zero-fill for genes absent from
Joung's axis) via load_norman_aligned. Perturbations are namespaced 'GENE__joung' / 'GENE__norman';
the per-cell modality_id is the DATASET/context (joung=0, norman=1) — the encoder's modality embedding
is the multi-context token that lets the model separate context-shared from TF-specific signal. The ESM
descriptor (built downstream from gene_names) is the cross-dataset bridge: the SAME TF gets the SAME
descriptor in both contexts, so a TF seen in one context can transfer to the other.

Eval-friendly held-out design: the gene-level train/val/test split is over the UNION of perturbed
genes and applied to BOTH contexts, so a held-out TF is held out in Joung AND Norman → enables
within-context zero-shot AND cross-context transfer on the same TFs.
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOD_JOUNG, MOD_NORMAN = 0, 1


def _split_genes(genes, train_frac=0.8, val_frac=0.1, seed=42):
    g = sorted(genes); rng = np.random.default_rng(seed); rng.shuffle(g)
    n = len(g); ntr = int(train_frac * n); nva = int(val_frac * n)
    return {'train': set(g[:ntr]), 'val': set(g[ntr:ntr + nva]), 'test': set(g[ntr + nva:])}


def _de(Xa, pert_idx, ctrl_mean):
    m = Xa[pert_idx]
    m = (m.toarray() if sp.issparse(m) else np.asarray(m)).mean(0)
    return (m - ctrl_mean).astype(np.float32)


def load_joung_norman(norman_path='/large_storage/goodarzilab/shervin/CausalBioRL/Norman/',
                      train_frac=0.8, val_frac=0.1, split_seed=42, min_cells_per_pert=5):
    from data.load_joung import load_joung
    from data.load_norman_aligned import load_norman_aligned

    print("[jn] loading Joung (full gene axis)...")
    J = load_joung(n_hvg=None, min_cells_per_pert=min_cells_per_pert)
    genes = [str(g) for g in J['gene_names']]; G = len(genes)
    Jx = J['adata'].X; Jx = Jx.toarray() if sp.issparse(Jx) else np.asarray(Jx)
    print(f"[jn] aligning Norman onto Joung's {G}-gene axis...")
    N = load_norman_aligned(norman_path, target_gene_names=genes, min_cells_per_pert=min_cells_per_pert)
    Nx = N['adata'].X; Nx = Nx.toarray() if sp.issparse(Nx) else np.asarray(Nx)

    nJ, nN = Jx.shape[0], Nx.shape[0]
    X = np.concatenate([Jx.astype(np.float32), Nx.astype(np.float32)], axis=0)
    modality = np.concatenate([np.full(nJ, MOD_JOUNG), np.full(nN, MOD_NORMAN)]).astype(np.int64)
    is_ctrl = np.zeros(nJ + nN, bool)
    for i in J['control_idx']:
        is_ctrl[i] = True
    for i in N['control_idx']:
        is_ctrl[nJ + i] = True
    # batch_id = dataset → the cell-sentence dataset pairs ctrl↔pert within the SAME gem_group, so
    # this GUARANTEES within-context pairing (Joung perts with Joung controls, Norman with Norman).
    obs = pd.DataFrame({'modality_id': modality, 'batch_id': modality, 'is_control': is_ctrl,
                        'dataset': ['joung'] * nJ + ['norman'] * nN})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))

    # namespaced single + combo perts (indices offset for Norman)
    single_perts, combo_perts = {}, {}
    for g, idx in J['single_perts'].items():
        single_perts[f'{g}__joung'] = list(idx)
    for g, idx in N['single_perts'].items():
        single_perts[f'{g}__norman'] = [nJ + i for i in idx]
    for k, idx in (J.get('combo_perts') or {}).items():
        combo_perts[(f'{k}')] = list(idx)
    for k, idx in (N.get('combo_perts') or {}).items():
        combo_perts[k] = [nJ + i for i in idx]
    control_idx = list(np.where(is_ctrl)[0])

    # per-namespaced-key de_single on the shared axis (within-context controls)
    Jc = Jx[J['control_idx']]; Jc = (Jc.toarray() if sp.issparse(Jc) else np.asarray(Jc)).mean(0)
    Nc = Nx[N['control_idx']]; Nc = (Nc.toarray() if sp.issparse(Nc) else np.asarray(Nc)).mean(0)
    de_single = {}
    for g, idx in J['single_perts'].items():
        de_single[f'{g}__joung'] = _de(Jx, idx, Jc)
    for g, idx in N['single_perts'].items():
        de_single[f'{g}__norman'] = _de(Nx, idx, Nc)
    adata.uns['de_single'] = de_single

    # shared gene-level split over the UNION of perturbed genes (un-namespaced)
    all_pert = {k.split('__')[0] for k in single_perts}
    pert_split = _split_genes(all_pert, train_frac, val_frac, split_seed)
    shared_tfs = sorted({k.split('__')[0] for k in single_perts if f'{k.split("__")[0]}__joung' in single_perts
                         and f'{k.split("__")[0]}__norman' in single_perts})
    print(f"[jn] cells J/N={nJ}/{nN} | single perts J/N="
          f"{sum('__joung' in k for k in single_perts)}/{sum('__norman' in k for k in single_perts)} | "
          f"shared TFs (both contexts)={len(shared_tfs)} | split train/val/test="
          f"{len(pert_split['train'])}/{len(pert_split['val'])}/{len(pert_split['test'])}")
    return {'adata': adata, 'control_idx': control_idx, 'single_perts': single_perts,
            'combo_perts': combo_perts, 'gene_names': genes, 'pert_genes': sorted(all_pert),
            'pert_split': pert_split, 'shared_tfs': shared_tfs, 'n_modalities': 2, 'n_batches': 2}


if __name__ == '__main__':
    dd = load_joung_norman()
    print("shared TFs (first 20):", dd['shared_tfs'][:20])

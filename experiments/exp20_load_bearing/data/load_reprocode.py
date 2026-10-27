"""REPROcode (Kurochkin et al. 2026, GSE298471) combinatorial immune-TF overexpression screen —
DOSE-AWARE per-cell world-model loader.

WHY REPROcode: the only dataset in the project with high-k combos (per-cell k up to ~40 TFs) AND
graded stoichiometry (per-TF barcode COPY COUNTS). This loader reads the pre-built PBMC proc h5ad
(`reprocode_pbmc_proc.h5ad`, 27,295 cells x 4,017 genes, X already log-normalized) whose obs carries
the parsed per-cell TF set + copy counts, so the raw CSV/h5ad merge in the old draft is no longer needed.

KEY DESIGN — PER-CELL variable-k, NOT exact-set pseudobulk grouping. At k up to 40 the in-vocab TF set
is almost always unique per cell, so grouping by exact set (the Briggs `combo_perts` pattern) would
collapse to singletons under any min-replicate floor. Instead EACH perturbed cell is its own training
example carrying (TF-set, dose-vector), consumed by DoseCellSentenceDataset. `de_single` (the per-TF
pseudobulk marginal) is kept only as a REFERENCE (drift / retrieval / probes), never as the target.

Controls: is_control (k=0) cells are pooled across panels (panel 9 = 1319 + panel 42 = 891 = 2210).
Panel 22 has NO controls of its own, so its perturbed cells borrow the pooled control (single ctrl pool).

Returns a dict mirroring load_joung_briggs:
  adata, control_idx, single_perts, de_single, gene_names, pert_genes, pert_split, dose_perts (the
  per-cell (cell_idx, gene_idx, copies) structure), fate labels, n_modalities/n_batches (0 -> single
  context, no modality/batch token), modality_gene_mask (None -> REPROcode is fully observed on its axis).
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPRO_VOCAB = "/large_storage/goodarzilab/shervin/CausalBioRL/REPROcode_2026/vocab"
REPRO_PROC_H5AD = os.path.join(REPRO_VOCAB, "reprocode_pbmc_proc.h5ad")
REPRO_EFF_TXT = os.path.join(REPRO_VOCAB, "effective_tf_genes_reprocode.txt")


def _split_genes(genes, train_frac=1.0, val_frac=0.0, seed=1, protect=None):
    """Seed-based train/val/test split over the modelable TFs (for the later leakage ladder). `protect`
    TFs are FORCED to train; train_frac applies to the splittable (non-protected) genes."""
    protect = set(protect or [])
    forced = sorted(g for g in genes if g in protect)
    g = sorted(g for g in genes if g not in protect); rng = np.random.default_rng(seed); rng.shuffle(g)
    n = len(g); ntr = int(train_frac * n); nva = int(val_frac * n)
    return {'train': set(g[:ntr]) | set(forced), 'val': set(g[ntr:ntr + nva]), 'test': set(g[ntr + nva:])}


def _densify(X):
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return np.asarray(X, dtype=np.float32)


def load_reprocode(proc_path=REPRO_PROC_H5AD, effective_tfs=REPRO_EFF_TXT,
                   train_frac=1.0, val_frac=0.0, split_seed=1,
                   min_single_cells=5, max_pert_cells=None, protect_tfs=None,
                   holdout_tfs=None, **kw):
    """Load the REPROcode PBMC proc h5ad into a dose-aware per-cell training structure.

    proc_path       : the pre-built log-normalized h5ad (obs: tf_set, tf_copies, k, panel, fate, is_control).
    effective_tfs   : path to the 42-TF list (only these, and only if on the gene axis, are modelable).
                      None -> every on-axis TF appearing in tf_set is modelable.
    train_frac/val_frac/split_seed : TF-level split for the leakage ladder (train_frac=1.0 -> all train).
    protect_tfs     : TFs FORCED into the train split (never held out). Use for the leave-TF-out rung to keep
                      ESM-uncovered TFs (no zero-shot descriptor) in train, so held-out TFs are a CLEAN
                      zero-shot test (every held-out TF has an ESM descriptor). List of names or a file path.
    min_single_cells: floor for a TF to get a de_single (marginal) reference vector.
    max_pert_cells  : optionally subsample the per-cell training examples (smoke / debugging).
    """
    if isinstance(protect_tfs, str) and os.path.exists(protect_tfs):
        protect_tfs = [l.strip() for l in open(protect_tfs) if l.strip()]
    if isinstance(holdout_tfs, str) and os.path.exists(holdout_tfs):
        holdout_tfs = [l.strip() for l in open(holdout_tfs) if l.strip()]
    proc_path = proc_path or REPRO_PROC_H5AD
    if effective_tfs is None:
        effective_tfs = REPRO_EFF_TXT
    adata = ad.read_h5ad(proc_path)
    genes = [str(g) for g in adata.var_names]
    g2i = {g: i for i, g in enumerate(genes)}
    X = _densify(adata.X)
    obs = adata.obs

    is_ctrl = obs['is_control'].values.astype(bool)
    control_idx = list(np.where(is_ctrl)[0])
    if not control_idx:
        raise RuntimeError("[repro] no control (is_control) cells found")
    control_mean = X[np.asarray(control_idx)].mean(0).astype(np.float32)

    eff = None
    if effective_tfs is not None and os.path.exists(str(effective_tfs)):
        eff = {l.strip() for l in open(effective_tfs) if l.strip()}

    tf_set = obs['tf_set'].astype(str).values
    tf_cop = obs['tf_copies'].astype(str).values

    single_perts = defaultdict(list)                  # TF -> cells containing it (in-vocab marginal)
    dose_perts = []                                   # (cell_idx, gene_idx[k], copies[k]) — per-cell example
    n_dropped_tf = 0
    for i in range(adata.n_obs):
        if is_ctrl[i]:
            continue
        s, c = tf_set[i], tf_cop[i]
        if not s or s == 'nan':
            continue
        tfs = s.split(';')
        try:
            copies = [float(x) for x in c.split(';')]
        except ValueError:
            continue
        if len(copies) != len(tfs):
            continue
        keep_g, keep_c = [], []
        for t, cp in zip(tfs, copies):
            t = t.strip()
            if t in g2i and (eff is None or t in eff):
                keep_g.append(g2i[t]); keep_c.append(cp)
            else:
                n_dropped_tf += 1
        if not keep_g:
            continue
        gidx = np.asarray(keep_g, np.int64); dose = np.asarray(keep_c, np.float32)
        dose_perts.append((int(i), gidx, dose))
        for j in keep_g:
            single_perts[genes[j]].append(int(i))

    # de_single: per-TF pseudobulk Δ = mean(cells containing TF) − pooled control mean (REFERENCE only).
    de_single = {t: (X[np.asarray(idx)].mean(0) - control_mean).astype(np.float32)
                 for t, idx in single_perts.items() if len(idx) >= min_single_cells}

    if max_pert_cells is not None and len(dose_perts) > int(max_pert_cells):
        rng = np.random.default_rng(split_seed)
        sel = sorted(rng.choice(len(dose_perts), int(max_pert_cells), replace=False).tolist())
        dose_perts = [dose_perts[i] for i in sel]

    # fate labels (per cell + int-encoded), for downstream fate readouts — not load-bearing for training.
    fate = obs['fate'].astype(str).values
    fate_cats = sorted({f for f in fate if f and f != 'nan'})
    fate_id = {c: i for i, c in enumerate(fate_cats)}
    fate_labels = np.array([fate_id.get(f, -1) for f in fate], np.int64)

    all_pert = set(single_perts)                      # modelable (on-axis, effective) TFs actually seen
    if holdout_tfs:                                   # EXPLICIT leave-TF-out set (reproducible; overrides train_frac)
        te = {t for t in holdout_tfs if t in all_pert}
        pert_split = {'train': all_pert - te, 'val': set(), 'test': te}
    else:
        pert_split = _split_genes(all_pert, train_frac, val_frac, split_seed, protect=protect_tfs)

    adata.uns['de_single'] = de_single
    adata.uns['de_single_reprocode'] = de_single

    ksz = [len(g) for _, g, _ in dose_perts]
    cop_all = np.concatenate([d for _, _, d in dose_perts]) if dose_perts else np.zeros(1, np.float32)
    print(f"[repro] {adata.n_obs} cells x {len(genes)} genes | control {len(control_idx)} "
          f"| per-cell perturbed examples {len(dose_perts)} | modelable TFs {len(all_pert)} "
          f"(dropped {n_dropped_tf} out-of-vocab TF occurrences) | de_single {len(de_single)} "
          f"| k med={int(np.median(ksz)) if ksz else 0} max={max(ksz) if ksz else 0} "
          f"| copies med={np.median(cop_all):.1f} max={cop_all.max():.0f} "
          f"| split train/val/test TFs = {len(pert_split['train'])}/{len(pert_split['val'])}/{len(pert_split['test'])}",
          flush=True)

    return {
        'adata': adata,
        'control_idx': control_idx,
        'single_perts': {g: list(v) for g, v in single_perts.items()},
        'combo_perts': {},                             # exact-set combos NOT used (per-cell dose path)
        'dose_perts': dose_perts,                      # THE per-cell (TF-set, dose) training structure
        'gene_names': genes,
        'pert_genes': sorted(all_pert),
        'pert_split': pert_split,
        'de_single': de_single,
        'de_single_reprocode': de_single,
        'fate': fate,
        'fate_labels': fate_labels,
        'n_modalities': 0,                             # single context -> no modality token
        'n_batches': 0,                                # pooled control -> no batch token
        'modality_gene_mask': None,                    # REPROcode fully observed on its own 4017 axis
    }


if __name__ == '__main__':
    dd = load_reprocode(max_pert_cells=500)
    print('smoke OK | examples:', len(dd['dose_perts']), '| controls:', len(dd['control_idx']),
          '| modelable TFs:', len(dd['pert_genes']), '| de_single:', len(dd['de_single']))

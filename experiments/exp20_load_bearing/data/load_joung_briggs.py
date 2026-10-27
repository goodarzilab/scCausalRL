"""Joung (hESC TF-OE, day 7) + Briggs (hESC->HSPC TF-OE combos, day 14/28) multi-context loader.

modality_id: joung=0, briggs=1 (the encoder's modality token separates context/timepoint-shared from
TF-specific signal; the ESM descriptor is the cross-context bridge). Joung contributes singles + its combos;
Briggs contributes COMBOS (2-3 TF gene-sets among screen cells) + pluripotent hESC-start "controls". Briggs
cells are aligned onto Joung's 4366-gene axis (matched CP10k-lognorm on shared genes, zero-fill the rest).

Held-out cocktails (the answer key) are excluded from TRAINING via `combo_holdout_keys` in the config
(train_vae's fixed-holdout path) — so recovery/synergy on them is honest, not memorization.
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOD_JOUNG, MOD_BRIGGS = 0, 1
BRIGGS_H5AD = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad"
BRIGGS_RAW = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026"
BRIGGS_SUP = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/Briggs_Supplementary"
# CLEAN cut: the 8 UNENRICHED raw files only (no _CD34 MACS, no _CFU — those are the HSPC-selected
# outcome). Keeps high-k combinatorial diversity; drops the HSPC-outcome supervision (population-level cut).
BRIGGS_UNENRICHED = ['GSE328457_103F_v1', 'GSE328457_103F_v2', 'GSE328457_130F_v1', 'GSE328457_13F_v1',
                     'GSE328457_773F_v1', 'GSE328457_773F_v2', 'GSE328457_773F_v3', 'GSE328457_773F_v4']
PLURI = ['POU5F1', 'NANOG', 'SOX2', 'LIN28A', 'DPPA4', 'ZFP42']


def _arrayed_combos():
    """The 221 arrayed-validated combos (mmc5 'Arrayed_screens') = the eval answer key. Returned as a set
    of frozensets of gene symbols (isoform-collapsed). Any Briggs combo matching one is a training leak."""
    try:
        d = pd.read_excel(f'{BRIGGS_SUP}/mmc5.xlsx', 'Arrayed_screens')
        out = set()
        for cid, sub in d.groupby('Combination_ID'):
            gs = frozenset(str(t).split('|')[0].strip() for t in sub['TF_names'])
            if len(gs) >= 2:
                out.add(gs)
        return out
    except Exception as e:
        print(f'[jb-clean] WARN could not read arrayed combos: {e}', flush=True); return set()
# Answer-key cocktails (held OUT of training so recovery is honest): arrayed-validated P1/P2 + the
# canonical HSPC synergy pairs. A high-k combo set is held out iff it is a SUBSET of P1 or P2 (a pure
# answer-key cocktail) or exactly an HSPC pair. It stays in de_combo (eval) but is dropped from training.
_P1 = {'TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'}
_P2 = {'DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85'}
_HSPC_PAIRS = {frozenset(p) for p in [('TAL1', 'LMO2'), ('TAL1', 'ETV2'), ('GATA2', 'ETV2'),
                                      ('GATA2', 'TAL1'), ('LMO2', 'ETV2'), ('GATA2', 'LMO2')]}


_ARRAYED = None
def _is_answerkey(gs):
    """STRICT: a combo leaks the P1/P2 answer if it shares >=2 members with P1 or P2 (e.g. {TAL1,LMO2,X}
    is 3/6 of P1), is an HSPC pair, or is one of the 221 arrayed-validated combos. Dropped from training."""
    global _ARRAYED
    if _ARRAYED is None:
        _ARRAYED = _arrayed_combos()
    s = set(gs)
    return (len(s & _P1) >= 2 or len(s & _P2) >= 2 or frozenset(s) in _HSPC_PAIRS or frozenset(s) in _ARRAYED)


def _lognorm(X):
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)


def _split_genes(genes, train_frac=1.0, val_frac=0.0, seed=1, protect=None):
    """Split single TFs into train/val/test. `protect` TFs are FORCED to train (never held out) —
    used to keep every combo-involved TF in training, so a single-TF holdout (leave-TF-out) doesn't
    silently gut combo coverage on a dense high-k screen where combos share hub TFs. The train_frac
    is applied to the SPLITTABLE (non-protected) genes."""
    protect = set(protect or [])
    forced = sorted(g for g in genes if g in protect)
    g = sorted(g for g in genes if g not in protect); rng = np.random.default_rng(seed); rng.shuffle(g)
    n = len(g); ntr = int(train_frac * n); nva = int(val_frac * n)
    return {'train': set(g[:ntr]) | set(forced), 'val': set(g[ntr:ntr + nva]), 'test': set(g[ntr + nva:])}


def _load_briggs(gene_names, min_combo_cells=10, kmax=8, plurip_pct=0.10):
    """Briggs cells on Joung's gene axis + per-cell TF gene-set + pluripotent controls + combo groups.

    HIGH-K: Briggs is a dense combinatorial screen (per-cell median 5 TFs, up to 17). We group cells by
    their IN-VOCAB TF set (the modelable TFs, projecting out the ~266/388 TFs not on the Joung axis) and
    keep sets with in-vocab size 2..kmax that have >= min_combo_cells replicates. Selecting on the IN-VOCAB
    set size (not the total-388 count) is the fix vs the old kmax=3-on-total that collapsed to 109 low-k
    sets: at kmax=8 / min_cells=10 this yields ~1000 sets (median k 5, up to ~65k combo cells). Replication
    over cells that share an in-vocab set but differ in out-of-vocab TFs marginalizes the unmodeled TFs."""
    g2i = {g: i for i, g in enumerate(gene_names)}
    B = ad.read_h5ad(BRIGGS_H5AD); bgenes = list(map(str, B.var_names))
    shared = [g for g in bgenes if g in g2i]
    shared_idx = np.array([g2i[g] for g in shared], np.int64)               # Briggs-observed axis positions
    bln = _lognorm(B[:, shared].X)                                          # [N, |shared|]
    Xb = np.zeros((B.shape[0], len(gene_names)), np.float32)
    Xb[:, shared_idx] = bln
    TF = B.obsm['TF_UMIs_filt']; TF = (TF.toarray() if sp.issparse(TF) else np.asarray(TF)) > 0
    iso = [str(t).split('|')[0] for t in B.uns['TF_names']]; ug = sorted(set(iso)); g2c = {g: i for i, g in enumerate(ug)}
    cg = np.zeros((B.shape[0], len(ug)), bool)
    for j, g in enumerate(iso):
        cg[:, g2c[g]] |= TF[:, j]
    pmk = [shared.index(g) for g in PLURI if g in shared]
    pscore = bln[:, pmk].mean(1); is_ctrl = pscore >= np.quantile(pscore, 1 - plurip_pct)
    # in-vocab TF set per cell (only modelable TFs); group singles (k==1) and high-k combos (k2..kmax)
    groups = defaultdict(list); singles = defaultdict(list)
    for i in np.where(~is_ctrl)[0]:
        gs = tuple(sorted(g for g in (ug[j] for j in np.where(cg[i])[0]) if g in g2i))
        if len(gs) == 1:
            singles[gs[0]].append(int(i))
        elif 2 <= len(gs) <= kmax:
            groups[gs].append(int(i))
    combo_perts = {gs: v for gs, v in groups.items() if len(v) >= min_combo_cells}
    briggs_singles = {g: v for g, v in singles.items() if len(v) >= min_combo_cells}   # in-vocab single-TF Briggs cells
    ksz = [len(gs) for gs in combo_perts]
    print(f"[jb] Briggs: {B.shape[0]} cells | shared {len(shared)}/{len(gene_names)} genes | "
          f"pluripotent-control {int(is_ctrl.sum())} | in-vocab combo sets (>= {min_combo_cells} cells, "
          f"k2-{kmax}) = {len(combo_perts)} | set-k med={int(np.median(ksz)) if ksz else 0} max={max(ksz) if ksz else 0} "
          f"| combo cells={sum(len(v) for v in combo_perts.values())} | in-vocab singles={len(briggs_singles)}", flush=True)
    return Xb, is_ctrl, combo_perts, shared_idx, briggs_singles


def _load_briggs_clean(gene_names, min_combo_cells=10, kmax=8, plurip_pct=0.10):
    """CLEAN Briggs: UNENRICHED raw files only (population cut #1) + drop the 221 arrayed combos (cut #2)
    + drop combos sharing >=2 TFs with P1/P2 (cut #3). Concatenate the 8 unenriched raw h5ads, align to
    Joung's axis, group by in-vocab TF set (k2..kmax, >=min_combo_cells). Genuinely leak-free HSPC training."""
    import gzip, shutil, tempfile
    g2i = {g: i for i, g in enumerate(gene_names)}
    Xs, TFsets_all = [], []
    tf_names_ref = None
    scratch = tempfile.mkdtemp(prefix='briggs_clean_')
    for stem in BRIGGS_UNENRICHED:
        gzf = f'{BRIGGS_RAW}/{stem}-raw-TF.h5ad.gz'
        if not os.path.exists(gzf):
            print(f'[jb-clean] missing {gzf}', flush=True); continue
        tmp = os.path.join(scratch, 'cur.h5ad')
        with gzip.open(gzf) as f, open(tmp, 'wb') as o:
            shutil.copyfileobj(f, o, 1 << 24)
        B = ad.read_h5ad(tmp); os.remove(tmp)
        bgenes = list(map(str, B.var_names)); shared = [g for g in bgenes if g in g2i]
        bln = _lognorm(B[:, shared].X)
        Xb = np.zeros((B.shape[0], len(gene_names)), np.float32)
        Xb[:, [g2i[g] for g in shared]] = bln
        Xs.append(Xb)
        TF = B.obsm['TF_UMIs']; TF = (TF.toarray() if sp.issparse(TF) else np.asarray(TF))
        TF = TF >= 2                                                    # confident TF calls (raw uses unfiltered UMIs; >=2 drops 1-UMI noise that fragments combo sets)
        iso = [str(t).split('|')[0] for t in B.uns['TF_names']]
        TFsets_all.append((TF, iso))
        print(f'[jb-clean] {stem}: {B.shape[0]} cells, shared {len(shared)}', flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    X = np.concatenate(Xs, 0); del Xs; import gc; gc.collect()
    # per-cell in-vocab TF set (collapse isoforms, union across the per-file TF panels)
    shared_idx = np.array(sorted({g2i[g] for g in gene_names if g in g2i and g in g2i}), np.int64)  # all axis
    # recompute shared_idx as Briggs-observed genes (union of shared across files) — use last file's shared set proxy:
    # simpler: mark every gene that is ever nonzero across Briggs cells as observed
    shared_mask = (np.abs(X).sum(0) > 0); shared_idx = np.where(shared_mask)[0].astype(np.int64)
    pmk = [g2i[g] for g in PLURI if g in g2i]
    pscore = X[:, pmk].mean(1) if pmk else np.zeros(X.shape[0], np.float32)
    is_ctrl = pscore >= np.quantile(pscore, 1 - plurip_pct)
    # combos
    from collections import defaultdict
    groups = defaultdict(list); singles = defaultdict(list); off = 0
    for TF, iso in TFsets_all:
        ug = sorted(set(iso)); g2c = {g: j for j, g in enumerate(ug)}
        cg = np.zeros((TF.shape[0], len(ug)), bool)
        for j, g in enumerate(iso):
            cg[:, g2c[g]] |= TF[:, j]
        for li in range(TF.shape[0]):
            gi = off + li
            if is_ctrl[gi]:
                continue
            gs = tuple(sorted(g for g in (ug[j] for j in np.where(cg[li])[0]) if g in g2i))
            if len(gs) == 1:
                singles[gs[0]].append(gi)
            elif 2 <= len(gs) <= kmax:
                groups[gs].append(gi)
        off += TF.shape[0]
    briggs_singles = {g: v for g, v in singles.items() if len(v) >= min_combo_cells}
    arrayed = _arrayed_combos()
    combo_perts, n_arr, n_p12 = {}, 0, 0
    for gs, v in groups.items():
        if len(v) < min_combo_cells:
            continue
        s = set(gs)
        if frozenset(s) in arrayed:                                    # cut #2: arrayed answer key
            n_arr += 1; continue
        if len(s & _P1) >= 2 or len(s & _P2) >= 2:                     # cut #3: >=2 P1/P2 overlap
            n_p12 += 1; continue
        combo_perts[gs] = v
    ksz = [len(k) for k in combo_perts]
    print(f'[jb-clean] UNENRICHED {X.shape[0]} cells | control {int(is_ctrl.sum())} | combo sets={len(combo_perts)} '
          f'(dropped {n_arr} arrayed + {n_p12} P1/P2-overlap) | set-k med={int(np.median(ksz)) if ksz else 0} '
          f'max={max(ksz) if ksz else 0} | combo cells={sum(len(v) for v in combo_perts.values())} '
          f'| in-vocab singles={len(briggs_singles)}', flush=True)
    return X, is_ctrl, combo_perts, shared_idx, briggs_singles


def load_joung_briggs(joung_proc_path=None, joung_combo_proc_path=None, min_cells_per_pert=5,
                      min_combo_cells=20, combo_kmax=3, train_frac=1.0, val_frac=0.0, split_seed=1,
                      effective_tfs=None, holdout_answerkey=True, briggs_clean=False,
                      briggs_control_only=False, holdout_tf_combos=None, **kw):
    from data.load_joung import load_joung
    _pp = {} if joung_proc_path is None else {'proc_path': joung_proc_path}
    J = load_joung(n_hvg=None, min_cells_per_pert=min_cells_per_pert, effective_tfs=effective_tfs,
                   load_combos=True, combo_proc_path=joung_combo_proc_path, min_combo_cells=min_combo_cells, **_pp)
    genes = [str(g) for g in J['gene_names']]
    Jx = J['adata'].X; Jx = (Jx.toarray() if sp.issparse(Jx) else np.asarray(Jx)).astype(np.float32)
    Jc = Jx[J['control_idx']].mean(0).astype(np.float32)
    _bl = _load_briggs_clean if briggs_clean else _load_briggs
    Xb, is_ctrl_b, briggs_combos, shared_idx, briggs_singles = _bl(genes, min_combo_cells, kmax=combo_kmax)
    if briggs_control_only:
        # Keep ONLY Briggs control (pluripotent) cells: they provide the modality-1 baseline for the
        # reconstruction loss (gene-masked to the ~2800 Briggs-observed genes) so the encoder learns to
        # embed the Briggs control state — with NO Briggs perturbed cells and NO Briggs Δ / perturbation
        # supervision at all. Used for the clean singles base that the combo-finetune later warm-starts from.
        _keep = np.where(is_ctrl_b)[0]
        Xb = Xb[_keep]; is_ctrl_b = np.ones(len(_keep), bool)
        briggs_combos = {}; briggs_singles = {}
        print(f"[jb] briggs_control_only=True: kept {len(_keep)} Briggs CONTROL cells; dropped ALL Briggs "
              f"perturbed cells (recon-only modality-1 baseline, no Briggs Δ/loss)", flush=True)
    Bc = Xb[is_ctrl_b].mean(0).astype(np.float32)
    nJ, nB = Jx.shape[0], Xb.shape[0]

    de_single = {g: (Jx[np.asarray(idx)].mean(0) - Jc).astype(np.float32) for g, idx in J['single_perts'].items()}
    de_single_briggs = {g: (Xb[np.asarray(idx)].mean(0) - Bc).astype(np.float32) for g, idx in briggs_singles.items()}
    combo_dataset = {}                                                      # tag each combo key by origin (for split-by-dataset eval)
    de_combo = {tuple(k): np.asarray(v, np.float32) for k, v in (J['adata'].uns.get('de_combo', {}) or {}).items()}
    for k in de_combo: combo_dataset[k] = 'joung'
    for gs, idx in briggs_combos.items():                                   # Briggs combo Δ vs Briggs pluripotent control
        de_combo[tuple(gs)] = (Xb[np.asarray(idx)].mean(0) - Bc).astype(np.float32); combo_dataset[tuple(gs)] = 'briggs'

    X = np.concatenate([Jx, Xb], axis=0); del Jx, Xb; import gc; gc.collect()
    modality = np.concatenate([np.full(nJ, MOD_JOUNG), np.full(nB, MOD_BRIGGS)]).astype(np.int64)
    is_ctrl = np.zeros(nJ + nB, bool)
    for i in J['control_idx']:
        is_ctrl[i] = True
    is_ctrl[nJ + np.where(is_ctrl_b)[0]] = True
    obs = pd.DataFrame({'modality_id': modality, 'batch_id': modality, 'is_control': is_ctrl,
                        'dataset': ['joung'] * nJ + ['briggs'] * nB})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.uns['de_single'] = de_single; adata.uns['de_combo'] = de_combo
    adata.uns['de_single_briggs'] = de_single_briggs; adata.uns['combo_dataset'] = combo_dataset
    # Per-modality gene mask [n_mod, G]: keep vocab=G but only score loss on genes each modality
    # actually observes. Joung (0) sees all G; Briggs (1) only the ~2800 shared genes (rest zero-filled).
    G = len(genes); mgm = np.ones((2, G), np.float32); mgm[MOD_BRIGGS] = 0.0; mgm[MOD_BRIGGS, shared_idx] = 1.0
    adata.uns['modality_gene_mask'] = mgm

    single_perts = {g: list(idx) for g, idx in J['single_perts'].items()}   # singles = Joung only
    combo_perts = {tuple(k): list(idx) for k, idx in (J.get('combo_perts') or {}).items()}
    _hto = {str(t) for t in (holdout_tf_combos or [])}                       # leakage-ladder: leave-TF-out set
    if _hto:                                                                # extend leave-TF-out to JOUNG singles + Joung combos (not just Briggs)
        _n0s, _n0c = len(single_perts), len(combo_perts)
        single_perts = {g: idx for g, idx in single_perts.items() if str(g) not in _hto}
        combo_perts = {k: idx for k, idx in combo_perts.items() if not (set(map(str, k)) & _hto)}
        print(f"[jb] leave-TF-out (JOUNG): dropped {_n0s-len(single_perts)} singles + {_n0c-len(combo_perts)} Joung combos touching {len(_hto)} held TFs (eval de_single kept)", flush=True)
    n_held = 0; n_tfheld = 0
    for gs, idx in briggs_combos.items():                                   # answer-key combos: eval-only
        if holdout_answerkey and _is_answerkey(gs):
            n_held += 1; continue                                            # kept in de_combo, dropped from training
        if _hto and (set(map(str, gs)) & _hto):                             # Rung2/3: any combo TOUCHING a held TF
            n_tfheld += 1; continue                                          # -> out of training, kept in de_combo for eval
        combo_perts[tuple(gs)] = [nJ + i for i in idx]
    if _hto: print(f"[jb] holdout_tf_combos: {len(_hto)} TFs -> {n_tfheld} combos held out of training (leave-TF-out)", flush=True)
    control_idx = list(np.where(is_ctrl)[0])
    all_pert = set(single_perts)
    combo_tfs = {g for k in combo_perts for g in k}                          # every combo-involved TF
    pert_split = _split_genes(all_pert, train_frac, val_frac, split_seed, protect=combo_tfs)
    print(f"[jb] cells J/B={nJ}/{nB} | singles(J)={len(single_perts)} | combos J/B="
          f"{len(J.get('combo_perts') or {})}/{len(briggs_combos)-n_held} (total {len(combo_perts)}) "
          f"| answer-key combos held out of training = {n_held}", flush=True)
    return {'adata': adata, 'control_idx': control_idx, 'single_perts': single_perts,
            'combo_perts': combo_perts, 'gene_names': genes, 'pert_genes': sorted(all_pert),
            'pert_split': pert_split, 'n_modalities': 2, 'n_batches': 2, 'de_single': de_single,
            'de_single_briggs': de_single_briggs, 'combo_dataset': combo_dataset,
            'modality_gene_mask': adata.uns['modality_gene_mask']}


if __name__ == '__main__':
    dd = load_joung_briggs()
    print("combos:", len(dd['combo_perts']), "| modalities:", dd['n_modalities'])

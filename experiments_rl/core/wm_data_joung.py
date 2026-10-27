"""Joung TF-atlas data backbone for the RL/planning experiments.

Parallel to `core.wm_data.WMData` (Norman), exposing the SAME public API so the env /
MPC / policies / validate are unchanged. Differences from Norman, all dataset-intrinsic:

  - The world model is trained on SINGLE TF overexpressions only; its checkpoint has no
    `combo_{train,test}_keys`. Combos/triples live in a SEPARATE h5ad
    (joung_combo_proc.h5ad), already aligned to the same 9923-gene axis as the singles.
  - Control = hESC GFP/mCherry cells from the SINGLES atlas; the combo Δ is measured
    against that same control baseline (consistent frame, matching combo_triple_eval).
  - Combo keys are sorted tuples of the real on-axis TFs (pairs (a,b) and triples
    (a,b,c); GFP/mCherry fillers dropped). Only ~40 pairs + 2 triples are usable
    (>=2 real TFs, >= min_combo_cells), so pairs are split train/test and the 2 triples
    are held out as a zero-shot generalisation probe (never trained on).

Public attributes (identical contract to WMData): de_single, de_combo (each value a
dict with lfc/effect/sig), z_target_single, z_target_combo, eps_ctrl, eps_ctrl_mean,
z0, A_ref, single_split{train,val,test}, combo_split{train,test,triples},
screened_genes/idx, full_idx, candidate_genes/idx. Methods: set_pool, sample_control_eps,
target_for.
"""
import os
import sys
import numpy as np
import torch


FILLERS = {'GFP', 'mCherry', 'mcherry', 'EGFP'}


def _parse_tfs(tf_set):
    """'TFORF0098-MSGN1,TFORF3549-GFP' -> ['MSGN1'] (real TFs, fillers dropped)."""
    out = []
    for tok in str(tf_set).split(','):
        sym = tok.split('-')[-1].strip()
        if sym and sym not in FILLERS:
            out.append(sym)
    return out


def _ensure_wm_root_on_path(meta):
    wm_root = os.path.abspath(os.path.join(os.path.dirname(meta['checkpoint']),
                                            "..", "..", "..", ".."))
    from core.wm_loader import DEFAULT_WM_ROOT
    for c in [wm_root, DEFAULT_WM_ROOT]:
        if os.path.isdir(os.path.join(c, "data")) and c not in sys.path:
            sys.path.insert(0, c)
            return c
    return wm_root


class WMDataJoung:
    """Task corpus over the Joung TF atlas, aligned to the frozen WM's gene axis."""

    def __init__(self, model, meta, op, singles_path=None, combo_path=None,
                 n_ctrl_cells=500, min_cells_per_pert=5, min_combo_cells=20,
                 seed=42, single_train_frac=0.8, single_val_frac=0.1,
                 combo_test_frac=0.3, deg_thresh=0.05, curated_topk=256):
        self._curated_topk = int(curated_topk)
        self.model = model
        self.meta = meta
        self.op = op
        self.device = meta['device']
        self.d = int(meta['latent_dim'])
        self.gene_to_idx = meta['gene_to_idx']
        self.gene_names = meta['gene_names']
        self.seed = seed
        self._n_ctrl_cells = n_ctrl_cells
        self._min_cells = min_cells_per_pert
        self._min_combo = min_combo_cells
        self._single_train_frac = single_train_frac
        self._single_val_frac = single_val_frac
        self._combo_test_frac = combo_test_frac
        self.thresh = float(deg_thresh)
        self._singles_path = singles_path
        self._combo_path = combo_path
        _ensure_wm_root_on_path(meta)
        self._build()

    # ------------------------------------------------------------------
    def _encode(self, X_rows):
        """Encode a [n, n_genes] expression block -> mean ε over its cells. [d]."""
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(X_rows), dtype=torch.float32, device=self.device)
            _, _, eps = self.model.encoder(xt)
            return eps.mean(0)

    def _de_entry(self, lfc):
        lfc = np.asarray(lfc, dtype=np.float32)
        return {'lfc': lfc, 'effect': np.abs(lfc), 'sig': np.abs(lfc) > self.thresh}

    def _encode_block(self, Xsparse, rows):
        """Densify only `rows` of a sparse matrix and encode → mean ε. [d]."""
        block = np.asarray(Xsparse[np.asarray(rows)].todense(), dtype=np.float32)
        return self._encode(block)

    def _build(self):
        import scipy.sparse as sp
        import anndata as ad
        # Make the WM experiment root importable regardless of cwd / how the model was
        # loaded, then import its data helpers (robust to path state across srun/sbatch).
        wm_root = os.path.abspath(os.path.join(
            os.path.dirname(self.meta['checkpoint']), "..", "..", "..", ".."))
        if os.path.isdir(os.path.join(wm_root, "data")) and wm_root not in sys.path:
            sys.path.insert(0, wm_root)
        try:
            from data.load_joung import _lognorm, DEFAULT_PROC
            from data.load_norman import split_single_perts
        except ModuleNotFoundError:
            import importlib.util

            def _load(modname, relpath):
                spec = importlib.util.spec_from_file_location(
                    modname, os.path.join(wm_root, relpath))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[modname] = mod
                spec.loader.exec_module(mod)
                return mod
            _ln = _load('data.load_norman', 'data/load_norman.py')
            split_single_perts = _ln.split_single_perts
            _lj = _load('data.load_joung', 'data/load_joung.py')
            _lognorm, DEFAULT_PROC = _lj._lognorm, _lj.DEFAULT_PROC

        # ---- singles atlas: load SPARSE, densify only slices (avoids the 45GB full
        # densify in load_joung; peak mem ~ sparse X + small dense slices) ----
        singles_path = self._singles_path or DEFAULT_PROC
        S = ad.read_h5ad(singles_path)
        assert list(map(str, S.var_names)) == list(self.gene_names), \
            "Joung singles gene axis != frozen WM gene axis"
        Xs = _lognorm(S.X)                                          # CSR, stays sparse
        is_ctrl = S.obs['is_control'].values.astype(bool)
        ctrl_idx = np.where(is_ctrl)[0]
        tfg = S.obs['TF_gene'].astype(str).values
        gset = set(self.gene_names)
        single_perts = {}
        for i in np.where(~is_ctrl)[0]:
            g = tfg[i]
            if g in gset:
                single_perts.setdefault(g, []).append(int(i))
        single_perts = {g: idx for g, idx in single_perts.items() if len(idx) >= self._min_cells}

        ctrl_mean = np.asarray(Xs[ctrl_idx].mean(0)).ravel().astype(np.float32)   # sparse mean
        self.de_single = {g: self._de_entry(np.asarray(Xs[np.asarray(idx)].mean(0)).ravel() - ctrl_mean)
                          for g, idx in single_perts.items()}

        # ---- control ε pool + control latent z0 (densify only sampled control cells) ----
        rng = np.random.default_rng(self.seed)
        sel = rng.choice(len(ctrl_idx), min(self._n_ctrl_cells, len(ctrl_idx)), replace=False)
        with torch.no_grad():
            xt = torch.tensor(np.asarray(Xs[ctrl_idx[sel]].todense(), dtype=np.float32),
                              device=self.device)
            _, _, eps_ctrl = self.model.encoder(xt)
        self.eps_ctrl = eps_ctrl.detach()
        self._ctrl_X = xt.detach()   # retained so the RL can re-encode control in a chosen modality (Briggs frame)
        self.A_ref = self.op.mean_A()
        self.z0 = self.op.scm_forward(self.eps_ctrl, self.A_ref).mean(0)
        self.eps_ctrl_mean = self.eps_ctrl.mean(0)

        # ---- data-grounded single-TF latent targets (densify each TF's cells only) ----
        self.z_target_single = {}
        for g, idxs in single_perts.items():
            eps_p = self._encode_block(Xs, idxs)
            self.z_target_single[g] = self.op.scm_forward(eps_p.unsqueeze(0), self.A_ref).squeeze(0)

        pert_split = split_single_perts(single_perts, train_frac=self._single_train_frac,
                                        val_frac=self._single_val_frac, seed=self.seed)

        # ---- combos / triples (separate h5ad, same gene axis; densify per-set) ----
        combo_path = self._combo_path or os.path.join(
            '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas',
            'data', 'joung_combo_proc.h5ad')
        C = ad.read_h5ad(combo_path)
        assert list(map(str, C.var_names)) == list(self.gene_names), \
            "Joung combo gene axis != frozen WM gene axis"
        Cx = _lognorm(C.X)                                          # sparse
        tf_set = C.obs['TF_set'].astype(str).values

        self.de_combo = {}
        self.z_target_combo = {}
        pairs, triples = [], []
        for setname in sorted(set(tf_set)):
            tfs = [t for t in _parse_tfs(setname) if t in self.gene_to_idx]
            tfs = tuple(sorted(set(tfs)))
            if len(tfs) < 2:
                continue
            m = tf_set == setname
            if int(m.sum()) < self._min_combo:
                continue
            rows = np.where(m)[0]
            block = np.asarray(Cx[rows].todense(), dtype=np.float32)
            lfc = block.mean(0) - ctrl_mean
            self.de_combo[tfs] = self._de_entry(lfc)
            eps_p = self._encode(block)
            self.z_target_combo[tfs] = self.op.scm_forward(eps_p.unsqueeze(0), self.A_ref).squeeze(0)
            (pairs if len(tfs) == 2 else triples).append(tfs)

        # ---- splits ----
        self.single_split = {k: [g for g in pert_split[k] if g in self.z_target_single]
                             for k in ('train', 'val', 'test')}
        pairs = sorted(pairs)
        # LEAK-FREE: if the frozen WM was combo-trained, its checkpoint stashed combo_test_keys
        # (the combos it did NOT see). Use exactly those as the RL held-out set so we never
        # evaluate on combos the WM trained on. Otherwise, a deterministic seed split.
        ckpt_test = [tuple(sorted(k)) for k in (self.meta.get('combo_test_keys') or [])]
        test_fs = {frozenset(k) for k in ckpt_test}
        if test_fs:
            test_set = {p for p in pairs if frozenset(p) in test_fs}
            print(f"[wm_data_joung] using WM checkpoint combo_test_keys for held-out "
                  f"({len(test_set)} of {len(pairs)} pairs) — leak-free vs combo-trained WM")
        else:
            prng = np.random.default_rng(self.seed)
            perm = prng.permutation(len(pairs))
            n_test = max(1, int(round(len(pairs) * self._combo_test_frac)))
            test_set = {pairs[i] for i in perm[:n_test]}
        self.combo_split = {
            'train': [p for p in pairs if p not in test_set],
            'test': [p for p in pairs if p in test_set],
            'triples': sorted(triples),
        }

        # ---- action pools ----
        # 'screened' = the single TFs the WM was trained on (the meaningful OE targets).
        # 'full'     = ALL axis genes (zero-shot via pert_label / ESM descriptor).
        # 'curated'  = top-K screened TFs by effect magnitude ∪ all combo-target genes — a
        #              SMALL, high-signal action space (the reward-landscape probe showed the
        #              per-action signal exists but is drowned in the 1689-way pool). The
        #              union guarantees every held-out combo's true genes stay reachable.
        self.screened_genes = sorted(self.z_target_single.keys())
        self.screened_idx = [self.gene_to_idx[g] for g in self.screened_genes]
        self.full_idx = list(range(len(self.gene_names)))
        eff = {g: float(np.linalg.norm(self.de_single[g]['lfc'])) for g in self.screened_genes}
        topk = sorted(self.screened_genes, key=lambda g: -eff[g])[:self._curated_topk]
        combo_genes = {g for key in (list(self.z_target_combo) ) for g in key}
        self.curated_genes = sorted(set(topk) | (combo_genes & set(self.screened_genes)))
        self.curated_idx = [self.gene_to_idx[g] for g in self.curated_genes]
        self.candidate_genes = self.screened_genes
        self.candidate_idx = self.screened_idx
        print(f"[wm_data_joung] singles train/val/test="
              f"{len(self.single_split['train'])}/{len(self.single_split['val'])}/"
              f"{len(self.single_split['test'])} | combo pairs train/test="
              f"{len(self.combo_split['train'])}/{len(self.combo_split['test'])} | "
              f"triples(zero-shot)={len(self.combo_split['triples'])} | "
              f"screened={len(self.screened_idx)} curated={len(self.curated_idx)} | "
              f"Nctrl={self.eps_ctrl.shape[0]}")

    # ------------------------------------------------------------------
    def set_pool(self, name):
        if name == 'full':
            self.candidate_idx = self.full_idx
            self.candidate_genes = list(self.gene_names)
        elif name == 'screened':
            self.candidate_idx = self.screened_idx
            self.candidate_genes = self.screened_genes
        elif name == 'curated':
            self.candidate_idx = self.curated_idx
            self.candidate_genes = self.curated_genes
        else:
            raise ValueError(f"unknown pool {name!r}")
        print(f"[wm_data_joung] action pool = '{name}'  (|candidates|={len(self.candidate_idx)})")
        return self

    def sample_control_eps(self, n=1, rng=None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.eps_ctrl.shape[0], size=n)
        return self.eps_ctrl[torch.as_tensor(idx, device=self.device)]

    def target_for(self, key):
        from core.scoring import Lineage
        if isinstance(key, Lineage):
            return self.z_target_lineage[key.name]
        if isinstance(key, tuple):
            return self.z_target_combo[key]
        return self.z_target_single[key]

    # ------------------------------------------------------------------
    def build_lineage_targets(self, tf_lineage_csv, qmax=0.10, min_drivers=4,
                              test_frac=0.3, seed=42):
        """Fate-target task: each developmental LINEAGE (Leiden cluster over Cao cell types,
        joung_tf_lineage.csv) becomes a goal whose target is the CENTROID of its confident
        driver TFs — observed-Δ centroid (de_lineage, what the reward scores) and latent
        centroid (z_target_lineage, the goal-conditioning state). Drivers (q<qmax, present in
        the singles atlas) are split train/test: the centroid is built from TRAIN drivers; TEST
        drivers are held out to ask 'does the planner pick OTHER TFs that reach this fate?'.

        Populates de_lineage, z_target_lineage, lineage_drivers{name:{train,test}},
        lineage_keys (Lineage objects, only lineages with >= min_drivers train drivers).
        """
        import pandas as pd
        from core.scoring import Lineage
        df = pd.read_csv(tf_lineage_csv); df['TF'] = df['TF'].astype(str)
        df = df[(df['fdr_qval'] < qmax) & (df['TF'].isin(self.z_target_single))]
        rng = np.random.default_rng(seed)
        self.de_lineage, self.z_target_lineage, self.lineage_drivers, self.lineage_keys = {}, {}, {}, []
        for name, sub in df.groupby('lineage_name'):
            tfs = sorted(sub['TF'].tolist())
            if len(tfs) < min_drivers:
                continue
            perm = rng.permutation(len(tfs))
            n_test = max(1, int(round(len(tfs) * test_frac)))
            test = sorted(tfs[i] for i in perm[:n_test])
            train = sorted(tfs[i] for i in perm[n_test:])
            if len(train) < max(2, min_drivers - n_test):
                train, test = tfs, []                         # too few → use all for centroid
            lfc = np.mean([self.de_single[g]['lfc'] for g in train], axis=0)
            self.de_lineage[name] = self._de_entry(lfc)
            self.z_target_lineage[name] = torch.stack([self.z_target_single[g] for g in train]).mean(0)
            self.lineage_drivers[name] = {'train': train, 'test': test}
            self.lineage_keys.append(Lineage(name))
        print(f"[wm_data_joung] lineage fate targets: {len(self.lineage_keys)} lineages "
              f"(>= {min_drivers} confident drivers) | "
              + " | ".join(f"{k.name.split(':')[0]}:{len(self.lineage_drivers[k.name]['train'])}+"
                           f"{len(self.lineage_drivers[k.name]['test'])}" for k in self.lineage_keys))
        return self.lineage_keys

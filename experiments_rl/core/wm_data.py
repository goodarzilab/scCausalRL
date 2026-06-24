"""Norman data backbone for the RL/planning experiments.

Builds, once, the task corpus the env / MPC / policies share:
  - a pool of control-cell exogenous noises ε_ctrl (start states),
  - per-perturbation DATA-GROUNDED latent targets z_target = scm_forward(mean ε of the
    REAL observed perturbed cells) — falsifiable: the planner must reach where the
    actual perturbation lands, not the operator's own prediction,
  - observed gene-space DE (de_single / de_combo) for the gene-space validation,
  - deterministic train/val/test splits (singles reconstructed with the training
    seed/fracs; combos taken from the checkpoint's stored combo_{train,test}_keys).

All data loading reuses the world-model experiment's own loaders
(load_norman_aligned, compute_observed_de, split_single_perts) so the axis,
control set, and DE match the validated eval exactly.
"""
import os
import sys
import numpy as np
import torch


def _ensure_wm_root_on_path(meta):
    wm_root = os.path.dirname(os.path.dirname(meta['checkpoint']))
    # checkpoint is .../<run>/checkpoints/x.pt → experiment root is 3 up from the file
    wm_root = os.path.abspath(os.path.join(os.path.dirname(meta['checkpoint']), "..", "..", "..", ".."))
    # Fall back to the known exp14 root via wm_loader's default if structure differs.
    from core.wm_loader import DEFAULT_WM_ROOT
    candidates = [wm_root, DEFAULT_WM_ROOT]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "data")) and c not in sys.path:
            sys.path.insert(0, c)
            return c
    return DEFAULT_WM_ROOT


class WMData:
    """Lazily-built task corpus over Norman, aligned to the frozen WM's gene axis."""

    def __init__(self, model, meta, op, norman_path=None,
                 n_ctrl_cells=500, min_cells_per_pert=5, seed=42,
                 single_train_frac=0.8, single_val_frac=0.1):
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
        self._single_train_frac = single_train_frac
        self._single_val_frac = single_val_frac
        self.norman_path = norman_path or meta.get('norman_path')

        _ensure_wm_root_on_path(meta)
        self._build()

    # ------------------------------------------------------------------
    def _encode(self, X_rows):
        """Encode a [n, n_genes] expression block → mean ε over its cells. Returns [d]."""
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(X_rows), dtype=torch.float32, device=self.device)
            _, _, eps = self.model.encoder(xt)
            return eps.mean(0)

    def _build(self):
        import scipy.sparse
        from data.load_norman_aligned import load_norman_aligned
        from data.load_norman import split_single_perts
        from evaluation.run_perturbation_eval import compute_observed_de

        dd = load_norman_aligned(self.norman_path, target_gene_names=self.gene_names,
                                 min_cells_per_pert=self._min_cells)
        X = dd['adata'].X
        X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
        self._X = X
        de_single, de_combo, _ctrl_mean = compute_observed_de(X, dd, self.gene_names)
        self.de_single = de_single
        # keep only combos with both genes on-axis AND observed singles (matches eval)
        self.de_combo = {k: v for k, v in de_combo.items()
                         if k[0] in self.gene_to_idx and k[1] in self.gene_to_idx
                         and k[0] in de_single and k[1] in de_single}

        # ---- control ε pool + control latent z0 ----
        rng = np.random.default_rng(self.seed)
        ctrl_idx = np.array(dd['control_idx'])
        sel = rng.choice(len(ctrl_idx), min(self._n_ctrl_cells, len(ctrl_idx)), replace=False)
        with torch.no_grad():
            xt = torch.tensor(X[ctrl_idx[sel]], dtype=torch.float32, device=self.device)
            _, _, eps_ctrl = self.model.encoder(xt)               # [Nc, d]
        self.eps_ctrl = eps_ctrl.detach()
        self.A_ref = self.op.mean_A()                              # fixed frame for targets
        self.z0 = self.op.scm_forward(self.eps_ctrl, self.A_ref).mean(0)   # [d] control centroid
        self.eps_ctrl_mean = self.eps_ctrl.mean(0)                # [d]

        # ---- data-grounded latent targets (encode REAL perturbed cells) ----
        self.z_target_single = {}
        for g, idxs in dd['single_perts'].items():
            if g not in self.de_single:
                continue
            eps_p = self._encode(X[np.asarray(idxs)])             # [d]
            self.z_target_single[g] = self.op.scm_forward(eps_p.unsqueeze(0), self.A_ref).squeeze(0)
        self.z_target_combo = {}
        for (ga, gb), idxs in dd['combo_perts'].items():
            if (ga, gb) not in self.de_combo:
                continue
            eps_p = self._encode(X[np.asarray(idxs)])
            self.z_target_combo[(ga, gb)] = self.op.scm_forward(eps_p.unsqueeze(0), self.A_ref).squeeze(0)

        # ---- splits ----
        single_split = split_single_perts(
            {g: dd['single_perts'][g] for g in self.de_single},
            train_frac=self._single_train_frac, val_frac=self._single_val_frac, seed=self.seed)
        self.single_split = {k: [g for g in v if g in self.z_target_single]
                             for k, v in single_split.items()}
        test_fs = {frozenset(k) for k in self.meta['combo_test_keys']}
        train_fs = {frozenset(k) for k in self.meta['combo_train_keys']}
        self.combo_split = {'train': [], 'test': []}
        for key in self.z_target_combo:
            fs = frozenset(key)
            if fs in test_fs:
                self.combo_split['test'].append(key)
            elif fs in train_fs:
                self.combo_split['train'].append(key)

        # ---- action pools ----
        # 'screened' = the 87 grounded Norman single-pert genes (default, small).
        # 'full'     = ALL genes on the axis (zero-shot via pert_label) — the large
        #              ~8000-gene pool where ~99% of candidates were never screened.
        self.screened_genes = sorted(self.z_target_single.keys())
        self.screened_idx = [self.gene_to_idx[g] for g in self.screened_genes]
        self.full_idx = list(range(len(self.gene_names)))
        self.candidate_genes = self.screened_genes          # active pool (default screened)
        self.candidate_idx = self.screened_idx
        print(f"[wm_data] singles train/val/test="
              f"{len(self.single_split.get('train',[]))}/{len(self.single_split.get('val',[]))}/"
              f"{len(self.single_split.get('test',[]))} | combos train/test="
              f"{len(self.combo_split['train'])}/{len(self.combo_split['test'])} | "
              f"candidate genes={len(self.candidate_genes)} | Nctrl={self.eps_ctrl.shape[0]}")

    # ------------------------------------------------------------------
    def set_pool(self, name):
        """Switch the active action pool: 'screened' (87) or 'full' (all axis genes)."""
        if name == 'full':
            self.candidate_idx = self.full_idx
            self.candidate_genes = list(self.gene_names)
        elif name == 'screened':
            self.candidate_idx = self.screened_idx
            self.candidate_genes = self.screened_genes
        else:
            raise ValueError(f"unknown pool {name!r}")
        print(f"[wm_data] action pool = '{name}'  (|candidates|={len(self.candidate_idx)})")
        return self

    def sample_control_eps(self, n=1, rng=None):
        """Sample n control-cell ε vectors (start states). Returns [n, d]."""
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.eps_ctrl.shape[0], size=n)
        return self.eps_ctrl[torch.as_tensor(idx, device=self.device)]

    def target_for(self, key):
        """Latent z_target for a single gene (str) or combo (tuple)."""
        if isinstance(key, tuple):
            return self.z_target_combo[key]
        return self.z_target_single[key]

"""
Paired cell-sentence dataset for Phase-1 perturbation-consistency training.

Each item returns a "sentence" — a set of S perturbed cells paired with S control
cells from the **same gem_group** (plate / library) — so the model trains on a
plate-matched ctrl ↔ pert distribution shift, not on cross-plate noise.

Returned per item:
    x_ctrl     : Tensor[S, n_genes]
    x_pert     : Tensor[S, n_genes]
    gene_idx   : LongTensor[]        index of the perturbed gene in gene_names
    batch_id   : LongTensor[]        gem_group index (same for ctrl and pert in the sentence)
    modality_id: LongTensor[]        0 for single-modality (Replogle-only) runs

Construction:
    1. Build a `(gene, gem_group) -> [cell indices]` map for every train-split
       perturbation gene.
    2. Build a `gem_group -> [control cell indices]` map.
    3. Drop (gene, gem_group) blocks with fewer than `min_block_cells` cells.
    4. Each `__getitem__` draws S perturbed cells from one block (with replacement
       if the block has < S cells) and S control cells from that same gem_group.

This is the scCausalRL analogue of STATE's cell-sentence loader
(`state-mcp-local/src/state/tx/data/`): same `[B, S, n_genes]` batch shape and the
same "control/pert paired within the sentence" contract.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset


class PairedCellSentenceDataset(Dataset):
    """One item = (ctrl_set, pert_set, gene_idx, batch_id) from one gem_group."""

    def __init__(
        self,
        adata,
        single_perts: dict,           # gene_name -> list of cell indices
        control_idx: list,            # indices of control cells
        gene_names: list,
        S: int = 64,                  # cell-sentence length
        n_sentences_per_block: int = 4,
        min_block_cells: int = 8,
        gene_filter: set | None = None,   # restrict to train-split genes
        seed: int = 42,
        modality_id_lookup: np.ndarray | None = None,
    ):
        self.S = int(S)
        self.n_per_block = int(n_sentences_per_block)
        self.min_block_cells = int(min_block_cells)

        # Materialise the expression matrix once (dense, float32).  For 186 k × 2000
        # this is ~1.5 GB — acceptable for an in-memory training run; revisit only if
        # we move to the full 33 k gene matrix.
        X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
        self._X = np.asarray(X, dtype=np.float32)

        # Per-cell batch_id (gem_group) — already an int64 column placed by load_replogle.py.
        if 'batch_id' in adata.obs.columns:
            self._batch_ids = adata.obs['batch_id'].values.astype(np.int64)
        else:
            self._batch_ids = np.zeros(adata.n_obs, dtype=np.int64)

        if modality_id_lookup is not None:
            self._modality_ids = np.asarray(modality_id_lookup, dtype=np.int64)
        else:
            self._modality_ids = np.zeros(adata.n_obs, dtype=np.int64)

        # Gene index lookup
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # 1) build (gene, gem_group) -> indices for perturbed cells
        # 2) build gem_group -> indices for control cells
        ctrl_idx_arr = np.asarray(control_idx, dtype=np.int64)
        self._ctrl_by_gg: dict[int, np.ndarray] = {}
        for gg in np.unique(self._batch_ids[ctrl_idx_arr]):
            mask = self._batch_ids[ctrl_idx_arr] == gg
            self._ctrl_by_gg[int(gg)] = ctrl_idx_arr[mask]

        # Build pert blocks
        self._blocks: list[tuple[int, int, np.ndarray]] = []  # (gene_idx, gg, indices)
        for gene, pert_indices in single_perts.items():
            base = gene.split('__')[0] if '__' in gene else gene
            if base not in gene_to_idx:
                continue
            if gene_filter is not None and base not in gene_filter:
                continue
            g_idx = gene_to_idx[base]
            pert_arr = np.asarray(pert_indices, dtype=np.int64)
            pert_ggs = self._batch_ids[pert_arr]
            for gg in np.unique(pert_ggs):
                gg = int(gg)
                if gg not in self._ctrl_by_gg:
                    # No controls on this plate — skip; the model has nothing to
                    # pair against without crossing plate boundaries.
                    continue
                block = pert_arr[pert_ggs == gg]
                if len(block) < self.min_block_cells:
                    continue
                self._blocks.append((g_idx, gg, block))

        if not self._blocks:
            raise RuntimeError(
                "PairedCellSentenceDataset: no (gene, gem_group) blocks survived "
                f"min_block_cells={self.min_block_cells}. "
                "Lower the floor or check the data loader."
            )

        # Build (block_idx, sentence_idx) flat index for __getitem__
        self._flat = [
            (b_idx, s_idx)
            for b_idx in range(len(self._blocks))
            for s_idx in range(self.n_per_block)
        ]
        # Per-block RNG seed for reproducibility across workers
        self._rng = np.random.default_rng(seed)

        # Diagnostics
        block_sizes = np.array([len(b[2]) for b in self._blocks])
        print(
            f"[PairedCellSentenceDataset] {len(self._blocks)} (gene, gem_group) blocks, "
            f"sizes min/median/max = {block_sizes.min()}/"
            f"{int(np.median(block_sizes))}/{block_sizes.max()}; "
            f"S={self.S}, n_per_block={self.n_per_block}, "
            f"total sentences/epoch = {len(self._flat)}"
        )

    def __len__(self) -> int:
        return len(self._flat)

    def _sample_set(self, pool: np.ndarray, k: int) -> np.ndarray:
        """Draw `k` indices from `pool`. With replacement only if the pool is too small."""
        if len(pool) >= k:
            return self._rng.choice(pool, k, replace=False)
        return self._rng.choice(pool, k, replace=True)

    def __getitem__(self, idx: int):
        b_idx, _ = self._flat[idx]
        gene_idx, gg, pert_pool = self._blocks[b_idx]
        ctrl_pool = self._ctrl_by_gg[gg]

        pert_sel = self._sample_set(pert_pool, self.S)
        ctrl_sel = self._sample_set(ctrl_pool, self.S)

        x_ctrl = torch.from_numpy(self._X[ctrl_sel])         # [S, n_genes]
        x_pert = torch.from_numpy(self._X[pert_sel])         # [S, n_genes]

        return {
            'x_ctrl':       x_ctrl,
            'x_pert':       x_pert,
            'gene_idx':     torch.tensor(gene_idx, dtype=torch.long),
            'batch_id':     torch.tensor(gg, dtype=torch.long),
            'modality_id':  torch.tensor(int(self._modality_ids[ctrl_sel[0]]),
                                          dtype=torch.long),
        }

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

This is the CausalBioRL analogue of STATE's cell-sentence loader
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
        combo_perts: dict | None = None,      # (gene_a, gene_b) -> list of cell indices
        combo_filter: set | None = None,      # restrict to train-split combo keys
        single_sentence_mult: int = 1,        # REBALANCE: single blocks get n_per_block*mult sentences/epoch
                                              # so many combo blocks (e.g. 743 Briggs) don't drown the singles
    ):
        self.single_sentence_mult = max(1, int(single_sentence_mult))
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

        # Build pert blocks. Each block stores a gene-index ARRAY (length 1 for a
        # single perturbation, length k for a k-way combo) so single + combo sentences
        # share one code path. __getitem__ pads to a fixed [self._pad_k] (−1 = no gene).
        # Pad width = max combo arity across the data (>=2), so pairs and triples coexist.
        self._pad_k = max([2] + [len(k) for k in (combo_perts or {})])
        self._blocks: list[tuple[np.ndarray, int, np.ndarray]] = []  # (gene_idx_arr, gg, indices)
        n_single, n_combo = 0, 0
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
                    continue
                block = pert_arr[pert_ggs == gg]
                if len(block) < self.min_block_cells:
                    continue
                self._blocks.append((np.array([g_idx], dtype=np.int64), gg, block)); n_single += 1

        # Combo blocks: (gene_a, gene_b[, gene_c, ...]) -> cells; every gene must be on axis,
        # and (if a combo_filter is given) the combo must be in the train split. Variable arity
        # (k>=2) is supported: pairs (Joung/Norman) and triples (Briggs) coexist; gene_idx is
        # padded to self._pad_k with -1 downstream, and the loss masks the pad slots.
        if combo_perts:
            for key, idxs in combo_perts.items():
                bases = [g.split('__')[0] if '__' in g else g for g in key]
                if any(b not in gene_to_idx for b in bases):
                    continue
                if combo_filter is not None and key not in combo_filter:
                    continue
                gi_arr = np.array([gene_to_idx[b] for b in bases], dtype=np.int64)
                pert_arr = np.asarray(idxs, dtype=np.int64)
                pert_ggs = self._batch_ids[pert_arr]
                for gg in np.unique(pert_ggs):
                    gg = int(gg)
                    if gg not in self._ctrl_by_gg:
                        continue
                    block = pert_arr[pert_ggs == gg]
                    if len(block) < self.min_block_cells:
                        continue
                    self._blocks.append((gi_arr, gg, block)); n_combo += 1
        self._n_single_blocks, self._n_combo_blocks = n_single, n_combo

        if not self._blocks:
            raise RuntimeError(
                "PairedCellSentenceDataset: no (gene, gem_group) blocks survived "
                f"min_block_cells={self.min_block_cells}. "
                "Lower the floor or check the data loader."
            )

        # Build (block_idx, sentence_idx) flat index for __getitem__. SINGLE blocks (gene_idx_arr length 1)
        # get n_per_block * single_sentence_mult sentences → upweight their gradient so combo blocks don't
        # dominate the shared shift-encoder/decoder (fixes Joung-single underfit in the joint model).
        self._flat = []
        for b_idx, (gi_arr, _gg, _blk) in enumerate(self._blocks):
            reps = self.n_per_block * (self.single_sentence_mult if len(gi_arr) == 1 else 1)
            self._flat.extend((b_idx, s % self.n_per_block) for s in range(reps))
        # Per-block RNG seed for reproducibility across workers
        self._rng = np.random.default_rng(seed)

        # Diagnostics
        block_sizes = np.array([len(b[2]) for b in self._blocks])
        print(
            f"[PairedCellSentenceDataset] {len(self._blocks)} blocks "
            f"({self._n_single_blocks} single + {self._n_combo_blocks} combo), "
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
        gene_idx_arr, gg, pert_pool = self._blocks[b_idx]
        ctrl_pool = self._ctrl_by_gg[gg]

        pert_sel = self._sample_set(pert_pool, self.S)
        ctrl_sel = self._sample_set(ctrl_pool, self.S)

        x_ctrl = torch.from_numpy(self._X[ctrl_sel])         # [S, n_genes]
        x_pert = torch.from_numpy(self._X[pert_sel])         # [S, n_genes]

        # gene_idx padded to a fixed [self._pad_k] with -1: single = [g, -1, ...],
        # pair = [ga, gb, -1...], triple = [ga, gb, gc]. Loss masks the -1 slots.
        gi = np.full(self._pad_k, -1, dtype=np.int64)
        gi[:len(gene_idx_arr)] = gene_idx_arr

        return {
            'x_ctrl':       x_ctrl,
            'x_pert':       x_pert,
            'gene_idx':     torch.from_numpy(gi),            # [2]
            'batch_id':     torch.tensor(gg, dtype=torch.long),
            'modality_id':  torch.tensor(int(self._modality_ids[ctrl_sel[0]]),
                                          dtype=torch.long),
        }


class DoseCellSentenceDataset(Dataset):
    """DOSE-aware, PER-CELL variable-k cell-sentence dataset (REPROcode).

    Unlike PairedCellSentenceDataset (which groups cells into (gene-set, gem_group) blocks that share an
    exact perturbation), here EACH perturbed cell is its own block carrying its own TF set + per-TF dose
    (barcode copy count). At high k (up to ~40 TFs/cell) exact in-vocab sets are almost all unique, so
    exact-set grouping would collapse under any replicate floor — hence per-cell examples.

    One item = (ctrl_set[S], pert_set[S], gene_idx[pad_k], dose[pad_k]). The pert set is the single
    perturbed cell tiled to S (its own point estimate); the ctrl set is S cells drawn from the single
    pooled control (all panels borrow it). __getitem__ returns `dose` (copies parallel to gene_idx,
    0 on pad slots) alongside gene_idx so the loss can thread it into model.compute_shift(..., dose=...).

    Item contract is a strict SUPERSET of PairedCellSentenceDataset (adds only the 'dose' key), so the
    training loop reads it via sent.get('dose') and every non-REPROcode caller is byte-for-byte unchanged.
    """

    def __init__(
        self,
        adata,
        dose_perts,                   # list of (cell_idx:int, gene_idx:np.ndarray[k], copies:np.ndarray[k])
        control_idx,                  # pooled control cell indices (all panels)
        gene_names,
        S: int = 32,
        n_sentences_per_block: int = 1,
        seed: int = 42,
        gene_filter: set | None = None,   # set of TRAIN-split gene NAMES; a cell is kept only if EVERY
                                          # in-vocab TF is in the filter (strict leave-TF-out, leakage-safe)
        max_blocks: int | None = None,    # optionally subsample the per-cell examples (smoke / debug)
        modality_id: int = 0,
    ):
        self.S = int(S)
        self.n_per_block = int(n_sentences_per_block)
        self.modality_id = int(modality_id)

        X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
        self._X = np.asarray(X, dtype=np.float32)
        self._ctrl = np.asarray(control_idx, dtype=np.int64)
        if self._ctrl.size == 0:
            raise RuntimeError("DoseCellSentenceDataset: empty control pool.")

        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        filt_idx = None
        if gene_filter is not None:
            filt_idx = {gene_to_idx[g] for g in gene_filter if g in gene_to_idx}

        rng = np.random.default_rng(seed)
        blocks = []
        n_dropped = 0
        for ci, gidx, dose in dose_perts:
            gidx = np.asarray(gidx, dtype=np.int64)
            dose = np.asarray(dose, dtype=np.float32)
            if gidx.size == 0:
                continue
            if filt_idx is not None and any(int(g) not in filt_idx for g in gidx):
                n_dropped += 1                                    # touches a held-out TF -> leakage, drop
                continue
            blocks.append((int(ci), gidx, dose))

        if max_blocks is not None and len(blocks) > int(max_blocks):
            sel = sorted(rng.choice(len(blocks), int(max_blocks), replace=False).tolist())
            blocks = [blocks[i] for i in sel]

        if not blocks:
            raise RuntimeError(
                "DoseCellSentenceDataset: no per-cell examples survived the gene_filter. "
                "Check the TF split (train_frac) / dose_perts.")

        self._blocks = blocks
        self._pad_k = max([1] + [len(b[1]) for b in blocks])
        self._flat = [(bi, s) for bi in range(len(blocks)) for s in range(self.n_per_block)]
        self._rng = rng

        ksz = np.array([len(b[1]) for b in blocks])
        print(
            f"[DoseCellSentenceDataset] {len(blocks)} per-cell examples "
            f"({n_dropped} dropped by gene_filter); k min/median/max = "
            f"{ksz.min()}/{int(np.median(ksz))}/{ksz.max()}; pad_k={self._pad_k}; "
            f"ctrl pool={self._ctrl.size}; S={self.S}, n_per_block={self.n_per_block}, "
            f"total sentences/epoch = {len(self._flat)}"
        )

    def __len__(self) -> int:
        return len(self._flat)

    def _sample_set(self, pool: np.ndarray, k: int) -> np.ndarray:
        if len(pool) >= k:
            return self._rng.choice(pool, k, replace=False)
        return self._rng.choice(pool, k, replace=True)

    def __getitem__(self, idx: int):
        b_idx, _ = self._flat[idx]
        cell_idx, gene_idx_arr, dose_arr = self._blocks[b_idx]

        ctrl_sel = self._sample_set(self._ctrl, self.S)
        pert_sel = np.full(self.S, cell_idx, dtype=np.int64)      # the single cell = its own point estimate

        x_ctrl = torch.from_numpy(self._X[ctrl_sel])              # [S, n_genes]
        x_pert = torch.from_numpy(self._X[pert_sel])              # [S, n_genes]

        gi = np.full(self._pad_k, -1, dtype=np.int64)
        gi[:len(gene_idx_arr)] = gene_idx_arr
        dk = np.zeros(self._pad_k, dtype=np.float32)              # 0 on pad slots (masked by gmask)
        dk[:len(dose_arr)] = dose_arr

        return {
            'x_ctrl':       x_ctrl,
            'x_pert':       x_pert,
            'gene_idx':     torch.from_numpy(gi),                 # [pad_k]
            'dose':         torch.from_numpy(dk),                 # [pad_k] copies parallel to gene_idx
            'batch_id':     torch.tensor(0, dtype=torch.long),
            'modality_id':  torch.tensor(self.modality_id, dtype=torch.long),
        }

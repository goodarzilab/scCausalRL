# scCausalRL — Project state handoff

Last update: 2026-06-03 (Claude session in `/home/sgoudarzi/CausalBioRL`, migrated to this repo).

This is a single-document handoff covering everything material that's happened — diagnostics, code changes, sweeps, retrains, evals, open items. Read sections in order; the architecture / training / eval changes build on each other.

---

## 0. TL;DR

- The Phase-1 Causal VAE has **input leakage at both train and eval**: the perturbation pipeline takes the observed differential-expression vector as `dx_input` and projects it through `D·D⁺` (rank-64 with orthogonal D), producing a low-rank reconstruction of the answer. The additive baseline returns `ctrl + dx_observed = pert_observed` literally, AUROC ≈ 1.0 on singles. Headline "+0.04 lift over correlation" was the projection identity.
- **Fix shipped in this branch**: `models/dx_input.py` builds sparse one-hot `dx_input` at the target gene under `target_lfc` (Option A — leaks one scalar at target, masked from scoring) or `canonical` (Option B — pure zero-shot). Plumbed through eval (`evaluation/run_perturbation_eval.py`) and training (`training/losses.py:perturbation_set_loss_u`). Two training variants are now supported: **V1 replacement** (replace dx_input with sparse) and **V2 auxiliary** (keep existing loss + add downstream-masked sparse-input MSE term, weight `aux_prop_weight`).
- **6 retrains in flight** (Replogle-K562_essential, Norman, Combined-K562essential+Norman × V1/V2), 2 finished (Norman V1 + V2). Eval after they finish under both input_modes A and B + ablation suite.
- **Phase 0 honest eval of the existing leaky-trained checkpoint** confirms the leakage diagnosis: legacy `observed_full` AUROC 0.619 → `target_lfc` 0.583 → `canonical` 0.587; correlation baseline 0.579 throughout. Lift over corr collapses from +0.040 to +0.004 once the answer isn't in the input.
- **Known unfixed bug**: encoder modality token defaults to 0 when `modality_id=None` ([models/encoder.py:73-75](models/encoder.py#L73-L75)); Norman eval path never passes it. CRISPRa cells are encoded in the CRISPRi/control slot. Fix queued (see § 9.4).

---

## 1. The model

`CausalVAE` in [models/causal_vae.py](models/causal_vae.py) — VAE with a linear SCM intervention layer.

- **Encoder** ([models/encoder.py](models/encoder.py)): MLP → `(μ, log σ²)`, optional additive `modality_emb` token + additive `batch_emb` (gem_group). 256-dim hidden. **Bug**: defaults `modality_id` to zeros when unset (line 73-75) — see § 9.4.
- **DAG** ([models/dag.py](models/dag.py)): factored `A = sigmoid(θ) · weights · (1 − I)`, where `θ` is Gumbel-Sigmoid edge logits and `weights` is a real-valued Xavier-init matrix. Sparsity penalty `sparsity_loss(A, mode)` supports two modes:
  - `mode='product'` (legacy): `mean(|gate · weight|)`.
  - `mode='gates'` (new, from sweep 2387217): `mean(sigmoid(θ/τ))` — penalises edge existence only, not magnitude. Sweep showed both modes give similar results at low λ; gates over-prune at high λ.
- **Decoder** ([models/decoder.py](models/decoder.py)): linear `x = D·z + b`. Orthogonality loss is **scale-invariant trace-normalised**: `||DᵀD / (||D||²/d) − I||²_F`. Drives equal column norms; does NOT prevent entry-sparsity, but the basin tends to dense small entries.
- **SCM**: `z = (I − A^T)⁻¹·ε` (observational), `z_post = (I − M·A^T)⁻¹·eps_tilde` (interventional). M = diag(1 − u).
- **u-score do-operator**: `u = clip(L[g, :], 0, u_clip_max)` where `L[g, i] = D[g,i]² / max_j D[g,j]²` is the row-max-normalised decoder loading. `u_clip_max = 0.5` in all current configs.

### Existing checkpoints

| Checkpoint | Trained on | Loss | Epochs | Use for |
|---|---|---|---|---|
| `results/phase1_promote/checkpoints/phase1_epoch150.pt` | Replogle train_v5.h5ad (CRISPRi, 187k K562, 1371 perts, 2000 HVG) | Leaky `match_delta` (observed dx as input) | 150 | THE existing "promotion" checkpoint. All Phase 0 honest evals run against this. |
| `results/phase1_norman_u_score/checkpoints/phase1_epoch150.pt` | Norman MTX (CRISPRa, 104k cells, 105 perts) | Leaky | 150 | Norman-only baseline before Task-A. |
| `results/phase1_union_hvg/checkpoints/phase1_epoch130.pt` | K562_gwps_filtered + force-include Norman pert genes (8248 → 2086 HVG axis) | Leaky | 130 (timed out at 24h) | K562_gwps baseline. Eval sweep on it failed; investigate. |

**Do not use `phase1_best.pt`** for evaluation — it's saved by `total_loss` criterion, which gets stuck at epoch 21 (right after Phase A ends and the regularizers kick in). Always use `phase1_epoch150.pt`.

---

## 2. The leakage diagnosis (the central finding)

### 2.1 What we found

In [evaluation/run_perturbation_eval.py:evaluate_single_perts](evaluation/run_perturbation_eval.py):
```python
dx_t = de_single_adata[g]    # observed lfc, length n_genes
z_post = model.intervene_gene_u(eps, dx_t, A, gene_idx=g, ...)
```

Inside `intervene_gene_u`: `dz = D⁺·dx_t` → `z_post = (I − M·A^T)⁻¹·(u·(z_prev + dz) + (1-u)·ε)` → `decode(z_post) ≈ D·D⁺·dx_t + (SCM correction)`.

Because λ_ortho=1.0 drives `DᵀD ≈ I·s`, `D·D⁺ ≈ DDᵀ` is a rank-64 orthogonal projection. So the model's prediction is essentially the answer, projected to rank 64.

The **additive baseline** is `ctrl + dx_observed = pert_observed_mean` — literally the ground truth. AUROC was reported as 0.998 on singles because of this.

The training loss in [training/losses.py:perturbation_set_loss_u](training/losses.py) has the same flaw:
```python
dx_all = (x_pert_set.mean(1) - x_ctrl_set.mean(1)).detach()    # observed dx per sentence
Dt_dx_T = D.T @ dx_all.T                                        # dz = D⁺·dx_observed
```
The `match_delta` mechanism cancels the bias path but NOT the projection path — the model is trained to denoise its own input into D's column space, leaving the SCM under-constrained.

### 2.2 Empirical confirmation (Phase 0)

Re-evaluated `phase1_promote/checkpoints/phase1_epoch150.pt` on the held-out Replogle test split under three input modes (jobs 2393172/3/4, all completed):

| input_mode | dx_input construction | held-out test AUROC | corr baseline | lift |
|---|---|---|---|---|
| `observed_full` | Full observed lfc passed through | **0.619** | 0.579 | +0.040 |
| `target_lfc` | Sparse one-hot at target with observed scalar | **0.583** | 0.579 | **+0.004** |
| `canonical` | Sparse one-hot at target with ±2.0 | **0.587** | 0.579 | **+0.008** |

40 test perts (the ones with ≥3 DEGs). `n_ctrl_cells=500`, `seed=42`, identical for all three. The +0.04 was the projection identity reading back leaked DEGs.

### 2.3 Diagnostic numbers on the same checkpoint (epoch 150)

Output of `evaluation/dag_diagnostics.py`:

| Metric | Value | Notes |
|---|---|---|
| `theta_std` | 1.613 | sharp differentiation; DAG topology IS learned |
| edges p > 0.7 | 2,568 / 4,032 | 64% high-confidence edges |
| edges p > 0.9 | 2,378 | dense topology |
| `mean |A_effective|` | **0.0003** | weights got crushed by `sparsity_loss` on (gates·weights) |
| max `|A_effective|` | 0.450 | only 4 entries > 0.10 |
| spectral_radius(A) | 0.453 | OK for stability |
| programs/gene (10% row-max) | 15.8 | hits biology target |
| programs/gene (30% row-max) | 6.4 | genuinely sparse at strong loading |
| u‖₁ (single-gene, after clip 0.5) | 1.06 | nearly one-hot |
| ‖u‖₁/‖u‖₂ | 1.80 | participation ratio ≈ 3 active dims/pert |
| ‖Δz‖₂ / ‖Δε‖₂ ratio | 0.727 | SCM dampens by ~27%, doesn't propagate |
| mean n_affected (mutilation per pert) | 35.7 / 64 | still flagged "dense dε" |

**Diagnosis**: gates learned topology, but `sparsity_loss = mean(|gate · weight|)` proportionally crushes weights on confident edges. Effective `A ≈ 0`. SCM acts as identity. The leakage explained the apparent AUROC lift; the SCM doesn't do useful work.

---

## 3. Code changes (this branch)

### 3.1 New file: `models/dx_input.py`

Central shared helper used by both eval and training loss.

```python
build_dx_input_single(input_mode, target_idx, lfc_observed, n_genes, pert_direction=-1, canonical_magnitude=2.0, device)
build_dx_input_batch(input_mode, gene_idx_batch, dx_full_batch, n_genes, pert_direction_batch=None, canonical_magnitude=2.0)
```

Three modes: `observed_full` (legacy, passes through), `target_lfc` (sparse with on-target observed scalar), `canonical` (sparse with `pert_direction · canonical_magnitude`).

### 3.2 Eval-side plumbing

[evaluation/run_perturbation_eval.py](evaluation/run_perturbation_eval.py):
- New CLI flags: `--input_mode {observed_full, target_lfc, canonical}`, `--pert_direction {-1, +1}` (auto from config.data.dataset), `--canonical_magnitude` (default 2.0), `--a_mode {learned, zero, shuffled}` (ablation), `--gene_idx_mode {correct, random}` (ablation), `--n_ctrl_cells` (default bumped 50 → 500), `--n_ctrl_combo` (default 20 → 200), `--seed`.
- `evaluate_single_perts` and `evaluate_combo_perts` accept all of the above and pass through to the build helper.
- `a_mode='zero'` replaces `A_eval` with zeros; `a_mode='shuffled'` permutes A's entries (preserves sparsity, destroys topology); `gene_idx_mode='random'` perturbs a random gene instead of the labelled one. **Scoring always masks the TRUE labelled gene** regardless of which gene the model was told to perturb.
- Additive baseline inside `evaluate_combo_perts` is now built from the **same** sparse `dx_a_t, dx_b_t` as the model — apples-to-apples. Under sparse input modes the additive baseline degenerates to `ctrl + on-target scalars` and its downstream-DEG AUROC drops to ~0.5.

[evaluation/run_norman_validation.py](evaluation/run_norman_validation.py): mirrors the same CLI flags; default `pert_direction=+1` (Norman is CRISPRa).

[evaluation/eval_combinatorial.py](evaluation/eval_combinatorial.py): `predict_combo` is input-agnostic — takes whatever dx tensors are passed in.

[scripts/eval_indist.sbatch](scripts/eval_indist.sbatch): reads `INPUT_MODE`, `PERT_DIRECTION`, `CANONICAL_MAGNITUDE`, `A_MODE`, `GENE_IDX_MODE`, `N_CTRL_CELLS`, `N_CTRL_COMBO`, `SEED` env vars and passes to the script.

### 3.3 Training-side fix

[training/losses.py:perturbation_set_loss_u](training/losses.py):
- New kwargs: `input_mode`, `pert_direction_batch`, `canonical_magnitude`, `aux_prop_weight`.
- **V1 (replacement)**: when `input_mode='target_lfc'` or `'canonical'`, builds sparse `dx_input` via `build_dx_input_batch`. The `dz_all = D⁺·dx_input` flow uses the sparse version. `match_delta` target stays as full observed dx → the SCM is structurally required to bridge sparse-input to dense-target.
- **V2 (auxiliary)**: when `aux_prop_weight > 0`, runs an **additional** forward pass with sparse `dx_input`, computes `delta_pred_sparse`, masks the target gene, and adds `aux_prop_weight × MSE(delta_pred_sparse, dx_full) on downstream genes` to the primary loss.
- The two variants can be combined (V1 input_mode + nonzero aux_prop_weight), but the cleanest experiments hold one variable at a time. V1 default uses `aux_prop_weight=0`; V2 default uses `input_mode='observed_full' + aux_prop_weight=1.0`.

[training/train_vae.py](training/train_vae.py): reads `phase1.input_mode`, `phase1.pert_direction`, `phase1.canonical_magnitude`, `phase1.aux_prop_weight` from the config. Builds `pert_direction_batch` per sentence (defaults to scalar from config; uses `sent['pert_direction']` if the dataset provides per-cell direction).

### 3.4 Dataset support

[data/load_replogle.py](data/load_replogle.py):
- Handles **ENSG → symbol rename** when `var['gene_name']` is present and `var_names` start with `ENSG`. Enables using the unfiltered K562_essential file at `/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/K562_essential_raw_singlecell_01.h5ad` (8,563 genes, 2,058 perts, 310k K562 cells, raw counts).
- Handles missing `cell_line` obs column (some files are already K562-only).
- `force_include` param: ORs a list of gene symbols into the HVG mask. Used to make Norman's perturbed gene set evaluable on Replogle-trained models.

[data/load_norman.py](data/load_norman.py): maps `gemgroup` → `batch_id` for encoder batch-effect conditioning + cell-sentence pairing.

[data/cell_sentence_dataset.py](data/cell_sentence_dataset.py): `PairedCellSentenceDataset` produces `(ctrl_set, pert_set, gene_idx, batch_id, [modality_id])` items for plate-matched perturbation-consistency training.

[data/load_combined.py](data/load_combined.py): joins Norman + Replogle on shared HVG axis, tags cells with modality (`control` / `CRISPRa` / `CRISPRi`).

[data/preprocess.py](data/preprocess.py): `force_include` plumbed; no double-log guard yet (separate fix queued).

### 3.5 New configs (this branch)

| File | Dataset | Variant | Notes |
|---|---|---|---|
| `configs/replogle_essential_taskA_V1.yaml` | K562_essential (8,563 genes, raw) | V1 replacement | `input_mode='target_lfc'`, `aux_prop_weight=0`, CRISPRi |
| `configs/replogle_essential_taskA_V2.yaml` | K562_essential | V2 auxiliary | `input_mode='observed_full'`, `aux_prop_weight=1.0`, CRISPRi |
| `configs/norman_taskA_V1.yaml` | Norman (CRISPRa) | V1 | |
| `configs/norman_taskA_V2.yaml` | Norman | V2 | |
| `configs/combined_taskA_V1.yaml` | K562_essential + Norman, `n_modalities=3` | V1 | shared HVG axis, modality-conditioned encoder |
| `configs/combined_taskA_V2.yaml` | K562_essential + Norman | V2 | |
| `configs/replogle_essential.yaml` | K562_essential | legacy (leaky) | for reference / ablation |

Common pinned hyperparameters: `λ_pert=10`, `λ_sparse=10` (mode='product'), `λ_ortho=1.0`, `λ_D_sparse=30`, `λ_prior=10`, margin=0.5, `u_clip_max=0.5`, 150 epochs, 200k max_cells.

### 3.6 New sbatches (this branch)

| File | Purpose |
|---|---|
| `scripts/train_phase1_taskA.sbatch` | Generic Task-A training. Takes `CFG`, `OUTDIR` env vars. Used for all 6 retrains. |
| `scripts/eval_indist.sbatch` | Generic single-checkpoint eval. Takes `CKPT`, `CFG`, `OUTDIR`, `INPUT_MODE`, `PERT_DIRECTION`, etc. env vars. |
| `scripts/eval_u_sweep.sbatch` | 9-cell array eval sweep (Replogle held-out + Norman external × parallel/sequential × 3 u_max). |
| `scripts/train_phase1_essential.sbatch` | Phase-1 on K562_essential under the LEAKY config (cancelled, see § 5). |
| `scripts/train_phase1_replogle_union_hvg.sbatch` | Phase-1 on K562_gwps_filtered + Norman force-include (timed out at epoch 130). |
| `scripts/train_phase1_norman_u_score.sbatch` | Norman-only baseline (leaky). |
| `scripts/train_phase1_sweep_lsparse_mode.sbatch` | 8-cell array: lambda_sparse mode × weight sweep. |
| `scripts/train_phase1_minigrid_lD.sbatch` | 2-cell mini-grid: lambda_D_sparse ∈ {30, 100}. |

---

## 4. Grid search sweeps

### 4.1 λ_D_sparse mini-grid (job 2372626, completed earlier)

Sweep `λ_D_sparse ∈ {30, 100}` with everything else pinned. Winner `λ_D=30` gave programs/gene = 12.9, recon=0.537, pert=0.153. This is what the promotion config uses.

### 4.2 λ_sparse mode × weight sweep (job 2387217_[0-7], completed)

Sweep `(lambda_sparse_mode, λ_sparse)` over `{product, gates} × {0.1, 1.0, 10.0, 100.0}`. Each cell: 80 epochs, 50k cells. Per-cell summaries in `results/grid_search/lsmode_<mode>_ls<val>_lp10p0_m0p5_ld30p0/summary_row.csv`.

Headline findings:

| mode | λ_sparse | edges p>0.5 | θ_std | mean prog/gene | recon | pert | Notes |
|---|---|---|---|---|---|---|---|
| product | 0.1 | 3005 | 0.53 | 12.6 | 0.529 | 0.153 | indistinguishable from 1/10/100 |
| product | 1.0 | 2989 | 0.52 | 12.8 | 0.527 | 0.152 | |
| product | 10.0 (default) | 2951 | 0.56 | 12.1 | 0.530 | 0.153 | |
| product | 100.0 | 2753 | 0.56 | 13.8 | 0.533 | 0.152 | |
| **gates** | 0.1 | 2844 | 0.57 | 12.6 | 0.528 | 0.152 | similar to product@0.1 |
| **gates** | 1.0 | 1131 | 0.57 | 12.2 | 0.542 | 0.152 | starts trimming topology |
| **gates** | 10.0 | 48 | 0.26 | 14.7 | **0.613** | 0.154 | over-pruning DAG |
| **gates** | 100.0 | **0** | 0.06 | 15.8 | **0.641** | 0.153 | DAG killed |

**Conclusion**: at low λ the modes are indistinguishable; at high λ `gates` over-prunes the topology. The `mode='product'` L1 was never the binding constraint we suspected — meaning the weight collapse on confident edges is downstream of something else (likely the leaky pert loss not requiring A magnitude).

---

## 5. Retrains in flight (Task-A, this branch)

Six Phase-1 retrains submitted on 2026-06-02 with the new training loss. Generic launcher: `scripts/train_phase1_taskA.sbatch` with `CFG` + `OUTDIR` env vars.

| Job | Dataset | Variant | Config | Out dir | Status |
|---|---|---|---|---|---|
| 2393175 | Replogle K562_essential | V1 replacement | `replogle_essential_taskA_V1.yaml` | `results/phase1_replogle_essential_taskA_V1` | RUNNING (~14h40m at writeup) |
| 2393176 | Replogle K562_essential | V2 auxiliary | `replogle_essential_taskA_V2.yaml` | `results/phase1_replogle_essential_taskA_V2` | RUNNING |
| 2393177 | Norman | V1 | `norman_taskA_V1.yaml` | `results/phase1_norman_taskA_V1` | **COMPLETED** 8h39m |
| 2393178 | Norman | V2 | `norman_taskA_V2.yaml` | `results/phase1_norman_taskA_V2` | **COMPLETED** 8h54m |
| 2393179 | Combined (K562_essential + Norman) | V1 | `combined_taskA_V1.yaml` | `results/phase1_combined_taskA_V1` | RUNNING (~7h) |
| 2393180 | Combined | V2 | `combined_taskA_V2.yaml` | `results/phase1_combined_taskA_V2` | RUNNING |

**Cancelled** (leaky-loss K562_essential): job 2393037. Re-submitted as Task-A above.

**Failed**: job 2393034_[0-8] (K562_gwps ep130 eval sweep, all 9 cells, exit code 1 after 1-5 min). Unrelated to Task-A; eval failed on the union-HVG checkpoint, likely a config mismatch. Investigate later — not blocking.

Important: when each retrain finishes, use `phase1_epoch150.pt` (NOT `phase1_best.pt` — same broken-best pathology as before).

---

## 6. Evaluation done so far

### 6.1 Phase 0 — honest eval of the existing leaky-trained checkpoint

Already covered in § 2.2. Three eval runs, all completed (jobs 2393172/3/4). Output under `results/eval_taskA/ep150_{observed_full,target_lfc,canonical}/`.

### 6.2 Pre-existing legacy evals (now superseded by Phase 0)

- `results/eval_sweep/replogle_single_u0p[5,8,9]/` — leaky-eval u_max sweep on ep150 (AUROC 0.618).
- `results/eval_sweep/norman_parallel_u0p[5,8,9]/` — leaky-eval Norman external (AUROC 0.730–0.737 single, 0.41 combo).
- `results/replogle_validation/` — earlier Norman-trained → Replogle external (AUROC 0.78, biased toward easy perts).

These are now compromised by the leakage diagnosis; we keep them for reference but the honest numbers are in `results/eval_taskA/`.

### 6.3 Phase 2 (planned — runs after retrains finish)

For each of the 6 retrained checkpoints + the original promotion ep150:
- Run `evaluation/run_perturbation_eval.py` with `--input_mode target_lfc` AND `--input_mode canonical` (two columns per checkpoint).
- For Replogle and Combined checkpoints: also run `evaluation/run_norman_validation.py` with the same flags + appropriate `pert_direction`.
- Ablation suite per checkpoint: `--a_mode zero`, `--a_mode shuffled`, `--gene_idx_mode random`, `--u_clip_max ∈ {0.0, 0.5, 1.0}`.
- Baselines: correlation (already Task A), GRN (`D·A·Dᵀ`), additive (now sparse so it degenerates to ~0.5 downstream).

Decision matrix:

| Best retrain under Task A | Best under Task B | Conclusion |
|---|---|---|
| > corr + 0.05 | > corr + 0.03 | Real predictive signal; pick winner, proceed to Phase-2 RL |
| > corr + 0.05 | ≈ corr | Uses on-target magnitude, no zero-shot generalisation |
| ≈ corr | ≈ corr | Linear-SCM no better than gene-gene correlation → pivot architecture |
| < corr | < corr | Architecture destroys signal → abort |

---

## 7. Known issues / open items

### 7.1 Modality token defaults to 0 silently — **bug, not fixed yet**

[models/encoder.py:73-75](models/encoder.py#L73-L75):
```python
if self.modality_emb is not None:
    if modality_id is None:
        modality_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
```

`evaluation/run_perturbation_eval.py:265` and `evaluation/eval_combinatorial.py:41` call `model.encoder(x_ctrl_t)` without `modality_id`. In `load_combined`'s convention (control=0, CRISPRa=1, CRISPRi=2), this means **all Norman eval cells get encoded as control modality**. Caught by the PR review.

**Fix queued**:
- Add `--modality_id` CLI to both eval scripts. Auto-detect from `config.data.dataset` (Replogle→2, Norman→1, combined→from per-cell `modality_id`).
- Plumb through `evaluate_single_perts`, `evaluate_combo_perts`, `predict_combo` → `model.encoder(x, modality_id=..., batch_id=...)`.
- Make the default in `encoder.py` raise a `RuntimeError` (or loud warning) when `n_modalities > 0` and `modality_id is None`. So this can't silently regress.

### 7.2 Leave-one-out across-pert correlation baseline

Current correlation baseline is computed from control cells only. The model has effectively trained on the full perturbation atlas. This handicaps the baseline. PR review (§ 3) recommends a leave-one-out across-perturbation correlation baseline as the headline comparator. **Not implemented yet** — separate plan item.

### 7.3 `phase1_best.pt` saved at epoch 21

`training/train_vae.py:610-618` saves "best" by `total_loss` criterion, which is lowest right after Phase A ends (epoch 21) — before regularizers kick in. **Always use `phase1_epoch150.pt`**. Fix: change criterion to `recon + pert` only (deferred).

### 7.4 `preprocess.py` double-log

`data/preprocess.py` calls `sc.pp.normalize_total + sc.pp.log1p` unconditionally. The Replogle `train_v5.h5ad` file is already log-transformed (max=6.67, P99=3.36), so we've been double-logging it during all training and evaluation on that file. K562_essential and Norman MTX are raw counts (correct). **Not blocking, but the existing promotion checkpoint's training data has been distorted**. Fix queued: add a `_looks_log_transformed` heuristic and skip normalize/log if already log.

### 7.5 K562_gwps eval sweep failures (job 2393034)

All 9 array cells exited with code 1 in 1–5 min. Probably a config or path mismatch when evaluating the union-HVG checkpoint with `replogle_u_score_union_hvg.yaml`. Low priority since the union-HVG checkpoint is itself a leaky-trained model that we wouldn't use for headline numbers. Look into logs at `logs/eval_u_sweep_2393034_*.{err,out}`.

---

## 8. File map (this branch only)

### 8.1 New files

```
models/dx_input.py                                  — central sparse-dx builder (eval + training)
configs/replogle_essential.yaml                     — K562_essential dataset, legacy loss
configs/replogle_essential_taskA_V1.yaml            — K562_essential + V1 replacement
configs/replogle_essential_taskA_V2.yaml            — K562_essential + V2 auxiliary
configs/norman_taskA_V1.yaml                        — Norman + V1
configs/norman_taskA_V2.yaml                        — Norman + V2
configs/combined_taskA_V1.yaml                      — combined + V1
configs/combined_taskA_V2.yaml                      — combined + V2
scripts/eval_indist.sbatch                          — generic single-checkpoint eval
scripts/train_phase1_taskA.sbatch                   — generic Task-A retrain launcher
scripts/train_phase1_essential.sbatch               — leaky K562_essential (cancelled job 2393037)
slides/make_eval_methodology_slides.py              — 4-slide writeup of the eval methodology
slides/eval_methodology.pptx                        — rendered slides
CODE_REVIEW_RESPONSE.md                             — PR review response
PROJECT_STATE.md                                    — this file
```

### 8.2 Modified files (versus previous main)

```
models/dag.py                  — factored A = sigmoid(θ)·weights, sparsity_loss accepts mode
data/load_norman.py            — gemgroup → batch_id mapping, n_batches
data/load_replogle.py          — ENSG → symbol rename, missing cell_line column, force_include
data/load_combined.py          — handles K562_essential via the load_replogle rename
data/cell_sentence_dataset.py  — gene_idx + batch_id paired sentences
data/norman_perturbed_genes.py — helper to extract Norman pert gene list without full load
training/losses.py             — perturbation_set_loss_u accepts input_mode + aux_prop_weight
training/train_vae.py          — reads + passes input_mode + pert_direction + aux_prop_weight
evaluation/run_perturbation_eval.py     — new CLI flags + ablation knobs + sparse-aware additive baseline
evaluation/run_norman_validation.py     — same CLI flags, pert_direction=+1 default
scripts/eval_u_sweep.sbatch    — OUTDIR_PREFIX env var
configs/replogle_u_score.yaml          — combo_mode field added
configs/replogle_u_score_union_hvg.yaml — K562_gwps + force_include
configs/norman_u_score.yaml    — Norman-only config
```

---

## 9. Run plan reference (commands)

### 9.1 Submit one Task-A retrain

```bash
CFG=configs/replogle_essential_taskA_V1.yaml \
OUTDIR=results/phase1_replogle_essential_taskA_V1 \
    sbatch scripts/train_phase1_taskA.sbatch
```

### 9.2 Eval a checkpoint under one input mode

```bash
CKPT=results/phase1_norman_taskA_V1/checkpoints/phase1_epoch150.pt \
CFG=configs/norman_taskA_V1.yaml \
OUTDIR=results/eval_taskA/norman_V1_target_lfc \
INPUT_MODE=target_lfc \
PERT_DIRECTION=1 \
N_CTRL_CELLS=500 \
    sbatch scripts/eval_indist.sbatch
```

### 9.3 Ablation eval

```bash
# Set A=0, otherwise identical:
CKPT=... CFG=... OUTDIR=...ablation_a_zero INPUT_MODE=target_lfc A_MODE=zero \
    sbatch scripts/eval_indist.sbatch

# Random gene index:
CKPT=... CFG=... OUTDIR=...ablation_random_gene INPUT_MODE=target_lfc GENE_IDX_MODE=random \
    sbatch scripts/eval_indist.sbatch
```

### 9.4 Phase-2 eval batch (the 7 checkpoints × 2 input modes)

```bash
for ckpt in \
  results/phase1_promote/checkpoints/phase1_epoch150.pt \
  results/phase1_norman_taskA_V1/checkpoints/phase1_epoch150.pt \
  results/phase1_norman_taskA_V2/checkpoints/phase1_epoch150.pt \
  results/phase1_replogle_essential_taskA_V1/checkpoints/phase1_epoch150.pt \
  results/phase1_replogle_essential_taskA_V2/checkpoints/phase1_epoch150.pt \
  results/phase1_combined_taskA_V1/checkpoints/phase1_epoch150.pt \
  results/phase1_combined_taskA_V2/checkpoints/phase1_epoch150.pt; do
  for mode in target_lfc canonical; do
    # determine cfg + pert_direction per ckpt
    ...
  done
done
```

(Wrap in a helper sbatch when the time comes.)

---

## 10. What the next Claude should pick up

In rough priority:

1. **Watch the 4 in-flight retrains** (jobs 2393175, 2393176, 2393179, 2393180). When they land, eval their `phase1_epoch150.pt` under `target_lfc` + `canonical` + the ablation suite. Build the Phase-2 comparison table.
2. **Fix the modality-token bug** before any Norman / Combined eval. Without this, all Norman external numbers are silently wrong. Pattern: assertion in encoder + CLI plumbing + auto-detect from config.
3. **Implement the leave-one-out correlation baseline** as the new headline comparator (Ivy's review § 3).
4. Write a follow-up plan for the **training-best-checkpoint criterion** (currently saves at epoch 21).
5. **Investigate job 2393034 failures** (K562_gwps eval sweep, all 9 cells exit 1).
6. The K562_essential ENSG→symbol rename has 2 duplicate symbols out of 8,563 (handled by `var_names_make_unique`); confirm those duplicates don't accidentally hide any Norman pert gene during the symbol-overlap check.
7. Phase 2 RL is **gated on Phase 2 evals showing real signal**. Don't proceed to RL until at least one Task-A retrain shows lift over the LOO correlation baseline.

---

## 11. Dataset reference

| Dataset | Path | Cells | Genes | Perts | Modality | Notes |
|---|---|---|---|---|---|---|
| Replogle train_v5 (legacy) | `/large_storage/goodarzilab/causal_state/pretrain_data/replogle/train_v5.h5ad` | 632k (187k K562) | **4,096** | 1,371 K562 | CRISPRi | **already log-transformed** (double-log bug). Used for the existing promotion checkpoint. |
| Replogle K562_essential (NEW) | `/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/K562_essential_raw_singlecell_01.h5ad` | 310,385 | **8,563** | **2,058** | CRISPRi | **raw counts**. var_names are ENSG, auto-renamed to symbols by load_replogle. Used for all Task-A retrains. |
| Replogle K562_gwps | `/large_storage/goodarzilab/causal_state/pretrain_data/replogle/K562_gwps_filtered.h5ad` | 1,064,891 | 8,248 | many | CRISPRi | raw counts. Was used in the union-HVG retrain (job 2386248, timed out). |
| Norman MTX | `/large_storage/goodarzilab/shervin/CausalBioRL/Norman/` | 111,668 raw → 104,507 after good_coverage filter | 33,694 → 2,000 HVG | **105 singles + 131 combos** | CRISPRa | raw counts. The only K562 dataset with combinatorial perturbations. |
| Norman pert gene overlap | — | — | — | 57/105 in K562_essential, 54/105 in K562_gwps_filtered, 27/105 in train_v5 | — | Cross-modality evaluable count. |

---

## 12. Glossary

- **Task A** (lenient zero-shot): `dx_input` is sparse one-hot at the target gene, value = on-target observed lfc. Leaks ONE scalar; target masked from scoring.
- **Task B** (strict zero-shot): `dx_input` is sparse one-hot at the target gene, value = `pert_direction · canonical_magnitude`. No observation leak.
- **V1 (replacement)**: replace the leaky training loss with sparse-input. `match_delta` target stays as full observed dx.
- **V2 (auxiliary)**: keep the existing (leaky) training loss + ADD a downstream-masked sparse-input MSE term, weight `aux_prop_weight`.
- **D⁺ projection identity**: with column-orthonormal D, `D·D⁺ ≈ DDᵀ` is a rank-64 projector. The model's prediction under `observed_full` input is essentially the leaked answer projected to rank 64.
- **`input_mode`** in code: `'observed_full' | 'target_lfc' | 'canonical'`. Threads through eval AND training via `models.dx_input.build_dx_input_{single,batch}`.
- **`aux_prop_weight`**: scalar multiplier on the V2 auxiliary loss. 0 = V1 only (or no aux).
- **`a_mode`** in eval: `'learned' | 'zero' | 'shuffled'`. Ablation knob for the SCM.
- **`gene_idx_mode`** in eval: `'correct' | 'random'`. Ablation: does gene identity carry signal?

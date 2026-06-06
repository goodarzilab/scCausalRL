# exp9 — Is the GRN learning structure, or copying the CollecTRI prior?

**Problem.** exp2's CollecTRI recovery AUROC (0.96) is a **supervised tautology** —
`biological_prior_loss` (margin-ranking, λ_prior=10) pushes *every* overlapping
CollecTRI edge up on `G=D·A_prob·Dᵀ`, then we "evaluate" recovery on those same
edges. We need to distinguish *copying the supervised edges* from *learning
transferable regulatory structure*.

**Base recipe (all cells)** = exp8 `sl0p5_ls0p0` (shift encoder + linear decoder,
MSE+MMD, `pert_shift_l1=0.5`, `lambda_sparse=0`, `lambda_spectral=0.1`) — the
A-active base where "structure learning" is a coherent claim (on the exp2 base
A is bypassed, so `G` is just the gate table θ).

## Cells

| Cell | config | λ_prior | prior_train_frac | split_mode | dag_weight_init_scale | epochs / cells |
|---|---|---|---|---|---|---|
| **C1 `unsup`** | `grid/exp9_c1_unsup.yaml` | **0** | — | — | 1.0 | 150 / 200k |
| **C2 `heldtf`** ★ | `grid/exp9_c2_heldtf.yaml` | 10 | **0.7** | **tf** | 1.0 | 150 / 200k |
| **C3 `fullsup`** | — | 10 | 1.0 | — | 1.0 | **reuse exp8 `sl0p5_ls0p0`** |
| **C4 `biginit`** | `grid/exp9_c4_biginit.yaml` | 10 | 0.7 | tf | **8.0** | 60 / 50k |

## What each answers
- **C1** — fully unsupervised: does the GRN recover CollecTRI (full AUROC) above
  the **0.535 correlation floor** from data alone? (Notebook §2.)
- **C2 ★** — the decisive test: supervise 70% of TFs, **hold out 30% of TFs
  entirely**. Held-out-TF AUROC vs train-edge AUROC vs correlation-on-held-out
  floor (Notebook §5b). Copying ⇒ held-out ≈ chance; generalizing ⇒ held-out ≫ floor.
- **C3** — full-supervision anchor (reuse exp8) confirms the loss "works" so a low
  C2 held-out number means memorization, not a broken loss.
- **C4** — short init probe: does a stronger initial A (×8) survive the dynamics
  (bypass ratio at epoch 60) and improve held-out recovery?

## Key design choices
- **Hold out whole TFs, not random edges** — random holdout leaks a supervised
  TF's out-degree to its held-out edges. `split_collectri_edges(mode='tf')` in
  `utils/bio_priors.py` (deterministic md5 hash on TF name, independent of edge
  order / gene set) is used identically by training and the notebook eval.
- **Init is one probe, not a sweep** — init is identical to CrossModalCausal-VAE
  (whose A survives via no-bypass + weak sparsity, not init); it can't fix a
  dynamics problem, so only C4 tests it, cheaply (60 ep / 50k cells).

## Code (vs exp8)
- `utils/bio_priors.py:split_collectri_edges` (shared train/eval split).
- `training/losses.py:biological_prior_loss(prior_train_frac, prior_split_seed, prior_split_mode)`.
- `models/dag.py:DAGModule(weight_init_scale, theta_init_scale)`; plumbed via `model.dag_weight_init_scale`.
- `train_vae.py` passes `phase1.prior_*` + `model.dag_*`.
- Shared notebook `notebooks/build_dag_analysis_notebook.py` §5b: held-out-TF recovery (no-op if frac=1).

## Runs
Each cell: `run_phase1_experiment.sbatch` → chained `run_shift_eval.sbatch` (held-out
pred AUROC — pred still works on A-active base?) + `run_dag_grad_diag_generic.sbatch`
(A still active?) + `run_dag_analysis.sbatch` (CollecTRI recovery / held-out-TF).
Outputs under `results/grid/<cell>/`.

## Decision (T=held-out-TF AUROC, R=train-edge, B=corr-on-test≈0.53, U=C1 full AUROC)
- `T≈0.5, R high` ⇒ pure copying.  `B≪T<R` ⇒ partial real structure.  `T≈R` high (tf) ⇒ strong.
- `U>~0.6` ⇒ data alone recovers GRN (strongest "not copying").  C4 bypass ~1e-5 ⇒ init not the lever.

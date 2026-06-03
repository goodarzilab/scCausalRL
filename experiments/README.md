# Experiments

Each subfolder is a **self-contained snapshot** of the codebase (its own `models/`,
`training/`, `data/`, `evaluation/`, `rl/`, `utils/`, `configs/`, `scripts/`,
`tests/`). Runs resolve imports from their own experiment root (`run_phase1.py`
inserts the experiment dir on `sys.path`), so changing one experiment never
affects another or the top-level repo. Artifacts (`checkpoints/`, `logs/`,
`results/`, `wandb/`) are git-ignored and written under each experiment's
`results/` dir.

## Launching

```bash
EXP_DIR=$(readlink -f experiments/exp2_shift_linear) \
CFG=configs/exp2_shift_linear.yaml \
OUTDIR=$EXP_DIR/results/phase1 \
    sbatch experiments/run_phase1_experiment.sbatch
```

Smoke test (tiny subsample, 3 epochs, W&B off) — swap `CFG=configs/smoke.yaml`
and add `--time=00:45:00`. A synthetic CPU smoke (no data load) is at each
experiment's `tests/smoke_shift.py`.

W&B logging goes to project **CausalBioRL** (set in each config's `wandb:` block).

## Catalogue

| Exp | Folder | Decoder | Perturbation pathway | Perturbation loss | D-regularizers |
|----|--------|---------|----------------------|-------------------|----------------|
| 1 | `exp1_baseline` | linear | `dz = D⁺·dx` (u-score do-operator) — **unchanged baseline** | match-delta MSE | ortho + D-L1 + GRN prior |
| 2 | `exp2_shift_linear` | linear | **learned shift encoder** `S^p=c·MLP(label_g)` → additive SCM shift | MSE(mean-shift) + MMD(sets) | ortho + D-L1 + GRN prior |
| 3 | `exp3_shift_mlp` | MLP (2×1024) | learned shift encoder | MSE + MMD | none (no D) |
| 4 | `exp4_shift_linear_energy` | linear | learned shift encoder | MSE + MMD + **energy** | ortho + D-L1 + GRN prior |
| 5 | `exp5_shift_mlp_energy` | MLP (2×1024) | learned shift encoder | MSE + MMD + **energy** | none (no D) |

## What changed vs the baseline (exp2–5)

The leaky `dz = D⁺·dx` intervention is replaced by a **learned perturbation
shift** (SCCVAE, Liu/Zhang/Uhler 2025): `S^p = c · ShiftEncoder(pert_label[g])`,
where `pert_label` is the top-PCA of the *transposed* control expression matrix —
a per-gene feature available for unseen genes (zero-shot capable) that contains
no observed-answer leakage. The shift is injected additively into the SCM:

```
z_post = (I − A^T)^{-1} (ε_ctrl + S^p)      # pure additive shift, no u-score mutilation
x_hat  = decoder(z_post)
```

- **New modules:** `models/shift_encoder.py` (`ShiftEncoder`); `models/decoder.py`
  gains `MLPDecoder`; `models/causal_vae.py` gains `decoder_type`,
  `use_shift_encoder`, `set_pert_labels`, `compute_shift`; `training/losses.py`
  gains `perturbation_set_loss_shift`; `training/train_vae.py` computes the PCA
  labels and dispatches the shift loss.
- **Config knobs (`model:`):** `decoder_type` (`linear`|`mlp`), `use_shift_encoder`,
  `shift_label_dim` (50), `shift_hidden` (1024), `shift_scale_init`.
- **Config knobs (`phase1:`):** `pert_mse_weight`, `pert_mmd_weight`,
  `pert_energy_weight`.
- Hyperparameters seeded from the promoted Task-A config
  (`replogle_essential_taskA_V1.yaml`): K562_essential CRISPRi, latent_dim 64,
  lr 1e-3, λ_pert 10, 150 epochs. Loss-term weights are first-pass starting
  points; W&B breaks out each term (`pert_diag/{mse,mmd,energy}_term`) for
  rebalancing.

Open follow-ups: bump `latent_dim` (64 kept for baseline comparability; SCCVAE
uses 512); rebalance loss-term weights from the W&B per-term breakdown.

## Evaluation (`evaluation/run_shift_eval.py`)

The shift-pathway eval. Takes a checkpoint + config and scores held-out
(test-split) perturbations with the leak-free shift prediction
`S^q = c·ShiftEncoder(pert_label[q]) → (I−A^T)⁻¹(ε+S^q) → decode`:

- **Metrics** (target gene masked): downstream-DEG AUROC, Spearman (|effect| &
  signed), direction accuracy, Pearson-on-DEGs, mean-shift MSE, and set-level
  **RBF-MMD + energy distance** vs a **control-vs-observed floor**.
- **Penetrance** `--c_mode fixed|search|both`: `fixed` (c=1, strict/leak-free,
  uses learned `shift_scale`) is the honest default; `search` grid-searches c on
  pseudo-bulk MSE (SCCVAE §2.2).
- **Baselines**: correlation (control |corr|) and a **co-expression-neighbour
  LOO** baseline (mean observed lfc of the k nearest train perts in the same
  control-PCA label space the shift encoder uses) — the fair zero-shot comparator.
- **Headline**: held-out test AUROC (shift c=1 / c-search) vs correlation vs co-exp.

Run via `run_shift_eval.sbatch` (set `EXP_DIR`/`CKPT`/`CFG`/`OUTDIR`/`CMODE`,
`WANDB_FLAG=--wandb`). The epoch-150 evals are submitted with
`--dependency=afterok:<train_job>` so they fire automatically when training
finishes. Synthetic CPU smoke: `tests/smoke_shift_eval.py`.

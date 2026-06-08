# exp10 — GRN sparsity + exogenous disentanglement ablations

Two diagnostics on the A-active winner (exp8 `sl0_ls0`) motivated this grid:

1. **GRN sparsity.** A real GRN is *few edges, each possibly strong*. The
   magnitude-L1 we used in exp1–7 (`lambda_sparse_mode=product`, `mean|A|`)
   killed A by shrinking *weights* (→ bypass). The right penalty is on edge
   **existence** — `lambda_sparse_mode=gates` = `mean(σ(θ/τ))` — which removes
   edges while leaving survivors at full magnitude (sparse-but-strong A).
2. **Exogenous entanglement.** The ε/z diagnostic showed the *exogenous* noise ε
   is ~0.43–0.47 off-diagonal correlated — i.e. not independent factors. Diagonal
   KL can't see cross-dim correlation, so raising β alone won't fix it; we add an
   explicit decorrelation term `eps_decorrelation_loss(μ)` = mean squared
   off-diagonal of cov(μ) (a total-correlation proxy).

**Base recipe (all cells)** = exp8 `sl0p0_ls0p0` (shift encoder, linear decoder,
MSE+MMD, `pert_shift_l1=0`, `lambda_sparse=0`, `lambda_spectral=0.1`,
`lambda_prior=10`) — the A-active winner (AUROC 0.708, bypass 0.42, ε off-diag 0.468).

## Cells

| Cell | knob | value | hypothesis |
|---|---|---|---|
| `exp10_g_ls2`       | `lambda_sparse` (mode=**gates**) | 2  | mild edge-existence penalty → sparser topology, A stays active |
| `exp10_g_ls10`      | `lambda_sparse` (gates) | 10 | medium |
| `exp10_g_ls30`      | `lambda_sparse` (gates) | 30 | aggressive — does the GRN get sparse without A dying? |
| `exp10_d_beta2`     | `beta_end` | 2.0 | does a stronger KL ceiling reduce ε off-diag corr? (expected: weakly, KL is diagonal) |
| `exp10_d_decorr100` | `lambda_eps_decorr` | 100  | TC proxy on μ → decorrelate ε without collapse |
| `exp10_d_decorr1000`| `lambda_eps_decorr` | 1000 | strong decorrelation |
| (control) | — | — | reuse exp8 `sl0p0_ls0p0` (AUROC 0.708, ε off-diag 0.468) |

## Read-outs (chained `run_exp10_evals.sbatch` per cell)
- **shift_eval** — Replogle held-out AUROC. Did prediction survive the new penalty?
- **dag_grad_diagnostic** — A-bypass ratio. For gate cells: is A still active
  (bypass ≫ 1e-5) at higher sparsity, or did edges all turn off → dead?
- **eps_z_diagnostic** — the ε/z table. For disentangle cells: did **ε** off-diag
  corr drop below 0.45 *without* killing z's info (silhouette z) or collapsing
  ε (active units)?

## Code (vs exp8)
- `training/losses.py:eps_decorrelation_loss(mu)` + `lambda_eps_decorr` in
  `compute_total_loss` (default 0 ⇒ identical to exp8).
- `training/train_vae.py` passes `phase1.lambda_eps_decorr`.
- Gate-sparsity needs **no** new code — `lambda_sparse_mode=gates` already exists.

## Decision
- Gate cell wins if: AUROC ≈ 0.70 **and** edge density / GRN nonzero-frac drops
  **and** bypass stays ≫ 1e-5 (sparse + active + predictive). All three → promote.
- Disentangle cell wins if: ε off-diag corr ↓ (toward <0.2) with AUROC held and
  active units = 64/64 (no collapse). decorr expected to beat β.

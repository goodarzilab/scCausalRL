# exp20 MASTER — all arms: drift-removed prediction (singles+combos) + A-load-bearing

disc_resid/pear_resid = held-out SINGLES drift-removed (chance disc 0.5). S_/C_real = drift-removed spearman singles/combos. **_Agap = real−max(zero,shuf)** (>0 ⇒ A helps). probe_A = 1−cos(Δ,Δ_{A=0}) (A used), probe_Dperm≈1 ⇒ decoder dominates, crossTF≈0 ⇒ mode collapse.

| arm | prior | disc_resid | pear_resid | S_real | **S_Agap** | C_real | **C_Agap** | C_disc | probe_A | probe_Dperm | crossTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sparse_esm | prior-ON | 0.597 | 0.213 | 0.444 | 0.024 | 0.480 | 0.015 | 0.050 | 0.007 | 1.007 | 0.000 |
| sparse_pertonly | prior-ON | 0.594 | 0.107 | 0.268 | -0.037 | 0.390 | 0.016 | 0.050 | 0.459 | 0.985 | 0.018 |
| sparse_con | prior-ON | 0.593 | 0.124 | 0.274 | -0.041 | 0.384 | 0.043 | 0.025 | 0.156 | 0.984 | 0.018 |
| sparse_residrecon | prior-ON | 0.593 | 0.097 | 0.367 | 0.010 | 0.384 | -0.006 | 0.050 | — | — | — |
| min_esm | prior-ON | 0.587 | 0.205 | 0.400 | 0.033 | 0.477 | 0.018 | 0.025 | 0.014 | 1.007 | 0.001 |
| sparse_resid | prior-ON | 0.504 | -0.175 | 0.436 | 0.014 | 0.406 | -0.006 | 0.025 | — | — | — |
| sparse_all | prior-ON | 0.502 | -0.180 | 0.453 | -0.002 | 0.447 | -0.003 | 0.025 | — | — | — |
| sparse_con | prior-off | 0.639 | 0.094 | 0.259 | -0.054 | 0.369 | 0.027 | 0.025 | — | — | — |
| sparse_pertonly | prior-off | 0.621 | 0.062 | 0.260 | -0.025 | 0.371 | 0.025 | 0.025 | — | — | — |
| staged_con_s26 | prior-off | 0.611 | 0.066 | 0.264 | -0.051 | 0.376 | 0.017 | 0.025 | — | — | — |
| sparse_residrecon | prior-off | 0.609 | 0.127 | 0.519 | 0.006 | 0.494 | 0.002 | 0.025 | — | — | — |
| min_esm | prior-off | 0.586 | 0.180 | 0.405 | 0.001 | 0.448 | -0.000 | 0.025 | — | — | — |
| staged_nonlinSCM_s26 | prior-off | 0.570 | -0.005 | 0.264 | -0.080 | 0.379 | 0.011 | 0.025 | — | — | — |
| sparse_esm | prior-off | 0.555 | 0.173 | 0.459 | 0.028 | 0.463 | 0.027 | 0.050 | — | — | — |
| staged_esm_s26 | prior-off | 0.553 | 0.215 | 0.423 | 0.050 | 0.461 | 0.027 | 0.025 | — | — | — |
| staged_esm_s36 | prior-off | 0.548 | 0.222 | 0.437 | 0.048 | 0.449 | 0.021 | 0.025 | — | — | — |
| sparse_all | prior-off | 0.502 | -0.187 | 0.462 | 0.018 | 0.456 | 0.005 | 0.025 | — | — | — |
| sparse_resid | prior-off | 0.502 | -0.182 | 0.445 | 0.034 | 0.440 | 0.018 | 0.025 | — | — | — |
| whiten | prior-off | 0.500 | -0.204 | 0.489 | 0.059 | 0.485 | 0.022 | 0.025 | — | — | — |

_exp19 best (linear, prior-on epsmix): disc_resid=0.647. Prior-on CollecTRI AUROC ~0.97 = memorization; prior-off ~0.57 = corr floor._

---
## VERDICT — what worked (2026-07-06)

**Only turning the prior OFF helped.** Best drift-removed disc = `sparse_con_po` **0.639** (↑ from 0.593 prior-on, ≈ exp19 epsmix 0.647). None of the 4 new architectural fixes beat it:

- **Whitening (mode-collapse fix): FAILED** — `whiten` collapsed to chance (disc 0.500, pear −0.20). Drift-removed + whitened target starves the pert signal (same failure as resid/all). Whitening did not fix mode collapse; it removed the signal.
- **Staged 3-phase L0 (gate-race fix): NO EFFECT** — `staged_esm_s26` 0.553 ≈ coupled `sparse_esm_po` 0.555 (no rescue of the no-contrastive arm); `staged_con_s26` 0.611 < coupled 0.639. The gate-vs-A race was not the bottleneck.
- **Nonlinear SCM (tanh-Picard): WORSE** — `staged_nonlinSCM_s26` 0.570 < its linear twin `staged_con_s26` 0.611; A actively hurts (S_Agap −0.08; combos c_pear real −0.26 vs +0.17 at A=0). Nonlinearity in A did not help on Joung.
- **Frozen GO+PCA decoder (absorption fix): DIVERGED (NaN)** — all 3 latent-256 frozen arms NaN'd at epoch 11 (DAG activation): forward-pass blowup of `(I−Aᵀ)⁻¹` at d=256 (spectral-clamp power-iteration under-estimated ‖A‖). **The central decoder-absorption question remains unanswered** pending a stabilized retrain (n_iter 10→20 + max_sn 0.9→0.85, mu-clamp, NaN-loss guard, lr 1e-3→5e-4).

**A is still not load-bearing in any completed arm** (every S_Agap ≤ 0.05 or negative). **Mode collapse persists** and could not be broken by whitening or contrastive. Combos are at chance (C_disc ~0.025) with negative drift-removed pearson for the best arms. The frozen-decoder retrain is the one remaining shot at making A useful.

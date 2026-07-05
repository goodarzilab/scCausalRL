# exp20 master results — did the structural+sparse fix make A load-bearing?

## 1. Drift-removed prediction (held-out singles) — the metric that matters

| arm | disc_cos_resid | pearson_δ_resid | deg_auroc_resid | disc_cos (raw) | retrieval_top1 |
|---|---|---|---|---|---|
| min_esm | 0.587 | 0.205 | 0.858 | 0.500 | 0.006 |
| sparse_esm | 0.597 | 0.213 | 0.885 | 0.500 | 0.006 |
| sparse_con | 0.593 | 0.124 | 0.861 | 0.505 | 0.000 |
| sparse_resid | 0.504 | -0.175 | 0.904 | 0.640 | 0.036 |
| sparse_pertonly | 0.594 | 0.107 | 0.864 | 0.504 | 0.000 |
| sparse_residrecon | 0.593 | 0.097 | 0.836 | 0.508 | 0.006 |
| sparse_all | 0.502 | -0.180 | 0.913 | 0.574 | 0.012 |

_exp19 best for reference: v2_esm_con_epsmix disc_cos_resid=0.647; ceiling ~0.55–0.65._

## 2. Is A load-bearing? (drift-removed spearman: real vs zero-A vs shuffled-A)

A_gap>0 ⇒ real A **improves** prediction. probe_Agap>0 ⇒ A **changes** the output (mechanistically used).

| arm | S_real | S_zero | S_shuf | **S_Agap** | C_Agap | gate_open | probe_Agap | probe_Agap(nt) |
|---|---|---|---|---|---|---|---|---|
| min_esm | 0.400 | 0.367 | 0.341 | 0.033 | 0.018 | 1.000 | 0.014 | — |
| sparse_esm | 0.444 | 0.419 | 0.398 | 0.024 | 0.015 | 0.016 | 0.007 | 0.018 |
| sparse_con | 0.274 | 0.315 | 0.277 | -0.041 | 0.043 | 0.749 | 0.156 | 0.164 |
| sparse_resid | 0.436 | 0.422 | 0.380 | 0.014 | -0.006 | 0.721 | 0.148 | 0.182 |
| sparse_pertonly | 0.268 | 0.305 | 0.276 | -0.037 | 0.016 | 0.788 | 0.455 | 0.432 |
| sparse_residrecon | 0.367 | 0.357 | 0.331 | 0.010 | -0.006 | 0.810 | 0.338 | 0.388 |
| sparse_all | 0.453 | 0.456 | 0.381 | -0.002 | -0.003 | 0.745 | 0.262 | 0.305 |

## 3. Component ablation (Δspearman when ablated; larger drop = more load-bearing)

| arm | shift→0 | D→random | A→0 |
|---|---|---|---|
| min_esm | -0.122 | 0.052 | 0.033 |
| sparse_esm | -0.078 | 0.096 | 0.024 |
| sparse_con | -0.231 | -0.040 | -0.043 |
| sparse_resid | -0.077 | -0.010 | 0.008 |
| sparse_pertonly | -0.254 | -0.057 | -0.037 |
| sparse_residrecon | -0.153 | 0.072 | 0.013 |
| sparse_all | -0.030 | -0.003 | -0.010 |

## 4. GRN biological-ness (projected G=D·A·Dᵀ vs priors, AUROC)

**⚠️ CAVEAT — the CollecTRI prior was ON during exp20 training (`lambda_prior:10`, carried over
from the forked exp19 configs; NOT off as the plan intended).** The prior loss activated at
epoch 11, spiked to 0.275, and was driven to ~0.008 — i.e. G was actively fit to the CollecTRI
margin. So `collectri_auroc` 0.97–0.98 is **prior memorization, not prior-free discovery**, and
this table is NOT a fair biological-recovery test. shuffA≈collectri (0.95≈0.97) shows the
memorization lives in the DECODER (D·Dᵀ), robust to A-shuffle; randomD→chance confirms D carries
it. `dorothea_held` (edges the prior never saw) 0.73–0.80 is the only partial generalization
signal. **A prior-OFF re-run is required for the honest biological-ness question.**

| arm | collectri | shuffA | dorothea_held | corrGRN(base) | randomD(chance) |
|---|---|---|---|---|---|
| min_esm | 0.983 | 0.951 | 0.799 | 0.568 | 0.477 |
| sparse_esm | 0.983 | 0.953 | 0.801 | 0.568 | 0.506 |
| sparse_con | 0.972 | 0.964 | 0.777 | 0.568 | 0.515 |
| sparse_resid | 0.973 | 0.961 | 0.783 | 0.568 | 0.463 |
| sparse_pertonly | 0.970 | 0.965 | 0.780 | 0.568 | 0.514 |
| sparse_residrecon | 0.981 | 0.957 | 0.731 | 0.568 | 0.479 |
| sparse_all | 0.971 | 0.965 | 0.779 | 0.568 | 0.527 |

---

## VERDICT — Phase 1 (structural intervention + sparse target gate)

**The fix works mechanically, but A still cannot carry the Joung signal.**

1. **A became mechanically load-bearing in 5/7 arms** (first time off zero in 30+ configs):
   probe_Agap = 1−cos(Δ_A, Δ_{A=0}) rose to 0.16 (con), 0.15 (resid), 0.26 (all),
   0.34 (residrecon), 0.45 (pertonly). Controls behave exactly as designed:
   `min_esm` (Fix-1 only, dense S) stays inert (0.014) → **Fix 2 (sparse gate) is the
   necessary piece, not Fix 1 alone**.

2. **But making A "used" did NOT improve prediction — it traded against it:**
   - `S_Agap` (real-A beats zero/shuffled-A on drift-removed spearman) ≈ 0 or **negative**
     for every arm — real ≈ zero, the exp19 result stands.
   - Best `disc_cos_resid` = **0.597** (`sparse_esm`, whose gate *collapsed* → A inert)
     < exp19 best **0.647**. **No arm beats the discrimination ceiling.**
   - Clear tradeoff: the high-probe_Agap arms have the *lowest* gene-space spearman
     (con/pertonly S_real ≈ 0.27 vs A-inert esm ≈ 0.44), and the aggressive-recon arms
     **collapsed to chance** (`sparse_resid` 0.504, `sparse_all` 0.502, both pearson_resid < 0).
   - Component ablation: A→0 barely moves spearman (±0.01–0.04); **shift→0 dominates** (−0.08…−0.25).

3. **The "biological" GRN is NOT a clean signal — the prior was accidentally ON.** CollecTRI
   AUROC 0.97–0.98 is prior memorization (`lambda_prior:10` carried over from exp19 configs),
   and it lives in the decoder (shuffling A only drops it to 0.95–0.96; randomD → chance). This
   arm-set cannot answer the prior-free biological-ness question; a prior-OFF re-run is needed.

**Fragility notes:** (a) L0=0.03 without contrastive over-sparsifies — `sparse_esm` gate
collapsed to 1.6% open; contrastive protects it (`con` 74.9% open). (b) drift-removed pert
target + down-weighted recon (`resid`, `all`) destabilize the fit to chance discrimination.

**Conclusion (the honest negative the plan anticipated):** at linear-Gaussian latent depth,
forcing effects through A makes A *used* but not *useful* — the shift+linear-decoder path
carries the predictable Joung signal, and the latent is rotationally degenerate so A adds
structure that isn't the right structure. **Pivot to Phase 2** (nonlinear SCM to give A a job
linear composition can't do + frozen-GO decoder to break the encoder/decoder rotational
symmetry so A can't be absorbed).

# Feature-probe gate — which INPUT carries generalizable TF-identity?

Model-free ridge probe (`evaluation/feature_probe.py`): fit Ridge(feature[train]→observed Δ
[train]) over the 1351 train TFs, retrieve on the 169 held-out test TFs. `_resid` = same on
the drift-removed residual Δ (subtract the shared across-TF mean) — the TF-SPECIFIC component.
chance: top1=0.006, disc=0.5.

## Result (epoch80 splits)

| feature | dim | RAW test disc | **RESID test disc** | RESID top1 |
|---|---|---|---|---|
| **esm_raw** | 1280 | 0.500 | **0.646** | 0.012 |
| esm_pca100 | 100 | 0.500 | 0.645 | 0.012 |
| esm+regulon_pca | 200 | 0.501 | 0.640 | 0.000 |
| ctrl+esm+regulon | 250 | 0.500 | 0.633 | 0.000 |
| esm+family | 175 | 0.502 | 0.635 | 0.006 |
| esm+regulon+family | 275 | 0.499 | 0.625 | 0.006 |
| tf_family | 75 | 0.500 | 0.585 | 0.012 |
| regulon_pca100 | 100 | 0.500 | 0.545 | 0.036 |
| regulon_raw | 9923 | 0.500 | 0.520 | 0.012 |
| ctrl_pca50 (baseline) | 50 | 0.500 | 0.498 | 0.000 |

## Verdict
1. **RAW retrieval: every feature at chance** — masked by the shared "differentiation drift"
   (~54% of observed-Δ variance). This is why the model looks hopeless.
2. **RESIDUAL (drift-removed): TF-identity features clear chance.** ESM is the single best
   (disc 0.645); TF family 0.585; CollecTRI regulon 0.545. Concats don't beat ESM alone.
3. **Control-coexpression PCA (the current input) is chance even on residual (0.498)** — the
   baseline input genuinely carries no generalizable TF-identity.

## Implication
The input is usable, but ONLY for the TF-SPECIFIC (drift-removed) component, and ONLY with a
real identity feature (ESM ≫ regulon ≫ control-PCA). To exploit it the model must:
  (a) use ESM as the shift-encoder descriptor (replace/augment control-PCA), and
  (b) target the DRIFT-REMOVED residual — the loss (contrastive) and the eval must operate on
      the residual, else drift masks everything.
Ceiling is modest (disc ~0.65), reflecting that hESC TF-OE effects are drift-dominated; the
shared differentiation program is most of the signal and the TF-specific part is small.

# exp18 — Causal world model (exp13 / exp16) vs Arc STATE on Replogle K562

Head-to-head on **Replogle K562 single-perturbation prediction**, scored with Arc's
**cell-eval**, against the **ST-SE-Replogle** STATE model and the **ContextMean** /
**PerturbMean** baselines.

## Key shortcut

STATE's evaluation is **already cached** in the HF model dir
(`models--arcinstitute--ST-SE-Replogle/.../fewshot/k562/eval_best.ckpt/`): `adata_real.h5ad`
+ `adata_pred.h5ad` (134 751 cells × 2000 HVG) + `k562_results.csv` (per-pert cell-eval on
the **968 K562 fewshot held-out singles**). So no STATE inference was run — we read its
per-pert metrics and use its `adata_real` as the shared ground truth.

## Matching (fairness)

- **Gene axis**: STATE's 2000 HVG (⊂ both world-model panels) — identical readout for all.
- **Pert sets**: each world model is scored only on Replogle test perts held out by **both**
  it and STATE — exp13 → **110**, exp16 → **92** (87 on-axis). STATE + baselines are subset
  to the same perts per panel, so every model-vs-STATE comparison is on an identical,
  fairly-held-out single-pert set. (Different seed-42 splits → these subsets differ, so the
  exp13 and exp16 panels are not on the identical pert set.)
- **Normalisation**: predictions are control-anchored — for each real perturbed cell,
  predicted expr = (sampled real non-targeting control) + model-Δ, clipped ≥0. So
  pred-Δ = model-Δ and the **scale-robust metrics** (pearson_delta, DE-direction,
  discrimination) are directly comparable to STATE. (MAE is normalisation-sensitive; report
  with care. `de_spearman_lfc_sig` hit a cell-eval polars dtype bug on the non-STATE runs →
  NaN there.)

## Pipeline

1. `prep.py` — STATE 2000-HVG axis + fair pert lists.
2. `predict_deltas.py` (base env) — exp13 (additive shift) / exp16 (mutilation+modality)
   per-pert gene-space Δ on the 2000 HVG via the shift encoder + do-operator.
3. `compute_baselines.py` — ContextMean (mean K562 perturbed Δ, constant) + PerturbMean
   (per-pert offset averaged over the other Replogle lines) Δ.
4. `score_celleval.py` (state_env) — control-anchor + `cell_eval.MetricsEvaluator` (the
   identical scorer STATE used).
5. `merge_and_plot.py` — head-to-head table + `results/state_comparison.png`.

## Result (means; n=110 exp13 panel / n=87 exp16 panel)

| method | pearson_delta | DE-direction | discrimination(cos) | overlap@100 |
|---|---|---|---|---|
| **STATE** | **0.32 / 0.34** | **0.69 / 0.74** | **0.73 / 0.76** | 0.12 / 0.13 |
| exp13 (K562-only) | 0.14 | 0.59 | 0.55 | 0.05 |
| exp16 (4-line) | 0.08 | 0.55 | 0.57 | 0.03 |
| ContextMean | 0.15 / 0.15 | 0.61 / 0.62 | 0.51 / 0.53 | 0.04 |
| PerturbMean | 0.04 / 0.05 | 0.53 / 0.55 | 0.55 / 0.58 | 0.02 |

**Takeaways.** (1) STATE clearly leads on every scale-robust metric (~2× the causal WMs on
pearson-delta). (2) **exp13 (K562-specialised) > exp16 (Replogle+Nadig, diluted on K562)** —
as expected. (3) The causal WMs barely match the **ContextMean** "predict-the-average-
perturbation" baseline on pearson-delta (exp13 0.14 ≈ ContextMean 0.15), whereas STATE
clearly beats it — i.e. on this single-pert task the causal WMs add little over the mean-
perturbation prior, while STATE captures real per-pert specificity. (4) **PerturbMean** is
weak here because only ~169/968 held-out K562 perts have a cross-cell-line signal to borrow.

## Zero-shot regime — STATE `zeroshot/k562` (never saw K562) vs WMs on their held-out perts

`merge_and_plot.py --regime zeroshot` (STATE row from `zeroshot/k562/eval_best.ckpt`, scored
vs the zeroshot `adata_real`). Note the regimes differ: STATE-zeroshot is **cell-line**
zero-shot (harder — never saw K562 at all); the WMs are **pert** zero-shot (held-out pert,
but trained on the K562 cell line).

| method | pearson_delta | DE-direction | discrimination(cos) |
|---|---|---|---|
| **STATE-zeroshot** | **0.21 / 0.20** | 0.64 / 0.64 | **0.74 / 0.71** |
| exp13 (K562-only) | 0.14 | 0.57 | 0.54 |
| exp16 (4-line) | 0.07 | 0.56 | 0.53 |
| ContextMean | 0.15 | 0.61 | 0.50 |
| PerturbMean | 0.04 / 0.06 | 0.55 | 0.54 / 0.56 |

**Takeaways.** STATE-zeroshot drops on pearson-delta (0.32→0.21, the cost of never seeing
K562) but **still edges the causal WMs and keeps high discrimination (~0.73)** — it retains
TF-specificity zero-shot. The causal WMs **do not beat ContextMean** on pearson-delta
(exp13 0.14 < ContextMean 0.15) and sit **at chance on discrimination** (~0.54 ≈ ContextMean
0.50) — i.e. like the Joung exp17 held-out eval, the causal WMs predict a near-generic
perturbation response with little per-perturbation specificity, while STATE's specificity
survives even the harder cell-line-zeroshot setting. Figures: `state_comparison.png`
(fewshot), `state_comparison_zs.png` (zeroshot).

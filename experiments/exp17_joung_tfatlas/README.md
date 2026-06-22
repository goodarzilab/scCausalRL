# exp17 — Joung TF-atlas world model (shift-encoder ablation)

Causal-VAE world model trained on the **Joung TF-atlas** (single-TF overexpression in hESCs).
The model trains on **single** TF over-expressions and is asked, zero-shot, to (a) predict
**held-out** TFs and (b) compose singles into **combos/triples** via the do-operator.
Everything is scored on **Δ = mean(perturbed) − mean(hESC control)** (raw expression is dominated
by the shared basal transcriptome and is uninformative).

## v2 ablation grid (`configs/ablations/`, `scripts/train_ablation.sbatch`)

| run | input | residual-MSE target | contrastive | latent |
|---|---|---|---|---|
| orig | control-PCA | ✗ | ✗ | 64 |
| esm | esm_pca100 | ✗ | ✗ | 64 |
| esm_resid | esm_pca100 | ✓ | ✗ | 64 |
| esm_resid_con | esm_pca100 | ✓ | ✓ | 64 |
| esm_con | esm_pca100 | ✗ | ✓ | 64 |
| esm_l128 / esm_con_l128 | esm_pca100 | ✗ | ✗ / ✓ | 128 |
| esm_resid_l128 / esm_resid_con_l128 | esm_pca100 | ✓ | ✗ / ✓ | 128 |

Submit: `CFG=configs/ablations/<run>.yaml TAG=v2_<run> sbatch scripts/train_ablation.sbatch`

## Evaluation (`evaluation/`, CPU sbatch in `scripts/`)

- **`heldout_zeroshot_eval.py`** — held-out TF singles. Reports **mutilation** (shift-native, headline)
  and **pure-additive** (`add_*`) predictors.
- **`combo_triple_eval.py`** — singles→combo/triple composition; **do-op** (mutilation) vs **additive** (Σ predicted singles).
- **`baseline_eval.py`** — non-learned baselines on the same metrics: **context_mean** (predict control → Δ=0),
  **perturb_mean** (predict mean training perturbation → Δ=drift), and **additive** (Σ *observed* single-TF Δ; combos/triples only).
- **`plot_summary.py`** — builds the panels in `../figures/joung/exp17_*.png`.

**Metrics** (all over the Δ/HVG axis, averaged over TFs): `pearson_delta`, `de_direction`,
`effect_spearman` (effect-size rank corr), `deg_auroc` (recover DEGs |Δ_obs|>0.1 using |Δ_pred|),
`discrimination_cos`/`retrieval_top1` (retrieval specificity), developmental `fate_*` (Cao-mouse NNLS).
Each also has a drift-removed **`*_resid`** variant (drift = mean training-TF Δ, leak-free).

## Key findings (epoch-60 checkpoints)

**Raw metrics are saturated by the drift baseline** — `perturb_mean` scores pearsonΔ **0.736**,
DEG-AUROC **0.950**, effect-Spearman **0.490**, all ≥ every model. So raw correlation/DEG/ranking is
*not* where the model earns its keep; it's the shared TF-overexpression direction.

**The model's value-add is drift-removed specificity** (baselines are at chance/zero by construction):

| run | pearsonΔ | pΔ_resid | deg_auroc | effect_spear | disc_resid |
|---|---|---|---|---|---|
| orig | 0.699 | 0.081 | 0.941 | 0.436 | 0.511 |
| **esm** | 0.729 | **0.238** | 0.944 | 0.461 | 0.591 |
| esm_resid* | 0.383 | −0.202 | 0.883 | 0.226 | 0.500 |
| esm_resid_con* | −0.087 | −0.196 | 0.875 | −0.027 | 0.501 |
| **esm_con** | 0.674 | 0.050 | 0.921 | 0.380 | **0.615** |
| BASE perturb_mean | 0.736 | 0 | 0.950 | 0.490 | 0.500 |

- **ESM input alone (orig→esm) helps** the residual signal a lot (pΔ_resid 0.081→0.238, disc_resid 0.51→0.59).
- **`esm_con` (ESM + contrastive, no residual target) is the winner** on discrimination (0.615).
- **Residual-*target* training (`*`) is unstable** — collapses by epoch 60 (negative pearson) and early-stops at
  epoch ~11; it also caused the latent-128 SCM solve to go singular (fixed via a spectral-norm clamp on A in
  `models/dag.py::sample_A`).

**Combos** (n=42): the **additive-of-observed-singles null = pearsonΔ 0.253** beats every learned do-op
(best `esm_con` do = 0.125); no model achieves combo-level discrimination (≈0.50). Composition is still weak.

Figures: `../figures/joung/exp17_{singles_panels,combos_panels,pairs_vs_triples,per_tf_dists}.png`.

# Singles→combos/triples generalization eval (Joung TF-atlas WM)

The core exp17 test: the model trains on **single** TF overexpressions; can the **do-operator**
compose single-TF shifts into correct **combo/triple** predictions, zero-shot, and does
coherent composition beat **additive**? Script: `evaluation/combo_triple_eval.py`
(`scripts/combo_eval.sbatch`, high-mem node — densifies the combo atlas).

For each held-out multi-TF set (real TFs only; GFP/mCherry fillers dropped; ≥20 cells, ≥2
on-axis TFs): observed Δ = mean(combo) − control; **do-op** Δ = decode(mutilation-solve(Σ
shift(TFᵢ))) − ctrl_recon (set-dependent u); **additive** Δ = Σ single-TF predicted Δ.
Metrics: gene-space trio + identity-space fate (Cao-mouse NNLS, nuisance fit on training
singles), split pairs vs triples.

## Results — epoch80 (40 pairs, 2 triples)

| set | predictor | pearson_delta | DE-direction | discrimination_cos | fate_top1 |
|---|---|---|---|---|---|
| pairs (40) | **do-op** | **0.091** | 0.580 | 0.497 | 0.000 |
| pairs (40) | additive | 0.071 | — | 0.499 | — |
| triples (2) | do-op | 0.079 | 0.579 | 0.500 | — |
| triples (2) | additive | 0.071 | — | 0.500 | — |

**Read.**
- **do-op > additive on pearson-delta** (0.091 vs 0.071, ~28% relative) — a small but
  consistent edge for coherent composition over naive sum-of-shifts. This is the only
  positive epistasis signal.
- **Discrimination ≈ 0.50 (chance)** and **fate_top1 = 0** — the model cannot distinguish
  combos, same as the singles eval; predicted combo profiles are near-generic.
- **Combo pearson-delta (0.09) ≪ singles (0.68)** — combo generalization is far weaker; the
  do-operator transfers little TF-specific structure into the combo regime at this stage.
- Triples n=2 — not interpretable.

## Caveats
- **epoch 80/150** — mid-training; re-run on the final checkpoint
  (`CKPT=... TAG=epoch150 sbatch scripts/combo_eval.sbatch`).
- **Cross-experiment baseline**: the combo atlas (GSE217066) is a different experiment from
  the singles atlas (GSE217460); observed combo Δ is anchored to the singles-atlas hESC
  control, so a batch/assay offset can deflate absolute pearson. The do-vs-additive
  comparison is internal (same observed target) and unaffected.

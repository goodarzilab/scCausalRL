# Held-out zero-shot eval (Joung TF-atlas WM)

Reusable eval of a Joung world-model checkpoint on held-out (seed-42 test) TFs, predicted
**zero-shot** via the shift encoder. All metrics on Δ = mean(OE) − mean(hESC control).

Run (high-mem node — `load_joung` densifies ~1.14M×9923):
```
sbatch scripts/heldout_eval.sbatch            # defaults to phase1_epoch80, tag=epoch80
CKPT=.../phase1_epoch150.pt TAG=epoch150 sbatch scripts/heldout_eval.sbatch
```
Outputs per tag: `summary.json`, `per_tf.csv`, `profiles.npz`.

## Metrics

**Gene-space trio** (Δ over the 9923-HVG axis)
- `pearson_delta` — corr(Δ_pred, Δ_obs), mean over held-out TFs. Generous floor: the shared
  "leaving-pluripotency" drift inflates it for everyone.
- `de_direction` — sign agreement on genes that move (|Δ_obs|>0.1). Robust but easy (most
  TF-OE hits are inductions).
- `discrimination_cos` — **load-bearing**: is Δ_pred_i closest (cosine) to its OWN Δ_obs_i
  among all held-out TFs? Reported as mean fraction-of-others-beaten + `retrieval_top1`.
  A generic-drift predictor scores ~0.5 / ~1-of-N here (chance).

**Identity-space fidelity** (developmental fate via the Cao-mouse NNLS panel; nuisance axes
fit on TRAINING TFs only → leak-free)
- `fate_top1` — predicted dominant cell type == observed dominant cell type (chance ≈ 1/20).
- `fate_betacos` — cosine of full fate-mixture β_pred vs β_obs.
- `fate_entropy_r` — corr of commitment strength H(β) across TFs.

## Results

| tag (epoch) | pearson_delta | de_direction | discrimination_cos | retrieval_top1 | fate_top1 | fate_betacos |
|---|---|---|---|---|---|---|
| epoch80 | 0.680 | 0.831 | 0.508 | 0.006 | 0.030 | −0.008 |

**epoch80 read:** high gene-space floor but **chance-level discrimination + identity** — the
model predicts the shared differentiation drift for every held-out TF (72/169 → "Early
mesenchyme") and has no TF-specific zero-shot signal. Mid-training (80/150); re-run on the
final checkpoint to see if specificity emerges.

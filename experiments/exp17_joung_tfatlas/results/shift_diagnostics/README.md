# Shift diagnostics — localizing the discrimination failure

`evaluation/shift_diagnostics.py` (sbatch `scripts/shift_diag.sbatch`, CPU high-mem).
Reusable on any checkpoint: `CKPT=... TAG=... sbatch scripts/shift_diag.sbatch`.

## epoch80 result (verdict)

| | train | test | chance |
|---|---|---|---|
| D1 model retrieval_top1 | 0.012 | 0.006 | 0.006 |
| D1 model discrimination | 0.514 | 0.508 | 0.50 |
| D2 shift pairwise-cos | 0.851 | 0.792 | — |
| D3 ridge fingerprint→Δ top1 | 0.136 | 0.006 | 0.006 |

**Verdict:** NOT just a generalization failure — the model is at chance even on TRAIN TFs
(D1), so it cannot represent TF-specificity in-sample. Two compounding bottlenecks:
1. INPUT (D3): the R^50 control-coexpression fingerprint carries no generalizable TF-identity
   (ridge: 0.136 in-sample → chance held-out). A TF's OE effect (motif/targets) isn't in
   baseline coexpression. → needs ESM/regulon descriptor.
2. PIPELINE/OBJECTIVE (D1 vs D3): ridge-on-fingerprint (0.136) BEATS the full model (0.012)
   in-sample → the SCM+linear-decoder + MSE-on-mean-Δ loss washes out signal; nothing rewards
   discrimination. → needs the contrastive loss.
Both fixes are under test in the C1 ablation grid (esm/regulon descriptor + InfoNCE).

# Why shift-magnitude + linear decoder are STILL load-bearing (and A is not useful)

Two probes on the trained checkpoints (`evaluation/component_probe.py`, `evaluation/gradient_flow.py`),
run 2026-07-05. Component probe = `1−cos(Δ_real, Δ_variant)` on the RAW do-op Δ (higher ⇒ that
component controls more of the prediction). Gradient flow = share of the held-out pert-Δ MSE
gradient landing on each block (learning pressure).

| arm | gate_open | probe_A (A≡0) | probe_Dperm | probe_shiftID | crossTF_div | dag_grad% | shift_scale_grad% |
|---|---|---|---|---|---|---|---|
| min_esm (dense S) | 1.00 | 0.014 | **1.01** | 0.001 | 0.001 | 5.4% | 47% |
| sparse_esm (gate collapsed) | 0.02 | 0.007 | **1.01** | 0.000 | 0.000 | 4.6% | 27% |
| sparse_con (healthy gate) | 0.75 | 0.156 | **0.98** | 0.018 | 0.018 | 9.2% | 33% |
| sparse_pertonly (open) | 0.79 | 0.459 | **0.99** | 0.018 | 0.019 | 18.3% | 28% |

## Finding 1 — the linear decoder absorbs the SCM operator (probe_Dperm ≈ 1.0 in EVERY arm)
`Δ_gene = D · Δz`. Shuffling D's gene-rows makes the prediction orthogonal to the real one
(1−cos ≈ 1.0) for every arm — D is the sole map to gene space and it alone determines the pattern.
Formally the model is degenerate: define `D_eff = D·(I−diag(1−u)Aᵀ)⁻¹`; since the operator is
invertible, for ANY A there is a D giving identical predictions (`D = D_eff·(I−MAᵀ)`). So a
**sparse** latent shift decoded through a **dense decoder column** already yields a dense gene
response — A is never needed to spread signal. **The L0 gate sparsified S but never removed D's
freedom to bypass A.** This is the escape hatch, and it is structural, not a loss-tuning issue.

## Finding 2 — the shift is load-bearing for MAGNITUDE, not IDENTITY (mode collapse)
`probe_shiftID ≈ crossTF_div ≈ 0.00–0.02`: swapping in another TF's shift barely changes the
prediction, and distinct TFs' predictions are ~98–100% identical. The model predicts ONE shared
direction scaled by size. The gradient confirms it: **a single scalar `shift_scale` absorbs 27–47%
of the entire pert-loss gradient** — the dominant lever for cutting error is global magnitude, not
per-TF direction. This mode collapse (not A) is what pins discrimination at chance. InfoNCE (on in
con/pertonly) lifts diversity only from ~0.001 → ~0.018 — present but ~20× too weak to matter.

## Finding 3 — A is now coupled but not useful
`probe_A` tracks gate openness (collapsed gate 0.007 → healthy gate 0.16–0.46) and dag receives
real gradient only where the gate is open (pertonly 18%, con 9% vs esm/min ~5%). So Fix 2 did
mechanically wire A into the loss. But (a) at convergence D carries the pattern (probe_Dperm≈1),
(b) A's contribution lands in the shared/magnitude direction that drift-removal and discrimination
are insensitive to (hence `S_Agap≈0`, no disc gain). A is *used*, redundantly with D, in a
direction that doesn't help.

Note: gradient magnitude ≠ load-bearing. decoder.D shows small grad (6–8%) because it is CONVERGED,
yet is maximally load-bearing (probe 1.0). Load-bearing = "shuffling it breaks the output"; gradient
= "the loss still wants to move it." D scores high on the first, low on the second.

## Should we penalize shift / decoder?  NO — penalties can't fix this.
- **Shift-magnitude penalty (shift_l1):** already tried in exp19 — fought the MSE and killed
  prediction without helping A. Magnitude is legitimately the real effect size; don't punish it.
- **More decoder penalty:** `lambda_D_sparse` is already 30 and `lambda_ortho` 1. Shrinking D does
  not break the D↔A absorption degeneracy (D stays free and full-rank enough to absorb the operator).
  A cannot become identified by penalizing D.

## The real levers (→ Phase 2)
1. **Freeze the decoder (frozen-GO D, `requires_grad=False`).** Removes D's absorption freedom →
   `D_eff` can no longer be anything → A becomes identified and MUST carry the propagation. This is
   the direct fix for Finding 1, now strongly motivated by probe_Dperm≈1.0.
2. **Attack mode collapse directly (Finding 2, orthogonal to A).** crossTF_div≈0.02 is the
   discrimination ceiling. Whiten/normalize the pert target so global magnitude isn't the easy win
   (starves the `shift_scale` shortcut), and/or a much stronger identity contrastive. Without this,
   even a perfectly-identified A won't raise discrimination.

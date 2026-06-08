# scCausalRL experiments — overview & narrative

A diagnostic-driven arc. Each experiment was motivated by what the previous one's
diagnostics revealed. All held-out numbers are downstream-DEG AUROC on the
Replogle K562 test split (~190 perts), no leakage, vs honest baselines
(correlation ≈0.56, co-expression-neighbour ≈0.66).

## The arc in one line
Baseline leaked → replace leak with a **learned shift encoder** (works, ~0.70) →
diagnose that the **SCM `A` is bypassed** (shift does everything, A collapses) →
**exp8** revives A → diagnose that the **GRN just copies the CollecTRI prior** →
**exp9** tests copying vs learning.

---

## exp1 — baseline (the leaky causal-VAE)
The original pathway lifted a gene-space effect to latent space via `dz = D⁺·dx`,
feeding the observed differential expression as input → the prediction was the
answer projected back. Honest re-eval (Task-A V1/V2, non-leaky input) on the same
held-out split: **AUROC ≈ 0.495 (chance)**, below the correlation baseline. So the
old pathway has no real zero-shot signal — the leak was doing all the work.

## exp2–7 — the shift encoder (the fix for the leak)
Replaced `D⁺·dx` with a **learned perturbation shift** (SCCVAE-style):
`Sᵖ = c·ShiftEncoder(pert_label[g])`, where `pert_label` is the gene's control
co-expression PCA (exists for unseen genes → zero-shot, no leak). Injected
additively into the SCM: `z = (I−Aᵀ)⁻¹(ε + Sᵖ)`. Six variants:

| exp | decoder | pert loss | Replogle | Norman |
|---|---|---|---|---|
| 2 | linear | MSE+MMD | **0.704** | **0.725** |
| 6 | linear | MSE+MMD (re-run) | 0.702 | 0.720 |
| 4 | linear | +energy | 0.696 | 0.717 |
| 7 | linear | cosine+MMD | 0.684 | 0.702 |
| 5 | MLP | +energy | 0.660 | 0.656 |
| 3 | MLP | MSE+MMD | 0.618 | 0.615 |

**Verdict:** linear MSE+MMD wins (~0.70), beats both baselines in-dist AND
cross-dataset/cross-modality on Norman (0.725 vs corr 0.61). Energy redundant
with MMD; cosine no better; **MLP decoder generalizes worse** than the
interpretable linear one. First genuine, non-leaky signal in the project.

---

## The diagnostics that motivated exp8 + exp9
Built a portable DAG-analysis notebook (`notebooks/dag_analysis.ipynb`) +
a gradient-flow probe (`dag_grad_diagnostic.py`) on exp2:

1. **GRN recovers CollecTRI at AUROC 0.96** — suspiciously high.
2. **But the SCM `A` is bypassed**: A's contribution to the prediction
   `‖z_post−(ε+S)‖/‖ε+S‖ ≈ 1e-5`, `dag.weights ≈ 1e-5`, spectral radius ≈ 0.
   The prediction is entirely `delta = D·Sᵖ`; the causal graph does nothing.
3. **Root cause:** `Sᵖ` is a free, unregularized path while `A` is shrunk by
   `lambda_sparse` (mean|A|, λ=10 — the dominant force; `lambda_spectral` is a
   relu(σ−0.95) ceiling, inactive at small A). The optimizer takes the free path
   and lets A decay.
4. **The 0.96 is a supervised tautology:** `biological_prior_loss` (margin-ranking,
   λ_prior=10) pushes *every* CollecTRI edge up on `G_feat=D·A_prob·Dᵀ` each step,
   and the imprint lands on the **gate topology θ** (prior-edge median 0.69 vs
   background ≈0, p≈0) — while the SCM weights are dead. The "recovery" measures
   the prior we supervised on, not discovery.

This split the problem in two → exp8 (make A do causal work) and exp9 (is the GRN
copying the prior or learning structure).

---

## exp8 — DAG-revival grid (make A useful, not bypassed)
**Motivation:** revive `A` so it carries the prediction (prerequisite for RL,
which plans *through* A). **Grid:** `pert_shift_l1 ∈ {0,0.1,0.5}` (L1 penalty on
the shift — force it through A) × `lambda_sparse ∈ {10,0}` (stop shrinking A).
Base = exp2 recipe (linear, MSE+MMD).

| cell | shift_l1 | λ_sparse | AUROC | A-bypass | mean‖S‖ | read |
|---|---|---|---|---|---|---|
| sl0_ls10 (≈exp2) | 0 | 10 | 0.709 | 1e-5 | 0.56 | good pred, **A dead** |
| **sl0_ls0** | 0 | 0 | 0.708 | **0.42** | 1.54 | ✅ A alive + good pred |
| sl0.1_ls10 | 0.1 | 10 | 0.485 | 0.01 | 0.00 | degenerate |
| **sl0.1_ls0** ★ | 0.1 | 0 | 0.697 | **0.45** | 0.08 | ✅ A alive, sparse shift, good pred |
| sl0.5_ls10 | 0.5 | 10 | 0.499 | 1e-5 | 0.00 | dead |
| sl0.5_ls0 | 0.5 | 0 | 0.489 | 0.45 | 0.00 | shift crushed → predict-control |

**Verdict — FIXED:** `lambda_sparse=0` revives A (bypass **1e-5 → ~0.45**, A now
does ~45% of the prediction; spectral 1e-5 → ~0.85) **without losing AUROC**
(~0.70). `shift_l1=0.5` is too strong (zeros the shift → predict-control collapse,
AUROC 0.49). **`sl0.1_ls0`** is the cleanest causal cell: a small (0.08) shift
*propagated through an active A*, AUROC 0.70. We now have a genuine causal world
model. (Figure: `figures/dag_revival_fix.png`.)

---

## exp9 — prior-generalization grid (copying vs learning?)
**Motivation:** the 0.96 CollecTRI AUROC is supervised. Does `G` learn
**transferable** regulatory structure, or just memorize the supervised edges?
**Key design:** hold out **whole TFs** (not random edges — random holdout leaks a
TF's out-degree to its held-out edges; TF-holdout forces cross-TF transfer).
Deterministic per-TF hash split shared by training + eval
(`utils/bio_priors.split_collectri_edges`). Cells: **C1** λ_prior=0 (unsupervised
floor), **C2** ★ hold out 30% of TFs, **C4** big-init probe. Base = the A-active
`sl0.1_ls0` (re-run after the first attempt mistakenly used the degenerate sl0.5).

First (degenerate-base) signal:
- **C1 unsupervised:** GRN recovers CollecTRI at **0.573 vs corr 0.535** — data
  alone barely beats correlation. So the structure is mostly prior-driven.
- **C2 held-out-TF:** held-out-TF AUROC **0.92** vs correlation floor 0.53 →
  the prior-shaped GRN **generalizes to TFs it never saw** (not pure memorization).
  *Caveat:* TF-holdout controls out-degree leakage but not shared-*target*
  in-degree leakage — read as "suggestive of generalization." Being re-confirmed
  on the A-active base.

**Status:** re-running C1/C2/C4 on `sl0.1_ls0` with Replogle + Norman evals + the
held-out-TF recovery notebook. Verdict pending (C2 ~30h).

---

## Where this leaves the RL (gated)
We finally have a world model where (a) prediction beats baselines (~0.70, +Norman
transfer) and (b) the SCM A is dynamically active (`sl0.1_ls0`). That clears the
"is the world model steerable" gate the RL was blocked on. The exp9 verdict (does A
encode real, not copied, structure) + the RL plan (keep u-score for multi-step
counterfactual rollout, feed it from the shift, validate on Norman combos) are the
next steps. See the RL strategy in the plan file.

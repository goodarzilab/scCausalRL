# exp12 — shift-encoder-fed mutilation training (Tier-2)

**Motivation.** The combo analysis (exp8) showed the linear additive shift is
*provably additive* (no epistasis), and the only non-additive operator (u-score
mutilation) *hurt* — but that was a **train/eval mismatch**: the model was trained
with the additive shift and the mutilation was bolted on at eval. The historical
"u-score didn't work" failure was the **`D⁺·dx` input**, not the mutilation
**operator** — the operator was never trained with a good (zero-shot, leak-free)
input. This grid does exactly that: **shift encoder (input) + mutilation do-operator
(propagation)**, trained together. It is the only coherent test of whether the
causal `do` captures epistasis, and is the RL world-model step.

**Forward (single-gene training; combo emerges at eval):**

    S        = c · ShiftEncoder(pert_label[g])          (learned, zero-shot input)
    z_target = z_prev + (I − A^T)^{-1} S
    u        = clip(L[g], u_max)   OR   min(1, β|S|)     (which programs to clamp)
    ε̃        = u ⊙ z_target + (1 − u) ⊙ ε
    z_post   = (I − M A^T)^{-1} ε̃,   M = diag(1 − u)     (per-sentence mutilated solve)

Loss = MSE(mean shift) + MMD(sets) — same as the additive shift loss, only the
operator changes. Base recipe = exp8 `sl0_ls0` (linear decoder, MSE+MMD,
lambda_sparse=0, pert_shift_l1=0, lambda_spectral=0.1, lambda_prior=10).

## Cells (4 = {Replogle, Norman} × {loading clamp, shift clamp})

| Cell | dataset | n_hvg | u_mode | params |
|---|---|---|---|---|
| `exp12_repl_loadclamp`    | Replogle K562_ess | 10000 | **loading** `u=clip(L[g],u_max)` | u_max=0.5 |
| `exp12_repl_shiftclamp`   | Replogle K562_ess | 10000 | **shift** `u=min(1,β|S|)` | β=1.0 |
| `exp12_norman_loadclamp`  | Norman (CRISPRa)  | 8000  | **loading** | u_max=0.5 |
| `exp12_norman_shiftclamp` | Norman (CRISPRa)  | 8000  | **shift** | β=1.0 |

**u_max choice.** The eval-time sweep favored *low* u_max (0.1) but only marginally
(0.693 vs 0.680) and on the *untrained* operator. For **training**, too-low u_max
caps how strongly programs pin → the perturbed distribution stays ≈(1−u)·control and
the MMD term can't fit the spread. So u_max=0.5 (the "keeps A involved" default); a
training-time u_max sweep is a follow-up.

## Read-outs (coherent eval = mutilation forward, `run_shift_mutilate_eval.py`)
- **Held-out single AUROC** via the *same* mutilation forward (`intervene_shift_mutilate`).
  Must stay ≈ the additive shift's ~0.70 (Replogle) — does the operator predict singles?
- **Norman combos**: mutilation `shift_uscore` vs `shift_add` vs `additive_obs` vs corr,
  **stratified by epistasis tertile**. The hypothesis: a *trained* mutilation operator
  beats additive specifically on high-epistasis combos (the test the eval bolt-on couldn't fairly run).

## Code (vs exp8)
- `training/losses.py:perturbation_set_loss_shift_mutilate` — batched per-sentence
  mutilated solve `(I − M A^T)^{-1}`, two u-modes.
- `models/causal_vae.py:intervene_shift_mutilate` — single/combo eval forward (shared semantics).
- `training/train_vae.py` — dispatch on `phase1.pert_loss_mode == 'shift_mutilate'`
  (+ `pert_mutilate_u_mode/u_max/beta`). Default `'shift'` ⇒ identical to exp8.

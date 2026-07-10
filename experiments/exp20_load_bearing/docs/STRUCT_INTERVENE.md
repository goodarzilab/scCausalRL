# Fix 1 + Fix 2 — structural intervention + sparse target gate

The core load-bearing fix. Two coupled changes that force downstream perturbation effects to
propagate through A and make perturbation identity matter.

## Fix 1 — single structural solve, shift into exogenous (`pert_intervene_mode: structural`)

**Legacy (exp19, the leak):**
```
z_S     = (I − Aᵀ)⁻¹ S              # shift pre-propagated
z_target = z_prev + z_S
ε̃       = ε_ctrl + u ⊙ (z_target − ε_ctrl)   # full target injected
z_post  = (I − diag(1−u)Aᵀ)⁻¹ ε̃
```
`z_target` already contains the propagated shift, so `ε̃` carries the intended downstream state and
`A=0` reproduces `z_post` → A inert.

**Structural (exp20):**
```
ε̃      = ε_ctrl + S                # shift injected ONLY into exogenous noise (dense-S caveat → Fix 2)
z_post = (I − diag(1−u)Aᵀ)⁻¹ ε̃     # ONE mutilated solve does the propagation
```
Row equation: `z_j = (ε+S)_j + (1−u_j) Σ_i A_ij z_i`. A non-target program (`S_j=0, u_j≈0`) moves
**only** via `Σ_i A_ij z_i` — exactly 0 at `A=0`. Implemented in both the training loss
(`losses.py::perturbation_set_loss_shift_mutilate`) and the model op (`causal_vae.py::intervene_shift_mutilate`),
mode-gated so `legacy` reproduces exp19.

## Fix 2 — sparse target gate (L0, not L1-on-magnitude) (`use_target_gate`, `pert_target_l0`)

Fix 1 alone is **insufficient**: `S` is dense (the shift encoder emits a value for every program),
so `ε_ctrl + S` still sets every exogenous node and A isn't needed. Fix 2 makes `S` sparse.

- A **second head** in `ShiftEncoder` (`gate_out`, sharing the trunk, reading the same ESM label →
  **zero-shot** for held-out TFs) emits per-program gate logits.
- Applied at the single choke point `CausalVAE.compute_shift`: `shift = shift_scale · (gate · raw)`.
  Every S-producer routes through here → S is gated everywhere with no caller edits.
- **Soft (train) / hard (eval)** via `self.training` (loaders call `model.eval()`), mirroring the
  DAG edge gates: train `g=σ((ℓ+Gumbel)/τ)`, eval `g=1{ℓ>0}`.
- **Dense-open init** (`zeros_(weight)`, `bias=+3` → σ≈0.95): starts at the proven dense-shift
  baseline; the L0 term carves sparsity.
- **L0 penalty** = `mean σ(ℓ/τ)` (expected open-gate probability), weight `pert_target_l0`. Unlike the
  failed `shift_l1` (which penalized *magnitude* and fought MSE), L0 penalizes gate *existence* —
  on-target magnitude stays free.

## Why the two together force A (the `u_mode='shift'` coupling)
With `u = clamp(β|S|)`, a sparse `S` ⇒ sparse `u` ⇒ the mutilation set `diag(1−u)` is confined to the
**target** programs (they get pinned, classic do), leaving all non-target programs solved through A.
Since observed TF Δ is dense (~474 DEGs), fitting it from a sparse injection **requires** A to
propagate → A becomes load-bearing by construction.

## Unit-verified (before training)
With a manually-sparsified hard gate + structural mode, on a random-init model:
```
target dims = [0, 3]
NON-target Δ:  with A = 0.381   |   A=0 = 0.00000
A=0 identity:  z_post0 == ε + S  (exact)
```
i.e. removing A kills all non-target movement — the mechanism is correct. The magnitude of `A_gap`
on a *trained* model (after L0 sparsifies the gate) is the Phase-1 result.

## Files
`models/shift_encoder.py` (gate head + `sample_gate`/`gate_prob`), `models/causal_vae.py`
(`compute_shift` choke, `compute_gate_prob`, structural branch in `intervene_shift_mutilate`,
`pert_intervene_mode`/`use_target_gate` ctor args), `training/losses.py` (structural branch +
`target_l0_weight`), `training/train_vae.py` (threading), `evaluation/agap_probe.py` (the A_gap check).

## Phase-1 smoke result (3 epochs, A_gap probe) — 2026-07-02
| arm | gate_open | A_gap (all) | A_gap (non-target) |
|---|---|---|---|
| `struct_min` (Fix1 only, dense S) | 1.00 | **0.075** (≈inert) | n/a |
| `struct_sparse` (Fix1+Fix2 gate) | 0.405 | **0.277** | **0.331** |

Decisive: dense-S control leaves A ~inert (0.075); the sparse target gate makes A load-bearing
(A_gap 0.28/0.33), with the L0 penalty sparsifying the gate to 40% open in just 2 DAG-active
epochs. Mechanism confirmed through real training → full 60-epoch runs submitted. (A_gap here =
1−cos(Δ_A, Δ_{A=0}); the prediction-quality real/zero/shuffled test + discrimination come from the
canonical eval on the full checkpoints.)

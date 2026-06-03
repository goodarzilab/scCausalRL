# Response to Ivy's review

Thanks @ivyliu — every one of these findings is real and we arrived at the same diagnosis in parallel last week. This PR includes the fixes for the eval leakage + training leakage + the related ablations; the modality-token bug is acknowledged below and will be patched in the next push.

## 1. Single-pert leakage — confirmed, fixed at both eval and train

You're right: with `dx = de_single[gene] = pert_mean − ctrl_mean` passed through `D⁺·dx → SCM → D·z`, the prediction is `D·D⁺·dx + (small SCM correction)`. With `λ_ortho = 1.0` driving `DᵀD ≈ I·s`, this is a rank-64 orthogonal projection of the answer. We confirmed empirically — `evaluation/run_perturbation_eval.py` re-run on `phase1_promote/checkpoints/phase1_epoch150.pt` with three input modes:

| input_mode | held-out test AUROC | corr baseline | lift |
|---|---|---|---|
| `observed_full` (legacy, leaky) | **0.619** | 0.579 | +0.040 |
| `target_lfc` (Option A — leak one scalar at the target only) | **0.583** | 0.579 | **+0.004** |
| `canonical` (Option B — strict zero-shot, ±2.0 at target) | **0.587** | 0.579 | **+0.008** |

So the entire "+0.04 over correlation" headline was the projection identity reading back the leaked answer. Agreed — the model bar in the figure should not be framed as zero-shot.

**What's in this PR for it**

- New `models/dx_input.py` with `target_lfc` (lenient — leaks only the on-target scalar; target is then masked from scoring) and `canonical` (strict zero-shot with `pert_direction · canonical_magnitude` at the target) builders, used identically in eval and training.
- `evaluation/run_perturbation_eval.py` gains `--input_mode {observed_full, target_lfc, canonical}`, `--pert_direction`, `--canonical_magnitude`, `--a_mode {learned, zero, shuffled}`, `--gene_idx_mode {correct, random}`, `--n_ctrl_cells` (default bumped 50 → 500), `--n_ctrl_combo` (20 → 200), `--seed`. All routed through `evaluate_single_perts` and `evaluate_combo_perts`. `run_norman_validation.py` plumbs the same args.
- `training/losses.py:perturbation_set_loss_u` accepts `input_mode`, `aux_prop_weight`, `pert_direction_batch`, `canonical_magnitude`. **V1 (replacement)** — `input_mode='target_lfc'` builds sparse dx and uses it for `dz = D⁺·dx_input`; `match_delta` target stays as the full observed dx so the SCM is structurally required to bridge sparse→dense. **V2 (auxiliary)** — keeps the existing leaky path and adds a downstream-only (target-masked) `L_prop` term computed from the sparse-input forward pass: `L_total = L_pert + aux_prop_weight · L_prop`.
- Six retrains were submitted last night under both variants on all three datasets — Replogle K562_essential, Norman, Combined. Norman V1 and V2 have already finished (8h each); the Replogle + Combined runs are at ~14h / ~7h and should land within a day. Phase-2 eval will report A and B as separate columns plus the ablation suite.

## 2. Combo "fair but additive wins" — same diagnosis, more nuance

You're right that the singles inputs `dx_a, dx_b` (used by additive and by the model) are not held out, only the combo is. That part is fair. **But** the combo prediction in the model also goes through `D·D⁺·(dx_a + dx_b)`, which projects the sum of single-pert observed dx's into D's column space — biasing the model output toward the additive baseline by architecture. So the comparison is still confounded. The model's combo AUROC being slightly below additive doesn't only reflect biology; it reflects projection error stacked on top of biology.

Under the new combo path the two-gene `dx_input` is sparse (target_lfc or canonical), so the additive baseline degenerates to `ctrl + (one or two on-target values)` — its downstream-DEG AUROC should drop to ~0.5, and the model finally has a real bar to clear. Phase 2 eval will report combo numbers under both old and new input modes for direct comparison.

## 3. Control-only correlation baseline — agreed, will switch to LOO

You're right that `compute_correlation_matrix` only uses control cells, while the model has effectively seen the entire training perturbation atlas. The honest baseline is `gene-gene correlation across perturbations, held-out per gene` — a leave-one-perturbation-out construction. **Not in this PR**, but it's the right next comparator and I'll open a follow-up. Until then, the most apples-to-apples pair in the current figure is GRN (`D·A·Dᵀ`) vs correlation, which you already noted.

## 4. CRISPRi→CRISPRa mismatch — agreed; the Combined retrain is the experiment

Norman is CRISPRa, our promotion checkpoint was trained on Replogle CRISPRi. We add `pert_direction = +1 / −1` per dataset and the encoder receives a modality token (`n_modalities = 3` in the combined config: control / CRISPRa / CRISPRi). The combined V1/V2 retrains in flight are the first real test of whether joint training closes that gap; we'll have the comparison in the Phase-2 results.

## 5. The modality-token default-to-0 bug

Confirmed real in [models/encoder.py:73-75](models/encoder.py#L73-L75) — when `modality_id=None` is passed, the encoder defaults to a zero-tensor (modality 0 = control in load_combined's convention). Then `evaluation/run_perturbation_eval.py:265` calls `model.encoder(x_ctrl_t)` without a modality_id, and Norman cells get encoded in the "control" slot — for a CRISPRa-trained model this is silently wrong.

**Fix queued** for the next push (kept out of this PR so it doesn't get tangled with the leakage fixes):
- Add a `--modality_id` CLI to both eval scripts (auto from config: Replogle → CRISPRi = 2, Norman → CRISPRa = 1, default ctrl = 0).
- Plumb through `evaluate_single_perts` / `evaluate_combo_perts` / `predict_combo` → `model.encoder(..., modality_id=...)`.
- Make the default in `encoder.py` raise (or warn loudly) when `n_modalities > 0` but `modality_id is None`, so this can't silently regress again.

## 6. The diagnostic script

`scripts/diagnose_modality_transfer.py` is exactly the right shape. I'll integrate the three checks (u_clip_max sweep, direction-acc + magnitude-calibration on combos, modality-token probe) into the Phase-2 eval harness when the retrains finish. If the combo Pearson rises as `u_clip_max → 1` (DAG bypassed), that's a direct confirmation that the transferred graph is hurting — which would explain why the existing checkpoint's combo numbers underperform additive even under the leaky protocol.

## TL;DR — status

| Concern | Status |
|---|---|
| Single-pert leakage in eval | Fixed in this PR (`input_mode` flag + sparse builders) |
| Single-pert leakage in training | Fixed (V1 replacement) AND mitigated (V2 auxiliary); 6 retrains in flight |
| Combo additive ≥ model | Reported honestly; new combo eval mode strips the projection-identity bias; awaiting Phase-2 numbers |
| Control-only correlation baseline | Acknowledged; LOO across-pert baseline queued as a follow-up |
| CRISPRi/CRISPRa modality mismatch | Combined V1/V2 retrains in flight; modality-token bug to be patched in next push |
| Modality-token defaults to 0 silently | Bug confirmed; fix queued (load-bearing assertion + CLI + config-driven dispatch) |

Honestly: the framing in the previous slides was overclaiming and your review caught it. We were already digging into the same issue when this landed — happy to merge once the modality-token patch and a smoke test of the new eval modes on the Norman external slice are in. Open to splitting that into a separate follow-up PR if you prefer.

# exp20 — Making the latent DAG A load-bearing (Joung TF atlas)

Fork of `exp19_nonlinear_compose` (exp19 left untouched) implementing the architectural fixes
from `docs/JOUNG_A_LOAD_BEARING_INVESTIGATION.md`, which proved **A is inert** in exp19
(`A_gap ≈ 0` across every loss/regularizer). All changes are behind config flags whose defaults
reproduce exp19 exactly. Same canonical eval metrics as exp17/exp19 (`gene_space`/`wm_metrics`).

## The leak (why A was inert) and the fix
Legacy do-operator pre-propagates the shift `z_S=(I−Aᵀ)⁻¹S` and injects the full target into the
effective noise `ε̃=ε+u⊙(z_target−ε)`; and `S` is **dense** over all latent programs. So the
intended downstream state reaches every latent directly and `A=0` reproduces predictions.

- **Fix 1 (`pert_intervene_mode: structural`)** — single mutilated solve, shift into exogenous:
  `z_post = (I − diag(1−u)Aᵀ)⁻¹ (ε_ctrl + S)`. No pre-propagation, no `z_target`. See `NONLINEAR_SCM.md`? no — see `STRUCT_INTERVENE.md`.
- **Fix 2 (`use_target_gate: true`, `pert_target_l0`)** — sparse per-TF target gate (L0, not L1),
  so `S` is nonzero at only a few programs and A must propagate to the rest. See `STRUCT_INTERVENE.md`.
- **Recon variants** (`lambda_recon`, `residual_recon`, `pert_residual_target`) — see `RECON_VARIANTS.md`.
- **Phase 2 (staged)**: nonlinear structural equations (`NONLINEAR_SCM.md`) and frozen-GO decoder
  (`FROZEN_DECODER.md`), run only after Phase-1 shows `A_gap > 0`.

## New config flags (all default to exp19 behavior)
| flag | default | effect |
|---|---|---|
| `model.pert_intervene_mode` | `legacy` | `structural` = Fix 1 (single solve, S→ε) |
| `model.use_target_gate` | `false` | Fix 2 sparse target gate (2nd ShiftEncoder head) |
| `model.target_gate_tau` | 1.0 | Gumbel-sigmoid temperature |
| `model.target_gate_bias_init` | 3.0 | gate bias init (sigmoid≈0.95 → dense-open start) |
| `phase1.pert_target_l0` | 0.0 | L0 penalty weight on open-gate probability |
| `phase1.lambda_recon` | 1.0 | per-cell reconstruction weight (`<1` → push load onto SCM) |
| `phase1.residual_recon` | false | reconstruct `x−drift` (near-no-op for linear decoder) |
| `phase1.kl_free_bits` | 0.0 | KL floor (guards posterior collapse when `lambda_recon`<1) |
| `model.model_impl` | — | `exp20` marker (informational) |

## Phase-1 sweep arms (`configs/ablations/lb/`)
| arm | base | key diffs | purpose |
|---|---|---|---|
| `struct_min_esm` | esm | Fix1 only (dense S) | **control** — verify leak persists (A_gap≈0) |
| `struct_sparse_esm` | esm | +gate, l0=0.03 | core fix (no contrastive) |
| `struct_sparse_con` | esm_con | +gate, l0=0.03 | core fix (with contrastive) |
| `struct_sparse_resid` | esm_con | +`pert_residual_target` | + specificity |
| `struct_sparse_pertonly` | esm_con | +`lambda_recon=0.1`,`kl_free_bits=0.5` | pert-only ("not reconstructing cells") |
| `struct_sparse_residrecon` | esm_con | +`residual_recon`,`separate_pert_decoder` | residual-reconstruction |
| `struct_sparse_all` | esm_con | gate+resid+`lambda_recon=0.1` | kitchen sink |

## Run
- Smoke (CPU, 3 ep + A_gap probe): `sbatch --export=ALL,TAG=<arm> scripts/smoke_cpu.sbatch` (configs in `configs/ablations/lbsmk/`).
- Full (GPU, 60 ep): `sbatch --export=ALL,CFG=configs/ablations/lb/<arm>.yaml,TAG=<arm> scripts/train_ablation.sbatch`.
- A_gap probe: `python evaluation/agap_probe.py --checkpoint <ckpt>` (PYTHONPATH=exp20).
- Canonical eval: `evaluation/heldout_zeroshot_eval.py` (same metrics as exp17/19).

## Success criteria
- **Primary:** `A_gap > 0` (real-A beats zero-A and shuffled-A) — A load-bearing.
- **Secondary:** `discrimination_cos_resid` above the ~0.61 exp19 ceiling.
- **Control:** `struct_min` shows `A_gap ≈ 0` (dense S → leak persists) → confirms the sparse gate is the necessary piece.

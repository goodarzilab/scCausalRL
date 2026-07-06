# The Latent DAG Is Not Load-Bearing: A Full Investigation (Joung TF-Atlas World Model)

**Date:** 2026-07-02 · **Scope:** exp19 causal-VAE world model on the Joung TF atlas (hESC, TF overexpression).
**One-line result:** Across ~30 model variants (old + 27 new), the latent DAG **A contributes nothing to prediction** (`A_gap ≈ 0` everywhere). The model's prediction is carried by the **shift-encoder magnitude + the linear decoder**, driving a dominant shared differentiation program; the **causal graph A and the perturbation identity are both inert**. This is an architectural property (per-cell reconstruction), not a tuning gap.

---

## 1. The model and the question

**Architecture (shift-encoder causal-VAE):**
```
x → encoder → ε (exogenous noise)         [KL to N(0,I)]
z = (I − Aᵀ)⁻¹ ε                           [linear-Gaussian SCM; A = latent DAG]
perturbation:  S = ShiftEncoder(ESM(TF));  z_post = mutilated SCM solve with shift S
x̂ = D·z + b                                [linear decoder; GRN read out as G = D·A·Dᵀ]
```
- **A** = `gates · weights · mask`, gates `= sigmoid(θ/τ)` (edge existence), weights ∈ ℝ (signed strength).
- The **do-operator** injects the shift S and propagates it through the SCM solve. **The exact do-operator (shift-add + mutilation, and its linear vs non-linear variants) differs across runs — see §1.1, which is central to why A is inert.**
- The intended story: A encodes program→program regulation; interventions propagate through A; G = D·A·Dᵀ is a causal GRN.

### 1.1 The do-operator: shift-add + mutilation (and its linear / non-linear variants)

The perturbation pathway is a **shift-fed mutilated SCM solve**. Given a perturbed set of TF(s), the operator (`WorldModelOperator.apply` / the training loss `perturbation_set_loss_shift_mutilate` / the eval used in the sweeps) does:

```
1.  Shift        S      = Σ_k  c · ShiftEncoder(ESM(TF_k))          # summed latent shift (per-gene ESM descriptor → shift)
2.  Propagate    z_S    = (I − Aᵀ)⁻¹ S ;   z_target = z_prev + z_S  # shift pushed through the graph
3.  Strength     u ∈ [0,1]^d  (the MUTILATION gate; see variants below)
4.  Eff. noise   ε̃      = ε_ctrl + u ⊙ (z_target − ε_ctrl)  [+ nonlinear correction]   (mix_eps)
5.  Mutil. solve z_post = (I − diag(1−u) · Aᵀ)⁻¹ ε̃            # scm_intervene_u
6.  Decode       Δ      = D·z_post − D·z_ctrl
```

**The mutilation itself** (step 5): with `M = diag(1−u)`, column *i* of A is scaled by `(1−u_i)`.
- `u_i = 1` → row *i* of `(I − M Aᵀ)` becomes `eᵢ` ⇒ `z_post[i] = ε̃[i]` — the classical do-operator: program *i* is **pinned**, its incoming parent edges **severed**.
- `u_i = 0` → the original SCM at coordinate *i* (no intervention).
- All new exp19 runs cap `u_max = 0.5` (`pert_mutilate_u_max`) with `u_mode='shift'` (`u = clamp(β·|S|)`), so columns are only *partially* attenuated — the solve is *meant* to still route the shift through A.

**Linear vs non-linear — two independent knobs.** The solve in step 5 is *always* linear given `(A, u)`. Non-linearity enters only through how `u` and `ε̃` depend on the shift:

| knob | LINEAR setting | NON-LINEAR setting |
|---|---|---|
| **u** (`refine_u`, step 3) | `u = clamp(β·|S|)` — no MLP (`u_gate_mlp=False, u_gate_pure=False`) | `u_gate_mlp`: `u = clamp(u_base + tanh(MLP(S)))` (residual, zero-init) · `u_gate_pure`: `u = sigmoid(MLP(S))` |
| **ε̃** (`mix_eps`, step 4) | `ε̃ = ε_ctrl + u⊙(z_target − ε_ctrl)` — linear interpolation (`nonlinear_eps_mix=False`) | `+ tanh(eps_mix_net(shift))` correction (`nonlinear_eps_mix=True`) |

**Why the non-linearity exists — combo (epistasis) modeling.** With linear `u` + linear `ε̃`, a *combination* of perturbations gives `z_post ≈ z_ctrl + Σ_k dz_k` — i.e. **additive** in the single-perturbation shifts. The MLP corrections (`u_gate_*`, `eps_mix`) make `u` and `ε̃` depend **non-linearly on the whole set**, so `z_post` is **non-additive** across the combined TFs → the model can express epistasis. Both MLPs are **zero-initialized**, so training *starts exactly at the linear-clamp baseline* and learns a cross-program correction. Crucially, this is the only non-additive channel that keeps the decoder/GRN linear.

**Which mechanism each run uses** (verified from checkpoint/config):

| run family | do-op | `u` (refine_u) | `ε̃` (mix_eps) | net mutilation |
|---|---|---|---|---|
| **new: all grnfix / sena / epsd** | shift-add + mutilate | `u_gate_mlp` (non-linear residual `u`) | linear | **non-linear via u** |
| **v2_esm_con_ugate** (old best) | shift-add + mutilate | `u_gate_mlp` | linear | non-linear via u |
| v2_esm_con_epsmix | shift-add + mutilate | linear (`β|S|`) | `nonlinear_eps_mix` | non-linear via ε̃ |
| v2_esm_con_ugate_epsmix | shift-add + mutilate | `u_gate_mlp` | `nonlinear_eps_mix` | **fully non-linear** |
| v2_combo_resid(_hicon) | shift-add + mutilate | `u_gate_mlp` | `nonlinear_eps_mix` | fully non-linear (+ combo-trained) |
| v2_orig / v2_esm / v2_esm_con | shift-add + mutilate | linear clamp, no gates | linear | **linear mutilation** (baseline) |

*(All exp19 models use `pert_loss_mode='shift_mutilate'`, `u_mode='shift'`, `u_max=0.5`, `β=1.0`. Legacy exps used other paths — pure-additive shift, or threshold-mutilation `intervene_gene(threshold_frac)` — but none of the models in this report do.)*

**Why none of this rescues A (the connection to the main result).** Step 5 is the *only* place A could carry the intervention downstream — if `A=0`, `z_post = ε̃` (no propagation). But two things route the effect *around* A:
1. **The shift already contains the propagation:** `z_target = z_prev + (I−Aᵀ)⁻¹S` and `ε̃ = ε + u⊙(z_target − ε)` inject the *target* directly. So even with `A=0` in the mutilated solve, `z_post = ε̃` already carries the intended downstream state. This is exactly why **`A_zero ≈ A_real`** in the ablation.
2. **The non-additivity is carried by the MLP gates, not A:** `u_gate_*` / `eps_mix` produce the combo epistasis; A is not needed for it. So even *fully non-linear* mutilation (e.g. `v2_esm_con_ugate_epsmix`, `v2_combo_resid`) shows `A_gap ≈ 0` — confirmed in §4a/§4b. Non-linear ≠ load-bearing.

Consequently the load-bearing/component results (§4) hold **across the entire linear→non-linear mutilation spectrum**: linear-clamp (`v2_orig`), non-linear-u (`v2_esm_con_ugate`, all new runs), non-linear-ε̃ (`epsmix`), and fully non-linear (`ugate_epsmix`, `combo_resid`) all leave A inert.

**The question:** is A actually *used* — i.e., does removing/scrambling it change predictions? A causal graph that predictions don't depend on is decorative.

---

## 2. How we test "load-bearing" (methodology)

All metrics are **drift-removed** (residual): `Pr = pred − drift`, `Or = obs − drift`, where `drift = mean train-TF Δ` (the shared program). Raw metrics are uninformative because the shared program is ~99% of the signal.

1. **A-inertness (real / zero / shuffled A):** score the same model with the real A, with `A = 0`, and with A's columns shuffled. **A is load-bearing only if `real ≫ {zero, shuffled}`.** Metric: `A_gap = spear_real − max(spear_zero, spear_shuf)`.
   - `real ≈ zero` → A does nothing.
   - `real ≈ shuffled` → only A's scale/sparsity matters, not its structure.
2. **Component ablation:** shuffle/zero each of the three components — **A**, the **shift encoder**, the **decoder** — and measure the prediction drop. Biggest drop = most load-bearing.
3. **Canonical held-out eval** (`heldout_zeroshot_eval.py`, same script/split as exp17): `pearson_delta`, `pearson_delta_resid`, `discrimination_cos` (rank specificity, chance 0.5), `retrieval_top1` (chance 1/T), `deg_auroc_resid`.
4. **Data QC + discriminability ceiling:** is the data even discriminable (independent of any model)?

---

## 3. Everything we tried to make A load-bearing

**Sparsity / topology**
- Fixed the sparsity parametrization: **product-mode → gates-mode** (penalize edge *existence*, not weights). Product-mode was degenerate — it shrank weights to ~0 leaving gates dense.
- **Sparsity-strength sweep:** λ_sparse = 0, 2, 5, 10, 20 (`gatesp5/10/20`, `dagutil_splo`, `dagutil_sp0`, incl. **zero sparsity**).
- **Decoder sparsity:** standard (30) and high (60, `dagutil_hiD`).

**Explicit pressure to use A**
- **DAG-utilization loss:** directly penalize when `A=0` predicts as well as real A (weights 10, 30; `dagutil`, `dagutil_hi`).

**Starve the competing pathway**
- **Shift-encoder L1 bottleneck:** force effects to route through A rather than the shift (L1 = 0.1, 0.2, 1.0; `*_shiftl1`, `*_shiftl1_hi`).

**Retire the prior**
- **CollecTRI eval-only** (prior-off arms): the CollecTRI margin loss was shaping A *via the decoder* (memorization), not causally.

**Identifiability anchoring**
- **SENA GO:BP masked encoder:** anchor latent programs to named pathways so the encoder can't rotate them freely (leak λ = 0, 0.1; `sena*`).
- **ε-decorrelation + orthonormal decoder:** the covariance-factorization recipe — with `Cov(ε)=I` and orthonormal D, program correlations must flow through A (A becomes the Cholesky factor of the latent precision). λ_eps = 0.1/0.5/1.0 × λ_ortho = 1/10 (`epsdecorr*`).

**Combinations & controls**
- SENA+shiftL1, SENA+shiftL1−dagutil, SENA+ε-decorr+ortho, etc.
- `noDAG` (A≡0) control; verified A is non-degenerate at init, gradients flow through A, spectral-clamp keeps the solve stable.

**Result: every one failed — `A_gap ≈ 0` in all ~30 runs.** See §4.

---

## 4. Eval results

### 4a. Master table — all sweeps + old best (drift-removed)

`disc_cos_res` = specificity (chance 0.5); `A_gap` = `spear_real − max(zero,shuf)` (>0 ⇒ A load-bearing); `densA` = solve-A density.

| run | λ_sparse | enc | pearson | pear_resid | **disc_cos_res** | deg_auroc | retr_top1 | A_real | A_zero | A_shuf | densA | **A_gap** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OLD: v2_esm_con** | orig | mlp | 0.674 | 0.050 | **0.615** | 0.858 | 0.006 | 0.265* | 0.332* | 0.310* | 0.827 | **−0.067** |
| **OLD: v2_esm** | orig | mlp | 0.729 | 0.238 | 0.591 | 0.876 | 0.006 | — | — | — | — | — |
| **OLD: v2_orig** | orig | mlp | 0.699 | 0.081 | 0.511 | 0.865 | 0.006 | — | — | — | — | — |
| dagutil_hiD | 10 | mlp | 0.617 | 0.015 | 0.584 | 0.872 | 0.012 | 0.287 | 0.305 | 0.293 | 0.05 | −0.018 |
| gatesp5 | 5 | mlp | 0.614 | 0.016 | 0.577 | 0.867 | 0.012 | 0.275 | 0.308 | 0.281 | 0.07 | −0.033 |
| dagutil | 10 | mlp | 0.620 | −0.001 | 0.560 | 0.860 | 0.012 | 0.257 | 0.283 | 0.253 | 0.04 | −0.026 |
| gatesp20 | 20 | mlp | 0.606 | 0.007 | 0.556 | 0.866 | 0.012 | 0.269 | 0.284 | 0.262 | 0.02 | −0.015 |
| dagutil_hi | 10 (util 30) | mlp | 0.610 | 0.007 | 0.552 | 0.861 | 0.012 | 0.260 | 0.283 | 0.267 | 0.04 | −0.022 |
| gatesp10 | 10 | mlp | 0.605 | −0.003 | 0.550 | 0.862 | 0.012 | 0.267 | 0.285 | 0.274 | 0.04 | −0.018 |
| **dagutil_sp0** | **0** | mlp | **0.684** | −0.045 | 0.528 | 0.858 | 0.012 | 0.282 | 0.324 | 0.301 | **0.75** | −0.042 |
| dagutil_noprior | 10 | mlp | 0.657 | −0.076 | 0.511 | 0.864 | 0.018 | 0.292 | 0.292 | 0.292 | 0.00 | 0.000 |
| gates10_noprior | 10 | mlp | 0.648 | −0.079 | 0.511 | 0.871 | 0.012 | 0.302 | 0.302 | 0.302 | 0.00 | 0.000 |
| epsdecorr | 10 | mlp | 0.640 | −0.073 | 0.510 | 0.866 | 0.012 | 0.291 | 0.291 | 0.291 | 0.00 | 0.000 |
| sena | 10 | sena | 0.641 | −0.105 | 0.510 | 0.873 | 0.012 | 0.312 | 0.312 | 0.312 | 0.00 | 0.000 |
| epsdecorr_ortho | 10 | mlp | 0.636 | −0.094 | 0.509 | 0.874 | 0.012 | 0.323 | 0.323 | 0.323 | 0.00 | 0.000 |
| dagutil_splo | 2 | mlp | 0.625 | −0.102 | 0.509 | 0.880 | 0.006 | 0.320 | 0.331 | 0.309 | 0.01 | −0.011 |
| sena_leak0 | 10 | sena | 0.652 | −0.087 | 0.507 | 0.872 | 0.012 | 0.319 | 0.319 | 0.319 | 0.00 | 0.000 |
| epsdecorr_ortho_hi | 10 | mlp | 0.644 | −0.094 | 0.505 | 0.872 | 0.012 | 0.312 | 0.312 | 0.312 | 0.00 | 0.000 |
| **noDAG** | 0 | mlp | 0.680 | −0.108 | 0.503 | 0.866 | 0.006 | 0.309 | 0.309 | 0.309 | 0.00 | 0.000 |
| sena_epsdecorr | 10 | sena | 0.505 | −0.205 | 0.500 | 0.926 | 0.000 | 0.514 | 0.514 | 0.514 | 0.00 | 0.000 |
| sena_shiftl1 | 10 | sena | 0.543 | −0.206 | 0.500 | 0.931 | 0.012 | 0.520 | 0.520 | 0.520 | 0.00 | −0.000 |
| dagutil_shiftl1 | 10 | mlp | 0.444 | −0.205 | 0.500 | 0.930 | 0.012 | 0.517 | 0.506 | 0.475 | 0.04 | 0.011 |
| sena_shiftl1_splo | 2 | sena | 0.560 | −0.206 | 0.500 | 0.932 | 0.006 | 0.521 | 0.520 | 0.520 | 0.01 | 0.000 |
| **sena_shiftl1_sp0** | **0** | sena | 0.402 | −0.206 | 0.500 | 0.932 | 0.000 | 0.520 | 0.518 | 0.517 | **0.72** | 0.002 |
| gates10_shiftl1 | 10 | mlp | 0.034 | −0.207 | 0.500 | 0.933 | 0.018 | 0.522 | 0.502 | 0.475 | 0.04 | 0.020 |
| dagutil_shiftl1_noprior | 10 | mlp | −0.004 | −0.207 | 0.500 | 0.933 | 0.018 | 0.522 | 0.522 | 0.522 | 0.00 | 0.000 |
| sena_shiftl1_hi | 10 (L1 1.0) | sena | −0.007 | −0.207 | 0.500 | 0.933 | 0.012 | 0.522 | 0.521 | 0.521 | 0.00 | 0.000 |
| sena_shiftl1_nodag | 10 (util 0) | sena | 0.118 | −0.207 | 0.500 | 0.933 | 0.024 | 0.521 | 0.521 | 0.521 | 0.00 | 0.000 |
| sena_epsdecorr_ortho | 10 | sena | 0.116 | −0.206 | 0.500 | 0.932 | 0.018 | 0.519 | 0.519 | 0.519 | 0.00 | 0.000 |

*(\* old-best A-shuffle from the base-grid sweep; A_real/zero/shuf are singles effect-spearman. Artifacts: `analysis/MASTER_eval_all.csv`, `heldout_canonical_all.csv`, `grn_dag_sweep_{grnfix,sena,epsd}.csv`, `grn_dag_sweep.csv`.)*

**Reading the table:**
- **No run beats the old best** `v2_esm_con` on specificity (`disc_cos_res` 0.615). The whole 27-run program produced *no* improvement on the metric that matters.
- **`A_gap ≈ 0` (or negative) everywhere.** The old best is even slightly *anti*-load-bearing (`−0.067`: `A=0` beats real A).
- **Sparsity sets A's density, not its usefulness:** zero-sparsity arms (`dagutil_sp0`, `sena_shiftl1_sp0`) keep A **dense** (`densA` 0.72–0.75) but `A_gap` is still ~0. Sparse-and-inert or dense-and-inert.
- **Prior-off collapses A to zero** (`densA=0`) and CollecTRI AUROC drops from ~0.97 to ~0.56 (correlation floor) — the GRN "recovery" was decoder/prior memorization.
- The **`shiftl1` family** shows fake-high raw `discrimination_cos` (0.61–0.66, not shown) but with **collapsed pearson** (~0/negative) and resid `disc_cos` pinned at **0.500** — a degenerate prediction, not real discrimination.

### 4b. Old models — A-inert confirmed (base-grid sweep)

| old model | s_spear real | zero | shuf | **A_gap** | c_pear real | c_pear zero | densA | CollecTRI | shufA |
|---|---|---|---|---|---|---|---|---|---|
| **v2_esm_con_ugate** (old best) | 0.265 | 0.332 | 0.310 | **−0.067** | −0.030 | 0.312 | 0.827 | 0.973 | 0.962 |
| v2_esm_con_epsmix | 0.260 | 0.372 | 0.319 | −0.112 | 0.073 | 0.489 | 0.828 | 0.973 | 0.962 |
| v2_esm_con_ugate_epsmix | 0.262 | 0.381 | 0.322 | −0.119 | 0.071 | 0.600 | 0.818 | 0.973 | 0.961 |
| v2_esm_con_ident | 0.310 | 0.423 | 0.357 | −0.113 | −0.030 | 0.473 | 0.869 | 0.974 | 0.961 |
| v2_combo_gate512 | 0.279 | 0.378 | 0.331 | −0.099 | 0.323 | 0.604 | 0.797 | 0.973 | 0.958 |
| v2_combo_resid | 0.486 | 0.425 | 0.382 | **+0.061** | 0.387 | 0.598 | 0.819 | 0.973 | 0.962 |
| v2_combo_resid_hicon_g512 | 0.465 | 0.415 | 0.364 | **+0.050** | 0.351 | 0.619 | 0.903 | 0.968 | 0.966 |

**A-inertness predates every fix.** The only arms with a *small positive* singles `A_gap` are the two **combo-trained resid** models (+0.05–0.06) — but they still fail on combos (A hurts: c_pear 0.39 vs 0.60), so not clean load-bearing. On the CollecTRI GRN: AUROC ~0.97 collapses to `shufA` ~0.96 when A is shuffled (A structure irrelevant), and to the correlation floor (~0.56) when the prior is off — i.e. memorization, not causality.

### 4c. Component ablation — what IS load-bearing

Singles, mean spearman **drop** when each component is ablated (baseline 0.380), across 26 runs. Artifact: `component_ablation_{grnfix,sena,epsd}.csv`.

| ablate | Δspear | interpretation |
|---|---|---|
| `shift_zero` (remove the shift) | **−0.142** | **shift existence/magnitude = #1 load-bearing** — no shift ⇒ predict control ⇒ lose most signal |
| `D_perm` (permute decoder genes) | **−0.095** | **decoder latent→gene map = #2** |
| `D_rand` (random decoder) | −0.050 | decoder matters |
| `shift_rand` (random shift, matched norm) | −0.039 | shift magnitude matters |
| **`shift_perm` (reassign TF↔shift)** | **−0.003** | **perturbation identity barely matters** ← smoking gun for discrimination failure |
| `A_zero` / `A_shuf` | −0.006 / +0.000 | **A does nothing** |

Combos (baseline 0.433): `A_zero/shuf` **improve** slightly (+0.005/+0.007 — A *hurts* combos); `shift_zero` −0.063; `shift_perm` +0.000. Same story.

**The actual load-bearing part = the shift-encoder magnitude + the linear decoder.** The model runs on *"a perturbation happened (shift magnitude) → apply the shared exit-pluripotency program (decoder)."* It does **not** use **which** TF (`shift_perm ≈ 0` — you can scramble the TF↔shift assignment with ~no effect) and does **not** use **A**.

---

## 5. Data QC — the data is good; the problem is the shared program

Full QC of the Joung singles atlas (`qc_joung_perts.csv`) vs Norman (`qc_norman_perts.csv`):

| metric | **Joung** (hESC, TF-OE) | **Norman** (K562, CRISPRa) |
|---|---|---|
| axis genes | 9,923 | 8,000 |
| controls | 127,813 | 11,183 |
| single perturbations | 1,769 | 105 |
| cells/pert (median / mean) | 93 / 574 | 469 / 515 |
| % ≥50 / ≥100 cells | 61% / 49% | 100% / 100% |
| on-target Cohen's D (median) | 0.57 | 2.16 |
| % strong on-target (\|D\|>0.5, up) | 56% | 93% |
| DEGs/pert (median / mean) | 474 / 596 | 79 / 134 |
| % ≥10 DEGs | 97% | 99% |
| ‖Δ‖ (median effect magnitude) | 27.7 | 3.75 |

**The Joung data is not garbage:** 100% of TFs are in-vocab, 76% show real on-target overexpression, median 474 DEGs/pert, 127k controls. `min_cells_per_pert=5` admits a noisy tail (39% have <50 cells), but the CausalGRN-style high-quality set is 911/1769.

**Discriminability test** (split each high-quality TF's cells in half; can half-A retrieve half-B?):
- median pairwise cosine of **raw** Δ across TFs = **0.987** → every TF produces a near-identical raw response (dominant shared differentiation program).
- median pairwise cosine of **drift-removed** Δ = 0.031 → TF-specific residuals *are* distinct (real, non-redundant signal exists).
- **DATA self-retrieval (drift-removed, top-1) = 0.054** (chance 0.0014, i.e. 38× chance) but only 5% — the residual is real but *small vs sampling noise* at ~93 cells/pert.
- MODEL retrieval_top1 = 0.006 (chance) → the model captures ~none of even that 5% ceiling.

**Latent-shift collinearity cascade** (ugate WM) — where the shared program lives:
| stage | pairwise cosine across TFs |
|---|---|
| latent shift `S` | 0.368 (diverse; effective rank ≈ 38/64) |
| decoded gene-space prediction `Ĝ` | 0.564 |
| observed data Δ | 0.987 |
| decoded `Ĝ` after drift-removal | −0.006 (distinct) |

⇒ The **latent space is not collapsed** — latent shifts are diverse. The shared-program dominance is a *gene-space / training-target* phenomenon, not a latent collapse. So modeling the shared program is safe for the latent graph.

*(Metric hygiene checked: all load-bearing metrics are drift-removed; excluding the perturbed gene itself changes `discrimination_cos` by 0.000 — negligible in a 9,923-gene readout — so numbers are honest downstream measures.)*

---

## 6. Root cause — why A is inert

**It is a per-cell reconstruction VAE.** The encoder produces ε *per cell*; `z = (I−Aᵀ)⁻¹ε` and reconstruction only needs `x̂ = D·z ≈ x` for each individual cell. Setting `A = 0` gives `z = ε`, and the encoder can produce whatever ε reconstructs the cell — so **`A = 0` remains globally optimal**, and D absorbs any structure A could carry. No term on the reconstruction path forces A to be used.

- **Sparsity / dagutil / shift-L1 / SENA / ε-decorr all leave this escape hatch open.** Even ε-decorrelation+orthonormal-D (which *should* force `A = Cholesky(precision)` under covariance-matching) fails, because the objective is per-cell reconstruction, not marginal-covariance matching — the encoder just makes a per-cell ε and `A=0` still works.
- A even *has* structure in some models (ugate: spectral 0.54, dense) — it's simply **not used** for prediction.

**Two separate problems (do not conflate):**
1. **A inert** → identifiability/architecture (per-cell reconstruction). Fix needs an architecture change: fit A only to explain the perturbed *distribution* (discrepancy-VAE protocol), or an encoder→z formulation with a distributional objective.
2. **Discrimination weak** (~0.61 disc_cos, top-1 at chance) → shared-program dominance + small, noise-limited residual. Fix: model the shared program explicitly (predict the residual) and/or add contexts (multiple environments).

---

## 7. What would actually move the needle

1. **Shared-program residual head (no new data).** Add a learned shared/global-program component (WT-PC1-style); train the TF-specific shift on the *residual*. The latent graph is diverse (§5) so this doesn't make A useless — it refocuses the loss off the ~99% shared mode and toward the residual, letting the model reach the ~5% ceiling and lift discrimination. Addresses problem #2.
2. **Multi-context to raise the ceiling.** Joung has one input context (hESC, 2 batches) → no leave-one-context-out. Add **Norman** (K562 CRISPRa — activation, same direction as OE, *has combos*) as a second context via a **data token** (encoder modality embedding), with the **ESM/TF descriptor** as the cross-dataset bridge and shared TFs as anchors. This is the multi-environment lever that separates TF-specific from context-shared signal. (Not Replogle — fitness-KD biology, no combos, off-goal.)
3. **Load-bearing A is a separate track** (discrepancy-VAE protocol) — only if a causal GRN is a goal in itself; it is not required for the RL fate-design application, which runs on the shift+decoder that already carry prediction.

---

## 8. Artifacts

- `experiments_rl/analysis/MASTER_eval_all.csv` — master table (§4a)
- `experiments_rl/analysis/heldout_canonical_all.csv` — canonical held-out metrics (same script as exp17)
- `experiments_rl/analysis/grn_dag_sweep_{grnfix,sena,epsd}.csv` + `grn_dag_sweep.csv` — A real/zero/shuffled
- `experiments_rl/analysis/component_ablation_{grnfix,sena,epsd}.csv` — shift vs decoder vs A ablation (§4c)
- `experiments_rl/analysis/qc_joung_perts.csv`, `qc_norman_perts.csv` — per-perturbation QC (§5)
- Scripts: `grn_dag_sweep.py`, `component_ablation.py`, `qc_joung.py`, `qc_norman.py`, `qc_discriminability.py`, `assemble_master_tables.py`, `aggregate_heldout.py`; canonical eval `experiments/exp19_nonlinear_compose/evaluation/heldout_zeroshot_eval.py`.

---

# PART II — exp20: architectural attack on the two failure modes (2026-07)

Part I (the 27-run exp19 sweep) proved no *loss/regularizer* makes A load-bearing, and named two separable problems: **(1) A inert** (per-cell reconstruction lets `A=0` stay optimal; the decoder absorbs any structure A could carry) and **(2) discrimination weak** (mode collapse onto the shared program). Part II builds a clean fork (`experiments/exp20_load_bearing/`, exp19 code untouched) and attacks the *architecture* directly, plus retires the CollecTRI prior. **Headline: none of the four architectural fixes beat a simple prior-off baseline; the one theoretically-decisive fix (frozen decoder) diverged and is being retrained.**

## 9. The exact fixes (code + math)

All behind config flags defaulting to exp19 behavior; the same eval scripts (mode-aware) score exp19 and exp20 checkpoints identically.

**Fix 1 — structural intervention (`pert_intervene_mode: structural`).** The exp19 do-operator pre-propagated the shift `z_S=(I−Aᵀ)⁻¹S` and injected the full target `ε̃ = ε + u⊙(z_target−ε)`, so `A=0` reproduces predictions (the leak). Replaced with a single mutilated solve, shift into exogenous noise only:
```
ε̃ = ε_ctrl + S ;   z_post = (I − diag(1−u)Aᵀ)⁻¹ ε̃
```
Row form: `z_j = (ε+S)_j + (1−u_j)·Σ_i A_ij z_i`. A non-target program (`S_j=0, u_j≈0`) then moves **only** through A — exactly 0 at `A=0`. Edited at every S→z_post site: training loss, model method, gate-aware operator, batched MPC `apply_many`, and the 4 eval do-ops.

**Fix 2 — sparse target gate (`use_target_gate`, L0 not L1).** A second head on the ShiftEncoder trunk emits per-program gate logits from the same ESM label (→ zero-shot). At the single choke point `compute_shift`: `S = shift_scale·(gate ⊙ raw)`. Gate is soft Gumbel-sigmoid `σ((ℓ+G)/τ)` in train / hard `1{ℓ>0}` in eval; dense-open init (bias +3, σ≈0.95). Penalty is **L0 on gate existence** `mean σ(ℓ/τ)` (not L1 on magnitude, which fought MSE in Part I). With `u=clamp(β|S|)`, sparse S ⇒ sparse u ⇒ mutilation confined to targets ⇒ non-target Δ must route through A. Fitting the dense observed TF Δ from a sparse injection *requires* A — by construction.

**Prior OFF (`lambda_prior:0`).** Part I's exp20 arms accidentally inherited `lambda_prior:10` (CollecTRI margin loss on `G=D·A·Dᵀ`, a 42,990-pair Python loop per step). Turned off for the honest GRN-bio test and to retire the prior from the loss.

**Recon knobs:** `lambda_recon` (down-weight per-cell recon), `residual_recon` (reconstruct x−drift; ~no-op for linear D), `pert_residual_target` (drift-removed pert target), `kl_free_bits`.

**Phase-2 architecture arms (attack the two failure modes at the architecture level):**
- **Frozen GO+PCA decoder (`decoder_type: frozen_go`)** — kills the decoder-absorption escape hatch. `D = [GO:BP membership (192 terms, unit-norm) | PCA-64 of expression]`, `requires_grad=False`, `latent_dim=256`. Since `D_eff = D·(I−MAᵀ)⁻¹` and D is now fixed and structured, A can no longer be rotated into D → A must carry propagation. GO gives interpretability; PCA gives full gene coverage (GO alone covers only 14–26%). "SENA for the decoder."
- **Whitened pert target (`pert_whiten_target`)** — attacks mode collapse. MSE computed on `(pred−target)/σ_gene` where `σ_gene` = per-gene std of the drift-removed training-TF Δ → rewards residual *direction*, not magnitude, starving the single-scalar `shift_scale` shortcut. Paired with `pert_residual_target`.
- **Decoupled 3-stage L0 schedule (`pert_target_l0_start_epoch`, `pert_target_l0_warmin`)** — attacks the gate-vs-A race. In the original coupled schedule the L0 penalty and A's structural learning share `dag_ramp` and both ramp from epoch 11, so gates can close before A is competent (this collapsed the no-contrastive esm arm: gate → 1.6% open). Decoupled: Phase A (1–10) recon; **Phase B1 (11→start) A learns on the DENSE shift, L0=0**; Phase B2 (start→+warmin) L0 ramps to sparsify once A is competent. Swept `start ∈ {26, 36}`.
- **Nonlinear SCM (`scm_nonlinear`, Picard)** — `z_i = tanh((Aᵀz)_i) + ε_i`, decoder stays linear. Solved by Picard fixed-point `z ← tanh(z@A)+ε` (mutilated: `z ← diag(1−u)⊙tanh(z@A)+ε̃`); **acyclicity off, spectral clamp on** — convergence guaranteed by `‖A‖₂≲0.9` (clamp) and `|tanh'|≤1` (Lipschitz <1). Verified: residual → 1e-6 by iter 40. Projected GRN uses the Jacobian `G(z*)=D·(I−diag(tanh')Aᵀ)⁻¹·D⁺`.

### 9.1 Training schedule — the 3 phases and the decoupled-L0 fix

Every exp20 run uses a phased schedule (con-base numbers: `reconstruction_phase_epochs=10`, `dag_start=11`, `dag_warmin=10`, `beta_anneal=30`, `n_epochs=60`):

- **Phase A (epochs 1–10) — reconstruction only.** `dag_active=False`; `dag.theta` **frozen** at init (θ≈−3 → ~4.7% dense sparse graph); the SCM solve runs but A doesn't learn (encoder/decoder adapt to a fixed sparse A). The perturbation block is **skipped entirely** (`if dag_active:` gates it) — no shift encoder, no gate, no L0.
- **Phase B (epochs 11–60) — DAG + perturbation on.** At epoch 11 `theta` unfreezes (A starts learning) **and** the pert loss (shift encoder + gate + L0) switches on. β anneals 0.01→0.5 and τ 1.0→0.25 over 30 epochs from epoch 11.

**The original coupling bug.** A single `dag_ramp = clamp((epoch−10)/10, 0, 1)` (0→1 over epochs 11–20) multiplied **all of** `lambda_dag` (A's acyclicity/sparse/spectral structural loss), `lambda_pert` (the MSE that teaches A to carry signal), **and** the L0 gate penalty `target_l0_weight = 0.03·dag_ramp`. So the gate-closing pressure and A's competence-building ramp **in lockstep**, both hitting full strength at epoch 20 — a race: L0 closes the target gate (forcing non-target signal through A) *while* A is still a near-random graph that has never seen a perturbation. The no-contrastive `sparse_esm` arm **lost the race** — the gate collapsed to **1.6% open** (over-sparsified before A could use the redirected signal), degenerating A_gap to ~0. Contrastive (`sparse_con`) accidentally protected it (gate stayed 75% open) by giving the magnitude head a reason to keep programs open — i.e. the coupled schedule relied on InfoNCE to win the race for it.

**The fix — decouple L0 onto its own timeline** (`pert_target_l0_start_epoch`, `pert_target_l0_warmin`, independent of `dag_ramp`; defaults reproduce the coupled behavior):
```
Phase A  (1–10):     recon only, A frozen
Phase B1 (11→start): pert + DAG structural ON, L0 = 0 (gate held DENSE-open) → A learns to carry a DENSE shift
Phase B2 (start→+warmin): L0 ramps 0→0.03 → sparsify the gate NOW that A is competent
```
`l0_ramp = 0 if epoch<start else clamp((epoch−start+1)/warmin, 0, 1)`, swept `start ∈ {26, 36}` (warmin=10) — a **15- or 25-epoch dense-shift head start** for A before any gate pressure:

| schedule | l0_ramp @ epochs 11 / 20 / 26 / 36 / 46 | A head-start |
|---|---|---|
| coupled (baseline) | 0.1 / 1.0 / 1.0 / 1.0 / 1.0 | 0 epochs |
| staged `s26` | 0 / 0 / 0.1 / 1.0 / 1.0 | 15 epochs |
| staged `s36` | 0 / 0 / 0 / 0.1 / 1.0 | 25 epochs |

The epoch log now prints `dag_ramp` and `l0_ramp`. **Result (§13): the decoupled schedule did not help** — `staged_esm_s26` (0.553) ≈ coupled `sparse_esm` (0.555); the gate-race was not the discrimination bottleneck (decoder-absorption + mode-collapse dominate regardless of when the gate closes).

## 10. Sweeps + methodology

Three sweep grids, all Joung, 60 epochs, latent 64 (except frozen=256), ESM descriptor, `u_mode=shift`, β up to 0.5. Eval on the exact Part I scripts (mode-aware):
- **`x20`** — 7 structural arms, **prior-ON** (min_esm control [Fix1 only, dense S], sparse_esm/sparse_con [Fix1+Fix2, ±InfoNCE], + resid / pertonly / residrecon / all recon variants).
- **`x20po`** — the same 7, **prior-OFF** (the honest baselines).
- **`x20arch`** — 8 Phase-2 arms: frozenD, frozenD_whiten, whiten (linear+whitened target), staged_esm_s26/s36, staged_con_s26, staged_frozenD_s26, staged_nonlinSCM_s26. All prior-off, `struct_sparse_con` base.

**Metrics.** `disc_resid` (held-out drift-removed discrimination_cos, chance 0.5) from `heldout_zeroshot_eval.py`; `S_Agap`/`C_Agap` = `spear_real − max(zeroA, shufA)` on drift-removed singles/combos from `grn_dag_sweep.py` (real vs zero vs col-shuffled A); GRN-bio CollecTRI/DoRothEA-held AUROC + shuffle/randomD controls. **New mechanism probes** (`evaluation/component_probe.py`, `gradient_flow.py`, `agap_probe.py`), raw-Δ cosine (not the drift-removed spearman, which is noise-dominated for mode-collapsed models):
- `probe_A = 1−cos(Δ, Δ_{A≡0})` — does removing A change the output (A mechanically used)?
- `probe_Dperm = 1−cos(Δ, Δ_{D-rows-permuted})` — decoder dominance.
- `probe_shiftID` / `crossTF_div` — does perturbation identity matter / are per-TF predictions distinct (mode collapse)?
- `gradient_flow` — share of the held-out pert-MSE gradient per parameter block.

## 11. Mechanism findings (why shift+decoder are load-bearing, A is not)

From `component_probe`/`gradient_flow` across arms (raw-Δ cosine; higher = more load-bearing):

| arm | gate_open | probe_A | probe_Dperm | crossTF_div | dag grad% | shift_scale grad% |
|---|---|---|---|---|---|---|
| min_esm (dense S) | 1.00 | 0.014 | **1.01** | 0.001 | 5% | 47% |
| sparse_esm (gate collapsed) | 0.02 | 0.007 | **1.01** | 0.000 | 5% | 27% |
| sparse_con (healthy gate) | 0.75 | 0.156 | **0.98** | 0.018 | 9% | 33% |
| sparse_pertonly | 0.79 | 0.459 | **0.99** | 0.018 | 18% | 28% |

1. **The linear decoder absorbs the SCM operator (probe_Dperm≈1.0 in EVERY arm).** Shuffling D's gene-rows makes the prediction orthogonal to the real one. Formally `D_eff = D·(I−MAᵀ)⁻¹` — for *any* A there's a D giving identical predictions, so a sparse latent shift decoded through a dense decoder column already produces a dense gene response. Fix 2 sparsified S but never removed D's freedom to bypass A. **This is the escape hatch, and it is structural — hence the frozen decoder.**
2. **The shift is load-bearing for MAGNITUDE, not IDENTITY.** `probe_shiftID ≈ crossTF_div ≈ 0.00–0.02`: swapping another TF's shift barely changes the prediction; distinct TFs' predictions are ~98–100% identical. A single scalar `shift_scale` eats 27–47% of the pert-loss gradient — the model reduces error by tuning global magnitude, which *is* the mode collapse that pins discrimination at chance. InfoNCE lifts crossTF only 0.001→0.018 (~20× too weak).
3. **Fix 2 makes A mechanically used but not useful.** `probe_A` tracks gate openness (0.007 collapsed → 0.16–0.46 healthy) and dag receives real gradient only when the gate is open — but A's contribution is redundant with D and lands in the shared/magnitude direction, so `S_Agap≈0` (no drift-removed gain). *Used, redundantly, in a direction that doesn't help.*

## 12. Full results — all 22 arms (drift-removed singles + combos + A-load-bearing)

`disc_resid` = held-out drift-removed discrimination (chance 0.5). `S_Agap`/`C_Agap` = real−max(zero,shuf) (>0 ⇒ A helps). `collectri` = GRN AUROC (prior-ON = memorization).

| arm | prior | fix | disc_resid | pear_resid | S_real | **S_Agap** | C_real | **C_Agap** | collectri |
|---|---|---|---|---|---|---|---|---|---|
| sparse_con | prior-off | baseline | **0.639** | 0.094 | 0.259 | −0.054 | 0.369 | 0.027 | 0.569 |
| sparse_pertonly | prior-off | baseline | 0.621 | 0.062 | 0.260 | −0.025 | 0.371 | 0.025 | 0.576 |
| staged_con_s26 | prior-off | staged-L0 (con) | 0.611 | 0.066 | 0.264 | −0.051 | 0.376 | 0.017 | 0.589 |
| sparse_residrecon | prior-off | baseline | 0.609 | 0.127 | 0.519 | 0.006 | 0.494 | 0.002 | 0.520 |
| min_esm | prior-off | baseline | 0.586 | 0.180 | 0.405 | 0.001 | 0.448 | −0.000 | 0.571 |
| staged_nonlinSCM_s26 | prior-off | **nonlinear-SCM** | 0.570 | −0.005 | 0.264 | **−0.080** | 0.379 | 0.011 | 0.567 |
| sparse_esm | prior-off | baseline | 0.555 | 0.173 | 0.459 | 0.028 | 0.463 | 0.027 | 0.564 |
| staged_esm_s26 | prior-off | staged-L0 (no-con) | 0.553 | 0.215 | 0.423 | 0.050 | 0.461 | 0.027 | 0.550 |
| staged_esm_s36 | prior-off | staged-L0 (no-con) | 0.548 | 0.222 | 0.437 | 0.048 | 0.449 | 0.021 | 0.570 |
| sparse_all | prior-off | baseline | 0.502 | −0.187 | 0.462 | 0.018 | 0.456 | 0.005 | 0.566 |
| sparse_resid | prior-off | baseline | 0.502 | −0.182 | 0.445 | 0.034 | 0.440 | 0.018 | 0.557 |
| **whiten** | prior-off | **whitening** | **0.500** | **−0.204** | 0.489 | 0.059 | 0.485 | 0.022 | 0.580 |
| sparse_esm | prior-ON | baseline | 0.597 | 0.213 | 0.444 | 0.024 | 0.480 | 0.015 | 0.983 |
| sparse_pertonly | prior-ON | baseline | 0.594 | 0.107 | 0.268 | −0.037 | 0.390 | 0.016 | 0.970 |
| sparse_con | prior-ON | baseline | 0.593 | 0.124 | 0.274 | −0.041 | 0.384 | 0.043 | 0.972 |
| sparse_residrecon | prior-ON | baseline | 0.593 | 0.097 | 0.367 | 0.010 | 0.384 | −0.006 | 0.981 |
| min_esm | prior-ON | baseline | 0.587 | 0.205 | 0.400 | 0.033 | 0.477 | 0.018 | 0.983 |
| sparse_resid | prior-ON | baseline | 0.504 | −0.175 | 0.436 | 0.014 | 0.406 | −0.006 | 0.973 |
| sparse_all | prior-ON | baseline | 0.502 | −0.180 | 0.453 | −0.002 | 0.447 | −0.003 | 0.971 |
| struct_frozenD | prior-off | **frozen-D** | NaN | — | — | — | — | — | — |
| struct_frozenD_whiten | prior-off | frozen-D+whiten | NaN | — | — | — | — | — | — |
| staged_frozenD_s26 | prior-off | frozen-D+staged | NaN | — | — | — | — | — | — |

Reference: exp19 best (linear, prior-ON epsmix) disc_resid **0.647**; Joung split-half top-1 discriminability ceiling ≈ 5%; CollecTRI prior-ON ≈0.97 (memorization) → prior-off ≈0.57 (corr floor).

## 13. Verdict — what worked

- **Only turning the prior OFF helped.** Best drift-removed disc = prior-off `sparse_con` **0.639** (↑ from 0.593 prior-on, ≈ exp19's 0.647). The prior was *distracting* D toward CollecTRI; removing it improves discrimination and is ~10× faster.
- **CollecTRI 0.97 was 100% memorization** — collapses to the 0.57 correlation floor prior-off (= shuffA); no biological GRN recovered from data.
- **Whitening (mode-collapse fix) → collapsed to chance** (0.500, pear −0.20). Drift-removed + whitened target starves the pert signal; it removed the signal rather than fixing collapse.
- **Staged 3-phase L0 (gate-race fix) → no effect** — staged_esm 0.553 ≈ coupled 0.555 (no rescue of the no-contrastive arm), staged_con 0.611 < coupled 0.639. The gate-vs-A race was not the bottleneck.
- **Nonlinear SCM → worse** than its linear twin (0.570 vs 0.611); A actively hurts (S_Agap −0.08; combos c_pear real −0.26 vs +0.17 at A=0). Nonlinearity in A did not help on Joung.
- **Frozen decoder (the decisive absorption fix) → diverged (NaN)** at epoch 11 (DAG activation): forward-pass blowup of `(I−Aᵀ)⁻¹` at latent-256 (spectral-clamp power-iteration under-estimated ‖A‖). **The central decoder-absorption question is still open.**

**A is not load-bearing in any completed arm** (every `S_Agap ≤ 0.05` or negative); **mode collapse persists** (crossTF ≈ 0.01–0.02) and neither whitening nor contrastive broke it; **combos at chance** (`C_disc ~0.025`, negative drift-removed pearson for the best arms).

### 13.1 Frozen-decoder divergence — UNRESOLVED
Diagnosis: latent-256 frozen arms NaN at epoch 11 (DAG activation); latent-64 arms fine. First stabilization attempt — spectral-clamp `n_iter 10→20`, `max_sn 0.9→0.85`; encoder `mu` clamp ±30 (KL `0.5·mu²` fp32-overflow guard); **skip optimizer step on non-finite loss**; `lr 1e-3→5e-4` — **did NOT fix it.** The full-latent-256 smoke shows the Phase-B forward is *deterministically* non-finite (not a rare transient): every batch trips the NaN guard, so the run silently trains nothing (`recon=0.0`, all batches skipped). So the NaN source is neither the mu-overflow nor the spectral clamp — it is a reproducible NaN in the frozen-256 Phase-B forward (candidates: the batched `[1,32,256,256]` mutilated solve in the pert loss, or an interaction of the fixed [GO|PCA] decoder with the SCM solve). **Root cause not yet isolated; needs a targeted single-batch forward trace.** Until then, the decoder-absorption fix — the one architectural change that could still make A load-bearing — is unverified.

**Standing conclusion (pending only the frozen-D result).** Every completed exp20 arm reproduces Part I: A not load-bearing (`S_Agap ≤ 0.05`), decoder absorbs prediction (`probe_Dperm≈1.0`), mode collapse persists (`crossTF≈0.02`), discrimination capped at the prior-off ~0.64 ceiling. If frozen-D also shows A-not-useful (once debugged), the architectural search closes: on Joung, keep the simple prior-off shift+linear-decoder; the causal-DAG machinery does not earn its place on this single-context, shared-program-dominated dataset. Part I §7 remains the path forward — shared-program residual head + multi-context (add Norman) — which targets discrimination (problem #2), not A-load-bearing (problem #1).

## 14. Artifacts (Part II)
- `experiments/exp20_load_bearing/` — the fork (models/training/configs/evaluation/docs).
- `experiments_rl/analysis/grn_dag_sweep_{x20,x20po,x20arch}.csv` — A real/zero/shuf + GRN-bio, per grid.
- `experiments/exp20_load_bearing/results/MASTER_ALL_x20.csv` + `docs/MASTER_ALL.md` — the §12 table (assembled by `evaluation/assemble_master_all.py`).
- Probes: `evaluation/{component_probe,gradient_flow,agap_probe}.py`; docs `MECHANISM.md`, `MASTER_RESULTS.md`, `STRUCT_INTERVENE.md`.
- Configs: `configs/ablations/{lb (prior-on), lbpo (prior-off), lb2 (frozen/whiten), lb3 (staged/nonlinear)}/`.

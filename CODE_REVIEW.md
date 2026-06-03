# Code Review — scCausalRL (baseline fairness focus)

**Reviewer:** Ivy (mentor) · **Date:** 2026-06-03 · **Commit:** `b78386a`
**Scope:** static review (no data/checkpoints run). Focus per request: how the
**additive** and **correlation** baselines are run, and whether they are fair
comparisons given the project's goals. Reasoning is from the code plus the
result numbers committed in `slides/make_slides.py`.

---

## TL;DR

- The codebase is well-organized, well-documented, and careful (clean module
  boundaries, honest docstrings, checkpoint gene-axis guards). Good work.
- **The single-perturbation "model vs correlation" comparison is not
  apples-to-apples.** The model is fed the *observed* differential-expression
  vector of the very perturbation it is being scored on, so its "prediction" is
  essentially a rank-64 reconstruction of the answer. The correlation baseline
  gets no such information. This inflates the headline 0.933-vs-0.677 gap.
- **The combo "model vs additive" comparison is a fair setup, but the result
  does not support the project's hypothesis:** additive ≥ the causal model on
  every metric and in every epistasis tercile.
- **The combo "model vs correlation" comparison is uninformative** (correlation
  does not receive the single-gene effects that model and additive both get).
- The **GRN vs correlation** comparison (0.832 vs 0.677) *is* fair and is the
  legitimate "does learned causal structure beat co-expression" result.

---

## 1. How the baselines are computed

**Additive** — `evaluation/eval_combinatorial.py:126`,
`evaluation/run_perturbation_eval.py:426-445`:
```
add_pred = ctrl_mean + de_single[A] + de_single[B]
```
`de_single[g] = mean(X_perturbed_g) − mean(X_ctrl)` (the observed single-gene
effects). Standard, appropriate null model for combinations.

**Correlation** — `evaluation/run_perturbation_eval.py:184-190`, scored in
`evaluate_baseline` (`:204-240`):
```
corr = |corrcoef(X_ctrl.T)|     # gene×gene, control cells only
scores = corr[pert_gene, :]     # rank genes by co-expression with the perturbed gene
```
Combos use `max(corr[A], corr[B])` (`:760`). This baseline sees **only control
co-expression** — no perturbation effect.

**GRN** — `compute_grn_matrix` (`:193-201`): `|D @ A_mean @ Dᵀ|`, scored the
*same way* as correlation (row indexed by perturbed gene, no `dx`).

---

## 2. Core issue: single-pert "model vs correlation" leaks the target into the input

Data flow for the model's single-pert prediction (`evaluate_single_perts`):

1. `run_perturbation_eval.py:281` — `dx_np = de_single_adata.get(pert_gene, de_data['lfc'])`
2. `de_single_adata[gene]` is set in `data/load_norman.py:215` to
   `pert_mean − ctrl_mean` — **the observed effect of that exact perturbation**.
3. `dx` is fed into `model.intervene_gene_u(...)` (`:292`):
   `dz = D⁺·dx`, `z_target = z_prev + dz`, then decode.

For the strongly-loaded programs (`u → u_clip_max`),
`z_post ≈ z_prev + D⁺·dx`, so:
```
pred_de ≈ D · D⁺ · dx   ≈  rank-64 projection of the observed effect
```
The ground truth it is scored against (`obs_lfc`, `:303`) is the **same**
`pert_mean − ctrl_mean` from the **same cells**. So the single-pert prediction
is a reconstruction of the answer it was handed as input. The correlation
baseline receives none of this and must predict from scratch.

Consequence: the slide-7 gap (model **0.933** vs correlation **0.677**) is
inflated, and the label "**zero-shot, never saw these genes**"
(`slides/make_slides.py:368`) is misleading. The *weights* never trained on the
test gene — true — but the *evaluation* hands the model that gene's observed DE
vector. The model is not predicting the perturbation; it is compressing and
reconstructing it.

**The fair comparison already exists in the same figure:** the **Learned GRN**
bar (`D@A@Dᵀ`) is scored exactly like correlation — a gene×gene matrix, never
sees `dx`. **GRN 0.832 vs correlation 0.677** is the honest "does learned causal
structure beat co-expression" result, and it is defensible. GRN and correlation
are the matched pair; the D⁺ "Causal VAE" column is a different kind of object
and should not share the axis.

### Fix
- Add a **given-`dx`** baseline to the single-pert panel to expose the leak:
  - **identity** — score by `|dx|` directly (≈ perfect; shows the task is
    trivial once `dx` is provided), and
  - **PCA-of-`dx`** — project `dx` onto the top-64 PCs of the control matrix.
  If the Causal VAE ≈ PCA-of-`dx`, the SCM adds little beyond generic 64-dim
  compression.
- Reframe the single-pert claim around **GRN vs correlation**; drop or heavily
  caveat the 0.933 bar; stop calling it "zero-shot."

---

## 3. Combo "model vs additive": fair setup, unsupportive result

Design is sound: in `intervene_combo_u` (`models/causal_vae.py:453-506`) both the
model and the additive baseline receive the two *single* effects (legitimately
available — singles are training data). The docstring is honest that the model
is "a strict generalisation of the additive baseline."

Reported numbers (`slides/make_slides.py:393-411`):

| Method      | AUROC | Spearman | Pearson |
|-------------|-------|----------|---------|
| Causal VAE  | 0.945 | 0.751    | 0.222   |
| Additive    | **0.954** | **0.789** | **0.929** |
| Correlation | 0.727 | 0.471    | —       |

Additive is **≥ the model on every metric**, dramatically so on Pearson. The
epistasis stratification — the place the causal model *should* win — shows
additive still ahead in **every** tercile (High: model 0.952 vs additive 0.960).
So the central hypothesis (a causal world model captures epistasis additive
misses) is **not supported** by these results.

The slide framing ("competitive with additive," "crushes correlation") reads a
null/negative result as a win. The slide-9 caveat ("additive uses real observed
data") is honest and should be applied to the single-pert slide too.

### Fix
- State plainly: additive ≥ model everywhere ⇒ no demonstrated epistasis capture
  on Norman yet. That is a legitimate finding and points to what to improve.
- Drop or relabel the combo-vs-correlation bar (correlation does not get the
  single effects that the other two methods get; the comparison is unmatched).

---

## 4. Is the correlation baseline allowed to see perturbed cells? (It should.)

As implemented, the correlation baseline is built from **control cells only**
(`run_perturbation_eval.py:184-190`: `np.corrcoef(X_ctrl.T)`). The Causal VAE,
by contrast, was trained on the **full Replogle perturbation atlas** (perturbed
+ control). So the comparison pits a model that learned from perturbation
responses against a baseline forbidden from seeing any. Control-only correlation
is therefore **unfair *to the baseline*** — it handicaps it by denying exactly
the data the model used, which flatters the model. The right fairness principle
is *information parity with what the model had access to*.

Two distinct baselines hide under one name:

1. **Control-cell co-expression** (implemented): homeostatic / technical
   covariation. Weak predictor of perturbation effects.
2. **Across-perturbation response correlation** (recommended headline): stack
   per-perturbation pseudobulk DE vectors and correlate genes across that panel
   — captures co-regulation, a strong and standard perturbation-effect predictor
   (the basis of co-expression GRN inference). Uses the same data regime the
   model trained on.

### Fix
- Make the **headline** correlation baseline the across-perturbation one, built
  from Replogle (the training atlas). Likely closes much of the 0.677→0.832 gap.
- Avoid circularity: when scoring gene *g*'s targets, build the network
  **leave-one-out** (exclude *g*'s own perturbation).
- Keep control-only correlation only as an explicitly-labeled "weakest
  reference," and *if* it is used, state that the model saw perturbation data the
  baseline did not — otherwise the comparison conflates data access with method.

---

## 5. CRISPRi (train) → CRISPRa (eval) modality/scale mismatch

The model trains on Replogle K562 **CRISPRi** (knockdown) and is evaluated on
Norman K562 **CRISPRa** (activation). Same cell line (good — controls for GRN
wiring), opposite perturbation direction and different effect-size regime. No
harmonization occurs: `preprocess.py:32-33` is just `normalize_total(1e4)` +
`log1p`; `dx = pert_mean − ctrl_mean` lives in log-normalized space. CRISPRi
log-FCs are negative and floor-bounded; CRISPRa log-FCs are positive and can be
larger / differently distributed. Nothing rescales between them.

**This plausibly explains why additive performs so well relative to the model:**

- **Additive is modality-agnostic.** `add_pred = dx_a + dx_b` uses the real
  Norman CRISPRa single effects directly and carries **zero transferred
  parameters**. Its quality depends only on how additive Norman combos truly are
  (empirically high — hence Pearson 0.93).
- **The model injects CRISPRi-trained operators.** It computes
  `D⁺(dx_a+dx_b) → (I−MAᵀ)⁻¹ → decode` (`causal_vae.py:453-506`). Both `D⁺` and
  `A` were fit to CRISPRi-magnitude, CRISPRi-signed latent shifts. On CRISPRa
  effects: (1) the `(I−MAᵀ)⁻¹` propagation perturbs an already-good additive
  starting point with a graph calibrated to the wrong modality → Pearson
  collapses to 0.22; (2) the `D D⁺` round-trip discards CRISPRa structure outside
  the 64-dim CRISPRi-fit column space. The causal machinery is a **liability**
  here, not an asset.

**Bug found while checking this.** The encoder conditions on a modality token
(`encoder.py:72-77`) but **silently defaults to modality 0 if none is passed**
(`encoder.py:73-75`, comment "Default to modality 0"). The Norman eval path never
passes one — `evaluate_single_perts` (`run_perturbation_eval.py:265`) calls
`model.encoder(x_ctrl_t)` and `eval_combinatorial.py:41` calls `model.encode(...)`
without `modality_id`. So if the model was trained multi-modally, **Norman
CRISPRa cells are encoded as modality 0 (the Replogle/CRISPRi slot) at eval** — a
train/eval conditioning mismatch in exactly the cross-modality direction that
hurts transfer. Confirm which index Norman had in training and pass it explicitly.

### Diagnostics (see `evaluation/dag_diagnostics.py` companion script
`scripts/diagnose_modality_transfer.py`)
1. **In-modality vs cross-modality:** run with `dataset='replogle'`
   (CRISPRi→held-out CRISPRi) and compare the model-vs-additive gap to Norman
   (→CRISPRa). If the model is competitive in-modality but collapses
   cross-modality, the gap is transfer, not method.
2. **`u_clip_max` sweep** (already wired via `--u_clip_max`): lower forces more
   `A` involvement, higher bypasses it. If combo Pearson *improves* as
   `u_clip_max → 1`, the transferred `A` is actively hurting on CRISPRa.
3. **Direction accuracy on Norman:** systematic sign errors ⇒ CRISPRi→CRISPRa
   sign mis-transfer.
4. **Magnitude calibration:** regress predicted log-FC on observed; slope far
   from 1 quantifies the scale mismatch.

### Fix
- Frame the experiment as "can knockdown-learned causal structure transfer to
  activation?" — not generic "perturbation prediction." Additive's dominance is
  then itself a result (transfer is failing).
- For the transfer to work, add per-modality effect-size handling (z-score `dz`
  within modality, or a learned per-modality gain). A shared `D`/`A` with only an
  encoder modality token — *not even passed at eval* — is insufficient.

---

## 6. Commutativity test is vacuous in the default mode

In `combo_mode='parallel'` (default), `predict_combo`
(`evaluation/eval_combinatorial.py:47-60`) does one symmetric solve and returns
the **same array** for AB and BA ("we return the same prediction twice"). So
`commutativity_error` (`run_perturbation_eval.py:422`) is identically 0 by
construction — it measures nothing. Only `combo_mode='sequential'` actually tests
it. Report commutativity only from sequential mode.

---

## 7. Minor code notes

- `pearson_on_degs` default `threshold=0.1` (`utils/metrics.py:52`) but callers
  pass `0.05`; AUROC's DEG set uses p<0.05 while Pearson's uses `|lfc|>0.05` —
  two DEG definitions in one results table. Harmless but worth unifying.
- `fast_mannwhitney_pvals` (`run_perturbation_eval.py:118`) is a normal
  approximation with no tie correction — fine for ranking, just note it's
  approximate.
- Module-level `warnings.filterwarnings('ignore')` can hide real numerical
  issues; prefer scoping it.

---

## 8. What can be claimed honestly today

1. The learned GRN (`D@A@Dᵀ`), which never sees a perturbation's effect, beats
   gene-gene control correlation at DEG detection (0.832 vs 0.677). ✅
   *Caveat:* the correlation comparator should be the across-perturbation
   baseline (§4), which may narrow this gap.
2. On held-out Norman combos, the causal model is **competitive but not better**
   than a simple additive model, and additive wins on effect-magnitude
   correlation. The causal model does not yet show added value on epistatic
   combos. ⚠️ (negative but real)
3. The single-pert 0.933 AUROC is **not** evidence of perturbation prediction —
   it reflects reconstruction of an observed effect supplied as input. ❌ as
   currently framed.

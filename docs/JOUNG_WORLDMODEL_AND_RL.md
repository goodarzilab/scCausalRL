# Joung TF-Atlas: World Model, Annotation, and RL — Full Walkthrough

*A single coherent narrative of everything built on the Joung TF atlas: the causal-VAE world
model and all its training variants, the developmental-fate annotation pipeline, the
goal-conditioned RL stack, every evaluation that was run, and the fate-target reframing —
with the motivation for each successive change and the numbers that drove it.*

Last updated: 2026-06-25. Scope: `experiments/exp17_joung_tfatlas`, `experiments/exp19_nonlinear_compose`, `experiments_rl/`.

---

## 0. The goal and the pieces

**Scientific goal.** Learn a *world model* of how transcription-factor (TF) overexpression moves a
human stem cell toward a differentiated fate, then use *reinforcement learning* over that frozen
world model to **plan TF perturbation sets that reach a target cell state**.

**The three layers:**
1. **World model (WM)** — a Causal-VAE: MLP encoder → linear-Gaussian SCM on latents → **linear**
   decoder, with a *shift-encoder do-operator* for interventions. Decoder kept linear on purpose
   (GRN / CollecTRI interpretability). `exp17` (original + ablation grid) and `exp19` (nonlinear
   composition grid).
2. **Annotation** — map each TF to the developmental **fate** it induces, against the human fetal
   cell atlas (Cao 2020). Provides supervision/targets for fate-directed RL.
3. **RL** — goal-conditioned planning over the frozen WM (`experiments_rl/`): MPC, sequential-PPO,
   set-PPO, distillation. W&B project `CausalBioRL`.

**Dataset.** Joung et al. 2023 (Cell) TF atlas: barcoded ORF overexpression of ~3,500 TF isoforms
in human embryonic stem cells (hESCs), single-cell RNA-seq after 7 days. Processed:
- singles `exp17/data/joung_singles_proc.h5ad` — 1.14M cells × 9,923 genes, 1,689 TFs with ≥5 cells
  (split 1,351 train / 169 val / 169 test), 127,813 GFP/mCherry controls.
- combos `joung_combo_proc.h5ad` — 40 usable pairs + 2 triples on the same gene axis.

---

## 1. The world model architecture

The WM is trained on **single-TF overexpressions only**; combos are held out for compositional
eval (and, in two variants, used for training — see §2.7).

**Latent SCM + do-operator (shift-native mutilation):**
```
z = (I − Aᵀ)⁻¹ ε                              latent SCM (A = sparse DAG via Gumbel/DAGMA)
S = Σ_i ShiftEnc(f_{g_i})                      summed latent shift for a TF set
z_S = (I − Aᵀ)⁻¹ S
ε̃ = u ⊙ z_target + (1−u) ⊙ ε                   interpolated exogenous noise
z_post = (I − diag(1−u) Aᵀ)⁻¹ ε̃               mutilated solve (the intervention)
Δx̂ = D z_post − D z_prev                       predicted expression change
```
`u = clamp(β|S|)` is the per-latent intervention strength. `f_g` is the TF *descriptor*
(control-PCA or **ESM-2** protein embedding) — this is what lets the model predict a held-out TF
zero-shot. The `u_gate` / `eps_mix` variants make `u` and `ε̃` nonlinear (§2.6).

**Evaluation frame (used everywhere).** Δ = mean(perturbed) − mean(control). `drift` = mean Δ over
training TFs (the shared "leaving pluripotency" program). `residual = Δ − drift`. Raw metrics are
**saturated by drift**; the load-bearing metrics are drift-removed and discrimination/retrieval.

---

## 2. World-model training versions — the genealogy

Two grids, 25 trained checkpoints. Each row says what it changes vs the `esm_con` baseline and why.
(latent dim 64 unless noted; `phase1_epoch{N}.pt` checkpoints every 10 epochs; eval uses the
**last** epoch, not `phase1_best.pt` — which tracks lowest *total* training loss and lands at
epoch ~11 because the DAG/pert penalties ramp up, so it is not the best perturbation model.)

### exp17 — descriptor / loss / regularization ablations (singles-only)

| tag | change vs baseline | why / hypothesis | outcome |
|---|---|---|---|
| `exp17_joung_tfatlas` | original, 150 ep, MMD+MSE set-loss | first full WM | baseline; superseded by v2 grid |
| `v2_orig` | control-PCA descriptor, no contrastive | cheapest descriptor | weakest |
| `v2_esm` | **ESM-2 descriptor** | protein-aware zero-shot generalization | better resid pΔ (0.238) |
| `v2_esm_con` ★ | ESM + **InfoNCE contrastive** (drift-removed) | force each TF's predicted Δ to retrieve its OWN observed Δ — a specificity signal baselines can't fake | **winner**: disc_resid 0.615 |
| `v2_esm_resid` | residual_target=true (MSE on Δ−drift) | retarget the cascade MSE to the driftless signal | **collapses** |
| `v2_esm_resid_con` | residual_target + contrastive | rescue residual via contrastive | unstable (epoch-14 blowup) |
| `v2_esm_mmd` / `_mmd_con` | MMD distributional set-loss (±contrastive) | match perturbed *distribution* not just mean | no gain over esm_con |
| `v2_esm_mmdmse` / `_mmdmse_con` | MMD+MSE (±contrastive) | restore exp14 recipe | no gain |
| `v2_esm_resid_sephead_lin/_mlp` | separate pert-decoder head (lin / MLP-1024×2) | decouple the residual target from the GRN decoder (resolve the decoder conflict that collapsed `resid`) | trains stably (loss 0.64) but no eval win |
| `v2_esm_l128`, `v2_esm_con_l128`, `v2_esm_resid_l128`, `v2_esm_resid_con_l128` | latent_dim=128 | more capacity | **all diverged to NaN** (singular SCM solve ~epoch 11–39); shelved |

**Why residual-target fails:** the shared **linear** decoder must satisfy reconstruction (full Δ,
*with* drift) AND the pert MSE (driftless) — a direct conflict. Contrastive (InfoNCE on
drift-removed Δ) supplies the same specificity signal *without* retargeting the decoder → it works
where residual-target collapses. **`esm_con` is the chosen exp17 WM.**

### exp19 — nonlinear composition grid (the RL world models)

| tag | change vs esm_con | why | notes |
|---|---|---|---|
| `v2_esm_con_ugate` ★ | **u = MLP(S)** (nonlinear gate), linear ε-mix, gate_h=256 | only *singles-trainable* path to non-additivity: the gate sees multi-program single shifts | **the RL WM** (best combos) |
| `v2_esm_con_epsmix` | nonlinear ε-mix, u=\|S\| | non-additivity via the ε-combination; zero-init → starts at linear baseline | |
| `v2_esm_con_ugate_epsmix` | both u-gate + ε-mix | max singles-only non-additivity capacity | |
| `v2_esm_con_ident` | + identity/fate-reg | encourage fate-discriminative latents | lower drift in pred-Δ (0.58 vs 0.80) but same fate separability |
| `v2_combo_gate512` | ε-mix, gate_h=512 | more gate capacity | lowest train loss (24.0) |
| `v2_combo_resid` | residual_target + ε-mix | "best singles" per earlier notes | **worst** for combo RL + fate (see §7) |
| `v2_combo_resid_hicon_g512` | residual + ε-mix + gate512 + contrastive_w=2 | push residual + specificity | |
| `v2_esm_con_ugate_combo` | u-gate + **combo training** (hash split, frac 0.7, 35/40 pairs) | learn real epistasis from combo data | **leakage** for RL eval (§7) |
| `v2_esm_con_ugate_combo_ho12` | u-gate + combo training **holding out the exact 12 RL-test pairs** | the *honest* combo-WM test | trained 2026-06-25, **28 train / 12 test**; RL-eval running |

**Families:** (descriptor) orig→esm; (specificity) contrastive; (distribution) MMD; (decoder
conflict) residual-target ± separate-head; (non-additivity) u_gate / eps_mix / gate512; (epistasis
data) combo-trained ± honest holdout; (capacity) latent-128 [failed]; (fate) ident-reg.

---

## 3. Perturbation evaluations (WM, not RL)

Four eval scripts in `exp17/evaluation/`, all on Δ, reporting raw + drift-removed (`*_resid`).

**Metrics.** `pearson_delta` (Δ corr; drift-saturated), `de_direction` (sign agreement),
`effect_spearman` (effect-size rank), `deg_auroc` (recover DE genes), **`discrimination_cos`**
(is Δ_pred_i nearest its OWN Δ_obs_i among held-out TFs — *the* uncheatable metric), `fate_top1`
(dominant-cell-type match vs a developmental reference).

**Baselines (`baseline_eval.py`)** — the bars the model must clear: `context_mean` (Δ=0),
`perturb_mean` (= drift), `corr` (gene–gene control correlation), `coexpression` (kNN of training-TF
Δ), and for combos `additive_obs` (Σ *observed* singles — an oracle null).

### 3.1 Held-out singles (v2_esm_con, 169 test TFs)

| metric | raw | resid | perturb_mean (drift) | coexp |
|---|---|---|---|---|
| pearson_delta | 0.674 | **0.050** | 0.736 | 0.715 |
| discrimination_cos | 0.504 | **0.615** | 0.500 | 0.500 |
| effect_spearman | 0.380 | 0.015 | — | — |
| fate_top1 | 0.053 | — | 0.024 | 0.030 |

**Reading:** raw pΔ (0.674) is *below* the drift baseline (0.736) — raw is not where the model adds
value. The model wins **only** on drift-removed **discrimination** (0.615 vs 0.5 chance): it can tell
held-out TFs apart on the residual axis. Effect-size ranking and absolute-fate prediction are weak.

### 3.2 Combos / triples (42 sets) — the model loses to the oracle

| | do-op (mutilation) | additive_obs (oracle) |
|---|---|---|
| pearson_delta | 0.125 | **0.253** |
| discrimination | 0.516 | 0.559 |
| fate_top1 | 0.048 | **0.405** |

The learned do-operator **loses to summing observed singles** on every axis. The mutilation operator
adds no compositional value over additivity.

### 3.3 Epistasis quartiles — epistasis-neutral

Binning combos by non-additivity E = ‖Δ_obs − Δ_add‖/‖Δ_obs‖: the do-op **never beats the additive
oracle in any quartile** (`do_wins_frac = 0` everywhere; gap stays negative). The apparent
"gap closes on epistatic combos" is the *oracle degrading*, not the model improving. Singles-only
training cannot learn interaction — confirmed.

### 3.4 Verdict on the WM

Solid single-TF *specificity* on the residual axis (disc 0.615); **discrimination-dead on combos
and epistasis-neutral**. This is the central limitation that the fate reframing later sidesteps.

---

## 4. The annotation arc (TF → developmental fate)

Goal: give each TF a fate label, both to interpret the atlas and to supply RL targets. Four stages,
each motivated by a flaw in the previous.

**Stage 1 — mouse MOCA (Cao 2019).** Cross-species orthology to early mouse trajectories
(leak-free nuisance removal). Used inside `heldout_zeroshot_eval.fate_space()`. Problem: mouse
orthology drops ~⅓ of genes and the directed-differentiation reference is a species mismatch.

**Stage 2 — human fetal Cao 2020 (S6 aggregate).** `human_annotate.py`: 1,084 Joung TFs (≥50 cells)
× 172 organ-cell-types, 6,557 shared genes. **Method shoot-out** (cosine vs NNLS/OLS/ridge): the
signature matrix is wildly ill-conditioned (cond(DᵀD)=1.1e17) → NNLS/OLS/ridge **degenerate**
(0 confident calls, <2% agreement with cosine). **Cosine wins decisively.** Permutation-FDR
(gene-shuffle null + BH): q<0.10 → 89 TFs.

**Stage 3 — cell-level matched-norm reference + paper-validated benchmark.** `finalize_annotation.py`:
switch from the S6 aggregate to **cell-level log1p-CP10k means** (`cao_type_mean_logexpr.csv`, 165
types). Then the benchmark was rebuilt around the **Joung paper's own experimentally-validated
TF→fate set** (Fig 3D nominated → Fig 4 validated; 18 pairs, e.g. FLI1→endothelial, MSGN1→smooth
muscle, GRHL3→trophoblast), scored against the **same Cao reference** the paper used, restricted to
the **~82 reachable cell types** (Table S4B — the fates a 7-day hESC differentiation can actually
produce; scoring against all 172 is unfair to every method).

| method | reachable r@1 | r@5 | med-rank | all-172 r@1 |
|---|---|---|---|---|
| cosine_S6 (shipped) | 0.25 | 0.56 | 3 | 0.19 |
| **cosine_celllevel** ★ | **0.31** | 0.69 | 3 | 0.19 |
| celltypist_LRsig | 0.31 | 0.75 | 2 | 0.25 |
| celltypist_percell ΔP | 0.00 | 0.18 | 21 | 0.00 |
| chance | 0.07 | 0.26 | 41 | 0.03 |

**Reading:** on the paper's validated set, cosine on cell-level human-fetal signatures reaches
**4.4× chance top-1, correct fate in top-2 by median** — far above the shipped aggregate; CellTypist's
intended per-cell mode *saturates* (one label for all TFs — shared hESC baseline). **Production
method = `cosine_celllevel`** (all 1,084 TFs; 84 at q<0.10, 109 top-decile by perm-FDR). Calls are
biologically right: FLI1→vascular endothelial, MSGN1→smooth muscle, HNF4A→intestinal epithelial,
GRHL3→trophoblast-like.

**Stage 4 — data-driven Leiden lineages.** `lineage_clusters.py`: replace the hand-written
keyword→lineage map with **Leiden clustering of the Cao cell-type signatures** (drift-removed,
res 0.4) → **7 emergent lineages** over 165 cell types: epithelial (L0), neural (L1, the largest TF
attractor), lymphoid (L2), myeloid (L3), stromal/mesenchyme (L4), vascular endothelial (L5),
mesothelial (L6). Each TF inherits the lineage of its annotated cell type. Of these, **5 lineages
have ≥4 confident driver TFs** (used as RL fate targets, §8).

---

## 5. RL — the goal-conditioned reach task

`experiments_rl/`. The frozen WM (gate-aware do-operator) is the environment; only the policy trains.

**State** `[z_t ; z_target]` where `z_target` = encoded held-out combo cells.
**Action** a gene SET applied in one step (set-PPO) / one gene per step (seq-PPO) / beam over sets (MPC).
**Reward** (`core/env.py`) potential Φ = signed-Pearson(decoded Δ, observed Δ) on the target's DEGs,
penalized by the hard-Gumbel DAG ensemble disagreement (pessimism). Three modes:
- `raw` — Φ on raw Δ. **Drift-saturated**: every TF aligns with the shared program → Φ≈const → no
  gradient → PPO learns only drift / collapses to entropy.
- `resid` — Φ on Δ−drift. The learnable signal.
- `blend` — ½raw+½resid.

**Critical gate-aware fix.** `intervene_shift_mutilate` does *not* apply ugate's `refine_u`/`mix_eps`
(they live in the loss); `wm_operator.apply/apply_many` were routed through them (hasattr-guarded →
Norman unchanged). Without this, freezing a ugate WM silently drops its epistasis.

**Methods.** MPC (beam, no training, re-plans per target); sequential-PPO; set-PPO (set = combo);
distillation (BC←MPC + PPO fine-tune). **Pools:** `screened` (~1,689 TFs), `curated` (top-256 by
effect ∪ combo genes), `full` (9,923).

**Value-add ladder / baselines** (`core/baselines.py`, `validate.py`): random → greedy-1 → additive
→ MPC, scored on the same DEG signed-Pearson `match` (+ recall of true drivers, dist_ratio, AUROC).

### 5.1 Reach-combo results (12 held-out combos; `rl_full_metrics`)

| method | pΔ raw | pΔ **resid** | disc_resid | Jaccard vs MPC |
|---|---|---|---|---|
| true-combo do-op (oracle) | 0.152 | −0.092 | 0.591 | 0.08 |
| **MPC · curated** | 0.395 | **0.489** | 0.50 | 1.00 |
| distill (BC+PPO) · curated | 0.395 | 0.489 | 0.50 | **1.00** |
| set-PPO · resid · curated | 0.315 | 0.421 | 0.50 | 0.33 |
| seq-PPO · resid · curated (stab, lr 1e-4) | — | 0.333 (match) | — | — |
| set-PPO · raw · curated | 0.209 | −0.017 | 0.50 | 0.00 |
| seq-PPO · raw · curated | 0.158 | −0.119 | 0.50 | 0.00 |
| set-PPO · raw · full (orig) | 0.102 | −0.265 | 0.49 | 0.00 |

**Findings:** (1) **MPC is the method**; distill = MPC by imitation (identical picks, Jaccard 1.0);
PPO fine-tune adds nothing because raw reward is flat. (2) **The drift fix works**: set-PPO·resid
reaches 0.421 ≈ 86% of MPC — a learned policy nearly matches planning *on the real signal*. It was
an objective bug (raw reward → drift), not a fundamental limit. (3) Raw-reward PPO only learns drift
(across-action reward std 0.093). (4) **Discrimination ≈ chance (0.50) for every method** — combos
share the drift, so no policy can *distinguish* them.

### 5.2 Newest reach-combo runs (2026-06-25)

- `set_resid_curated_anneal` (2534571): entropy_final 0.002 → **collapsed, match −0.05** (over-annealing).
- `seq_resid_curated_stab` (2534572): lr 1e-4 → **match 0.333** (stabilized seq-PPO, recall 0.21).
- `set_resid_curated_comboWM` (2534573): on the combo-trained WM → **match 0.716 but n=5** (only 5
  pairs were held out by that WM's hash split → leakage; see §7).

---

## 6. Transfer + readiness

- **Joung→Norman transfer** (`transfer_norman`): zero-shot do-op pΔ 0.166 raw / 0.024 resid,
  discrimination ≈ chance → context-specific; no cross-dataset generalization.
- **RL readiness** (`docs/RL_READINESS_ASSESSMENT.md`): clears the discrimination gate over trivial
  baselines for single-step DE-state-reaching RL; **not** ready for fate-directed RL on the original
  framing (combo discrimination ≈ chance). This motivated the WM bake-off and the fate reframing.

---

## 7. "Does a better world model help RL?" — the WM bake-off

All combo-RL had only ever run on `v2_esm_con_ugate`. MPC (needs no per-WM training) was run on
every WM trained alongside the combo model, reporting raw + drift-removed pΔ + 12-way discrimination.

| WM | pΔ raw | pΔ resid | disc | honest? |
|---|---|---|---|---|
| **ugate** (current RL WM) | 0.427 | **0.503** | 0.083 | ✅ singles-only |
| ~~ugate_combo~~ | 0.933 | 0.921 | 0.083 | ❌ **leakage: 10/12 test pairs in WM training** |
| combo_resid_hicon_g512 | 0.456 | 0.490 | 0.083 | ✅ |
| combo_resid ("best singles") | 0.514 | 0.437 | 0.083 | ✅ |
| esm_con_ident | 0.324 | 0.439 | **0.167** | ✅ |
| combo_gate512 | 0.338 | 0.418 | 0.083 | ✅ |

(chance disc = 1/12 = 0.083.)

**Findings:** (1) **No honest WM beats ugate** on combo-reach (pΔ_resid 0.503). (2) `ugate_combo`'s
0.92 is **memorization** — the WM's hash combo-split overlapped the RL test set by 10/12. (3)
**Discrimination is at chance for *every* WM**, including the leaked combo-trained one — no world
model makes the held-out combos distinguishable; `ident` is the only one nudging above (0.167), via
lower-drift predictions. (4) `combo_resid` (the "best singles" model) is *worse* for combo RL and
worst for fate separability — residual training doesn't transfer to planning.

**The honest combo-WM test.** Because the leakage made `ugate_combo` uninterpretable, a new WM
(`v2_esm_con_ugate_combo_ho12`) was trained holding out the *exact* 12 RL-test pairs (confirmed
**28 train / 12 test**). Its RL-eval (set-PPO·resid·curated, 12 truly-held-out pairs) is **running
(job 2537050)** — the first valid measurement of whether combo training helps RL. *[append result]*

**Net:** the gain is not in the world model — every WM is discrimination-dead on combo-reach. The
lever is the **task**.

---

## 8. The fate-target reframing

**Motivation.** Joung is *directed differentiation*: each single TF maps to a cell **fate**, and the
combos are lineage-driver combinations chosen to reach fates — not abstract epistasis probes. The
original RL ("reproduce combo #7") throws away the fate annotations, has only ~12 noisy targets, and
runs straight into the drift problem (every combo shares "leaving pluripotency"). The Joung-native
task: **target = a developmental-lineage centroid; reward = reach that fate's residual signature;
supervision = the TF→fate atlas; eval = did the planner reach the intended lineage.**

### 8.1 Separability ceiling — is there signal? (before investing)

`fate_separability.py` on the ugate WM, confident drivers (q<0.10), 8 coarse fates, chance 0.12.
Nearest-lineage leave-one-out accuracy in the spaces RL uses, raw vs drift-removed:

| space (what it is for RL) | RAW r@1 | RESID r@1 | RESID r@3 | drift frac |
|---|---|---|---|---|
| latent_target (the RL goal state) | 0.08 (**below chance**) | 0.21 | 0.55 | 0.92 |
| pred_delta (what the reward sees) | 0.24 | 0.25 | 0.54 | 0.80 |
| obs_delta (**data ceiling**) | 0.39 | 0.33 | 0.60 | 0.78 |

**Reading:** (1) **drift is 80–95% of every representation** — the quantified core problem. (2) fates
are separable **only in the residual and only at coarse lineage** (fine 165-type → chance); the RL
goal state is *below chance* raw, 0.21 resid. (3) The ceiling is **modest**: top-1 ≈ 0.25 (WM) / 0.39
(data), **top-3 ≈ 0.55**. (4) The WM delivers ~70% of the data ceiling. (5) Confident drivers only —
on all 1,084 TFs everything is ~chance. *Bake-off (above): ugate ≈ ident on this; combo_resid worse.*

### 8.2 Fate-MPC — it works

`fate_rl.py`, 5 Leiden lineages, curated pool, **drift-removed reward**:

| lineage | MPC picked | resid-match | near@1 | near@3 |
|---|---|---|---|---|
| stromal/mesenchyme | **MSGN1** | 0.97 | ✓ | ✓ |
| vascular endothelial | **FLI1** | 0.88 | ✓ | ✓ |
| lymphoid | FOXE3, KLF4 | 0.88 | ✓ | ✓ |
| epithelial | **CDX1** | 0.88 | ✗ | ✓ |
| neural | CLOCK | 0.86 | ✗ | ✓ |

**nearest-lineage recall@1 = 0.60, recall@3 = 1.00** (chance 0.20; random sets 0.21). The planner
recovers **real master regulators** (FLI1→endothelial, MSGN1→mesoderm, CDX1→gut). Contrast the
reach-combo task on the same WM: discrimination ≈ chance. **The fate framing converts a
discrimination-dead problem into one where the planner hits the right lineage every time in top-3.**

### 8.3 Fate set-PPO — collapses (and why)

| variant | recall@1 | recall@3 | behavior |
|---|---|---|---|
| fate-MPC | **0.60** | **1.00** | per-target |
| set-PPO · resid · curated, **5 centroid goals** | 0.20 | 0.60 | near-constant set (DDB1+KLF4) |
| set-PPO · resid · curated, **~60 per-driver goals** | 0.12 | 0.76 | 2 generic sets (FOXJ2+POU4F3) |

Both PPO variants **collapse to target-agnostic generic sets**, even with 12× more goals — so it is
**not** a goal-diversity problem. It is the **reward landscape**: because lineages *overlap* (the
separability silhouette is negative), a few generic "strong-differentiation" sets get high
drift-removed match to *many* lineage targets at once, so PPO has no gradient to specialize and
exploits the generic basin (training reward ~0.80). **MPC wins only because it re-plans per target.**
This is the same MPC≫PPO story as combos, made starker.

**The principled fix (next experiment):** a **discriminative/contrastive reward**
`r = match(target lineage) − max_{other} match(other lineage)` — directly rewards specialization,
aligns reward with the nearest-lineage eval metric, and kills the generic-set exploit. *(Not yet run.)*

---

## 9. Where things stand & recommendations

**Established:**
- **WM:** `esm_con` (specificity via contrastive) is the singles WM; `ugate` adds the only
  singles-trainable non-additivity and is the RL WM. Residual-target collapses (decoder conflict);
  latent-128 diverges; MMD/sep-head/ident give no eval win.
- **WM is the limit on combos:** every honest WM is discrimination-dead on combo-reach (~chance);
  the apparent combo-WM win was leakage. Honest combo-WM test running (§7).
- **RL on combos:** MPC is the method; distill = MPC; PPO matches MPC *only* with the drift-removed
  reward; raw reward learns drift.
- **Annotation:** `cosine_celllevel` on the human fetal atlas, validated against the Joung paper's
  own TF→fate set (4.4× chance top-1); data-driven Leiden lineages replace the keyword map.
- **Fate reframing:** the better-posed task. **Fate-MPC reaches the right lineage reliably**
  (recall@3 1.00, real master TFs); a learned fate-policy needs a discriminative reward (PPO
  collapses on absolute-match reward because lineages overlap).

**Recommended next steps (in order):**
1. **Discriminative-reward fate set-PPO** — the principled fix for the PPO collapse (§8.3).
2. **Honest combo-WM RL eval** (running) — settle whether combo training helps RL at all.
3. **Fate-MPC is the deployable planner now** — for "which TFs reach cell type X," at coarse lineage
   (recall@3 ≈ 1.0). Do *not* chase fine-grained fate (out of reach for this WM) or a better combo WM
   (no honest WM beats ugate).
4. (Optional) more combo data is the only real lever for genuine epistasis — Joung has 40 pairs;
   Norman has ~107.

**Running jobs (2026-06-25):** honest combo-WM set-PPO·resid·curated (2537050); per-driver fate
set-PPO done (collapsed). Append the honest-WM number to §7 when it lands.

---

## Appendix — file map

- WM grids: `experiments/exp17_joung_tfatlas/results/grid/`, `experiments/exp19_nonlinear_compose/results/grid/`; configs in `…/configs/ablations/`.
- WM evals: `exp17/results/{heldout_eval,combo_eval,baseline_eval,epistasis_quartile}/`; `exp19/evaluation/fate_separability.py` (mouse-ref WM fate eval).
- Annotation: `exp17/analysis/human_dev_annotation/` — `human_annotate.py`, `finalize_annotation.py` (cosine_celllevel), `paper_ground_truth.py` + `paper_gt_rescore.py`, `lineage_clusters.py`; outputs `joung_human_annotation_celllevel.csv`, `joung_paper_ground_truth.csv`, `cao_celltype_to_lineage.csv`, `joung_tf_lineage.csv`, `human_annotation_stats_v2.json`.
- RL: `experiments_rl/core/` (`wm_operator.py` gate-aware do-op, `wm_data_joung.py` leak-free combo split + lineage targets, `env.py` reward modes, `scoring.py` `Lineage` key, `setup_stack.py`); methods `rl0_mpc/`, `rl1_seq_ppo/`, `rl2_set_ppo/`, `rl3_mpc_distill/`; analyses `analysis/fate_separability.py`, `fate_rl.py`, `fate_set_ppo.py`, `mpc_wm_bakeoff.py`.
- Aggregate RL metrics: `experiments_rl/rl0_mpc/results/rl_full_metrics/summary.json`; bake-off `analysis/mpc_wm_bakeoff.csv`; fate `analysis/fate_rl_mpc.csv`.

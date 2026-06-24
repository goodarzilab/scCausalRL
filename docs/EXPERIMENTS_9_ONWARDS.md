# Experiments 9 → 17 + the RL/planning suite — methods, evals, and results

This document explains every experiment from **exp9 onward**, the **evaluation harnesses**
we built, the **main results** (with the path to each figure / result CSV), and how they
connect. All numbers are pulled from the on-disk `summary.csv` files unless noted.

**Recurring metrics (defined once):**
- **DEG-AUROC** — can the predicted per-gene effect magnitude separate the perturbation's
  true differentially-expressed genes from the rest (direction-agnostic; target gene masked).
- **pearson-DEG / DE-match** — signed Pearson of predicted vs observed effect on the DEGs.
- **Baselines:** `corr` = LOO gene-gene correlation from control cells; `coexp` =
  co-expression-neighbour; `additive_obs` = sum of the two *observed* single effects
  (an oracle, not zero-shot).
- **The do-operators:** `shift_add` = additive composition of single-gene shift predictions
  (zero-shot, provably additive for a linear SCM+decoder); `shift_uscore` = the shift-native
  **mutilation** do-operator `z=(I−MAᵀ)⁻¹ε̃`, the only zero-shot predictor that can be
  non-additive.

---

## exp9 — Is the GRN *learning* regulatory structure or *copying* the CollecTRI prior?
`experiments/exp9_prior_generalization/`

**Question.** exp2's 0.96 CollecTRI "recovery" was a supervised tautology (the prior is
pushed onto the same edges we then score). Does the SCM `A` generalize beyond the supervised
edges? **Design:** hold out **whole TFs** (not random edges) and measure recovery of the
held-out TFs' targets. Grid: **C1 unsup** (no prior), **C2 held-TF** (prior on 70% of TFs,
score the held-out 30%), **C4 big-init**.

**Main results** (`results/grid/exp9_c*/{shift_eval,norman_eval}/summary.csv`):
- Held-out single-pert prediction is preserved across cells: C1 K562-test **0.719**, C2
  **0.712** (corr 0.558); Norman external C1 **0.736**, C2 **0.728** (corr 0.607).
- **GRN held-out-TF recovery AUROC ≈ 0.889 vs a correlation floor ≈ 0.53** (the
  `dag_analysis.ipynb` held-out-TF cell) → the GRN **generalizes** to unseen TFs; the prior
  *corroborates* structure rather than *creating* it. C4 big-init broke (spectral blow-up,
  AUROC 0.50).
- **Figure / result:** `experiments/notebooks/dag_analysis.ipynb` (held-out-TF ROC);
  per-cell numbers in `experiments/exp9_prior_generalization/results/grid/*/`.

---

## exp10 — Sparsity + latent disentanglement (can we sparsify `A` and de-correlate ε?)
`experiments/exp10_sparsity_disentangle/`

**Question.** (a) Does **gate-sparsity** on the DAG help or hurt? (b) Can we **disentangle**
the exogenous ε so it doesn't carry all the information (we don't want z≈ε)?

**Main results** (`results/grid/exp10_g_ls*/shift_eval/summary.csv`):
- **Gate-sparsity is net-negative:** as λ rises, held-out AUROC falls **0.704 → 0.689 →
  0.621** (g_ls2/g_ls10/g_ls30) — sparsity kills the active `A` (spectral → 0). Use
  `lambda_sparse=0` (the exp8 DAG-revival setting) instead.
- **ε-decorrelation (decorr100) is the disentanglement win:** off-diagonal cov(μ)
  **0.468 → 0.332** while keeping `A` active and z informative (decorr1000 over-shrinks).
- Diagnostic: `experiments/exp10_sparsity_disentangle/evaluation/eps_z_diagnostic.py`
  (ε-vs-z silhouette / Wasserstein / off-diag table).

---

## exp11 — Norman-native Tier-1 (train the shift model directly on Norman; additive combos)
`experiments/exp11_norman_shift/`

**Main results** (`results/phase1/{shift_eval,norman_combo}/summary.csv`):
- Held-out Norman **singles 0.898** — but the **co-expression baseline (0.935) beats it**
  (Norman's broad responses make coexp very strong; honest caveat).
- **Combos:** `shift_add` **0.886**, `additive_obs` (oracle) 0.953, `shift_uscore` (bolted-on
  mutilation) **0.829** (hurts), `corr` 0.781.
- **Lesson:** mutilation applied *only at eval* (not trained) *hurts* — motivating exp12/14.
- **Figure:** `experiments/figures/norman_native_eval_summary.png` (singles + combos across
  the Norman models).

---

## exp12 — Shift-native **mutilation**, trained coherently (the do-operator that works)
`experiments/exp12_shift_mutilate/`

**Question.** If the mutilation operator is *trained* (not bolted on), does it beat additive
on combos? Grid: Norman × {loading-clamp, shift-clamp} + a Replogle cell.

**Main results** (`results/grid/exp12_*/{shift_eval,norman_combo}/summary.csv`):
- **`norman_shiftclamp`:** singles 0.891; **combos `shift_uscore` 0.896 > `shift_add` 0.858**
  (first coherent mutilation win), `additive_obs` 0.953.
- `norman_loadclamp`: 0.847 / 0.838 (loading-clamp `u` near-degenerate → ties additive).
- `repl_shiftclamp` (Replogle K562): held-out single **0.714** (corr 0.558).
- **Lesson:** shift-native mutilation, trained, is the do-operator to carry into RL.
- **Figures:** `experiments/figures/combo_methods_auroc_spearman.png`;
  `combo_epistasis_scatter_shift_native.png`;
  `combo_epistasis_tertiles_nrm_mut_shiftclamp.png` (vs `_loadclamp` / `_additive_tier1`).

---

## exp13 — Cross-cell-line transfer, in-modality (K562-trained → RPE1)
`experiments/exp13_crosscell_rpe1/`

**Main result** (`results/rpe1_eval/summary.csv`): a K562-only model transfers **weakly** to
RPE1 — **model 0.694 vs corr 0.679** (barely above baseline). This weak transfer motivated
the multi-cell-line model (exp16).

---

## exp14 — Norman **combo-trained** (Tier-3) — the frozen world model for RL
`experiments/exp14_norman_combotrain/`

**Design.** Train singles + a 70% subset of combos (shift-native mutilation); hold out 30% of
combos. **This checkpoint (`phase1_epoch150.pt`) is the frozen world model used by the
`experiments_rl` planners.**

**Main results** (`results/grid/exp14_norman_combotrain/{shift_eval,norman_combo}/summary.csv`):
- Held-out **singles 0.918** (best single-pert; coexp 0.935 still edges it).
- **Combos (all 107):** `shift_uscore` **0.910 > `shift_add` 0.869**, `additive_obs` 0.953.
- **Held-out 31 test combos:** `shift_uscore` **0.899 > `shift_add` 0.856** (+0.043) — the
  validated mutilation edge that the RL **falsification gate** must reproduce.
- **Honest ceiling:** mutilation beats zero-shot additive but **not** the observed-singles
  oracle (0.953) → the operator is *epistasis-neutral*, not yet *epistasis-capturing*.

---

## exp15 — Modality-conditioned **combined** model (Replogle CRISPRi + Norman CRISPRa)
`experiments/exp15_modality_combined/`

**Design.** One model over two modalities, with a per-modality shift scale; tests whether a
combined model serves both screens.

**Eval we built:** `evaluation/run_modality_eval.py` — **modality-aware** Norman eval (the
stock combo eval omitted the modality and silently used the global scale). Encodes controls
with modality CONTROL, scores Norman perts with the CRISPRa scale.

**Main results** (`results/grid/exp15_modality_combined/modality_eval_norman/summary.csv`):
- Norman **singles 0.732**; **combos** `shift_add` 0.741, `shift_uscore` 0.744 (mutilation
  edge ≈ 0), `additive_obs` 0.846, `corr` 0.575.
- **Lesson:** the combined model **underperforms the Norman-native exp14** (0.73 vs 0.92
  singles; 0.74 vs 0.91 combos) — cross-modality conditioning costs per-modality fidelity →
  **exp14 stays the RL world model.**
- **Figure:** `experiments/figures/exp15_modality_eval.png` (2×2: combo/single × AUROC/Spearman).

---

## exp16 — 4-cell-line CRISPRi world model + cross-cell eval
`experiments/exp16_multicellline/`

**Design.** Joint training on **K562 + RPE1 (Replogle) + HepG2 + Jurkat (Nadig)**, all CRISPRi,
on a shared 6,642-gene axis with the cell line as a **modality token** (encoder modality
embedding + per-line shift scale). Timed out at ~ep83/150 (3-day walltime); the **ep80
checkpoint is healthy** (spectral 0.94).

**Eval we built:** `evaluation/run_crosscell_eval.py` — per-cell-line held-out single-pert
prediction (seed-42 test split, genes the model never trained on in any line), modality
threaded into control encoding + shift.

**Main results** (`results/grid/exp16_multicellline/crosscell/summary.csv`):

| line | n | model AUROC | corr AUROC |
|---|---|---|---|
| K562 | 158 | 0.654 | 0.523 |
| HepG2 | 184 | 0.647 | 0.538 |
| Jurkat | 168 | 0.649 | 0.536 |
| **RPE1** | 171 | **0.707** | 0.570 |

- **Joint training beats correlation in every context and *rescued* RPE1** (0.707 vs 0.570,
  +0.14 — vs exp13's K562-only 0.694≈0.679). Small K562 dilution (0.654 vs exp2's 0.704) is
  the usual multi-task tradeoff.
- **Figure:** `experiments/figures/exp16_crosscell.png` (per-line model vs corr AUROC).

---

## exp17 — Joung TF Atlas world model (TF overexpression → differentiation) — TRAINING
`experiments/exp17_joung_tfatlas/`

**Design.** Train the shift+mutilation world model on **single TF overexpressions** from the
Joung TF Atlas (GSE217460), hold out combos + triples (GSE217066) for compositional eval.
**Data engineering:** the raw atlas is dense (1.15M × 37,528 ≈ 172 GB) → a preprocessing job
(`data/preprocess_joung.py`) chunk-reads it via h5py to a compact **9,923-gene** axis (8,506
combinatorial ∪ 1,769 TF genes). **GFP/mCherry cells are the controls** (~128k).

**Run:** full dataset **1,144,579 × 9,923**; **1,689 TFs split 80/10/10 → 1,351/169/169**
(test = held-out singles). Training cells = 1,144,390. Live: wandb
`CausalBioRL/exp17_joung_tfatlas`. Combos/triples held out: **44 pairs + 3 triples**.

**Planned evals** (built once a checkpoint lands): held-out singles (169 test TFs) + combos
(44 pairs) + **triples (3)** — additive vs mutilation vs observed-oracle.

---

# The RL / planning suite — `experiments_rl/`

A frozen-world-model planner that finds gene interventions to reach a target perturbation
state. Full design + math: `experiments_rl/RL_TRAINING_DESIGN.md` and `RL_ARCHITECTURES.md`.

**Shared setup (all methods).** World model = **frozen exp14** (no co-training). State
`s=[z_t ; z_target]` (current latent + the data-grounded target perturbation's latent).
Transition = the **shift-native mutilation do-operator** under the eval-A point estimate,
with a hard-Gumbel DAG ensemble for a pessimism penalty. Reward = potential shaping on the
**gene-space DE-match** `Φ` (decode the reached state, signed-Pearson vs the observed DE on
DEGs), penalized by ensemble disagreement. Action vocabulary = the candidate gene pool (87
screened, or the full 8,000-gene zero-shot axis).

### The four architectures (clearly)

**M0 — MPC (oracle / upper bound)** · `rl0_mpc/`
No learning. **Beam search with pruning** over the do-operator: score every candidate single
(batched), keep the top-B, expand each against all partners (B·C pairs), keep the best 2-set,
re-rank finalists by the pessimistic (mean − α·std) DE-match. `beam=1,k=1` ⇒ greedy-1. It is
the planning **ceiling** the learned policies chase and the **teacher** for distillation.

**M1 — Sequential PPO** · `rl1_seq_ppo/`
Goal-conditioned actor-critic (MLP on `[z_t,z_target]`, action head = #candidate genes). One
gene per step; a combo is reached as a **sequence** of single-gene steps (ε threaded across
steps → genuine multi-step counterfactual). Standard PPO+GAE on the gene-space reward. Tests
whether a learned, amortized policy can match the MPC ceiling; imposes an (artificial) order
on the gene set.

**M2 — Set-based PPO** · `rl2_set_ppo/` ★
The policy emits an **unordered set** of ≤k genes applied *simultaneously* (matching how a
combo is delivered). It's drawn as an **autoregressive token recipe** over
`V={STOP}∪candidate_genes`: a shared encoder feeds k slot-heads; `π(set)=∏_i π(a_i|s,a_{<i})`,
each factor a masked softmax (STOP forbidden at slot 0; already-chosen genes masked → no
duplicates); the chosen-gene context is an **order-invariant sum** of gene embeddings. `log π`
and entropy are per-slot sums (the only change PPO sees). STOP lets the policy pick its own
cardinality. Removes the false ordering of M1.

**M3 — MPC → distillation** · `rl3_mpc_distill/`
Bridges the oracle to a learned set policy for the thin task distribution: (1) collect MPC
plans over the train targets, (2) **behavior-clone** the set policy to the MPC-chosen set
(cross-entropy on the token sequence), (3) **PPO fine-tune** on the gene-space reward. The
warm start is what lets the policy approach the oracle.

### Two key methodological findings (built into the harness)
1. **Latent-distance objective is gameable** → we **pivoted to a gene-space objective**
   (`core/scoring.py`: signed Pearson of the decoded reached-state DE vs observed DE on DEGs).
   On the small (87-gene) pool, gene-space MPC = **0.799 DE-match vs additive 0.535** —
   planning beats additive when it optimizes the right metric.
   (`experiments_rl/rl0_mpc/results/ladder_combo.csv`)
2. **Gene recovery ≈ 0** everywhere — the perturbation→state map is **many-to-one**, so the
   planner reaches the target DE via *different* genes. Reported as an identifiability finding,
   not a failure (it's fine for "reach a target state").

### Large zero-shot pool (8,000-gene axis; ~99% of candidates never screened)
The action pool expands from 87 screened genes to the full axis (shift encoder embeds any gene
zero-shot). Batched do-operator (`core/wm_operator.apply_many`) makes beam search at scale
feasible (~0.3 s/combo).

**Ladder, held-out combos** (`experiments_rl/rl0_mpc/results/fullpool/ladder.csv` + the
per-method `rlN_*/results/*_test.csv`):

| method | DE-match | gene recall |
|---|---|---|
| random (2 of 8000) | 0.327 | 0.00 |
| greedy-1 (best single) | 0.801 | 0.00 |
| top2-marginal | 0.800 | 0.00 |
| **MPC** (beam pairs) | **0.811** | 0.02 |
| seq-PPO | 0.586 | 0.00 |
| set-PPO | 0.659 | 0.00 |
| **distill** (BC+PPO) | **0.753** | 0.00 |

- **MPC ≥ distill > set-PPO > seq-PPO > random.** Distillation closes most of the
  amortization gap (0.59→0.75); set > sequential. At 8,000-pool scale a single zero-shot gene
  (greedy-1, 0.80) already mimics a combo's DE → MPC's edge over greedy is marginal (the large
  pool helps the cheap baseline most). wandb runs: `seq_ppo_fullpool`, `set_ppo_fullpool`,
  `set_ppo_distill_fullpool` under `CausalBioRL`.
- **Falsification gate (operator):** reproduces exp14's held-out mutilation edge
  **0.899 > 0.856** — confirmed before any policy work.
- **Figure:** `experiments/figures/rl_value_add_ladder.png` (DE-match per method, families
  colour-coded: baselines grey/green, learned policies blue/purple, MPC oracle red).

---

## Consolidated headline table

| exp | what | held-out metric | result | figure / CSV |
|---|---|---|---|---|
| 9 | GRN copy-vs-learn | held-out-TF recovery | 0.889 vs ~0.53 floor → generalizes | dag_analysis.ipynb |
| 10 | sparsity/disentangle | K562 single AUROC | gate-sparsity 0.70→0.62 (hurts); decorr fixes ε | exp10 shift_eval CSVs |
| 11 | Norman Tier-1 | combos | shift_add 0.886 > bolt-on mutil 0.829 | norman_native_eval_summary.png |
| 12 | trained mutilation | combos | **shift_uscore 0.896 > shift_add 0.858** | combo_methods_auroc_spearman.png |
| 13 | K562→RPE1 | RPE1 single | 0.694 ≈ corr 0.679 (weak) | rpe1_eval CSV |
| 14 | Norman combo-trained (RL WM) | held-out combos | **0.899 > 0.856**; singles 0.918 | norman_combo CSV |
| 15 | combined modality | Norman | 0.73/0.74 (< exp14) | exp15_modality_eval.png |
| 16 | 4-cell-line | per-line singles | RPE1 0.707 vs 0.570 (rescued) | figures/exp16_crosscell.png |
| 17 | Joung TF Atlas | singles+combos+triples | training (1.14M cells, 80/10/10) | wandb exp17_joung_tfatlas |
| RL | MPC/PPO planning | combo DE-match | MPC 0.81 ≥ distill 0.75 > set 0.66 > seq 0.59 | figures/rl_value_add_ladder.png |

## Evaluation harnesses we built (clearly)
- **`run_shift_eval.py`** (reused) — held-out single-pert: DEG-AUROC / pearson-DEG / Spearman /
  distributional MMD, vs `corr` + `coexp` baselines, target gene masked.
- **`run_norman_combo_eval.py`** — combo prediction with the full method set (`shift_add`,
  `additive_obs`, `shift_uscore`, `uscore_obs`, `corr`) + per-combo epistasis / non-additivity.
- **`eps_z_diagnostic.py`** — ε-vs-z entanglement (silhouette, Wasserstein, off-diag cov).
- **`exp15/.../run_modality_eval.py`** — modality-aware singles + combos (CRISPRa scale).
- **`exp16/.../run_crosscell_eval.py`** — per-cell-line held-out singles, modality-conditioned.
- **`experiments_rl/core/{validate.py,value_add_ladder.py}`** — the planner value-add ladder
  (random→greedy→additive→MPC→PPO) on the gene-space DE-match objective, + reachability /
  OOD-honesty probes.
- **Joung (exp17, planned):** held-out singles (169 test TFs) + combos (44) + triples (3).

> **Figures generated** (`experiments/plot_exp16_rl_figures.py`): `exp16_crosscell.png`,
> `rl_value_add_ladder.png`. Still pending: the **exp17 Joung combo/triple eval** figure
> (after the world model finishes training).

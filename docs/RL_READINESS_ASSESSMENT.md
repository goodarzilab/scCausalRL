# RL-Readiness Assessment — Causal-VAE world model + intervention planner

**Date:** 2026-06-21 · **Scope:** synthesizes exp17 (Joung TF-atlas ablation + baselines), exp14
(the live RL world model, Norman), the `experiments_rl/` planner suite, and the mouse-developmental
fate annotation. **Question:** *do we have a model ready to do RL on yet, and what are the proper
evals?*

> Note on sourcing: this report is reconstructed from the repo (evals, `docs/EXPERIMENTS_9_ONWARDS.md`,
> the annotation pipeline). It does **not** include content from the separate "Review causal RL
> architecture" chat (Claude sessions are isolated) — if that chat fixed decisions not reflected in
> the repo, reconcile against §2/§6.

---

## 0. TL;DR verdict

There are **two different RL problems**, and the answer differs:

| RL problem | World model | Ready? | Why |
|---|---|---|---|
| **A. Reach a target perturbation DE-state** (Norman, gene-space reward) | exp14 (frozen) | **Partially — yes enough to keep iterating** | Planner beats additive/random on the gene-space DE-match it optimizes (MPC 0.811); operator falsification gate passes (0.899 > 0.856). **But epistasis-neutral**: never beats the observed-singles oracle (0.953). |
| **B. Directed differentiation to a target developmental fate** (Joung + our mouse annotations) | exp17 (best = `esm_con`) | **No** | The world model predicts held-out fate **at chance** (`fate_top1` ≈ 0.05 = 1/20; β-cosine ≈ 0); composition loses to the observed-additive null; confident fate labels exist for only ~1% of TFs. A fate-based reward is **not estimable** by the current model. |

**Bottom line:** We can (and do) run RL for *DE-state reaching* on exp14, and it produces meaningful
plans — but it's an *epistasis-neutral* world model, so RL mostly rediscovers strong single/greedy
moves. We are **not** ready for the more ambitious, biologically meaningful goal — *planning TF
interventions to drive a target cell fate* — because the world model cannot yet predict fate above
chance. §6–§7 lay out the gates and the path.

---

## 1. What "ready for RL" actually requires

RL/planning amplifies whatever the world model believes. Four prerequisites, each with a measurable gate:

1. **Accurate transition operator** — the do-operator must predict held-out interventions better than
   naive composition. *Gate:* mutilation do-op > zero-shot additive on held-out combos.
2. **A reward the model can estimate** — the reward must be computable from the model's predictions
   *and* track reality on held-out data. *Gate:* the metric the reward is built on beats baselines
   zero-shot. (DE-match: ✅. Fate: ❌.)
3. **Composition that beats the additive null** — multi-step plans = compositions; if Σ(observed
   singles) beats the operator, the planner has nothing to exploit beyond picking good singles.
   *Gate:* do-op ≥ `additive_obs`. **Currently fails** (both Norman and Joung).
4. **OOD honesty** — RL pushes into unseen intervention space; the model must flag where it's
   guessing. *Gate:* ensemble disagreement correlates with held-out error on OOD interventions.

We pass (1) for Norman, partially (2) for DE-reach, and **fail (3)** everywhere; (4) is implemented
(DAG-ensemble pessimism) but not yet validated as calibrated.

---

## 2. Evidence — aggregated evals

### 2.1 exp17 held-out singles (epoch 60) — model vs **all** baselines
Δ = mean(OE) − mean(hESC control). `*_resid` = drift-removed (drift = mean training-TF Δ). Figure:
`experiments/figures/joung/exp17_singles_panels.png`.

| predictor | pearsonΔ | pΔ_resid | effect_spearman | deg_auroc | discrim_resid |
|---|---|---|---|---|---|
| orig | 0.699 | 0.081 | 0.436 | 0.941 | 0.511 |
| **esm** | 0.729 | **0.238** | 0.461 | 0.944 | 0.591 |
| esm_resid* | 0.383 | −0.202 | 0.226 | 0.883 | 0.500 |
| esm_resid_con* | −0.087 | −0.196 | −0.027 | 0.875 | 0.501 |
| **esm_con** | 0.674 | 0.050 | 0.380 | 0.921 | **0.615** |
| BASE perturb_mean (drift) | **0.736** | 0 | **0.490** | **0.950** | 0.500 |
| BASE coexpression (kNN) | 0.715 | 0.044 | — | 0.934 | 0.523 |
| BASE corr (gene–gene) | 0.056 | −0.205 | — | 0.660 | 0.500 |
| BASE context_mean (Δ=0) | nan | −0.207 | nan | nan | 0.500 |

**Reading:** raw pearson / Spearman / DEG-AUROC are **saturated** — `perturb_mean` (just predict the
mean training perturbation) scores ≥ every model, and `coexpression` is nearly as strong. The model's
genuine value-add is only on **drift-removed pearson + discrimination**, where it beats *all*
baselines including the coexpression prior-knowledge baseline (esm pΔ_resid 0.238 vs coexp 0.044; esm_con
disc_resid 0.615 vs coexp 0.523). `* = residual-target runs collapsed` (see exp17 README).

### 2.2 exp17 composition (combos, n=42; triples n=2 — not meaningful)
Figure: `exp17_combos_panels.png`. `additive_obs` = Σ **observed** single-TF Δ.

| predictor | combo pearsonΔ | combo deg_auroc |
|---|---|---|
| esm_con do-op | 0.125 | 0.834 |
| esm_resid_con* do-op | 0.107 | 0.828 |
| esm do-op | 0.071 | 0.852 |
| BASE perturb_mean | 0.087 | 0.860 |
| **BASE additive_obs** | **0.253** | 0.850 |
| BASE coexpression | 0.092 | 0.849 |

**Reading:** the learned do-op beats the *zero-shot* additive and perturb_mean, but **loses badly to
`additive_obs`** (0.25 vs 0.13). Composition is epistasis-neutral at best — exactly criterion (3)
failing.

### 2.3 exp14 — the live RL world model (Norman) + planner ladder
From `docs/EXPERIMENTS_9_ONWARDS.md` / `experiments_rl/`:
- Held-out **singles 0.918** (coexp 0.935 still edges it — same saturation pattern).
- Held-out 31 combos: `shift_uscore` **0.899 > shift_add 0.856** (operator falsification gate ✅) but
  **< `additive_obs` 0.953** ("epistasis-neutral, not epistasis-capturing").
- Planner value-add ladder (8000-gene pool, held-out combos, gene-space DE-match):

| method | DE-match | gene recall |
|---|---|---|
| random | 0.327 | 0.00 |
| greedy-1 | 0.801 | 0.00 |
| **MPC (beam)** | **0.811** | 0.02 |
| distill (BC+PPO) | 0.753 | 0.00 |
| set-PPO | 0.659 | 0.00 |
| seq-PPO | 0.586 | 0.00 |

- **Two findings baked into the harness:** (i) latent-distance reward is *gameable* → pivoted to a
  gene-space DE-match reward; (ii) **gene recovery ≈ 0** (the intervention→state map is many-to-one →
  the planner reaches the target DE via different genes; reported as identifiability, not failure).
- **Caveat:** at 8000-pool a single zero-shot gene (greedy-1, 0.80) already mimics a combo's DE, so
  MPC's edge over greedy is marginal — i.e. the task barely *needs* multi-step planning yet.

---

## 3. The developmental-fate annotation (mouse MOCA) — methodology + performance

This is what would turn the Joung atlas into a *fate-labeled* environment for problem B, so its quality
is load-bearing. Pipeline: `experiments/exp17_joung_tfatlas/analysis/mouse_dev_annotation/`.

### 3.1 How we did it
Goal: label each overexpressed TF with **which mouse developmental cell-type direction it drives cells
toward**, by aligning the TF's induced *shift* to Cao et al. 2019 MOCA (mouse organogenesis, ~1.3M cells,
E9.5–E13.5) cell-type signatures.

1. **Induced shift** `Δ_TF = mean(OE) − mean(control)` — the shift, not the absolute profile (all cells
   are hESC-derived → absolute profiles share one huge baseline that hides TF identity).
2. **Cell-type signatures** `D_c = mean(type c) − mean over types`, `D ∈ ℝ^{G×C}`.
3. **Score** `cosine(Δ_TF, D_c)` per cell type; `specificity = peak cosine − mean cosine` = per-TF
   **confidence**. An NNLS mixture is also written.
4. **Three corrections that make it work** (naïve deconvolution is degenerate — every TF → same answer):
   (a) **MGI 1:1 orthology** (`mgi_homologene.rpt`, ~18.8k orthologs) instead of symbol-uppercasing
   (~3× more mapped genes); (b) **shared OE-axis removal** — every lentiviral-OE population shares a
   dominant selection/stress/proliferation axis (pairwise shift corr 0.9+); we center per TF and strip
   the **top-4 SVD components**; (c) **early-progenitor reference** — hESC + ~7 d OE reaches early
   progenitors, not E9.5–13.5 terminal types, so restrict to ~20 progenitor `Main_cell_type`s.
5. **n ≥ 50 cells** per TF (small pseudobulks are noise). **1,084 / 1,595 TFs survive.**

### 3.2 Which estimator — NNLS vs cosine vs OLS/ridge
Tested head-to-head. **On the raw shift all four are indistinguishable** (they collapse onto the shared
OE-axis). **After axis removal, cosine is the most stable** (the cell-type signatures are collinear →
unconstrained OLS/NNLS are ill-conditioned). **The estimator is a red herring — the three corrections
in §3.1 are what matter.** The shipped annotation (`joung_tf_developmental_annotation.csv`) uses the
**cosine** score with a specificity confidence column; NNLS/cos CSVs are both kept for audit.

### 3.3 How well it performed
**As an annotation of OBSERVED data — works, but only for confident calls.** High-specificity TFs match
known biology: `MSGN1→mesoderm/jaw-tooth+chondrocyte` (spec 0.17), `ATOH7→neural progenitors` (0.21),
`SCRT2→inhibitory-neuron progenitors` (0.25), `GRHL3→epithelial` (0.16), `SNAI1/2→mesenchyme (EMT)`,
`LMX1B→isthmic organizer`. **But the confidence distribution is thin:** median specificity 0.051,
**only 13/1084 TFs exceed 0.12**. Terminal-fate masters (MYOD1, GATA4, HNF4A) score low *by design*
(their fates aren't in the early-progenitor whitelist). So the annotation is a **gated** label: trust
the top ~1–5%, treat the rest as "not annotatable."

**As a PREDICTION metric (`fate_top1`, held-out singles) — at chance.** Per the exp17 eval
(`fate_space` deconvolves pred & obs Δ to a fate vector, nuisance fit on train):

| | fate_top1 | fate_betacos |
|---|---|---|
| all 5 models | 0.041–0.059 (chance = 1/20 = 0.05) | ≈ 0 |
| esm_con, specificity-gated >0.05 (n=60) | 0.083 | — |
| esm_con, specificity-gated >0.10 (n=7) | 0.000 | — |

Gating does **not** rescue it, and only **1 of 169 held-out test TFs** is even high-confidence
annotatable. **Conclusion:** the annotation is usable as a *label for a small confident subset of
observed TFs*, but the world model **cannot predict fate for a held-out intervention** — so it cannot
yet serve as an RL reward.

---

## 4. Verdict (detailed)

- **World-model quality (exp17 ablation):** best config is **`esm_con`** (ESM input + contrastive, no
  residual target). It clears every baseline on the load-bearing metrics — including the strong
  coexpression prior — but only modestly (disc_resid 0.615 vs 0.50 chance). Residual-*target* training
  is unstable and should be dropped. This is an incremental, not step-change, world model.
- **Problem A (DE-reach):** **ready enough to keep doing RL** on exp14. The reward is estimable, the
  operator gate passes, the planner ladder is sane. The honest limit: epistasis-neutral, and on the big
  pool greedy already ≈ MPC, so the *current task distribution barely rewards planning*.
- **Problem B (fate-directed differentiation):** **not ready.** Reward not estimable (fate at chance),
  composition < additive_obs, labels sparse. Building fate-RL now would optimize a reward the model
  can't compute → reward hacking in an inaccurate model.

---

## 5. Proper RL evals (the design)

Split into **world-model gates** (pass *before* RL) and **planner evals** (given a passing WM).

### 5.1 World-model prerequisite gates
- **G1 Operator falsification:** do-op > zero-shot additive on held-out combos (exp14 ✅ 0.899>0.856;
  exp17 ✅ small). *Keep as a regression test.*
- **G2 Composition vs observed-additive:** do-op ≥ `additive_obs`. **Currently fails everywhere — this
  is THE bar that gates real multi-step planning value.**
- **G3 Reward-estimability:** the held-out metric the reward is built on must beat baselines. DE-match ✅;
  **fate ❌** (do not build a fate reward until this passes).
- **G4 OOD honesty:** bin held-out (and progressively OOD) interventions by ensemble disagreement; show
  disagreement tracks true error (calibration curve). Needed because the planner exploits low-uncertainty
  regions.

### 5.2 Planner evals (given a WM that passes G1–G4)
- **Value-add ladder:** random < additive < greedy < MPC ≤ learned policy, on the *gene-space* reward
  (exists: `experiments_rl/core/value_add_ladder.py`). Headline = does the policy beat greedy-1 and
  additive_obs?
- **Reachability / coverage:** fraction of held-out target states reachable within budget k.
- **Recovery vs identifiability:** report gene-recall but interpret via the many-to-one map (recall ≈ 0
  is expected; score "reached the state," not "found the gene").
- **Counterfactual transfer (the key external check):** validate planned interventions against **held-out
  REAL combos/triples** — see §6.2.

### 5.3 Annotation-grounded evals (problem B, once unblocked)
- **TF→fate recovery:** for each high-confidence target fate, does the planner propose the known driver
  TF(s)? Small (≈13-TF) but clean benchmark from the annotation.
- **Fate-reach:** plan to a target cell-type signature; score fate-cosine of the decoded reached state vs
  the target (only meaningful after G3-fate passes).

---

## 6. How to approach the RL problem with our annotations + the dataset's own annotations

### 6.1 Reward designs (two tiers)
- **Tier-1 (now): gene-space DE-match** — reach a target *perturbation* state; reward = signed-Pearson of
  decoded reached-state DE vs observed DE on DEGs, pessimism-penalized. This is what works.
- **Tier-2 (aspirational): fate-space reward** — reach a target *developmental fate*; reward = fate-cosine
  to a target MOCA cell-type signature. **Blocked on G3-fate.**

### 6.2 Using the dataset's own combo/triple annotations ("state → state")
The Joung combo/triple data (`TF_set`, 44 pairs + 3 triples held out) is **ground-truth multi-TF →
endpoint-state** data. It is **not trajectories** (no intermediate states; endpoints only), so use it as:
- **(a) Offline planner validation:** target = observed combo Δ-state; check whether the planner, given
  the target, (i) reaches its DE-state (DE-match) and (ii) — as a *bonus, not a requirement* — selects the
  constituent TFs (identifiability-aware).
- **(b) Held-out reachability test set:** the combos the WM never trained on (exp14 holds out 30%;
  exp17 holds out all combos) are the honest reach targets.
- **(c) Imitation / offline-RL seed:** treat the observed `TF_set` as the "expert action" that produced the
  observed state → behavior-clone a set policy (this is exactly the exp14 RL `rl3_mpc_distill` pattern, but
  with *data* experts instead of MPC experts). Strongest if we first combo-train the Joung WM (see §6.4).

### 6.3 Using our mouse-fate labels
- Defines a **discrete directed-differentiation problem:** "find the TF set that drives hESC → target
  early-progenitor fate." The ~13 high-confidence TF→fate pairs are a small, clean **recovery benchmark**
  and a source of **target states** for fate-reach.
- Use specificity as a **confidence gate** on both targets and rewards (don't reward toward fates the
  annotation can't call).

### 6.4 Concrete steps to unblock fate-RL (problem B)
1. **Combo-train the Joung WM** (mirror exp14's Tier-3 on Norman): train singles + a subset of Joung
   combos with shift-native mutilation; this is the most direct lever on **G2** (composition) and likely
   on fate too.
2. **Make fate predictable (G3-fate):** add a **fate-space auxiliary loss** (predict the MOCA-cosine
   vector from the decoded Δ) on the *high-confidence* TFs; and/or evaluate/operate only on the confident
   subset. Re-measure `fate_top1` gated.
3. **Calibrate uncertainty (G4):** validate the DAG-ensemble disagreement as an error predictor on the
   held-out combos before trusting the pessimism penalty in fate-space.
4. **Then, and only then,** stand up Tier-2 fate RL with the annotation-grounded evals in §5.3, reusing
   the existing `experiments_rl` MPC/set-PPO/distill stack with the fate reward swapped in.

---

## 7. Appendix — pointers & repro

- exp17 evals: `experiments/exp17_joung_tfatlas/evaluation/{heldout_zeroshot_eval,combo_triple_eval,baseline_eval}.py`;
  results under `results/{heldout_eval,combo_eval,baseline_eval}/`; figures `evaluation/plot_summary.py`
  → `experiments/figures/joung/exp17_*.png`.
- Baselines: `context_mean` (Δ=0), `perturb_mean` (drift), `additive_obs` (Σ observed singles; combos),
  `corr` (gene–gene control correlation), `coexpression` (k-NN of training-TF Δ in control-corr space).
- Annotation: `analysis/mouse_dev_annotation/{README.md,nnls_annotate.py,build_mouse_reference.R}`;
  table `joung_tf_developmental_annotation.csv` (cols: confidence, main_fate, main_specificity, …).
- RL suite: `experiments_rl/` (`rl0_mpc`, `rl1_seq_ppo`, `rl2_set_ppo`, `rl3_mpc_distill`,
  `core/{scoring,validate,value_add_ladder}.py`); design in `experiments_rl/RL_TRAINING_DESIGN.md`,
  `RL_ARCHITECTURES.md`; world model = exp14 `phase1_epoch150.pt`.
- Cross-experiment context: `docs/EXPERIMENTS_9_ONWARDS.md`.

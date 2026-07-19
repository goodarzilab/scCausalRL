# Session handoff — Joung WM ablations, RL training, fate discrimination

## 0. Context / setup
- **Goal:** RL over a frozen causal-VAE **world model (WM)** to plan TF perturbation SETS that
  reach a target cell state; plus WM ablations to improve single-TF specificity & combo epistasis;
  plus developmental-fate "identifiability."
- **Repos/dirs:**
  - WM training/eval: `experiments/exp19_nonlinear_compose` (active; has the v2 grid),
    `experiments/exp17_joung_tfatlas` (data, annotation pipeline).
  - RL stack: `experiments_rl/` (core/ + rl0_mpc/rl1_seq_ppo/rl2_set_ppo/rl3_mpc_distill).
  - W&B project: `CausalBioRL`.
- **Data:** Joung TF atlas — singles `experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad`
  (1.14M cells × 9923 genes, 1771 TFs OE in hESCs + GFP/mCherry controls); combos
  `joung_combo_proc.h5ad` (168k cells, same 9923-gene axis, 40 usable PAIRS + 2 triples,
  ≥20 cells). Norman at `/large_storage/goodarzilab/shervin/CausalBioRL/Norman/`.
- **Frozen WMs (exp19 v2 grid, `results/grid/<tag>/checkpoints/phase1_epoch60.pt`):**
  `v2_esm_con` (baseline), `v2_esm_con_ugate` (gate, best combos), `v2_esm_con_epsmix`,
  `v2_esm_con_ugate_epsmix`, `v2_esm_con_ident` (fate reg), `v2_combo_gate512`,
  `v2_combo_resid` (residual-trained, BEST singles), `v2_combo_resid_hicon_g512`,
  and NEW (training): `v2_esm_con_ugate_combo` (few-shot combo-trained, job 2533394).
  Descriptor = ESM2 (esm_pca100). latent_dim 64. Metric basis caveat: **residual-trained**
  models scored on RAW disc; raw-trained on drift-removed (resid) — else double-subtraction.

## 1. How the Joung RL training works (mechanics)
Goal-conditioned planner over the frozen WM (identical machinery to Norman):
- **WM do-operator** (`core/wm_operator.py`): takes a TF set → sums latent shifts
  (`model.compute_shift`) → mutilated SCM solve → perturbed z → decode → predicted Δ.
  **Gate-aware fix (critical):** `intervene_shift_mutilate` does NOT apply ugate's `refine_u`/
  `mix_eps` (those live in the loss + `epistasis_quartile_eval`), so the operator was made
  gate-aware (`apply`/`apply_many` route through refine_u+mix_eps when present; hasattr-guarded
  so Norman is byte-identical). Validated: pair pred non-additive (rel-diff 0.235).
- **State** `[z_t ; z_target]`; **z_target** = encode the observed cells of a held-out combo.
- **Action:** seq-PPO (one TF/step), set-PPO (autoregressive set, max_steps=1), MPC (beam over sets).
- **Reward:** gene-space DE match = signed Pearson(pred Δ, observed Δ) on DEGs + ensemble pessimism + step cost.
- **Train corpus:** singles_train + combo-pairs_train; eval on held-out combo pairs.
- **Backbone:** `core/wm_data_joung.py` (WMDataJoung, memory-light: sparse h5ad, densify only
  slices — avoids 45GB full densify); `core/setup_stack.py` (`--dataset/--checkpoint/--wm_root/
  --pool/--curated_topk` + `build_stack`). Pools: `screened` (~1689 trained TFs), `full` (9923),
  `curated` (top-256 by effect norm ∪ all combo genes).

### Difference from Norman & the key conceptual issue
Mechanically identical. **Semantically different:** Norman = genetic-interaction (epistasis) screen,
no target cell state. **Joung = directed differentiation** — single TFs map to cell FATES, and
the combos are lineage-driver combinations chosen to reach fates. We are NOT using that: our
processed h5ads were stripped to bare `TF_gene`/`is_control` and `TF_set`/`set_size` (no state
labels). **The natural Joung RL target is an annotated cell FATE/state, not "reproduce combo #7."**

## 2. Experiments attempted + results

### 2a. RL ladder (frozen ugate WM, 12 held-out combo pairs, match = signed-Pearson on DEGs)
Jobs 2532361/2/3 (orig full/screened pool) + 2533403-407 (fixed) + 2533743/4 (resid).
| method | match (raw) | match_resid (drift-removed) | notes |
|---|---|---|---|
| MPC (planner) | 0.432 | **0.489** | best; planning over frozen WM |
| distill (BC←MPC + PPO) | 0.428 | **0.489** | == MPC: picks IDENTICAL gene sets (Jaccard 1.00) |
| distill BC-only (no PPO) | — | 0.471 | already ≈MPC (0.92 Jaccard) → PPO adds ~nothing |
| set-PPO raw · full (orig) | 0.092 | −0.25 | collapsed (entropy stuck ~13/14.8); raw=pure drift |
| set-PPO raw · curated | 0.20–0.23 | −0.01 | curated pool unstuck it on RAW; but resid≈0 (drift) |
| seq-PPO raw · curated | 0.17 | −0.11 | learns weakly; resid≈0 |
| set-PPO blend · curated | 0.071 | — | blend reward BACKFIRED (collapsed) — dropped |
| **set-PPO RESID · curated** | 0.345 | **0.423** | **FIX WORKS** — ≈MPC on residual; learned policy |
| seq-PPO resid · curated | ~0.13 | 0.4 transient | unstable (bounces −0.3..+0.4) |

**RL conclusions:**
1. **MPC is the method**; distill = MPC by imitation (identical picks; PPO fine-tune adds nothing
   because raw reward is flat → no signal to exceed teacher).
2. **PPO-from-scratch on RAW reward only learns DRIFT** (raw pearson 0.1–0.2 is all the shared
   differentiation program; drift-removed resid ≈ 0). Diagnosed via `diag_reward_landscape.py`
   (raw reward across-action std only 0.093; training is 98% single-TF targets where every TF's
   drift-aligned pred scores equally → flat reward → entropy bonus dominates).
3. **THE FIX (user's idea):** train on **drift-removed reward** (`reward_mode=resid`) + curated
   pool → set-PPO reaches **match_resid 0.42 ≈ MPC 0.49**. So a learned, amortized policy can
   nearly match planning **on the real (TF-specific) signal** — it was an objective bug, not a
   fundamental limit. seq-PPO works transiently but unstable.
- Eval scripts: `eval_rl_full_metrics.py` (RL planners scored on WM metric suite +
  drift-removed + MPC-agreement), `transfer_joung_to_norman.py`.

### 2b. WM ablations (held-out singles; combo do-op vs additive on 40 pairs)
| model | disc | pearson_Δ | eff-spear | combo do-op | combo additive |
|---|---|---|---|---|---|
| esm_con | 0.615 | 0.050 | 0.015 | 0.127 | 0.084 |
| ugate | 0.576 | 0.027 | 0.017 | **0.159** | 0.132 |
| ident (fate reg) | 0.562 | −0.038 | 0.009 | — | — |
| combo_gate512 | 0.613 | 0.031 | 0.031 | 0.100 | −0.065 |
| **combo_resid*** | 0.628 | 0.248 | 0.153 | 0.005 | −0.130 |
| **combo_resid_hicon_g512*** | 0.611 | **0.302** | **0.169** | −0.037 | −0.045 |
(* residual-trained, raw-basis disc.)
- **Residual-training = best singles specificity** (pearson_Δ 0.25–0.30, 3–6×). **ugate = best
  combos.** The singles↔combo tradeoff is unbroken — no combined config got both.
- **Identifiability (fate-SupCon) regularizer FAILED — it's MEMORIZATION:** train-TF shift
  kNN-fate-purity 0.075→**0.985**, but held-out **0.057 = chance**. Don't increase it (deepens
  overfitting). Verdict: retire as a discrimination lever.
- **Few-shot combo training** added: `load_joung(load_combos=True)` merges combo pairs;
  `combo_train_frac=0.7` config knob. Submitted `v2_esm_con_ugate_combo` (job 2533394) — pending.

### 2c. Joung→Norman transfer (context-invariance)
`transfer_joung_to_norman.py`: frozen Joung (hESC) WM on 55 Norman (K562) combos with both TFs in
Joung's vocab, 5799 shared genes. do-op raw pearson 0.166 ≈ additive 0.161, but drift-removed
~0.02 and discrimination ≈ chance. → Only the shared/coarse component transfers; TF-specific
composition does NOT cross cell context.

## 3. FATE annotation pipeline + discrimination (THE PART TO REDO)
- **Pipeline** (`experiments/exp17_joung_tfatlas/analysis/mouse_dev_annotation/`):
  `nnls_annotate.py` → annotate each TF vs **Cao 2019 MOUSE organogenesis** reference,
  ortholog-mapped mouse→human via **MGI 1:1** (`mgi_homologene.rpt`), early cell types only.
  Method = **COSINE primary**: `score_c = cosine(Δ_TF after shared-PC removal & z-score,
  z-scored cell-type deviation signature D_c)` on discriminative (high-var) shared genes; NNLS
  reported as secondary mixture. `main_fate = argmax cosine`; specificity = peak − mean.
  `make_annotation_table.py` → `joung_tf_developmental_annotation.csv` (1084 TFs, 20 main fates,
  confidence tiers from specificity percentile). Held-out scorer = `fate_space()` in
  `exp19/evaluation/heldout_zeroshot_eval.py` (same cosine pipeline, nuisance PCs fit on TRAIN).
- **NOT human fetal** (user thought human fetal — it's mouse Cao 2019). Could redo vs human fetal.
- **Faithful separability** (`exp19/evaluation/fate_separability.py`, FIXED to use fate_space —
  first version wrongly used kNN on raw Δ):
  | metric (held-out singles, 20 fates) | ugate | combo_resid | chance |
  |---|---|---|---|
  | top-1 | 0.046 | **0.101** | 0.05 |
  | top-3 | 0.202 | 0.266 | 0.15 |
  | top-5 | 0.312 | **0.413** | 0.25 |
  | β-cosine | 0.079 | 0.110 | 0 |
  | fate retrieval (per-TF) | 0.0 | 0.009 | 0.009 |
  | obs-β margin (top1−top2) median | 0.039 | 0.039 | — |
  | frac TFs margin>0.1 (confident) | 0.11 | 0.11 | — |
- **Findings:** (1) labels are intrinsically **LOW-MARGIN** — obs-β top1−top2 median 0.039, only
  11% confident → 20-way top-1 is near-coin-flip BY CONSTRUCTION (this is why top-1 is hard).
  (2) Real **above-chance signal on SOFT metrics**: combo_resid top-1 2× chance, top-5 1.65×
  chance, βcos 0.11 → fate IS partially recoverable; residual-training is the lever.
  (3) Per-TF fate **retrieval ≈ chance** (each TF's fate not uniquely identifiable).

## 4. WHAT TO REDO — fate discrimination / RL training pipeline (open thread)
The fate-discrimination + fate-target-RL idea needs rebuilding because:
- Top-1 over 20 low-margin Cao-mouse fates is the wrong target/metric. **Use top-k / β-cosine /
  soft β-regression** instead, and/or **coarsen to higher-margin super-fates** or the
  confident-subset (margin>0.1, ~11%).
- Consider **in-species labels** (Leiden/Ward clusters on Joung's OWN expression, or a HUMAN
  fetal reference) so the label ceiling isn't cross-species-NNLS-limited.
- **Fate-target RL** (the reframing): target = annotated fate/cluster CENTROID (not a specific
  combo's expression). Cleaner & more numerous targets (~20–40), targets diverge in the residual
  (the discriminative axis), real ground truth from Joung's design. Build on **combo_resid WM**
  (best residual Δ fidelity) + **resid reward** (the fix that worked), not ugate + raw reward.
- Real limiting factor throughout = WM **held-out Δ fidelity / per-TF discrimination** (~0.6,
  retrieval ~chance). Residual-training moved it most. The fate regularizer does NOT help (memorizes).

## 5. Key files created/modified this session
- `core/wm_data_joung.py` (NEW), `core/setup_stack.py` (NEW), `core/wm_operator.py` (gate-aware),
  `core/env.py` (reward_mode raw/resid/blend + drift).
- `rl1_seq_ppo/train.py`, `rl2_set_ppo/train.py`, `rl3_mpc_distill/distill.py`,
  `run_fullpool_ladder.py` (— `--dataset/--checkpoint/--pool curated/--reward_mode` + eval/match
  raw&resid wandb logging + quick_match).
- `eval_rl_full_metrics.py`, `transfer_joung_to_norman.py`, `diag_reward_landscape.py` (NEW evals).
- `scripts/joung_{mpc_ladder,seq_ppo,set_ppo,distill}.sbatch`, `joung_rl_fixed.sbatch`
  (parameterized METHOD/POOL/REWARD/ITERS), `diag_ppo_smoke.sbatch`, `eval_rl_metrics.sbatch`,
  `transfer_norman.sbatch`.
- exp19: `data/load_joung.py` (+load_combos), `training/train_vae.py` (+fate_labels ident reg,
  +load_combos), `training/losses.py` (+ident SupCon term), `evaluation/fate_separability.py` (NEW),
  `configs/ablations/esm_con_ident.yaml`, `esm_con_ugate_combo.yaml` (NEW).

## 6. Jobs still running (check status)
- 2533394 `v2_esm_con_ugate_combo` (few-shot combo-trained WM) → then combo/epistasis eval on
  HELD-OUT combos (filter `combo_test_keys` from ckpt config).
- 2533480 `set_raw_curated` extended 200k (plateau check).
- 2533743/4 resid-reward set/seq PPO (set converged match_resid 0.42).

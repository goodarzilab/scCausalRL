# experiments_rl — RL / Planning Architectures

Four methods for steering the **frozen causal-VAE world model** (exp14 Norman
combo-trained shift+mutilation) toward a target cell state, all sharing one
do-operator, one env, and one data-grounded validation so the value-add ladder is a
fair comparison. This file explains each architecture and how it is trained + evaluated.

> **What "the world model" is.** Encoder→ε→linear SCM `z=(I−Aᵀ)⁻¹ε`→linear decoder.
> A *shift encoder* maps any gene to a latent program shift `Sᵍ=c·ShiftEncoder(pert_label[g])`
> (`pert_label` = top-50 PCA of transposed control expression → zero-shot, works for
> unseen genes). The validated do-operator is **shift-native mutilation**:
> `intervene_shift_mutilate` solves `z=(I−MAᵀ)⁻¹ε̃` with `M=diag(1−u)`,
> `ε̃=u⊙z_target+(1−u)⊙ε`, `u=min(1,β|ΣS|)`. It is genuinely set-dependent (non-additive)
> and on held-out Norman combos beats additive shift composition **0.899 vs 0.856 AUROC**
> (zero-shot). The model is frozen for all RL — only the planner/policy learns.

---

## Shared foundation (`core/`)

| Piece | What it is |
|---|---|
| **State** `s_t = [z_t ; z̄_target]` | current latent gene-program vector ⧺ the data-grounded target (a perturbation-induced z, or a desired cell-state centroid). Goal-conditioned: the same network serves any target. |
| **Action** | a gene SET applied at one step — singleton (sequential PPO), k-set (set PPO / MPC), or empty (no-op). |
| **Transition** (`wm_operator.apply`) | shift-native mutilation do-operator; shifts summed over the set; ε threaded per-DAG via `recover_epsilon_u` (persisted intervention → genuine multi-step counterfactual). Bootstrapped over **N Gumbel-sampled DAGs** (epistemic uncertainty). |
| **Reward** (`env.py`) | potential shaping `Φ(z)=−‖z−z̄_target‖²`; penalized over the ensemble: `r = mean_N(γΦ′−Φ) − α·std_N(·) − λ_cost`. Entropy-for-exploration is the PPO entropy bonus (policy entropy); pessimism is the env's `α·std`. So "mean − pessimism + entropy" holds across all methods. Per-plan `std_N` is also the OOD-honesty signal. Continuous latents — no Leiden. |
| **Targets / data** (`wm_data.py`) | Norman aligned to the WM axis: control-ε pool (start = control mean), per-perturbation **data-grounded latent targets** (encode the REAL perturbed cells → ε → z), observed gene-space DE, train/test splits (combos = checkpoint's 31 held-out). |
| **Validation** (`validate.py`, `value_add_ladder.py`) | every planner is `fn(target_key)→gene_set`; scored by (1) **gene recovery** (did it find the true perturbation's genes?), (2) **latent distance** reached vs control baseline, (3) **gene-space DE fidelity** (decode the chosen set through the operator, score vs observed DE with the same AUROC/Spearman/Pearson-DEG used by the combo eval). |

**Falsification gate (already passed):** the operator's zero-shot combo prediction
reproduces **0.899 > 0.856** on the 31 held-out combos (and the observed-singles oracle
`additive_obs`=0.952 still beats it — the operator is epistasis-*neutral*, the central
risk the ladder makes explicit).

---

## M0 — MPC (oracle / upper bound) · `rl0_mpc/`

- **What it does.** Stateless beam search over the candidate genes; scores each gene
  SET by applying the do-operator under the N-DAG ensemble and taking the pessimistic
  cost `mean_N‖z_post−z_target‖² + α·std_N`. Returns the best set (≤ `max_set` genes).
- **No training** — pure planning. `beam=1,max_set=1` ⇒ greedy-1.
- **Roles:** ladder rung, **distillation source** for M3, and the planning **upper
  bound** the learned policies must match-or-beat.
- **Tests:** beam1/max_set1 == exhaustive argmin; an exact target (the operator's own
  z_post for a gene) is recovered at ~0 cost; reaches closer than control on held-out
  combos; pessimism raises cost monotonically in α.

## M1 — Sequential PPO · `rl1_seq_ppo/`

- **Action.** One categorical draw over candidate genes per step; a combo is a
  *sequence* of single-gene steps, with ε threading + persisted intervention making it
  a genuine multi-step counterfactual (not additive forward-sim).
- **Networks.** Goal-conditioned actor-critic, MLP `[z_t, z̄_target] → 256 → 256`, action
  head = #candidate genes (~87, shrunk from the dead run's ~4000 — a key fix).
- **Training.** PPO + GAE on the unified penalized reward; `log π` = a single
  log-softmax per step; standard entropy bonus.
- **What it tests.** Can a learned policy match MPC on singles and reach combo targets
  across steps? Does imposing an ORDER on an order-free combo hurt synergy capture?

## M2 — Set-based PPO · `rl2_set_ppo/` ★

The policy emits an **unordered set of ≤ k_max perturbations applied simultaneously** —
matching how a Norman double is actually delivered, so synergy can live in the joint
action instead of being split across steps.

- **Shared encoder** `h = enc(s_t)` feeds `k_max` decision slots (BDQ-style branched
  outputs) → cost grows **linearly** in k_max, not combinatorially.
- **Vocabulary** `V = {STOP} ∪ ({1..G}×{KO,OE})`, `|V| = 2G+1`. An action is an
  autoregressive token sequence read until STOP or k_max, then collapsed to a set.
- **Policy factorization** `π_θ(a|s) = ∏_i π_θ(a_i | s, a_{<i})`: each factor a masked
  softmax over V; masks forbid STOP at slot 0 (no empty action) and forbid
  already-chosen genes at later slots (no duplicates). The chosen-gene context is an
  **order-invariant sum** of gene embeddings (matches set semantics).
  `log π = Σ_i log π_θ(a_i|·)`; entropy = Σ per-slot entropies.
- **Transition.** The collapsed set → one mutilation solve (`apply(ε, set)`), so the
  set action *is* a combo prediction; non-additivity = the set-dependent `u`.
- **Training.** PPO + GAE; only `log π`/entropy become per-slot sums — the rest of PPO
  is unchanged (it only ever consumes that one scalar log-prob).
- **What it tests.** Does the order-free set formulation capture combo synergy better
  than sequential? Can it pick its own cardinality via STOP (singles vs doubles)?

## M3 — MPC → policy distillation · `rl3_mpc_distill/`

- **Bridge** the oracle to a learned policy (works for M1 and M2): (1) collect MPC
  trajectories over the train-split task corpus; (2) **behavioral-clone** the policy to
  MPC's chosen action per step (set-BC for M2 = cross-entropy over the token sequence) —
  a non-random warm start; (3) **PPO fine-tune** against the same targets + reward so
  the policy can *exceed* MPC by amortizing across tasks.
- **What it tests.** Does distillation beat cold PPO, and let the policy beat the MPC
  oracle on held-out targets?

---

## The value-add ladder (the headline comparison)

For each held-out perturbation (single, then combo), run every planner and report gene
recovery / latent distance / DE fidelity:

`random → greedy-1 → additive → MPC → sequential-PPO → set-PPO (+ distilled)`

- **random** — k random candidate genes.
- **greedy-1** — best single gene by latent distance.
- **additive** — greedy ADDITIVE latent composition (sum of single-shift deltas), no
  operator epistasis — the additive bar the do-operator planning must beat.
- **MPC / PPO / set-PPO** — as above.

Plus: **set-vs-sequential** on combos (does order-free help?), **reachability** (reach
an unseen pert's state using only train-split genes), and the **OOD honesty** check
(per-plan DAG disagreement should correlate with held-out error).

---

## Status + headline result (live)

- ✅ `core/` (loader, operator, data, env+reward, baselines, validate, ladder) built and
  verified on GPU; operator reproduces the held-out **0.899 > 0.856** combo edge.
- ✅ **Objective pivot (important):** the latent-distance objective is *gameable* — MPC
  reached the closest latent state but produced the worst gene-space DE (and gene
  recovery ≈ 0). Fixed by optimizing + scoring in **gene space** (signed Pearson of the
  reached DE vs the observed DE on the target's DEGs). Env reward and MPC cost both use
  it now; pessimism = hard-Gumbel DAG-ensemble std (the soft Concrete gates were a
  second bug — they made a dense A ≠ eval-A).
- ✅ **M0 value-add ladder (31 held-out combos, gene-space DE-match):**

  | method | DE-match | gene recall | pearson-DEG | dist_ratio |
  |---|---|---|---|---|
  | random | 0.509 | 0.04 | 0.504 | 1.37 |
  | greedy-1 | 0.559 | 0.02 | 0.554 | 0.58 |
  | additive | 0.535 | 0.02 | 0.531 | 0.56 |
  | **MPC** | **0.799** | 0.10 | **0.789** | 0.95 |

  **MPC beats additive 0.80 vs 0.54** — searching the world model finds interventions
  that reproduce a target state's DE far better than additive composition. This is a
  **controllability/reachability** result (the reprogramming-relevant capability), NOT
  held-out prediction.
- ✅ **Identifiability finding (reported, not a failure):** gene recovery ≈ 0 — the
  perturbation→state map is many-to-one in the 64-dim latent, so the planner reaches the
  target DE via *different* genes ~90% of the time. (DEG-AUROC is saturated ~0.90 even
  for random genes → non-discriminative; DE-match / pearson-DEG are the real metrics.)
- 🔄 M1 sequential PPO + M2 set-PPO training (gene-space reward); M3 distillation next.
- World models: frozen WM = exp14 (Norman). exp16 (4-cell-line) still training
  (~ep40/150) → cross-cell/multi-modality WM when done.

# Methodology deep-dive — answers to the 12 questions

Companion to `docs/EXPERIMENTS_9_ONWARDS.md`. Every concept defined to the bone, with the
exact code paths and math.

---

## Q1 — `correlation` vs `coexpression` baseline

Both are control-only baselines (no model), but they predict a perturbation's downstream
effect differently:

- **`corr` (LOO gene-gene correlation).** From control cells, compute the gene-gene Pearson
  correlation matrix $R\in\mathbb R^{G\times G}$, $R_{ij}=\mathrm{corr}(x_i,x_j)$ over control
  cells. The predicted effect of perturbing gene $g$ is **row $g$ of $|R|$**: i.e. "genes
  co-varying with $g$ in controls are the ones that move when you perturb $g$." For a combo
  $(a,b)$ the signature is $|R_a|+|R_b|$. It is a *correlational* guess — no causal direction,
  no propagation.
- **`coexp` (co-expression neighbour).** Same correlation matrix, but instead of using $g$'s
  own row, it predicts the perturbation's effect as the **average expression profile of $g$'s
  top co-expression neighbours** (the genes most correlated with $g$). It captures "perturbing
  $g$ looks like shifting the module $g$ belongs to." Empirically it is a *stronger* baseline
  than `corr` on Norman (coexp ≈ 0.93 vs corr ≈ 0.74 on Norman singles) because Norman TFs
  drive coherent co-expression modules.

In one line: **`corr` = "$g$'s correlates move"; `coexp` = "$g$'s module's average profile."**
`coexp` is the harder bar; our model beats `corr` everywhere but only beats `coexp` on
Replogle, not on Norman (stated honestly throughout).

---

## Q2 — DEG-AUROC: how many DEGs, what cutoff?

Code: `evaluation/run_perturbation_eval.py::compute_observed_de`.

- A gene is called **differentially expressed (DEG)** for a perturbation if its **Mann–Whitney
  U test p-value < 0.05** comparing perturbed-cell vs control-cell expression — *raw* p, **no
  FDR correction**, with `MIN_CELLS=5` cells required. Stored as the boolean mask `sig`.
- **How many DEGs?** Norman combos are **broad**: **median ≈ 1,346 DEGs, mean ≈ 1,279, range
  153–2,082 out of 8,000 genes** (~16%). (Computed from `norman_combo/combo_per_pert.csv`.)
- **DEG-AUROC** = `roc_auc_score(sig, |predicted_effect|)` over non-target genes — can the
  predicted magnitude rank the true DEGs above non-DEGs. **Consequence:** because there are
  ~1,300 DEGs and the response is global, *almost any* strong perturbation prediction scores
  AUROC ≈ 0.90 (even random genes in the 8000-pool hit ~0.87) → **AUROC is saturated and
  weakly discriminative.** That is why we lean on **`pearson-DEG`** (signed Pearson of
  predicted vs observed log-fold-change on the DEGs, $|\text{lfc}|>0.05$) and the RL
  **DE-match** — those *do* separate methods.

---

## Q3 — What is ε-decorrelation (decorr100)? Was it used downstream?

The encoder produces an exogenous latent $\epsilon\in\mathbb R^{64}$ (≈ $\mu(x)$). We want the
latent *programs* to be statistically independent — a total-correlation (TC) proxy — so the
SCM does real work and we avoid $z\approx\epsilon$. The diagonal KL of a VAE cannot decorrelate
dimensions, so we add an explicit penalty on the **off-diagonal covariance of the batch
$\mu$**:

$$
\mathcal L_{\text{decorr}}=\frac{1}{d(d-1)}\sum_{i\neq j}\mathrm{Cov}(\mu)_{ij}^2,\qquad
\mathrm{Cov}(\mu)=\tfrac1N(\mu-\bar\mu)^\top(\mu-\bar\mu),
$$

added as $\lambda_{\text{decorr}}\mathcal L_{\text{decorr}}$. "decorr100" = $\lambda=100$.
In exp10 it reduced off-diagonal $|\mathrm{cov}(\mu)|$ from **0.468 → 0.332** while keeping
$A$ active and $z$ informative (decorr1000 over-shrank).

**Was it used downstream? No.** I grepped `exp14_norman_combotrain` (config + `losses.py` +
`train_vae.py`) — there is **no `lambda_eps_decorr`**. So decorr100 was an **exp10-only
finding** and was **not** carried into exp14/15/16, the RL world model, or Joung (exp17). It
*should* be revisited as an add-on, but as of now none of the production world models use it.
(`code: exp10_sparsity_disentangle/training/losses.py::eps_decorrelation_loss`.)

---

## Q4 — Shift-native mutilation: how is the u-score computed? Full math.

The world model is a linear SCM with a linear decoder:
$$
z=(I-A^\top)^{-1}\epsilon,\qquad \hat x=Dz+b .
$$
Perturbing a gene = injecting a **latent shift** from the shift encoder. For perturbed gene
set $G$ with per-gene control-PCA label $\ell_g$,
$$
S=\sum_{g\in G} c\cdot\mathrm{ShiftEncoder}(\ell_g)\in\mathbb R^{64}\quad(\text{summed shift}).
$$

**Why a mutilation operator at all.** If you just add the shift, $z=(I-A^\top)^{-1}(\epsilon+S)$,
then because solve∘decode are both linear the combo prediction is **exactly the sum of the two
single predictions** — provably additive, no epistasis. To get non-additivity you must
*mutilate* the graph: partially clamp the perturbed programs to a target (a soft do-operator),
so the propagation through $A$ changes with the set.

**The operator** (`models/causal_vae.py::intervene_shift_mutilate`, `u_mode='shift'`):

1. shift-induced target latent: $z_S=(I-A^\top)^{-1}S,\quad z_{\text{target}}=z_{\text{prev}}+z_S$
   (with $z_{\text{prev}}=(I-A^\top)^{-1}\epsilon$).
2. **the u-score** (intervention strength per latent program):
   $$
   \boxed{\,u=\min\!\big(\mathbf 1,\ \beta\,|S|\big)\in[0,1]^{64}\,}\qquad(\beta=1)
   $$
   i.e. each program $k$ is clamped in proportion to *how hard the summed shift drives it*,
   $|S_k|$, saturating at 1. (The alternative `u_mode='loading'` instead sets
   $u=\mathrm{clip}(\sum_{g}L[g],0,u_{\max})$ from the decoder loading matrix
   $L_{g,i}=D_{g,i}^2/\max_j D_{g,j}^2$ — but that was near-degenerate and tied additive, so we
   use `shift`.)
3. mutilated exogenous + solve with $M=\mathrm{diag}(1-u)$:
   $$
   \tilde\epsilon=u\odot z_{\text{target}}+(1-u)\odot\epsilon,\qquad
   \boxed{\,z_{\text{post}}=(I-MA^\top)^{-1}\tilde\epsilon\,}
   $$

**Reading the limits:** $u_k\to1$ ⇒ program $k$ is pinned to $z_{\text{target},k}$ and its
incoming edges are cut (a hard $\mathrm{do}$); $u_k\to0$ ⇒ free (control). **Non-additivity for
a combo** arises because $u$ (hence $M$) depends on the **summed** shift $S=S_a+S_b$ — the
pair's clamp pattern differs from either single's, so the propagation through $A$ is not the
sum. The u-score is therefore the single quantity that encodes "which programs this particular
intervention set hijacks, and how hard."

---

## Q5 — exp11 additive vs exp12 mut-loading vs exp12 mut-shift-native; what's the figure?

All three bars in your figure (`combo_methods_auroc_spearman.png`) are models **trained on
Norman SINGLES only**, then evaluated **zero-shot on the 107 combos** (combos were *not* in
training for exp11/12 — that only happens in exp14):

- **exp11 additive** — shift encoder trained with the **additive** shift only (no mutilation).
  Its combo prediction is `shift_add` (sum of singles). Combo AUROC 0.886.
- **exp12 mut-loading** — same singles training, but the **mutilation operator is trained** with
  `u_mode='loading'` (u from decoder loadings). The loading-`u` came out near-degenerate ⇒ its
  `shift_uscore` (0.838) ≈ additive, no gain.
- **exp12 mut-shift-native ★** — mutilation trained with `u_mode='shift'` ($u=\min(1,\beta|S|)$).
  This is the winner: `shift_uscore` **0.896 > shift_add 0.858** — the first coherent
  non-additive gain. The five bars per group are the predictors `shift_add`, `additive_obs`
  (observed-singles oracle, 0.95), `shift_uscore`, `uscore_obs`, `corr`.

**Where are the Replogle results?** Replogle is a **single-perturbation CRISPRi screen — it has
no combinatorial perturbations**, so there is *nothing to validate combos on* in Replogle. The
Replogle analogue is the **held-out single-pert** number: `exp12_repl_shiftclamp` →
**0.714** (`results/grid/exp12_repl_shiftclamp/shift_eval/summary.csv`), and the broader
Replogle single-pert results live in exp2/8/10 (~0.70) and the exp16 per-line table. The
combo/triple validation only exists for **Norman** (here) and **Joung** (exp17, in progress).

---

## Q6 — exp14 figures; did combo-training help and how is it combined with singles?

**How combo-training is combined with singles** (`training/train_vae.py` +
`data/cell_sentence_dataset.py`):
- The perturbation loss uses a **cell-sentence dataset** of "blocks." A block holds cells of
  one perturbation with a padded gene-index array: **single = `[g, -1]`, combo = `[gₐ, g_b]`**.
- Combos are split by a **deterministic md5 hash** on `seed|geneA|geneB` with
  `combo_train_frac=0.7` → **70% combos train / 30% test** (`combo_filter` restricts training to
  the train keys). Training then mixes **all single blocks + the 70%-train combo blocks**; the
  combo block's loss uses the **summed** shift $S=S_a+S_b$ through the same mutilation operator.
  The **30% held-out combos are never seen** and are the eval.
- So "combo-trained" = the shift encoder + DAG see real 2-gene cells during training (the
  operator learns set-dependent $u$ from data), not just singles.

**Did it help?** Yes, modestly, on both axes (numbers from the `summary.csv`):

| | held-out **singles** AUROC | **combos** `shift_uscore` AUROC | combos `shift_add` |
|---|---|---|---|
| exp12 shift-native (singles-only) | 0.891 | 0.896 (all 107) | 0.858 |
| **exp14 (combo-trained)** | **0.918** | **0.910** (all 107) / 0.899 (held-out 31) | 0.869 |

→ combo-training lifted singles **0.891→0.918** and combos **0.896→0.910**, and it **flattened
the epistasis-aversion pathology** (advantage-vs-epistasis ρ went −0.40→−0.10). It still does
not beat the observed-singles oracle (0.953) — the epistasis-neutral ceiling. **exp14 is the
better model and is the one frozen for RL.**

**Figures:** I generated `experiments/figures/exp14_combo_eval.png` (singles + combo bars for
exp14) and `experiments/figures/exp11_12_14_combo_compare.png` (the did-combo-training-help
comparison) — see Q-figures note at the end.

---

## Q7 — How exactly is the modality-conditioned combined model (exp15) built?

Goal: one model serving **CRISPRi (Replogle, knockdown)** and **CRISPRa (Norman, activation)**.
Mechanism (`exp15_modality_combined/models/causal_vae.py`):
- Every cell gets a **modality id** $m\in\{0=\text{control},1=\text{CRISPRa},2=\text{CRISPRi}\}$
  (`data/load_combined.py`). The **encoder adds a learned modality embedding**
  $\text{modality\_emb}[m]$ to the cell representation.
- The shift **direction** is modality-agnostic (a gene's identity), but the **sign/magnitude**
  is modality-specific — knockdown ≈ −activation. So there is a **per-modality scalar**
  $\texttt{shift\_scale\_mod}\in\mathbb R^{n_{\text{mod}}}$, and
  $$
  S^g_m=\texttt{shift\_scale\_mod}[m]\cdot\mathrm{ShiftEncoder}(\ell_g),
  $$
  (`compute_shift(gene_idx, modality_id=m)`), with $m$ = the perturbation's screen. The loss
  computes the shift with the perturbed cells' modality; the eval
  (`run_modality_eval.py`) encodes controls with $m=0$ and scores Norman perts with $m=1$.
- **Result/verdict:** sharing capacity across modalities **hurt per-modality fidelity** (Norman
  0.73/0.74 vs exp14's 0.92/0.91), so it confirmed exp14 (Norman-native) as the RL world model.
  Figure: `experiments/figures/exp15_modality_eval.png`.

---

## Q8 — Cross-cell eval: what is n=158 for K562? Use exp16 for RL?

`exp16_multicellline/evaluation/run_crosscell_eval.py`. The exp16 model trained jointly on 4
lines with an 80/10/10 **TF-gene split** (seed 42). The eval reconstructs that split over the
**union** of single-pert genes and scores, **per line, only the test-split genes** — i.e.
**genes the model never trained the shift on, in any line** (zero-shot generalization to unseen
perturbations *within* a trained cell context). **n=158 for K562 = the number of held-out
(test-split) single-pert genes present in K562** (HepG2 184, Jurkat 168, RPE1 171). For each:
encode that line's control cells with the line's modality id, predict the held-out gene's effect
via the modality-conditioned shift + mutilation, decode, score DEG-AUROC vs the observed effect
and vs `corr`. (It is **not** a held-out *cell line* — all 4 lines were in training; that
stronger test would need train-on-3/test-on-4th.)

**Should we use exp16 for RL?** Caveats: (a) it's **CRISPRi singles only — no combos**, so it
can't drive the combo/triple value-add story; (b) it **timed out at ep80/150** (usable but not
final); (c) per-line AUROC is modest (0.65–0.71). **Verdict:** **not** the primary RL world
model (exp14/Joung are, because they have combos). exp16's *right* RL role is the
**cell-context-conditioned** planner later — "find interventions to reach a target state *in a
chosen cell line*," and ask whether the same target needs different genes across lines. So: keep
it for the cross-context RL extension, not the combo-planning experiments.

---

## Q9 — z_target, the forward loop, and the MPC beam search (full math)

**The latent target $z_{\text{target}}$** (`experiments_rl/core/wm_data.py`). For a held-out
perturbation $p$ with observed cells $\mathcal C_p$, we encode the **real** perturbed cells and
solve the SCM at the eval-A point estimate $A^\star$:
$$
\bar\epsilon_p=\tfrac1{|\mathcal C_p|}\sum_{i\in\mathcal C_p}\mu_\phi(x_i),\qquad
z^{\text{tgt}}_p=(I-A^{\star\top})^{-1}\bar\epsilon_p .
$$
This is **data-grounded** (the encoder's view of where the actual perturbation lands), not the
model's own prediction — so reaching it is falsifiable. Control start:
$\bar\epsilon_{\text{ctrl}}$, $z_0=(I-A^{\star\top})^{-1}\bar\epsilon_{\text{ctrl}}$.

**The forward loop (one rollout step)** given current $\epsilon$, action set $G$:
$$
z_{\text{prev}}=(I-A^{\star\top})^{-1}\epsilon
\;\xrightarrow{\text{do-op }G}\;
z_{\text{post}}=(I-MA^{\star\top})^{-1}\tilde\epsilon,\qquad
\epsilon_{\text{next}}=(I-MA^{\star\top})z_{\text{post}}
$$
(the last is `recover_epsilon_u` — it threads a consistent noise vector into the next step so
multi-step rollouts persist the intervention).

**MPC scoring — what exactly is scored** (`rl0_mpc/mpc_planner.py`). MPC optimizes the
**gene-space DE-match** (Q12), not latent distance. For a candidate set $G$, decode the reached
state and score signed-Pearson vs the target's observed DE on its DEGs $\mathcal S_p$:
$$
\hat d(G)=D\,z_{\text{post}}(G)+b-\hat x_{\text{ctrl}},\qquad
\Phi_p(G)=\mathrm{corr}_{\,j\in\mathcal S_p}\big(\hat d(G)_j,\ d^{\text{obs}}_{p,j}\big).
$$
Pessimism over the hard-Gumbel DAG ensemble $\{A^{(n)}\}_{n=1}^N$:
$$
\text{cost}(G)=-\,\Phi_p^{A^\star}(G)\;+\;\alpha\,\operatorname{std}_n\big[\Phi_p^{A^{(n)}}(G)\big].
$$

**Beam search with pruning** (the 8000-gene pool is too big to enumerate pairs,
$\binom{8000}{2}\!\approx\!3.2\times10^7$):
1. **Depth-1:** score **all** $C$ candidate singles in one batched do-operator call
   (`apply_many`), keep the top-$B$ by $\Phi$.
2. **Depth-2:** for each of the $B$ beam genes, score it paired with **all** $C$ partners
   ($B\!\cdot\!C$ pairs, batched), keep the best pair.
3. **Re-rank** the $B{+}1$ finalists (best single + best pair per beam) by the **pessimistic**
   cost and return the argmin.
`beam=1, depth=1` reduces to greedy-1. The batched solve of $C$ per-set mutilated systems
$(I-\mathrm{diag}(1-u_k)A^\top)$ via one `torch.linalg.solve` is what makes scoring 8,000
candidates take ~0.3 s.

---

## Q10 — Full PPO reward; full set-PPO equations; seq-PPO vs set-PPO

**The reward (shared by all PPO variants)** (`experiments_rl/core/env.py`). Potential
$\Phi(z)=\mathrm{corr}_{\mathcal S}(\hat d(z),d^{\text{obs}})$ (gene-space DE-match). Per step:
$$
\boxed{\,r_t=\underbrace{\gamma\,\Phi(z_{t+1})-\Phi(z_t)}_{\text{potential shaping (eval-}A^\star)}
\;-\;\underbrace{\alpha\,\operatorname{std}_n\big[\gamma\,\Phi(z^{(n)}_{t+1})\big]}_{\text{pessimism}}
\;-\;\underbrace{\lambda\,\mathbf 1[|G_t|>0]}_{\text{step cost}}\,}
$$
plus the standard PPO **entropy bonus** $c_{\text{ent}}\,\mathcal H(\pi)$ inside the loss (the
"+entropy for exploration"). So "mean − pessimism + entropy" holds across all methods.

**PPO objective (both variants).** With ratio
$\rho_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$, GAE advantage $\hat A_t$:
$$
\mathcal L=-\mathbb E_t\big[\min(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1{-}\varepsilon,1{+}\varepsilon)\hat A_t)\big]
-c_{\text{ent}}\mathbb E_t[\mathcal H(\pi_\theta)]+c_v\mathbb E_t[(V_\psi(s_t)-\hat R_t)^2].
$$

**Sequential PPO (M1).** Action $a_t\in\{1..C\}$ = one gene; $\pi_\theta(a|s)$ = a single
Categorical (one log-softmax); a combo is reached over $k$ sequential steps; GAE over the steps.

**Set-based PPO (M2) — the full set policy.** Action = an unordered set, drawn as an
autoregressive token recipe over $\mathcal V=\{\texttt{STOP}\}\cup\{1..C\}$. With shared encoder
$h=\mathrm{enc}([z_t;z_{\text{target}}])$ and order-invariant chosen-gene context
$c_i=\sum_{j<i}e(g_j)$:
$$
\pi_\theta(a\mid s)=\prod_{i=1}^{|a|}\pi_\theta(a_i\mid s,a_{<i}),\qquad
\pi_\theta(\cdot\mid s,a_{<i})=\mathrm{softmax}\big(\mathrm{head}[\,h;c_i;\text{slot}(i)\,]+\text{mask}_i\big),
$$
where $\text{mask}_i$ sets logits to $-\infty$ for **STOP at slot 0** (no empty action) and for
**already-chosen genes** (no duplicates). Hence
$$
\log\pi_\theta(a\mid s)=\sum_{i}\log\pi_\theta(a_i\mid s,a_{<i}),\qquad
\mathcal H=\sum_i\mathcal H\big(\pi_\theta(\cdot\mid s,a_{<i})\big).
$$
The set is applied **simultaneously** in one do-operator call (one env step), so GAE collapses
to $\hat A=r-V_\psi(s)$. **The only change PPO sees** is that $\log\pi$ and $\mathcal H$ are
per-slot **sums** — everything else (clip, value, entropy) is identical. A unit test confirms
`evaluate(sampled tokens)` reproduces the sampled $\log\pi$ to $2\times10^{-7}$.

**Seq-PPO vs set-PPO (the difference):** sequential imposes an *order* on an order-free combo
(do A, then B) and reaches it over multiple steps; set-PPO emits the *whole unordered set at
once* with self-chosen cardinality (via STOP), so synergy lives in the joint action and there's
no spurious ordering. Empirically set (0.66) > seq (0.59) DE-match.

---

## Q11 — MPC → distillation algorithm; how PPO fine-tuning works; which PPO

`experiments_rl/rl3_mpc_distill/distill.py`. **It distills into the SET policy** (M2's
architecture). Three stages:

1. **Collect MPC plans.** For each train-split target $p$, run the MPC beam search →
   $a^{\text{MPC}}_p$ (the chosen gene set), stored as a token row $(\le k_{\max}$, STOP-padded).
   Cached to disk (the slow step).
2. **Behavioral cloning (warm start).** Maximize the log-likelihood of the teacher's set under
   the set policy — cross-entropy on the autoregressive tokens:
   $$
   \mathcal L_{\text{BC}}(\theta)=-\,\mathbb E_p\Big[\textstyle\sum_i\log\pi_\theta\big(a^{\text{MPC}}_{p,i}\mid s_p,a^{\text{MPC}}_{p,<i}\big)\Big].
   $$
   This uses the exact same `SetPolicy.evaluate` used by PPO, so the warm start is consistent.
3. **PPO fine-tuning.** Initialize from the BC weights and run **set-PPO** (M2's `SetPPO`, the
   1-step set objective above) against the gene-space reward + pessimism, so the policy can
   *exceed* the teacher by amortizing across targets and exploiting the dense shaped reward.
   It is the *set-based* PPO (not sequential).

Why this works: cold PPO over a thin task distribution (a few hundred targets) and a large
action head is slow/unstable; the MPC teacher gives a strong initialization, and PPO then
polishes. Result: distill **0.753** ≫ cold set-PPO 0.659 ≫ cold seq-PPO 0.586 — closing most of
the gap to the MPC oracle (0.81).

---

## Q12 — gene recovery (why 0), gene-space objective, DE-match, the baselines, the 8000-axis

**What the task is.** Given a held-out perturbation's **target state** (its data-grounded
$z^{\text{tgt}}_p$ / observed DE), find a gene intervention set whose do-operator-reached state
**reproduces that DE**. The candidate pool is the **full 8,000-gene axis**: the shift encoder
embeds *any* gene zero-shot from its control-PCA label $\ell_g$, so ~99% of candidates were
never screened. **Nothing is trained on the 8000-axis for MPC** (MPC is search over the frozen
exp14 model); the PPO policies *are* trained — their action head is over the 8,000 genes and
they learn, from the reward, which genes reach which targets.

- **DE-match** = the scoring/objective: signed Pearson between the **predicted** DE (decode the
  reached latent, minus control reconstruction) and the **observed** DE of the target, on the
  target's DEGs. Range $[-1,1]$; higher = the reached expression state matches the real one.
- **Gene recovery / recall** = a *different* question: of the **true** genes that produced the
  target perturbation, how many did the planner pick? $\text{recall}=|G^{\text{chosen}}\cap
  G^{\text{true}}|/|G^{\text{true}}|$.
- **Why recall ≈ 0.** The perturbation→state map is **massively many-to-one** in a 64-dim latent
  with 8,000 candidate shifts: many different gene sets decode to nearly the same broad DE
  pattern. So the planner reliably **reaches the target state** (DE-match 0.80) but via
  **different genes** than the real perturbation — it cannot invert a many-to-one map. This is a
  genuine **identifiability** result, not a bug, and it's fine for the application ("reach a
  target cell state" — many protocols do), while being honest that "recover the exact driver
  genes" is not achievable here. (Confirmed: when the target is the operator's *own* prediction
  for a known gene, MPC recovers it exactly — so the loop is correct; it's the data-grounded,
  many-to-one targets that aren't gene-identifiable.)

**The planning baselines (all scored by the same DE-match, differ only in how they pick genes):**
- **random** — $k$ genes drawn uniformly from the 8,000 pool (≈ chance: DE-match 0.33).
- **greedy-1** — the single candidate gene whose do-operator reached-state has the **highest
  DE-match** to the target (MPC depth-1, beam-1). Surprisingly strong (0.80) because with 8,000
  zero-shot options some single gene mimics the combo's broad DE.
- **top2-marginal** — the **two best independent singles** by DE-match, taken as a set (no joint
  search). Tests whether picking the 2 best marginals matches joint beam search.
- **MPC** — full beam search over pairs with pessimism (0.81). Its edge over greedy is small at
  8000-pool scale (the large pool helps the cheap baselines most).

---

---

# Follow-up clarifications (round 2)

## C1 — MPC→distillation: the cross-entropy / behavior-cloning loss, precisely

The MPC teacher gives, for each train target $p$, a chosen gene set serialized as a token row
$a^{\text{MPC}}_p=(t_1,\dots,t_{k})$ over $\mathcal V=\{\texttt{STOP}\}\cup\{1..C\}$ (e.g.
`[geneA+1, geneB+1, STOP]`). **Behavior cloning = maximum-likelihood of those teacher tokens
under the set policy** — i.e. supervised next-token classification, identical in form to
language-model training:
$$
\mathcal L_{\text{BC}}(\theta)=-\,\mathbb E_p\big[\log\pi_\theta(a^{\text{MPC}}_p\mid s_p)\big]
=-\,\mathbb E_p\Big[\textstyle\sum_{i}\log\,\mathrm{softmax}\big(\text{logits}_\theta(s_p,a^{\text{MPC}}_{p,<i})\big)_{\,a^{\text{MPC}}_{p,i}}\Big].
$$
Each term is a **categorical cross-entropy** between the policy's slot-$i$ distribution and the
one-hot teacher token, conditioned on the teacher's *previous* tokens (teacher forcing). It
reuses the exact same `SetPolicy.evaluate(s, tokens)` that PPO uses, so the warm-started weights
are immediately PPO-compatible. Intuition: "given this target state, output the same gene set
the search oracle found." It does **not** clone the latent or the reward — only the *action
distribution*.

## C2 — Does set-PPO start with two genes already chosen? What does it fine-tune?

**No.** Every episode the set policy builds its set **from scratch**: slot 0 samples a gene
(STOP forbidden), slot 1 samples another gene **or** STOP, …, up to $k_{\max}$. Nothing is
pre-seeded; the policy chooses *which* genes and *how many* (cardinality via STOP) afresh each
time. What "**fine-tuning**" updates is the **policy network weights $\theta$** (the shared
encoder + slot heads + gene embeddings) — *not* a fixed gene set. So:
- **BC stage:** $\theta$ is trained so the policy's set distribution matches the MPC teacher's.
- **PPO stage:** $\theta$ is further updated by policy-gradient on the gene-space reward, so it
  can find sets that score *higher* than the teacher (amortizing across targets). At eval, the
  trained policy greedily emits a set per target (argmax tokens). There is never a "two genes
  fixed, tweak around them" step — the whole set is re-generated each time; fine-tuning shifts
  the *probabilities* over which genes get generated.

## C3 — How is "reached the target latent" defined?

It is **not a binary "reached/not"** — there is no success threshold by default. The episode
applies the chosen gene set (set-PPO: one step; sequential: $k$ steps), producing $z_{\text{post}}$,
and we score the **continuous DE-match** $\Phi$ between the *decoded reached state* and the
target's observed DE. The reward is potential-shaped on $\Phi$ (so "closer" = more reward),
and the ladder reports $\Phi$ itself. (The env *does* expose an optional `tol` that ends an
episode early if $\Phi>$ tol, but that's just an early-stop convenience — the headline number
is the continuous DE-match, not a hit-rate.) So "reached" = "how well the reached expression
state matches the target," measured continuously, after the policy emits its set.

## C4 — Gene recovery (true vs chosen genes); aren't genes "always chosen"? Should we add a
## gene-supervised / ESM auxiliary loss?

- **true genes** = the actual perturbed genes of the held-out target (e.g. the combo's 2 TFs).
  **chosen genes** = the genes the planner/policy emitted. Recall $=|G^{\text{chosen}}\cap
  G^{\text{true}}|/|G^{\text{true}}|$.
- Yes, the policy **always chooses some genes** — recall asks whether they are the *right* ones.
  recall ≈ 0 means it reaches the target *state* via **different** genes than the real
  perturbation (the perturbation→state map is many-to-one in 64-dim), not that it fails to act.
- **Should we add an intermediary loss / ESM-embedding similarity to push toward the true
  combinations?** My honest take, nuanced:
  - It *would* raise recall — but it changes the objective from "reach the state" to "reach the
    state **using genes like the annotated ones**," and you can only supervise on genes you
    *know* (the training targets), so it risks **memorizing training combos** and **biasing
    against genuinely valid alternative solutions** (for reprogramming, many gene sets reach the
    same fate — that's biology, not error).
  - A **safer, defensible version** I'd recommend instead of hard gene supervision: (a) an
    **action prior / smoothed recall** — give partial credit when the chosen gene is
    *functionally close* to a true gene in an embedding (ESM2 protein embeddings, or the
    control-PCA pert_label space, or a TF-family/GO/CollecTRI graph) — so "picked a TF in the
    same regulatory module" counts; (b) for the **Yamanaka / TF-Atlas** setting where canonical
    protocols exist, report **pathway-enrichment of the top-k action distribution** (is the
    OSKM / known-combo set enriched among the policy's high-probability genes?) rather than exact
    set membership. This measures "biologically sensible genes" without forcing one answer.
  - So: **good idea as an auxiliary/eval signal (especially ESM-smoothed recall for known
    pathways), not as a hard training constraint.** It will help the *recall-style* metric and
    interpretability; it won't (and shouldn't) override the fundamental many-to-one identifiability
    for state-reaching. A concrete plan: add `recall@ESM-sim` and `pathway-enrichment` as
    reported metrics first; only add a *small* embedding-prior reward term if we specifically
    want the planner to prefer canonical drivers.

## C5 — greedy-1 vs greedy-2 for combos (you're right)

greedy-1 picks **one** gene for a 2-gene target — it's apples-to-oranges as a *combo* planner
(it was informative only as "a single zero-shot gene already reaches DE-match 0.80," a statement
about the broad many-to-one structure). For a fair 2-gene baseline we want **greedy-2**, of
which there are two flavors:
- **top2-marginal** (already in the ladder) — the 2 genes with the highest *independent*
  single-gene DE-match, taken as a set (no interaction considered).
- **sequential greedy-2** — pick the best gene, then the best *second* gene **conditioned on the
  first** (joint). This is exactly `MPCPlanner(max_set=2, beam_width=1)`.
I've added an explicit **`greedy2`** rung (= MPC beam-1, max_set-2) to the full-pool ladder
runner so the comparison is gene-count-matched; it slots between top2-marginal and full MPC
(beam-8). (Re-run the ladder job to populate its number.)

## Generated figures referenced above
`experiments/plot_exp16_rl_figures.py` → `figures/exp16_crosscell.png`,
`figures/rl_value_add_ladder.png`. `experiments/plot_exp14_combo.py` →
`figures/exp14_combo_eval.png`, `figures/exp11_12_14_combo_compare.png` (Q6).
Joung data analysis: `notebooks/joung_tf_atlas_analysis.ipynb`.

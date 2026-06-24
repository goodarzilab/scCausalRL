# RL / Planning on the frozen exp14 world model — detailed training design

This document specifies, in full, how each planning/RL method is trained and evaluated
on the **frozen exp14 causal world model**, the data and action space they use, the
reward/loss functions, and the baselines. No method trains the world model — it is a
fixed, differentiable simulator; only the planner/policy is learned (MPC learns
nothing). Math is rendered for GitHub/VS-Code markdown.

---

## 1. The frozen world model (exp14)

Checkpoint: `experiments/exp14_norman_combotrain/.../phase1_epoch150.pt` — a causal
LDVAE trained on **Norman** (CRISPRa, K562) with the shift-native mutilation operator.
Dimensions: $G=8000$ genes, latent dim $d=64$.

**Generative path.** For a cell $x\in\mathbb{R}^{G}$:

$$
\epsilon \sim q_\phi(\epsilon\mid x)=\mathcal N\!\big(\mu_\phi(x),\,\operatorname{diag}\sigma_\phi^2(x)\big),\qquad
z=(I-A^{\top})^{-1}\epsilon,\qquad
\hat x = Dz+b .
$$

- $\epsilon\in\mathbb R^{d}$ is the **exogenous noise** (cell-intrinsic state).
- $A\in\mathbb R^{d\times d}$ is the weighted **DAG adjacency** of the latent SCM; the
  solve $(I-A^{\top})^{-1}$ propagates exogenous noise through the regulatory graph to
  the **endogenous latent** $z$.
- $D\in\mathbb R^{G\times d}$ is the **linear decoder**; $\hat x_{\text{ctrl}}=Dz_0+b$
  is the reconstructed control profile, where $z_0$ is the control latent (§2).

**The DAG.** $A=\text{gate}\odot W\odot M$ with weights $W$, no-self-loop mask
$M=\mathbf 1-I$, and a Gumbel-sigmoid gate with logits $\theta$ and temperature $\tau$.
The deterministic **eval-time** adjacency (used for the validated combo metric and for
the planner's mean dynamics) is

$$
A^\star=\mathbf 1[\theta>0]\odot W\odot M .
$$

**Why frozen.** The previous joint RL co-trained the VAE and learned nothing; freezing
gives the planner a single stationary simulator and isolates the planning question. The
exp14 operator is the validated one (held-out combos: zero-shot mutilation AUROC
$0.899$ vs additive $0.856$).

---

## 2. Data and the task distribution

The RL "data" is **not raw cells** — it is a distribution of **goal-conditioned tasks**.
Each task is a *target cell state* the planner must reach by intervening.

**Norman screen (aligned to the 8000-gene axis):** 105 single-gene CRISPRa
perturbations, 131 two-gene combos, and ~11k control cells. After requiring genes to be
on-axis and have $\ge 5$ observed cells: **87 evaluable singles, 107 evaluable combos.**

**Control ε pool & control latent.** Encode $N_c{=}300$ control cells, take the mean
$\bar\epsilon_{\text{ctrl}}=\tfrac1{N_c}\sum_i\epsilon_i$ (every episode starts here — a
single random cell is noise-dominated, see §5), and $z_0=(I-A^{\star\top})^{-1}\bar\epsilon_{\text{ctrl}}$.

**Data-grounded targets.** For a perturbation $p$ (single $g$ or combo $(g_a,g_b)$) with
observed cells $\mathcal C_p$, the **latent target** and the **observed differential
expression** are

$$
\bar\epsilon_p=\tfrac1{|\mathcal C_p|}\!\!\sum_{i\in\mathcal C_p}\!\!\epsilon_i,\qquad
z^{\text{tgt}}_p=(I-A^{\star\top})^{-1}\bar\epsilon_p,\qquad
d^{\text{obs}}_p = \tfrac1{|\mathcal C_p|}\!\!\sum_{i\in\mathcal C_p}\!\! x_i \;-\; \bar x_{\text{ctrl}} .
$$

$z^{\text{tgt}}_p$ is the encoder's view of the **real** perturbed cells — falsifiable,
not the model's own prediction. $d^{\text{obs}}_p$ is the gene-space ground truth used by
the reward and the eval.

**Splits (deterministic).** Singles: 70 train / 9 val / 8 test (hash split, seed 42).
Combos: the checkpoint's stored split → **76 train / 31 test** (evaluable). Training uses
**train** tasks; the value-add ladder reports **held-out test combos**.

---

## 3. The action space — why 87 genes (not 8000)

An action is a **set of perturbed genes** drawn from a fixed candidate list. The
candidates are the **87 Norman single-perturbation genes that are on-axis and have
observed single-cell support**. Rationale:

1. **Grounding / measurability.** These are the genes actually screened in Norman, so
   each has (a) an observed single-perturbation effect to validate against and (b) the
   building blocks of the 131 observed combos — required for the combo-reaching task and
   for the gene-recovery metric. (The shift encoder generalizes zero-shot to *any* gene
   via its PCA label, so this is a deliberate *evaluation* restriction, not a model
   limitation.)
2. **Tractability / not repeating the dead run.** The original PPO used
   $2G=\,$~$4000$ actions (gene $\times$ {KO,OE}) and never learned. Restricting to 87
   single-direction (CRISPRa) actions shrinks the policy head ~50$\times$ and makes
   exhaustive/beam MPC feasible ($87$ singles, $\binom{87}{2}{=}3741$ pairs).
3. **Single modality.** Norman is CRISPRa only, so the {KO,OE} doubling of the general
   vocabulary collapses to the live (OE) half.

Formally the candidate set is $\mathcal G=\{g_1,\dots,g_C\}$, $C=87$, with gene indices
into the 8000-axis.

---

## 4. The do-operator (shared transition)

Every method steps the world model through the **shift-native mutilation** operator.
For a perturbed gene set $G\subseteq\mathcal G$ from exogenous state $\epsilon$ under
adjacency $A$:

$$
S=\sum_{g\in G} c\cdot \mathrm{ShiftEncoder}(\ell_g)\ \in\mathbb R^{d},\qquad
z_S=(I-A^{\top})^{-1}S,\qquad
z^{\text{tgt}} = z_{\text{prev}}+z_S,
$$

$$
u=\min\!\big(\mathbf 1,\ \beta\,|S|\big)\in[0,1]^d,\qquad
M=\operatorname{diag}(\mathbf 1-u),\qquad
\tilde\epsilon = u\odot z^{\text{tgt}} + (\mathbf 1-u)\odot\epsilon,
$$

$$
\boxed{\,z_{\text{post}}=(I-M A^{\top})^{-1}\tilde\epsilon\,}\qquad(\text{`intervene\_shift\_mutilate`, }u\text{-mode}=\text{shift},\ \beta{=}1).
$$

- $\ell_g$ is the gene's **perturbation label** = top-50 PCA of the transposed control
  expression matrix (a zero-shot, leak-free gene descriptor).
- $u$ is the per-program **intervention strength**; $u_k\to1$ clamps program $k$ to its
  target (a hard $\mathrm{do}$), $u_k\to0$ leaves it free. Because $u$ depends on the
  **summed** shift of the whole set, the operator is **set-dependent / non-additive** —
  this is the only source of epistasis (a linear SCM + linear decoder makes plain shift
  composition exactly additive).

**Multi-step (ε threading).** To persist an intervention across steps,

$$
\epsilon_{t+1}=(I-MA^{\top})\,z_{\text{post}}=z_{\text{post}}-(\mathbf1-u)\odot(A^{\top}z_{\text{post}}),
$$

so step $t{+}1$ starts from a noise vector consistent with the mutilated regime
(`recover_epsilon_u`). Sequential PPO uses this; the set operator applies the whole set
in one solve.

**DAG ensemble (uncertainty).** The pessimism term needs *diverse, discrete* DAGs whose
mean matches $A^\star$. We draw $N$ **hard-Gumbel** samples

$$
A^{(n)}=\mathbf 1\!\big[\theta+G^{(n)}_1-G^{(n)}_2>0\big]\odot W\odot M,\qquad
G^{(n)}_1,G^{(n)}_2\overset{\text{iid}}{\sim}\mathrm{Gumbel}(0,1),
$$

so each edge fires with probability $\sigma(\theta)$ — diverse on uncertain edges
($\theta\!\approx\!0$), unanimous on confident ones. (Using the model's *soft*
train-mode gates was a bug: $\sigma((\theta+G)/\tau)\in(0,1)$ is dense and its mean
$\ne A^\star$ — a different dynamical system.)

---

## 5. The objective / reward — gene-space DE-match (shared)

**Why gene-space, not latent.** Minimizing latent distance $\lVert z-z^{\text{tgt}}\rVert$
is *gameable*: the 64-dim bottleneck is many-to-one, so a planner can reach the target
$z$ via gene sets that **decode to the wrong expression**. Empirically MPC-on-latent hit
the *closest* $z$ but the *worst* gene-space DE. We therefore score and optimize in gene
space.

**Predicted DE and the potential.** For a latent $z$, the predicted differential
expression is $\hat d(z)=Dz+b-\hat x_{\text{ctrl}}$. With the target's DEG set
$\mathcal S_p=\{\,j:\ j\notin G_p,\ |d^{\text{obs}}_{p,j}|>\tau\,\}$, $\tau=0.05$ (target
genes masked), the **potential** is the signed Pearson correlation on the DEGs:

$$
\Phi_p(z)\;=\;\operatorname{corr}_{\,j\in\mathcal S_p}\!\big(\hat d(z)_j,\ d^{\text{obs}}_{p,j}\big)\ \in[-1,1].
$$

$\Phi=1$ means the reached state's DE perfectly matches the real perturbation's DE.

**Penalized, potential-shaped reward.** With discount $\gamma$, pessimism weight
$\alpha$, step cost $\lambda$, and the ensemble $\{A^{(n)}\}_{n=1}^N$:

$$
\boxed{\,r_t=\underbrace{\big(\gamma\,\Phi_p(z_{t+1})-\Phi_p(z_t)\big)}_{\text{potential shaping (mean, eval-}A^\star)}
\;-\;\underbrace{\alpha\,\operatorname{std}_{n}\!\big[\gamma\,\Phi_p\big(z^{(n)}_{t+1}\big)\big]}_{\text{epistemic pessimism (MOPO-style)}}
\;-\;\underbrace{\lambda\,\mathbf 1[\,|G_t|>0\,]}_{\text{step cost}}\,}
$$

where $z_{t+1}$ is the operator output under $A^\star$ and $z^{(n)}_{t+1}$ under $A^{(n)}$.
Potential shaping is policy-invariant (Ng et al.): it densifies learning without changing
the optimal policy. The **entropy-for-exploration** term is added inside the PPO loss
(§8–9) as $c_{\text{ent}}\mathcal H(\pi)$, so the full objective is the requested
"mean $-$ pessimism $+$ entropy" across all methods. The per-state ensemble std
$\operatorname{std}_n[\Phi^{(n)}]$ is also logged as the **OOD-honesty** signal.

---

## 6. Baselines (planning ladder)

All planners output a gene set; all are scored identically by decoding the do-operator on
the chosen set and computing $\Phi$ (so they differ only in *how* they pick genes). Let
$\delta_g=z_{\text{post}}(\{g\})-z_0$ be a single gene's latent effect.

- **random** — $k$ genes drawn uniformly from $\mathcal G$ (averaged over seeds).
- **greedy-1** — the single gene closest in latent to the target:
  $\;\arg\min_{g}\lVert z_0+\delta_g-z^{\text{tgt}}\rVert$.
- **additive** — greedy additive latent composition: repeatedly add the $\delta_g$ that
  most reduces $\lVert z_0+\sum\delta-z^{\text{tgt}}\rVert$ (no operator epistasis).
- **prediction-side baselines** (the falsification gate, *given the true genes*, not
  planners): **shift_add** $=\sum_{g\in G}\hat d(g)$ (additive composition), and
  **additive_obs** $=\sum_{g}d^{\text{obs}}_g$ (observed-singles oracle).

> Honest caveat: `greedy-1`/`additive` currently *select* by latent geometry but are
> *scored* in gene space, so part of MPC's edge is that MPC selects on the scored metric.
> A stricter test (planned) adds a **gene-space additive planner** that selects genes to
> maximize $\Phi$ of the additive prediction.

---

## 7. M0 — MPC (oracle / upper bound)

No learning. Beam search over candidate gene **sets** $|G|\le k_{\max}$ that maximizes
the pessimistic gene-space match:

$$
J(G)=\underbrace{\Phi_p\big(z_{\text{post}}(G;A^\star)\big)}_{\text{match, eval-}A^\star}
-\;\alpha\,\operatorname{std}_n\!\big[\Phi_p\big(z_{\text{post}}(G;A^{(n)})\big)\big]
-\;\rho\,|G|,\qquad G^\*=\arg\max_G J(G).
$$

Beam: start empty; at each depth expand every beam by each unused candidate, decode +
score, keep the top-$B$ sets; return the best set found over depths $1..k_{\max}$.
$B{=}6$–$8$, $k_{\max}{=}2$, $N{=}8$–$16$. ($B{=}1,k_{\max}{=}1\Rightarrow$ greedy-1.)
Roles: ladder rung, **distillation teacher** (M3), and the **upper bound** the policies
chase. Cost: $\sim(C+B\cdot C)$ decode-and-score calls per target.

---

## 8. M1 — Sequential PPO

**MDP.** State $s_t=[\,z_t\,;\,z^{\text{tgt}}\,]\in\mathbb R^{2d}$ (goal-conditioned).
Action $a_t\in\{1,\dots,C\}$ = one gene per step. Transition = the do-operator on $\{g_{a_t}\}$
with ε threading (a combo is reached as a **sequence** of single-gene steps, length =
target cardinality $k\in\{1,2\}$). Reward = §5.

**Networks.** Actor $\pi_\theta(a\mid s)=\mathrm{Categorical}\big(\mathrm{MLP}_\theta([z_t;z^{\text{tgt}}])\big)$
and critic $V_\psi(s)$, each a $2d\!\to\!256\!\to\!256\!\to\!(\,C \text{ or }1)$ MLP. The
$C$-way head (vs the dead run's ~4000) is a key fix.

**PPO + GAE.** Advantages

$$
\delta_t=r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t),\qquad
\hat A_t=\sum_{l\ge0}(\gamma\lambda_{\text{GAE}})^l\delta_{t+l},\qquad
\hat R_t=\hat A_t+V_\psi(s_t).
$$

With probability ratio $\rho_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$, the clipped surrogate + value + entropy losses:

$$
L^{\text{CLIP}}(\theta)=\mathbb E_t\!\Big[\min\!\big(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t,1{-}\varepsilon,1{+}\varepsilon)\hat A_t\big)\Big],
$$
$$
\mathcal L=\;-\,L^{\text{CLIP}}\;-\;c_{\text{ent}}\,\mathbb E_t[\mathcal H(\pi_\theta(\cdot\mid s_t))]\;+\;c_v\,\mathbb E_t\big[(V_\psi(s_t)-\hat R_t)^2\big].
$$

Defaults: $\gamma{=}0.99,\ \lambda_{\text{GAE}}{=}0.95,\ \varepsilon{=}0.2,\ c_v{=}0.5,\
c_{\text{ent}}{=}0.02$, Adam $\text{lr}_\pi{=}3\!\times\!10^{-4},\ \text{lr}_V{=}10^{-3}$,
4 epochs/update, minibatch 64, update every 128 env steps, grad-clip 0.5.

**As a planner (for the ladder).** Greedy roll-out: at each step take
$\arg\max_a \pi_\theta(a\mid s)$ with already-chosen genes masked, for $k$ steps → gene set.

---

## 9. M2 — Set-based PPO (★ the novel architecture)

The policy emits an **unordered set** of $\le k_{\max}$ genes applied **simultaneously**
(matching how a combo is delivered — synergy lives in the joint action, not split across
steps). Episodes are **one step**: the policy builds a set, the env applies it once.

**Vocabulary & autoregressive recipe.** $\mathcal V=\{\texttt{STOP}\}\cup\{1,\dots,C\}$
(single-direction, so $|\mathcal V|=C+1$; token 0 = STOP). An action is tokens read until
STOP or $k_{\max}$ slots, collapsed to a set. A **shared encoder** $h=\mathrm{enc}(s)$
feeds $k_{\max}$ slot heads (BDQ-style branched outputs → cost **linear**, not
combinatorial, in $k_{\max}$). The set probability factorizes by the chain rule:

$$
\pi_\theta(a\mid s)=\prod_{i=1}^{|a|}\pi_\theta(a_i\mid s,a_{<i}),\qquad
\pi_\theta(a_i\mid s,a_{<i})=\operatorname{softmax}\!\Big(\operatorname{head}\big[\,h\ ;\ \textstyle\sum_{j<i}e(g_j)\ ;\ \text{slot}(i)\,\big]\Big),
$$

with two masks before the softmax (set logits to $-\infty$): **STOP forbidden at slot 0**
(no empty action), and **already-chosen genes forbidden** at later slots (no duplicates).
The context $\sum_{j<i}e(g_j)$ is an **order-invariant sum** of chosen-gene embeddings —
matching set semantics. Hence

$$
\log\pi_\theta(a\mid s)=\sum_{i=1}^{|a|}\log\pi_\theta(a_i\mid s,a_{<i}),\qquad
\mathcal H(\pi_\theta(\cdot\mid s))=\sum_{i=1}^{|a|}\mathcal H\big(\pi_\theta(\cdot\mid s,a_{<i})\big).
$$

The STOP token lets the policy choose its **own cardinality** (singles vs doubles).

**PPO.** Identical clipped-surrogate machinery as §8; the *only* change is that
$\log\pi$ and $\mathcal H$ are the per-slot **sums** above (PPO only ever consumes that one
scalar log-prob). Episodes are one step, so GAE collapses to $\hat A=r-V_\psi(s)$.
A unit test confirms `evaluate(sampled tokens)` reproduces the sampled $\log\pi$ to
$2\!\times\!10^{-7}$ (the factorization is consistent between sampling and re-evaluation —
required for a correct importance ratio).

**As a planner.** Greedy: $\arg\max$ token per slot (with masks), STOP-terminated → set.

---

## 10. M3 — MPC → policy distillation

Cold PPO over a thin task distribution (146 tasks) with a $C$-way head is slow/unstable;
distillation warm-starts the policy from the MPC oracle.

1. **Collect** MPC plans $\{(s_p,\,a^{\text{MPC}}_p)\}$ over the train tasks ($a^{\text{MPC}}_p$
   as a token row for the set policy).
2. **Behavioral cloning** (cross-entropy = max-likelihood of the teacher's set):
   $$
   \mathcal L_{\text{BC}}(\theta)=-\,\mathbb E_{p}\big[\log\pi_\theta\big(a^{\text{MPC}}_p\mid s_p\big)\big]
   =-\,\mathbb E_p\Big[\textstyle\sum_i\log\pi_\theta\big(a^{\text{MPC}}_{p,i}\mid s_p,a^{\text{MPC}}_{p,<i}\big)\Big].
   $$
3. **PPO fine-tune** from the BC init against the §5 reward, so the policy can **exceed**
   the teacher by amortizing across targets (and exploiting the dense shaped reward).

MPC collection is the slow step (each plan is a beam search) → cache to disk; use modest
$B,N$ during collection.

---

## 11. Evaluation — the value-add ladder + findings

On the **held-out 31 combos**, every planner reports:

- **DE-match** $\Phi$ (primary) — signed Pearson of the reached DE vs observed, on DEGs.
- **gene recovery** — $|G^\*\cap G^{\text{true}}|/|G^{\text{true}}|$ and full-recovery rate.
- **pearson-DEG, Spearman, DEG-AUROC** — secondary.

Ladder: `random → greedy-1 → additive → MPC → seq-PPO → set-PPO (→ distilled)`.

**Established (M0):** MPC $\Phi=0.80$ vs additive $0.54$, random $0.51$ — gene-space
search over the world model substantially beats additive composition at **reaching a
target state's DE** (a controllability result). Two findings carried forward:

- **Identifiability (reported, not a failure):** gene recovery $\approx0$ — the
  perturbation→state map is many-to-one in 64-dim, so the planner reaches the target DE
  via *different* genes ~90% of the time. Relevant + acceptable for reprogramming (many
  protocols reach a target).
- **AUROC saturated** ($\approx0.90$ even for random genes; Norman's broad overlapping
  responses) → DE-match / pearson-DEG are the discriminative metrics.

**Planned extra probes (`reachability_ood.py`):** (i) reach an *unseen* (test-split)
pert's state using only *train-split* candidate genes (compositional generalization);
(ii) OOD honesty — Spearman$\big(\operatorname{std}_n[\Phi^{(n)}],\ \text{held-out error}\big)>0$.

---

## 12. Training protocol & cost (how each is run *properly*)

| method | learns | "data" | proper run | wall-clock (1 GPU) |
|---|---|---|---|---|
| baselines | — | — | — | seconds |
| **M0 MPC** | nothing (search) | 31 test targets | n/a | ~10–15 min (beam over combos) |
| **M1 seq-PPO** | $\pi_\theta,V_\psi$ | 146 train tasks | $\ge$ 50k–200k iters to convergence | hours |
| **M2 set-PPO** | set-$\pi_\theta,V_\psi$ | 146 train tasks | $\ge$ 50k–200k iters | hours |
| **M3 distill** | set-$\pi_\theta$ | MPC plans on 146 tasks + reward | BC (few hundred epochs) + 50k+ PPO | MPC-collect (slow) + hours |

> **Honesty on the runs so far:** the M1/M2 runs to date used `--iters 5000` (~8 min) —
> a **pipeline-validation / first pass, NOT a converged PPO**. With an 87-way head and a
> 146-task distribution, that is under-trained; treat those numbers as a smoke check, not
> the policy's real ceiling. The credible protocol is a **long SLURM GPU job**, and given
> the thin task distribution the recommended order is **distillation-first** (M3): BC from
> MPC, then long PPO fine-tune, with seq/set cold-PPO as ablations.

---

## 13. Limitations (carried into any conclusion)

1. **Thin task distribution** — 146 train tasks; goal-conditioning must do the
   generalization. (Can't pad with Replogle singles: wrong modality for a Norman WM.)
2. **Epistasis-neutral operator** — exp14's mutilation beats additive on combos but loses
   to the observed-singles oracle ($0.95$); the planner inherits this ceiling.
3. **Identifiability** — state-reaching ≠ gene recovery (many-to-one).
4. **Selection/scoring asymmetry** in the baselines (§6) — add a gene-space additive
   planner for a fully fair ladder.
5. **In-distribution only** — exp14 is K562/CRISPRa; cross-cell/reprogramming is the
   separate exp16 / differentiation-atlas track.

---

## 14. Validation — exactly how each model is evaluated (step by step)

**Shared harness** (`core/validate.py::evaluate_planner`). For every held-out combo target
$p$ (the 31 test combos), the loop is:
1. `chosen = planner(target_key)` → a gene set (each method's "produce a plan" below).
2. `pred = predict_de(chosen)` → the gene-space prediction (`core/baselines.py`):
   $z_{\text{post}}=\text{do-op}(\bar\epsilon_{\text{ctrl}},\text{chosen};A^\star)$,
   $\hat d=D z_{\text{post}}+b-\hat x_{\text{ctrl}}$. **Note: scoring always decodes the chosen
   set through the SAME shift-native mutilation operator under the eval-A point estimate** —
   so methods differ only in *which genes* they pick, never in how the prediction is made.
3. **Metrics:** `match = de_match(pred, observed_DE_p, mask)` (signed Pearson on DEGs,
   target genes masked) — the headline; plus `recall = |chosen ∩ true|/|true|` and latent
   `dist_ratio`. Mean ± SEM over the 31 combos.

**The cardinality question ("do you stop at two genes?") — explicitly:**
- The combo targets are 2-gene perturbations, and we ask each planner for a 2-gene
  intervention. *How* the "2" is enforced differs by model:
  - **MPC:** `max_set=2` is a **cap**; beam search explores 1- and 2-gene sets and returns the
    lowest-cost one (for combo targets the best *pair* wins, so it returns 2).
  - **sequential PPO:** the env is run for **exactly `max_steps = k` steps**, with
    `k = 2` for a combo target (cardinality given) → it emits exactly 2 genes.
  - **set-PPO / distill:** `k_max=2` cap, but the policy **chooses its own cardinality via
    STOP** — it can emit 1 *or* 2 genes (slot-0 gene, then slot-1 gene *or* STOP). So it is
    *not* forced to 2.
  - **baselines:** random→2, top2-marginal→2, greedy-2→2, greedy-1→1.

### Per-model "produce a plan" forward pass

**M0 — MPC** (`rl0_mpc/mpc_planner.py::plan`), no learned weights:
1. fetch the target's observed DEGs $\mathcal S_p$ + observed DE $o_p$; sample the hard-Gumbel
   DAG ensemble $\{A^{(n)}\}$.
2. **depth-1:** compute summed shifts $S_g$ for **all** $C$ candidates at once; batched
   `apply_many` → $z_{\text{post}}^{(g)}$ for all $g$; decode + `torch_match` → $\Phi(g)$ for
   all; keep the top-$B$ singles.
3. **depth-2:** for each beam gene $b$, batched-score $b{+}g$ for **all** $g$ → best pair.
4. **pessimistic re-rank** the $B{+}1$ finalists by $\Phi^{A^\star}-\alpha\,\mathrm{std}_n\Phi^{A^{(n)}}$;
   return the argmax set (≤2 genes).

**M1 — sequential PPO** (`rl1_seq_ppo/train.py::make_planner`), trained actor frozen:
1. `env.max_steps = k` (= target cardinality, 2 for combos); `obs = env.reset(target)`
   → start state $[z_0; z_{\text{target}}]$ from control-mean ε.
2. for step $t=1..k$: split obs → $(z_t, z_{\text{target}})$; **greedy** action
   $a=\arg\max_a \pi_\theta(a\mid z_t,z_{\text{target}})$ with already-chosen genes masked
   ($-\infty$); apply gene $a$ via the do-operator; **ε is threaded** (`recover_epsilon_u`)
   so step $t{+}1$ starts from the persisted state. Collect the $k$ genes → the set.

**M2 — set-PPO** (`rl2_set_ppo/train.py::make_planner`), trained set-policy frozen:
1. `env.max_steps = 1`; encode $h=\mathrm{enc}([z_0;z_{\text{target}}])$ once.
2. autoregressive **greedy** decode: slot 0 → $\arg\max$ over $\{1..C\}$ (STOP masked at slot 0);
   slot 1 → $\arg\max$ over $\{\texttt{STOP}\}\cup\{1..C\}\setminus\text{chosen}$ — if STOP,
   the set is a single; else add the gene. Cap at $k_{\max}=2$. The collapsed set is applied
   in **one** do-operator call.

**M3 — distill** (`rl3_mpc_distill`): the policy *is* a set-policy, so its validation forward
pass is **identical to M2** (greedy autoregressive set, one env step). The only difference is
how its weights got there (BC from MPC, then PPO fine-tune).

### What I'd add next (your question — Yamanaka / differentiation evals)

Short answer: **yes, but on the right world model.** The current MPC/distill run on the
**exp14 Norman/K562 CRISPRa** world model, which has **no notion of pluripotency, iPSC, or the
Yamanaka factors** — K562 is not a reprogramming system and OSKM aren't in its perturbation
set. So validating the *Norman* planner against Yamanaka/differentiation would be **far OOD and
not meaningful**. The principled places to add those evals:

1. **On the Joung WM (exp17) — the natural fit.** Joung is TF-overexpression in **hESCs with a
   directed-differentiation readout**, and it ships the **44 pair + 3 triple** combinatorial
   ground truth. Once it trains, the goal-conditioned RL there supports two strong evals
   beyond combo DE-match: **(a) known-protocol recapitulation** — for a target lineage state,
   does the planner's top-$k$ actions recover the **canonical differentiation TFs** the Joung
   paper validated (e.g. MYOD1→muscle, NEUROD/ASCL1→neural, GATA→endoderm)? **(b) ESM-smoothed
   recall / pathway-enrichment** — credit for picking TFs functionally near the true drivers
   (ESM2 / CollecTRI module), which sidesteps the exact-recall identifiability wall.
2. **A true Yamanaka (fibroblast→iPSC, OSKM) eval needs a reprogramming atlas** (e.g.
   Schiebinger MEF→iPSC) trained as its own WM — that's the DiRL setting, with the
   self-evaluation caveat. Worth doing as a separate track, *not* on the Norman/Joung models.
3. **On exp14/Norman now (cheap):** add the **ESM/pathway-enrichment recall** as a reported
   metric — it's the only "beyond combos" eval that's in-distribution for the Norman planner.

So: route the Yamanaka/differentiation validations to the **Joung WM** (and a reprogramming
atlas for OSKM specifically); keep the Norman planner's extra eval to the ESM-smoothed recall.

---

## 15. Perturbation-effect soft recall (the effect-cluster metric)

The "ESM-smoothed recall" above is one way to relax exact gene recovery. The version we
**actually use** rewards the thing we care about directly — *picking genes with similar
perturbation effects* — rather than similar protein sequence. It is **diagnostic /
reported, with no harsh penalty** (the planner is not punished for a low score).

**Why.** Exact gene recall is ≈0 because perturbation→state is many-to-one: the planner
reaches a target via different-but-functionally-equivalent drivers. Recall=0 conflates
"picked a gene with the *same downstream effect*" with "picked junk". The effect-cluster
metric separates them.

**Construction** (`experiments_rl/core/effect_clusters.py`, `EffectClusters`):
- **Posthoc effect modules.** Cluster the screened genes by their **observed** DE profiles
  into $K$ modules. We z-score (mean-centre + unit-norm) each gene's profile and run
  **Ward-linkage** agglomerative clustering on it (Euclidean on unit-norm profiles $\propto
  \sqrt{2(1-\text{corr})}$, so it is correlation-based but **variance-balanced**). Genes that,
  when perturbed, move the cell the same way land together. *Note:* the first version used
  average-linkage on the raw `1−corr` distance, which **chained** into one 64-of-87-gene
  mega-cluster + singletons and made `clust` near-trivial (random ≈ 0.73); Ward fixes this
  (balanced modules, e.g. sizes `[11,5,3,4,15,5,17,5,7,15]`, random ≈ 0.16).
- **Universal query embedding.** A chosen gene may be unscreened (zero-shot 8000-pool), so
  it has no observed profile. Embed it by the **model-predicted** single-gene DE
  $e_{\text{pred}}(g)=\text{decode}(\text{do}(\varepsilon_{\text{ctrl}},\{g\}))-\text{decode}(z_{\text{ctrl}})$,
  which is defined for any gene and lives in the same gene-expression frame as the observed
  profiles — so a chosen gene's predicted profile compares directly to a true gene's
  observed profile and assigns to the nearest module.

**Two measures** (both in `evaluate_planner`, reported per-target + summary):
- `soft_recall_corr` (graded): for each true gene, the **max correlation** of its observed
  profile with any chosen gene's predicted profile, averaged. Continuous; exact recovery → 1.
- `soft_recall_clust` (cluster-hit): fraction of true drivers whose **effect module** is hit
  by some chosen gene. 1 iff every true driver's module is represented.

**Optional reward use.** `effect_bonus = relu(soft_recall_corr) ≥ 0` is exposed as a small
**additive** term MPC/RL may add to the gene-space reward — *never* a penalty (per the "no
harsh penalty" requirement). Default is report-only, so the planner is not biased toward
canonical genes; the bonus is a knob for experiments that want to nudge toward
effect-similar picks.

**Dataset-agnostic** — built from `data.de_single` + the operator, so the same metric serves
the Norman ladder (`run_fullpool_ladder.py` reports both columns) and the Joung TF-atlas
planner once exp17 trains.

### 15.1 Ladder analysis — which method picks effect-equivalent genes?

`analyze_effect_clusters_ladder.py` applies the metric **post-hoc** across the whole Norman
RL ladder (no re-planning): it loads the frozen WM + `EffectClusters`, reads each method's
already-saved per-combo `(target, chosen)` rows, maps gene names → indices, and scores
`soft_recall_{corr,clust}` of the **chosen** genes against the **true combo** genes for every
held-out combo. Methods scored (same 7 as the DE-match ladder): `random`, `seq-PPO`,
`set-PPO`, `MPC→distill`, `top2-marginal`, `greedy-1`, `MPC`.

**Result** (Norman held-out combos, $n=31$; 10 Ward modules over 87 screened genes; mean):

| method | cluster-hit | graded corr |
|---|---|---|
| random | 0.16 | 0.45 |
| seq-PPO | 0.16 | 0.45 |
| set-PPO | 0.23 | 0.52 |
| greedy-1 | 0.40 | 0.59 |
| top2-marginal | 0.42 | 0.60 |
| MPC→distill | 0.44 | 0.60 |
| **MPC** | **0.45** | **0.62** |

**Reading.** The ordering mirrors the DE-match ladder. **MPC and the planning/distilled
methods pick genes in the true effect-module ~2.8× more often than chance** (0.45 vs the
≈0.16 random floor) — they choose *effect-equivalent* drivers, not just lucky state matches.
**`seq-PPO` sits at the random floor on both measures** despite reaching DE-match 0.59: it
reaches the target state via genes that are effectively arbitrary, while MPC's are
biologically aligned. This is the metric's value-add — exact gene recall is ≈0 for everyone
(identifiability), but the effect-cluster view cleanly separates "effect-equivalent drivers"
(MPC family) from "arbitrary genes that happen to reach the state" (`seq-PPO`).

**Artifacts** (`experiments_rl/results/effect_clusters/`): `effect_cluster_softrecall.png`
(2-panel cluster-hit + graded-corr bar chart, baselines hatched), `effect_cluster_map.png`
(MDS of the screened genes coloured by effect module), `per_method.csv`, `summary.csv`.

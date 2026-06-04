# The u-score (decoder-loading) intervention scheme

This document specifies the new do-operator used by `intervention.scheme = "u_score"`
and proves the four claims about it that make the scheme principled:

1. **Endpoint sanity (u = 0)**: at $u = 0$, the u-score do-operator reduces to the unperturbed SCM.
2. **Endpoint sanity (u = 1)**: at $u_i = 1$, it reduces to the classical do-operator $\mathrm{do}(z_i = z^{\text{target}}_i)$.
3. **The previous threshold scheme is biased** — it only equals the intended $\mathrm{do}$ when $A \approx 0$.
4. **Numerical conditioning** of $(I - M A^\top)$ is never worse than the original SCM.

It also describes how persistence (the accumulating $u_t$ and the gene-level
knockout clamp) carries causal state across episode steps.

Code references (line numbers as of this commit):
- [`models/dag.py:scm_intervene_u`](../models/dag.py) — the SCM solve.
- [`models/causal_vae.py:compute_loadings`, `intervene_gene_u`, `recover_epsilon_u`](../models/causal_vae.py) — the wrapper that combines $D^+$, $L$, and step-level state.
- [`rl/env.py:step`](../rl/env.py) — the dispatch and episode-level $u_t$ + knockout-set updates.
- [`tests/test_u_intervention.py`](../tests/test_u_intervention.py) — invariants 1, 2, 3, 6, 7 from this doc, all verified to numerical zero on a controlled toy model.

---

## 1. Setup: the linear SCM and what we are intervening on

The Causal VAE maps a cell's gene-expression vector $x \in \mathbb{R}^G$ to a
latent program vector $z \in \mathbb{R}^d$ via the encoder, applies a linear
structural causal model on $z$, then decodes back to $x$. The SCM is

$$
z \;=\; A^\top z \;+\; \varepsilon
\qquad\Longleftrightarrow\qquad
z \;=\; (I - A^\top)^{-1}\,\varepsilon ,
$$

where $A \in [0, 1]^{d\times d}$ is the (Gumbel-sigmoid-sampled) adjacency
matrix with $A_{ij}$ = probability of edge $i \to j$, $\varepsilon$ is exogenous
noise, and $A$'s diagonal is masked to zero. The decoder is

$$
x \;=\; D\,z \;+\; b , \qquad D \in \mathbb{R}^{G\times d}, \quad b \in \mathbb{R}^{G} .
$$

The decoder is regularised toward column-orthogonality
($D^\top D / (\|D\|_F^2/d) \approx I$), so each column of $D$ is the gene-space
"signature" of one latent program and columns have near-uniform energy.

**A perturbation in the experiment is a gene-level action** — e.g. CRISPRi
knockdown of gene $g$, or CRISPRa overexpression of $g$. It gives a target
change $dx \in \mathbb{R}^G$ supported on the row $g$ of the gene axis.
The world model has to lift this to a latent-level intervention.

---

## 2. Why "argmax decoder column" or threshold-mutilation are unsatisfying

**Argmax pick** (`_get_latent_for_gene`): set $i^* = \arg\max_i |D_{gi}|$ and
do $\mathrm{do}(z_{i^*} = v)$. This throws away the long tail of programs that
$g$ partially drives — a problem the diagnostic exposed: with an orthogonal
decoder, $D^+ dx$ has near-uniform energy across many programs.

**Threshold multi-mutilation** (`scm_intervene_multi`): compute
$dz = D^+\,dx$, then $d\varepsilon = (I - A^\top) dz$, then mutilate the set of
columns $\mathcal{S} = \{i : |d\varepsilon_i| > \tau \cdot \max_j |d\varepsilon_j|\}$
and shift $\tilde\varepsilon = \varepsilon + d\varepsilon$, then solve
$\tilde z = (I - \tilde A^\top)^{-1}\tilde\varepsilon$. The mathematical
problem with this scheme is the topic of §4 below.

The diagnostic on `phase1_best_no_dag.pt` showed 49 out of 64 programs
"affected" per single-gene perturbation — i.e. half the latent dim got
mutilated for every single-gene experiment. That number is a feature of the
threshold rule, not of the DAG; with $A \approx 0$ we have $d\varepsilon \approx dz$,
and $dz = D^+ dx$ for a single gene is dense.

---

## 3. The u-score scheme

### 3.0 Why the decoder also gets an L1 penalty

Orthogonality regularises **columns** of $D$ (programs are mutually orthogonal directions
in gene space) but does **not** regularise rows (a single gene's loadings across
programs). With $G \!=\! 2000$ and $d \!=\! 64$:

$$
\underbrace{\tfrac{d(d-1)}{2} \;=\; 2016}_{\text{orthogonality constraints}}
\;\ll\; \underbrace{G\,d \;=\; 128{,}000}_{\text{entries of }D}.
$$

So $D$ can be column-orthogonal and *still* have every row dense. That is exactly the
observed regime (`ortho off-diag = 0.0006`, `col-norm CV = 0.0007`, yet
`dz L1/L2 = 6.17` ≈ $\sqrt{d}$ = dense). For single-gene perturbations,
$dz = D^{+}\,dx$ is essentially a row of $D^+$; orthogonality forces no sparsity on it.

**Penalty.** We add a mean-normalised L1 on every entry of $D$:

$$
L_{D\text{-sparse}} \;=\; \frac{1}{G\,d}\,\sum_{g=1}^{G}\sum_{i=1}^{d}\,|D_{g,i}| .
$$

This pushes both row-sparsity (each gene loads on few programs) and column-sparsity
(each program defined by few genes). Sparse-orthogonal matrices exist (e.g. block-
supported columns), so the penalty is **compatible** with the existing orthogonality
loss — the equilibrium is the standard *sparse factor analysis* solution.

**Effect on the u-score scheme.** $L_{g,i} = D_{g,i}^2 / T_i$ inherits the sparsity
of $D$:
- Sparse $D$ → sparse $L$ rows → single-gene $u = L[g, :]$ is sharper.
- $\sum_g L_{g,i} = 1$ still holds (the proof only uses the definition of $L$).
- Tested in `test_loadings_with_sparse_D`.

**Effect on identifiability.** Orthogonality alone fixes $D$ up to rotations in $O(d)$
plus sign flips. Adding $L_1$ on $D$ further restricts the unidentifiable group to
permutations + sign flips — programs become *labelable* (up to ordering), and the DAG
$A$ becomes reproducible across seeds.

**Default weight.** `lambda_D_sparse = 0.001` with the same warm-in schedule as the
DAG losses (epoch ≥ `dag_start_epoch`, ramped over `dag_warmin_epochs`). The grid
search sweeps it in $\{0.0, 0.001, 0.01\}$.

**Diagnostic.** `dag_diagnostics.py` reports `mean_programs_per_gene` (target ~5–10
for biologically reasonable programs) and `mean_genes_per_program`, plus a
histogram in `decoder_sparsity_hist.png`.

### 3.1 The loading matrix $L$ and the strength vector $u$

For each latent program $i$ define its decoder energy and each gene's
contribution to that energy:

$$
T_i \;=\; \sum_{h=1}^{G} D_{hi}^{2},
\qquad
L_{gi} \;=\; \frac{D_{gi}^{2}}{T_i} .
$$

**Properties of $L$**, verified in `test_loadings_column_sum`:

- $L_{gi} \ge 0$ for all $g, i$.
- $\sum_g L_{gi} = 1$ — each column of $L$ is a probability distribution over genes.
- $L_{gi}$ is the **fraction of program $i$'s decoder energy contributed by gene $g$**. Perturbing $g$ disturbs program $i$ in proportion to $L_{gi}$.

For an episode with a sequence of gene actions $a_1, a_2, \dots$ we accumulate

$$
u_t \;=\; \mathrm{clip}\Big(u_{t-1} + L_{g_t,\,:}, \;0,\; u_{\max}\Big),
\qquad u_0 = 0,
$$

where $g_t$ is the gene perturbed at step $t$ and $u_{\max} \in (0, 1]$ is a
clip ceiling (default 1.0). $u_t \in [0, 1]^d$ is monotone non-decreasing in
$t$ — see `test_u_monotonic`.

### 3.2 The do-operator

Given $\varepsilon_{t-1}$, $z_{t-1}$, and the gene action $a_t = (g_t, \delta_t)$
("knockout" or "overexpress"), the u-score do-operator computes:

#### (a) Gene-level effect.
For knockout: $dx = -x_{t-1}^{(g_t)} e_{g_t}$, where $x_{t-1}$ is the decoded
gene-expression vector and $e_g$ is the gene-axis unit vector. **Persistence
clamp**: before computing $dx$, set $x^{(g)} = 0$ for every $g$ in the running
knockout set $\mathcal{S}^{\text{ko}}_{t-1}$. For overexpression:
$dx = (x^{\max}_{g_t} - x_{t-1}^{(g_t)}) e_{g_t}$.

#### (b) Latent target.
Project to the latent space via the regularised decoder pseudoinverse
([`compute_dz_from_dx`](../models/causal_vae.py)):

$$
dz \;=\; (D^\top D + \lambda I)^{-1} D^\top dx,
\qquad
z^{\text{target}}_t \;=\; z_{t-1} + dz .
$$

#### (c) Effective noise (the key step).
Mix $\varepsilon_{t-1}$ with the target *per program* using $u_t$:

$$
\boxed{\;\;\tilde\varepsilon_t \;=\; u_t \odot z^{\text{target}}_t \;+\; (1 - u_t) \odot \varepsilon_{t-1}\;\;}
$$

For program $i$: $\tilde\varepsilon_{t,i} = u_{t,i}\, z^{\text{target}}_{t,i} + (1-u_{t,i})\,\varepsilon_{t-1,i}$.

#### (d) Mutilated graph.
With $M_t = \operatorname{diag}(\mathbf 1 - u_t)$,

$$
\tilde A^{(n)}_t \;=\; A^{(n)} M_t \quad\text{(scale column $i$ of $A$ by $1 - u_{t,i}$)}.
$$

The product is taken over each bootstrap DAG sample $n$.

#### (e) Solve and recover.
For each bootstrap $n$:

$$
\big(I - M_t\,A^{(n)\top}\big)\,z^{(n)}_t \;=\; \tilde\varepsilon_t ,
\qquad
z_t \;=\; \frac{1}{N}\sum_n z^{(n)}_t .
$$

Consensus $z_t$ is used to assign next-state (KMeans/Leiden centroid) and to
recover the next-step noise (in the mutilated regime, since the intervention
persists going forward):

$$
\varepsilon^{(n)}_{t+1} \;=\; \big(I - M_t\,A^{(n)\top}\big)\,z^{(n)}_t .
$$

---

## 4. Math — proofs of correctness

### 4.1 Lemma 1 (endpoint $u = 0$ ⇒ identity)

**Claim.** If $u_t = 0$ then $z_t = z_{t-1}$ and $\tilde A_t = A$.

**Proof.** $M_t = I$, so $\tilde A_t = A I = A$.
The mixing step gives $\tilde\varepsilon_t = 0 \odot z^{\text{target}}_t + 1 \odot \varepsilon_{t-1} = \varepsilon_{t-1}$.
The solve becomes $(I - A^\top) z_t = \varepsilon_{t-1}$, which is the
unperturbed SCM, so $z_t = (I - A^\top)^{-1}\varepsilon_{t-1} = z_{t-1}$.   $\blacksquare$

Verified to numerical zero in `test_u_zero_identity`.

### 4.2 Lemma 2 (endpoint $u_i = 1$ ⇒ classical $\mathrm{do}$)

**Claim.** Suppose $u_t = e_i$ (one-hot at coordinate $i$). Then
$z_{t,i} = z^{\text{target}}_{t,i}$ and column $i$ of $\tilde A_t$ is zero.
This is the classical $\mathrm{do}(z_i = z^{\text{target}}_{t,i})$.

**Proof.** With $M_t = \operatorname{diag}(\mathbf 1 - e_i) = I - e_i e_i^\top$,
the column scaling $\tilde A_t = A M_t$ zeroes column $i$ of $A$ (multiplies it
by $0$) and leaves all other columns untouched. So $\tilde A_t e_i = 0$, i.e.
column $i$ is removed — exactly Pearl's truncation of incoming edges to $i$.

For the value: row $i$ of $M_t A^\top$ is $(1 - u_{t,i})\,(A^\top)_{i,:} = 0$.
So row $i$ of $I - M_t A^\top$ is $e_i^\top$. The system

$$
(I - M_t A^\top) z_t \;=\; \tilde\varepsilon_t
$$

therefore reduces row $i$ to $z_{t,i} = \tilde\varepsilon_{t,i}$. From the mixing
rule, $\tilde\varepsilon_{t,i} = u_{t,i}\,z^{\text{target}}_{t,i} + (1-u_{t,i})\,\varepsilon_{t-1,i} = z^{\text{target}}_{t,i}$.
Hence $z_{t,i} = z^{\text{target}}_{t,i}$.   $\blacksquare$

Verified to numerical zero in `test_u_one_pins_target`.

These two endpoint properties — the *only* sane fixed points for a soft
do-operator — pin down $\tilde\varepsilon_t$ uniquely as a linear interpolation
under the constraints (Lemmas 1 & 2 + linearity in $u$). That is the answer
to **"is $\tilde\varepsilon_t = u \odot z^{\text{target}} + (1-u) \odot \varepsilon_{t-1}$ the right formula?"** — yes; it is the unique linear mix that
realises both endpoints under the column-scaled mutilation.

### 4.3 Theorem 3 (the threshold scheme is biased away from the intended do)

**Claim.** Let $\mathcal{S}$ be a non-empty mutilation set in the threshold
scheme. For every $i \in \mathcal{S}$,

$$
z^{\text{threshold}}_{t,i} \;=\; z_{t-1,i} + dz_i \;-\; \big(A^\top (z_{t-1} + dz)\big)_i .
$$

The intended target is $z_{t-1,i} + dz_i$; the threshold scheme adds an
unwanted residual $-\big(A^\top(z_{t-1}+dz)\big)_i$ that only vanishes when
$A \to 0$ (or when no parent of $i$ has any mass in $z_{t-1} + dz$).

**Proof.** The threshold scheme sets

$$
\tilde A_t \;=\; \mathrm{mutilate}(A, \mathcal{S}), \qquad
\tilde\varepsilon_t \;=\; \varepsilon_{t-1} + d\varepsilon ,
\qquad
d\varepsilon \;=\; (I - A^\top)\,dz .
$$

For $i \in \mathcal{S}$, column $i$ of $\tilde A_t$ is zero, so row $i$ of
$\tilde A_t^\top$ is zero, hence row $i$ of $I - \tilde A_t^\top$ is $e_i^\top$
and $z^{\text{threshold}}_{t,i} = \tilde\varepsilon_{t,i}$. Expand:

$$
\tilde\varepsilon_{t,i} \;=\; \varepsilon_{t-1,i} + d\varepsilon_i
\;=\; \varepsilon_{t-1,i} + dz_i - (A^\top dz)_i .
$$

From the unperturbed SCM, $\varepsilon_{t-1,i} = z_{t-1,i} - (A^\top z_{t-1})_i$.
Substituting,

$$
z^{\text{threshold}}_{t,i}
\;=\; z_{t-1,i} - (A^\top z_{t-1})_i + dz_i - (A^\top dz)_i
\;=\; z_{t-1,i} + dz_i - \big(A^\top(z_{t-1} + dz)\big)_i . \quad\blacksquare
$$

The residual is exactly the parental contribution to node $i$ that we
**should have removed** when we mutilated $A$ — but the formula
$d\varepsilon = (I - A^\top) dz$ was computed with the **un-mutilated** $A$,
so it bakes in a parental term that the post-mutilation SCM also evaluates,
producing the double-count. In short: $d\varepsilon$ was the right shift for a
DAG we then dismantled. The u-score scheme avoids this by pinning $z_{t,i}$
to $z^{\text{target}}_{t,i}$ *directly* via $\tilde\varepsilon_{t,i}$ in the
mutilated regime.

### 4.4 Lemma 4 ($\varepsilon$ recovery inverts the do)

**Claim.** $\varepsilon_{t+1} := (I - M_t A^\top) z_t$ satisfies
$\varepsilon_{t+1} = \tilde\varepsilon_t$.

**Proof.** By construction $(I - M_t A^\top) z_t = \tilde\varepsilon_t$ —
the left-hand side is exactly the residual of the SCM equation at step $t$
under the mutilated graph.   $\blacksquare$

This is the persistence guarantee: starting step $t+1$ from
$\varepsilon_{t+1}$ in the same mutilated regime reproduces $z_t$. The
recovered $\varepsilon_{t+1}$ is what the env feeds into the next step.

Verified to numerical zero in `test_recover_epsilon_u_inverts`.

### 4.5 Proposition 5 (numerical conditioning is never worse than the SCM)

**Claim.** Let $L(u) = I - M(u) A^\top$ with $M(u) = \operatorname{diag}(\mathbf 1 - u)$, $u \in [0, 1]^d$. Then

$$
\sigma_{\min}(L(u)) \;\ge\; \sigma_{\min}(L(0)) \cdot \min_i (1 - u_i) \;\text{ — informally, conditioning improves toward identity as }u \to 1.
$$

More precisely, the rows of $L(u)$ where $u_i \to 1$ become $e_i^\top$
(identity rows), which contribute optimal singular structure. The remaining
sub-block $L(u)_{\bar S, \bar S}$ where $\bar S = \{i : u_i < 1\}$ has the
same shape as a smaller-dimensional version of $L(0)$ with row scaling by
$(1 - u_i)$, which can only shrink off-diagonals.

A practical consequence: the SCM solve in `scm_intervene_u` never has worse
conditioning than the original `scm_forward` (which is what gets used at
$u = 0$). The implementation also clamps $z$ to $[-20, 20]$ as a final guard.

---

## 5. Persistence across episode steps

### 5.1 Strength accumulation

$u_t = \mathrm{clip}(u_{t-1} + L_{g_t,:}, 0, u_{\max})$. The first hit on
gene $g$ gives $u_1 = L_{g,:}$ — verified in `test_u_monotonic`. Repeated hits
saturate component-wise at $u_{\max}$.

### 5.2 Gene-level knockout clamp

[`rl/env.py:step`](../rl/env.py) carries a `knockout_set` of gene indices that have
been knocked out in the episode so far. Before computing the next $dx$, it
zeros those entries of the decoded $x_{t-1}$:

```python
if intervention_scheme == 'u_score' and knockout_clamp and len(knockout_set) > 0:
    ko_idx = torch.tensor(sorted(knockout_set), dtype=torch.long, device=device)
    x_current = x_current.clone()
    x_current[:, ko_idx] = 0.0
```

This is idempotent (`test_knockout_clamp_logic`) and prevents the world model
from "re-discovering" a gene that has already been silenced.

### 5.3 Noise carry-over

`recover_epsilon_u` uses the mutilated graph (Lemma 4) so that step $t+1$
starts from $\varepsilon_{t+1}$ in the same post-intervention regulatory
regime that ended step $t$. At the start of an episode, `env.reset()` zeros
both `current_u` and `knockout_set`.

---

## 6. Why this fixes the diagnostic

The diagnostic reported "mean # programs mutilated = 49/64 for single-gene
perturbations". Under the u-score scheme, single-gene perturbations produce
$u = L_{g,:}$, a column of $L$. Because $\sum_g L_{gi} = 1$ across $G \gg d$
genes, the typical entry $L_{gi}$ is $O(1/G)$ unless gene $g$ is one of the
few that loads heavily onto program $i$. So $u$ is sparse by construction — there is no threshold to tune and no degenerate "everyone affected" regime.

The new diagnostic ([`evaluation/dag_diagnostics.py:analyze_u_sparsity`](../evaluation/dag_diagnostics.py))
prints, per test gene:

- mean number of programs with $u_i > 0.5$ (strongly intervened)
- mean $\|u\|_1$ (total decoder-energy fraction allocated by the perturbation)
- $\|u\|_1 / \|u\|_2$ as a "1 = one-hot, $\sqrt{d}$ = uniform" sparsity proxy

These replace the "mean_n_affected (threshold)" number that was always
diagnostically degenerate at $A \approx 0$.

---

## 7. What the u-score scheme does NOT fix

The u-score scheme is a correct do-operator under the linear-SCM + linear-
decoder modelling assumptions. It does **not** solve, on its own, the
upstream problem that the DAG never escaped its $\theta \approx -5$
initialisation. That problem is hyperparameter — `lambda_sparse = 1.0` is too
aggressive at init, and the margin-ranking biological prior at
`lambda_prior = 0.1, margin = 0.1` is too weak to push edges up. The
grid-search driver ([`scripts/run_phase1_grid.py`](../scripts/run_phase1_grid.py))
sweeps these and the winning configuration should be promoted into
`configs/replogle.yaml` and `configs/replogle_u_score.yaml` before running a
full Phase-1 job. The u-score scheme then operates on a DAG that has
actually learned non-trivial $A$.

---

## 8. Implementation pointers

| Concept | Code | Test |
|--------|------|------|
| Loading matrix $L$, $T$ | `CausalVAE.compute_loadings` | `test_loadings_column_sum` |
| Soft do-operator (SCM solve) | `DAGModule.scm_intervene_u` | `test_u_zero_identity`, `test_u_one_pins_target` |
| Gene → latent wrapper with $u$ accumulation | `CausalVAE.intervene_gene_u` | `test_u_monotonic` |
| Noise recovery in mutilated regime | `CausalVAE.recover_epsilon_u` | `test_recover_epsilon_u_inverts` |
| Episode state in RL env | `CausalWorldModelEnv.step`, `.reset` | `test_knockout_clamp_logic` |
| Threshold path preserved | `DAGModule.scm_intervene_multi`, `CausalVAE.intervene_gene` | `test_threshold_path_unchanged` |
| Combinatorial eval dispatch | `evaluation/eval_combinatorial.py:predict_combo` (`scheme=` argument) | — |
| Single-pert eval dispatch | `evaluation/run_perturbation_eval.py:evaluate_single_perts` (`scheme=` argument) | — |
| Sparsity diagnostic | `evaluation/dag_diagnostics.py:analyze_u_sparsity` | — |

Switching schemes at the config level:

```yaml
intervention:
  scheme: u_score        # or 'threshold' for legacy
  u_clip_max: 1.0
  u_accumulate: true
  knockout_clamp: true
```

The flag is read by `rl/env.py`, `evaluation/eval_combinatorial.py`, and
`evaluation/run_perturbation_eval.py`. **Phase-1 perturbation consistency
loss also reads it** — see §5 below for why we reversed an earlier decision
to keep Phase-1 on the threshold path.

---

## 5. Phase-1 training pipeline (post-grid-failure rewrite)

After the 18-cell grid (slurm job `2349145`) failed to move θ off init in any
cell, the architecture itself — not the hyperparameters — had to change.
The empirical finding is consistent with Theorem 3 (§4.3): with $A \approx 0$,
the threshold-mutilation pert loss has zero leading-order gradient on θ.

### 5.1 Phase-1 default switches to `scheme = u_score`

When `intervention.scheme == 'u_score'`, Phase-1 uses
[`perturbation_set_loss_u`](../training/losses.py) instead of
`perturbation_consistency_loss_v2`. The forward pass:

1. Draw a paired cell sentence from
   [`PairedCellSentenceDataset`](../data/cell_sentence_dataset.py) — $S$
   perturbed cells and $S$ control cells from the **same gem_group** (plate).
2. Encode all $2 \cdot B \cdot S$ cells with `batch_id` (gem_group) as additive
   conditioning so plate effects do not leak into $\varepsilon$.
3. Per sentence, compute $dx = \mathrm{mean}(x_{\mathrm{pert}}) - \mathrm{mean}(x_{\mathrm{ctrl}})$.
4. Apply the u-score do-operator (Eq. 3.5 / §3.2) at strength $u = L[\text{gene}_b,:]$
   to produce a *predicted* perturbed set.
5. Compare the predicted set's empirical distribution to the observed set's
   via energy distance (default) or RBF MMD (fallback).

This single change reinstates the gradient signal on θ. Verified in
[`test_pert_set_loss_u_gradient_to_theta`](../tests/test_u_intervention.py):
θ receives nonzero gradient norm at init.

### 5.2 Why paired cell sentences (gem_group-matched)

The legacy [`sample_interventional_batch`](../training/train_vae.py) drew control
cell $i$ and perturbed cell $j$ independently with `np.random.choice`. With 48
gem_groups (plates) in Replogle K562, $i$ and $j$ come from different plates
~95 % of the time. The "predicted perturbation effect" is then contaminated
with a plate shift the model has never been trained to disentangle.

`PairedCellSentenceDataset` indexes the data by `(gene, gem_group)` and only
ever pairs cells within the same block. The dataset shape is
$[B, S, n_{\text{genes}}]$, matching STATE's cell-sentence contract
([state-mcp-local/src/state/tx/models/state_transition.py:769–785](../state-mcp-local/src/state/tx/models/state_transition.py)).

### 5.3 Why energy distance, not per-cell MSE

Within a sentence, the $S$ control cells and $S$ perturbed cells are random
samples from their respective conditional distributions — they are **not**
1-to-1 paired. Per-cell MSE on `(x_hat_ctrl[i], x_pert[i])` therefore matches
arbitrary index alignment, wasting the set structure. Energy distance
(`geomloss.SamplesLoss(loss="energy", blur=0.05)`) compares the empirical
*distributions* directly:

$$
\mathcal{L}_{\text{energy}}(\hat{X}, X) \;=\; 2\,\mathbb{E}\|\hat{x} - x\| \;-\; \mathbb{E}\|\hat{x} - \hat{x}'\| \;-\; \mathbb{E}\|x - x'\|,
$$

which is what STATE uses
([state-mcp-local/src/state/tx/models/state_transition.py:264–279](../state-mcp-local/src/state/tx/models/state_transition.py)).
RBF MMD (with median-heuristic bandwidth) is provided as a fallback in
[`training/losses.py:mmd_rbf`](../training/losses.py) and is also a grid axis
so we can empirically compare.

**Recon loss stays MSE.** Reconstruction is a per-cell identity ($\hat x \approx x$
for *each* cell), not a distribution match — replacing it with a distributional
loss would tell the encoder/decoder "any encoding that matches the overall
distribution is fine," destroying identifiability.

### 5.4 Batch-effect conditioning on `gem_group`

[`CausalEncoder`](../models/encoder.py) accepts an `n_batches` argument; when
positive it allocates `nn.Embedding(n_batches, 256)` whose output is **added**
to the post-MLP hidden activations (mirrors STATE's
`batch_encoder` pattern: additive, not concatenated, so encoder input
dimensions remain stable). Replogle K562 has 48 gem_groups; `n_batches = 48`
in the default config. Without this, the encoder learned to fold plate
effects into $\varepsilon$ and the pert loss was contaminated.

### 5.5 Frozen-θ Phase A through the SCM solve

Old Phase A: `bypass_scm = True` → $z = \varepsilon$ (identity). At DAG
activation in Phase B, the SCM solve suddenly engages — different statistics,
encoder/decoder unprepared — and recon regresses 5–10× ([slurm log 2349145]
shows `recon: 0.45 → 4.3` from epoch 20 → 40 in every cell).

New Phase A under the u-score path: `bypass_scm = False` from epoch 1, but
`model.dag.theta.requires_grad_(False)` during Phase A. The forward path
solves $z = (I - A^\top)^{-1}\varepsilon$ with $A = \sigma(-3) \approx 4.7\\%$
dense (fixed at init). Encoder/decoder learn to operate on this regime
*before* θ starts moving. At `dag_start_epoch` we unfreeze θ and apply the
full DAG loss block. Verified in the smoke run: recon went $4.08 \rightarrow 3.17$
across epochs 1 → 2, not up.

### 5.6 Theta init: $-5 \rightarrow -3$

$\sigma(-5) \approx 0.0067$ is too far from anywhere the pert loss can pull
θ; $\sigma(-3) \approx 0.047$ gives a small but real edge mass with
$\mathrm{spectral\\_penalty}$ active to keep $(I - A^\top)$ well-conditioned.
The diagnostic in [evaluation/dag_diagnostics.py:analyze_dag] now expects
`theta_mean` to drift to ≤ $-2$ after training; if it does not, training
failed (see §5.8 acceptance).

### 5.7 What the grid axes look like now

The grid sweeps **only** axes that we know matter, on the now-fixed
architecture (24 cells):

| Axis | Values |
|---|---|
| `lambda_sparse` (A L1) | $\{0.1,\ 0.01\}$ |
| `(lambda_prior, margin)` | $\{(1.0, 0.5),\ (10.0, 0.5)\}$ |
| `lambda_D_sparse` (D L1) | $\{0.0,\ 0.001,\ 0.01\}$ |
| `pert_loss_kind` | $\{\text{energy},\ \text{mmd\_rbf}\}$ |

All on the **full 186 k K562 dataset** (no subsample). Run via
[`scripts/train_phase1_grid.sbatch`](../scripts/train_phase1_grid.sbatch)
(36-hour wall-time).

### 5.8 Acceptance criteria

A grid cell is considered a "DAG-learning success" iff **all** of:

- `theta_mean ≤ -2.0` (θ moved at least 1.0 off init)
- `edges_gt_0.1 ≥ 100`
- `recon_active_final / recon_phase_a_final ≤ 2.0` (no recon explosion)
- `mean_programs_per_gene ≤ 15` (decoder sparsity actually took effect)
- `ortho_off_diag_mean_abs ≤ 0.05` (orthogonality preserved)

These are computed automatically per cell in
[`scripts/run_phase1_grid.py`](../scripts/run_phase1_grid.py) and dumped to
`results/grid_search/summary.csv`. If zero cells pass, the model architecture
needs further investigation before another sweep.

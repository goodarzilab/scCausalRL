# CausalBioRL

A causal structural equation model (SCM) as a dynamics model for reinforcement learning over single-cell perturbation biology.

The system learns a directed acyclic graph (DAG) over latent gene programs from Perturb-seq data, then trains a PPO-based RL agent to plan multi-step perturbation sequences using the causal world model as its environment.

---

## Architecture Overview

```
Perturb-seq data (Norman / Replogle)
        │
        ▼
┌─────────────────────────────────┐
│        CausalVAE                │
│  ┌──────────────────────────┐   │
│  │ CausalEncoder            │   │
│  │  x → (μ, log σ², ε)     │   │
│  └────────────┬─────────────┘   │
│               │ε (exogenous)    │
│  ┌────────────▼─────────────┐   │
│  │ DAGModule (SCM)          │   │
│  │  z = (I - A^T)⁻¹ ε      │   │
│  │  A sampled via Gumbel-   │   │
│  │  Sigmoid + DAGMA         │   │
│  └────────────┬─────────────┘   │
│               │z (gene programs)│
│  ┌────────────▼─────────────┐   │
│  │ LinearDecoder            │   │
│  │  x̂ = D z + b            │   │
│  │  GRN = D A Dᵀ           │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
        │
        ▼ (trained world model)
┌─────────────────────────────────┐
│  CausalWorldModelEnv            │
│  State: discrete cell cluster   │
│  Action: (latent_idx, value)    │
│  Reward: bootstrapped + shaped  │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PPO Agent                      │
│  Actor: π(a | s, s_target)      │
│  Critic: V(s, s_target)         │
│  Goal-conditioned on PCA        │
│  centroid embeddings            │
└─────────────────────────────────┘
```

---

## Project Structure

```
CausalBioRL/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── preprocess.py             # Shared: HVG selection, normalization, PCA
│   ├── load_norman.py            # Norman 2019: K562 combinatorial Perturb-seq (MTX)
│   ├── load_replogle.py          # Replogle 2022: genome-wide Perturb-seq (H5AD, 22GB)
│   └── load_waddington.py        # Waddington-OT: iPSC reprogramming time course
├── models/
│   ├── dag.py                    # DAGModule: Gumbel-Sigmoid sampling, DAGMA, SCM
│   ├── encoder.py                # CausalEncoder: expression → exogenous noise ε
│   ├── decoder.py                # LinearDecoder: z → expression (D, b); GRN projection
│   └── causal_vae.py             # CausalVAE: full model, interventions, bootstrapping
├── training/
│   ├── losses.py                 # All loss functions: recon, KL, DAGMA, pert, prior
│   ├── train_vae.py              # Phase 1: 3-stage warm-start training
│   └── train_joint.py            # Phase 2: joint VAE + PPO co-training
├── rl/
│   ├── env.py                    # CausalWorldModelEnv: bootstrapped reward, do-operator
│   ├── actor_critic.py           # Goal-conditioned Actor and Critic MLPs
│   ├── ppo.py                    # PPO with GAE, clipped surrogate, entropy bonus
│   └── replay.py                 # RolloutBuffer for PPO
├── evaluation/
│   ├── eval_combinatorial.py     # Aim 3a: predict combos from single perturbations
│   ├── eval_planning.py          # Aim 3b: multi-step trajectory planning
│   └── eval_dag.py               # DAG quality vs CollecTRI ground truth
├── utils/
│   ├── metrics.py                # MSE on DEGs, Pearson, cosine, direction accuracy, MMD
│   └── bio_priors.py             # CollecTRI loading, GRN AUPRC
├── scripts/
│   ├── run_phase1.py             # Entry point: train causal VAE
│   ├── run_phase2.py             # Entry point: joint RL + world model
│   └── run_eval.py               # Entry point: all evaluations
├── test.py                       # Component smoke tests (no data required)
├── requirements.txt
└── checkpoints/                  # Saved model weights (created at runtime)
└── logs/                         # Training curves, evaluation results (created at runtime)
```

---

## Datasets

### Norman et al. 2019 (K562 combinatorial Perturb-seq)
- **Location**: `/large_storage/goodarzilab/shervin/CausalBioRL/Norman/`
- **Format**: Raw MTX (barcodes.tsv.gz + genes.tsv.gz + matrix.mtx.gz + cell_identities.csv.gz)
- **Size**: ~111,668 cells × 33,694 genes
- **Perturbations**: Single-gene and two-gene combinatorial (CRISPRa activation)
- **Usage**: Primary training and combinatorial evaluation dataset. **All combo perturbations are held out** from training — used exclusively as a test set to evaluate whether the model can predict two-gene combinations from single-gene training data.
- **Guide identity format**: `"GENE1_GENE2__GENE1_GENE2"` where `NegCtrl0` = negative control guide

### Replogle et al. 2022 (genome-wide Perturb-seq)
- **Location**: `/large_storage/goodarzilab/shervin/CausalBioRL/Replogle/`
- **Format**: Single H5AD file (22 GB, dense float32)
- **Size**: 643,413 cells × 6,642 genes across 4 cell lines (hepg2, jurkat, k562, rpe1)
- **Perturbations**: ~2,024 single-gene knockdowns + non-targeting controls
- **Usage**: Large-scale world model training (K562 cells subset, downsampled to ~50k for memory)

### Waddington-OT (iPSC reprogramming time course)
- **Location**: `/large_storage/goodarzilab/shervin/CausalBioRL/Waddington_OT/SCP295/`
- **Format**: gzipped TSV (genes × cells), metadata TXT files, GMT cell type annotations
- **Size**: ~15,000 cells (downsampled), mouse genes
- **Timepoints**: Day 0 (fibroblast) → Day 18 (iPSC), continuous trajectory
- **Usage**: **No perturbation labels** — used only for evaluating whether the RL agent can plan trajectories that recapitulate real developmental progression. NOT used for perturbation consistency training.

---

## Key Mathematical Components

### Structural Equation Model (SCM)
The core innovation: a linear SCM over latent gene programs.

**Observational forward pass:**
```
ε = Encoder(x)              # exogenous noise, independent per node
z = (I - A^T)⁻¹ ε          # solve the linear SCM
x̂ = D z + b                # linear decoder
```

**Interventional forward pass (do-calculus):**
```
do(z_i = v):
    A_tilde = A with column i zeroed   # remove incoming edges (mutilation)
    ε_tilde = ε with ε_i = v           # fix the noise at node i
    z_post = (I - A_tilde^T)⁻¹ ε_tilde
    x̂_pert = D z_post + b
```

### DAGMA Acyclicity Constraint
Ensures A is a DAG during learning:
```
h(A) = tr((I + s/d · A◦A)^d) - d = 0  iff A is a DAG
```
Enforced via augmented Lagrangian:
```
L_dag = λ·h(A) + (ρ/2)·h(A)²
λ, ρ updated every 20 epochs when h(A) > 1e-4
```

### Gene-Space GRN Projection
```
G_feat = D @ A @ D^T    (shape: n_genes × n_genes)
G_feat[i,j] > 0  →  gene i regulates gene j
```
This allows direct comparison against CollecTRI ground truth TF-target networks.

### Bootstrapped Reward
The RL environment samples N different DAGs and computes a pessimistic reward:
```
r_n = r_goal_n + γ·φ(s') - φ(s) - λ_cost
r_penalized = mean(r_n) - α · std(r_n)   # pessimism under model uncertainty
r_explore = β · H(P(s' | DAG samples))   # entropy bonus over predicted next states
```

---

## Training

### Phase 1: Warm-start Causal VAE (run first)

```bash
python scripts/run_phase1.py --config configs/default.yaml
```

Three-stage schedule:
| Epochs | Active Components | Key Losses |
|--------|-------------------|------------|
| 1–20   | Encoder + Decoder | Reconstruction only, DAG frozen |
| 21–50  | All (DAG unlocked)| + DAGMA, L1 sparsity, pert consistency. β annealed 0.01→0.5 |
| 51–150 | All               | Full loss. Gumbel τ annealed 1.0→0.25. β→1.0 |

**Combined loss:**
```
L = L_recon + β·L_KL + λ_pert·L_pert + L_dagma + λ_sparse·L_sparse
  + λ_spectral·L_spectral + λ_ortho·L_ortho + λ_prior·L_prior
```

Where `λ_pert = 10.0` is the most important term — it forces the SCM to predict interventional distributions P(x | do(a)), not just observational P(x).

Outputs:
- `checkpoints/phase1_best.pt` — best model by total loss
- `checkpoints/phase1_epoch{N}.pt` — every 10 epochs
- `logs/phase1_training.csv` — full training curves

### Phase 2: Joint World Model + PPO (run after Phase 1)

```bash
python scripts/run_phase2.py \
  --config configs/default.yaml \
  --vae_checkpoint checkpoints/phase1_best.pt
```

**Joint training loop:**
1. **RL phase**: Sample starting cells from control population, random target states. Collect rollouts via PPO in `CausalWorldModelEnv`. Update actor + critic.
2. **World model phase** (every 10 RL iterations): Sample (x_ctrl, gene, x_pert) triples. Compute VAE losses with **policy-relevance upweighting** — perturbations targeting latent programs frequently used by the policy receive up to 3× higher weight, focusing the DAG on policy-critical causal edges.

**Cell state space**: KMeans clustering (K=15) on PCA embeddings of training cells. Cluster centroids serve as state representations for the Actor/Critic networks.

Outputs:
- `checkpoints/phase2_policy_final.pt` — final PPO policy
- `checkpoints/phase2_vae_final.pt` — fine-tuned VAE + cluster centroids
- `logs/phase2_training.csv` — reward curves, VAE losses

---

## Evaluation

```bash
python scripts/run_eval.py \
  --config configs/default.yaml \
  --vae_checkpoint checkpoints/phase1_best.pt \
  --policy_checkpoint checkpoints/phase2_policy_final.pt \
  --evals combinatorial dag planning
```

### Aim 3a: Combinatorial Prediction (`--evals combinatorial`)
- **Task**: Train on single-gene perturbations, predict held-out two-gene combination outcomes
- **Method**: Sequential do-operators: do(gene_A) → do(gene_B) on the same SCM
- **Metrics**: MSE on DEGs, Pearson correlation, cosine similarity, direction accuracy
- **Baselines**: Additive (effect_A + effect_B + ctrl_mean), non-causal ablation (A=0)
- **Commutativity test**: If gene_A and gene_B map to latents with no directed path, predicted(A→B) ≈ predicted(B→A)

### Aim 3b: Multi-step Planning (`--evals planning`)
- **Eval 1**: On Norman combo data — does the policy discover the correct (gene_A, gene_B) combination?
- **Eval 2**: On Waddington-OT — does the trajectory match real developmental progression (day 0→18)?
- **Eval 3**: Greedy baseline comparison — RL should outperform one-step lookahead for multi-step problems
- **Eval 4**: Uncertainty calibration — Spearman correlation between bootstrap std and actual prediction error

### DAG Quality (`--evals dag`)
- Projects latent DAG to gene space: G_feat = D @ A @ D^T
- AUPRC and Precision@K against CollecTRI TF-target reference network
- Top-5 genes per latent program (decoder D columns)
- Heatmap visualizations saved to `logs/dag_eval/`

---

## Configuration (`configs/default.yaml`)

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.latent_dim` | 64 | Number of latent gene programs (DAG nodes) |
| `model.tau_init` | 1.0 | Initial Gumbel-Sigmoid temperature |
| `model.tau_min` | 0.25 | Final temperature after annealing |
| `model.n_dag_samples` | 16 | DAG bootstrap samples for uncertainty |
| `phase1.lambda_pert` | 10.0 | Perturbation consistency loss weight |
| `phase1.lambda_spectral` | 10.0 | Spectral radius stability weight |
| `phase2.n_cell_states` | 15 | Number of discrete cell states (K) |
| `phase2.alpha_pessimism` | 1.0 | Pessimism coefficient in bootstrapped reward |
| `phase2.policy_relevance_upweight` | 3.0 | Upweight for policy-critical perturbations |
| `phase2.max_episode_steps` | 3 | Horizon H for RL episodes |

---

## Running Tests

```bash
python test.py
```

Tests all core components without loading data (random tensors only):
- DAGModule: sampling, DAGMA constraint, SCM forward/intervene, mutilation
- CausalEncoder / LinearDecoder
- CausalVAE: observational forward, interventional forward, bootstrapped, multi-step
- Loss functions: reconstruction, KL, perturbation consistency, combined
- RL: Actor, Critic, RolloutBuffer, PPO batching
- Metrics utilities

---

## Dependencies

```
torch>=2.0
scanpy>=1.9
anndata>=0.10
numpy, scipy, scikit-learn, pandas
pyyaml, matplotlib, seaborn, tqdm
decoupler>=1.4    # for CollecTRI loading
leidenalg         # for Leiden clustering
```

Install:
```bash
pip install -r requirements.txt
```

---

## Implementation Notes

1. **Numerical stability**: `(I - A^T)⁻¹ ε` is computed via `torch.linalg.solve` rather than explicit matrix inversion. The spectral radius penalty (`relu(λ_max(A◦A) - 0.99)`) prevents near-singular systems.

2. **Gumbel-Sigmoid**: Soft (differentiable) during training via `(theta + G1 - G2) / τ` where G1, G2 ~ Gumbel(0,1). Hard binary threshold during evaluation.

3. **Perturbation mapping**: Gene index → latent index via `argmax_i |D[gene, i]|` (the most aligned decoder column). Recomputed periodically as D changes.

4. **Combo holdout**: All Norman combo perturbations are **strictly held out** from Phase 1 and Phase 2 training. The model never sees combo data during training; combos are test-only.

5. **Memory management**: Replogle (22 GB) is subset to K562 cells and downsampled to 50k cells by default. Norman (~100k cells, sparse) fits in RAM easily.

6. **Mixed precision**: The matrix solve stays in float32 for stability. Other operations can use AMP if needed.

7. **Waddington-OT**: Mouse genes — NOT used for perturbation consistency training (no intervention labels). Used only for trajectory evaluation in Aim 3b.

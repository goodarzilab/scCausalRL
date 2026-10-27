# Compositional Perturbation Planning with a Generative Structural Causal World Model

[![Paper](https://img.shields.io/badge/paper-OpenReview-b31b1b.svg)](https://openreview.net/pdf?id=N2ewxvaRBT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Venue](https://img.shields.io/badge/NeurIPS%202026-Workshop-6f42c1.svg)](https://openreview.net/pdf?id=N2ewxvaRBT)

Reference implementation for **"Compositional Perturbation Planning with a Generative
Structural Causal World Model"** (NeurIPS 2026 workshop).

> **Shervin Goudarzi** (UC Berkeley; Arc Institute), **Nuo Liu** (Arc Institute),
> **Hani Goodarzi** (Arc Institute; UC San Francisco)
>
> Paper: https://openreview.net/pdf?id=N2ewxvaRBT

---

## Overview

Directing a stem cell to a target fate is a combinatorial design problem: fates are reached
by over-expressing *sets* of transcription factors (TFs) at high multiplicity of infection.
This repository implements two components that together turn that into an amortized design task.

1. **A generative structural causal world model (WM).** A cell is a latent state `z ∈ ℝ^d`
   over `d = 357` Gene-Ontology biological-process (GO:BP) programs, following a linear
   structural equation model on a learned signed directed graph `A`. A TF perturbation enters
   as a program-space *shift* produced by a learned ESM2 shift-encoder, plus a per-program
   *mutilation gate* on `A`. Any `k`-TF cocktail is the **exact closed-form solve** of the same
   SEM (Eq. 2 of the paper), so composition of unseen combinations is well defined rather than
   an off-distribution query. A low-rank interaction head supplies the non-additive (epistatic)
   part of a combination.

2. **A goal-conditioned GFlowNet planner.** Over the *frozen* WM, a trajectory-balance
   GFlowNet builds a cocktail one TF at a time and is rewarded by the WM's do-operator fate
   match under an **ensemble-pessimistic** reward (`K = 6` WMs, `λ = 0.25`). A goal-conditioned
   `log Z(g)` head lets the policy sample diverse cocktails in proportion to reward rather than
   collapsing to a single argmax.

### Headline results

| Claim | Result |
| --- | --- |
| Held-out combination prediction vs. a fair same-context additive baseline | **+0.18 – +0.23** drift-removed Pearson_DEG (margin grows with cocktail order, up to `k = 8`) |
| Held-out epistasis (interaction) recovery | `r = 0.70`, concentrated at high order (`r = 0.83` at `k = 6–8`) |
| Directed graph `A` is load-bearing for prediction | single-TF Pearson collapses `0.54 → −0.16` under a shuffled `A` (`p < 0.001`, 1000-shuffle null); reversing edges is as damaging as shuffling |
| Planner vs. MPC beam search on Briggs recovery | parity on recall@20 (**0.58** vs 0.51), **wins** recall@5 / precision@5, at **≈36× lower** inference cost |
| Planner diversity at fixed quality | **~160** distinct high-quality cocktails vs **1** for beam search |
| Zero-shot HSPC fate | surfaces **15/54** arrayed answer-key drivers (incl. LYL1, KLF1, KLF2, IKZF1) vs 3–4 for baselines |

See the paper for the full protocol, baselines, statistics, and honest negatives.

---

## Repository layout

This repository tracks **only** the source, configs, figure scripts, and manuscript behind the
paper. Large artifacts (checkpoints, datasets, result dumps), the 19 precursor experiments, and
abandoned model arms are intentionally not tracked — see [`.gitignore`](.gitignore).

```
experiments/exp20_load_bearing/     # the structural causal world model
├── models/                         #   causal VAE, latent DAG, SENA decoder, ESM2 shift encoder
├── data/                           #   Joung / Briggs / Cao loaders, cell-sentence dataset
├── training/                       #   train_vae.py (world-model objective), losses.py
├── utils/                          #   GO:BP masks, ESM embeddings, metrics, bio priors
├── evaluation/                     #   held-out prediction, epistasis, edge-direction ablation
├── configs/ablations/joungbriggs/  #   the reported Joung+Briggs training recipes (see below)
└── scripts/run_phase1.py           #   world-model training entry point

experiments_rl/                     # the goal-conditioned GFlowNet planner
├── analysis/
│   ├── briggs_multigoal_rl.py      #   planner driver (flagship entry point)
│   ├── gflownet_hstep.py           #   trajectory-balance objective
│   ├── fate_rl.py                  #   MPC beam-search baseline + residual scoring
│   ├── hstep_mdp.py                #   candidate masking / MDP helpers
│   └── retriever.py                #   goal-conditioned candidate retriever
├── core/                           #   frozen-WM operator, ensemble, env, reward, scoring
├── rl2_set_ppo/                    #   shared autoregressive SetPolicy (+ set-PPO baseline)
└── rl0_mpc/                        #   MPC planner utilities

current_figures/                    # paper figure scripts (fig1–fig9) + precomputed metric JSON
paper_v2_current/paperv3/           # LaTeX source, references, and compiled main.pdf
```

**Not tracked (kept locally):** precursor experiments `exp1`–`exp19`; the abandoned sequential-PPO
(`rl1_seq_ppo`) and MPC-distillation (`rl3_mpc_distill`) arms; `*.pt`/`*.npz`/`*.h5ad`/`*.csv` and
other binaries; `wandb/`, `logs/`, `results/`, `checkpoints/`, `_archive/`; and internal working notes.

---

## Installation

```bash
git clone https://github.com/goodarzilab/scCausalRL.git
cd scCausalRL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.10 and PyTorch ≥ 2.0 are recommended. The world model trains on CPU high-memory
nodes (~340 GB RAM, see App. C of the paper) or a GPU; the planner trains on a single GPU.

## Data

The model is trained on three public resources. Download them and point the loaders in
`experiments/exp20_load_bearing/data/` at your local copies (data files are not shipped):

| Dataset | Role | Accession / reference |
| --- | --- | --- |
| Joung TF atlas | single-TF over-expression (hESC, day 7) | Joung et al., 2023 (Cell) |
| Briggs combinatorial screen | high-MOI TF combinations → HSPC fate | GEO **GSE328457** (Briggs et al., 2026) |
| Cao fetal cell atlas | target-fate marker signatures | Cao et al., 2020 (Science) |

Per-TF descriptors are 100-dim PCA-reduced, z-scored ESM2 protein-language-model embeddings.

## Reproducing the results

**1. Train the world model** (two stages: a singles base, then a combination finetune that
oversamples singles 8×). Run from inside `experiments/exp20_load_bearing/`:

```bash
cd experiments/exp20_load_bearing
# stage 1: singles base
python scripts/run_phase1.py --config configs/ablations/joungbriggs/jb_singles_base_co8.yaml --device cuda
# stage 2a: deployed production checkpoint (all data except the 222 answer-key combos held out)
python scripts/run_phase1.py --config configs/ablations/joungbriggs/jb_prod_wm.yaml --device cuda
# stage 2b: genuine held-out sibling for the Fig. 2 generalization numbers (0.85/0.8 split)
python scripts/run_phase1.py --config configs/ablations/joungbriggs/jb_ft_replay_co8.yaml --device cuda
```

Seed variants (`..._seed2/3/4`, `..._s2/s3`) and the leakage-ladder rungs
(`jb_ft_replay_co8_rung2_leaveTFout.yaml`, `..._rung3_leaveP1out.yaml`) are in the same folder.

**2. Train the planner** against the frozen WM ensemble. Run from inside `experiments_rl/`:

```bash
cd experiments_rl
python analysis/briggs_multigoal_rl.py \
    --gflownet --gflownet_logz cond --gflownet_pess 0.25 \
    --kmax 8 --fold 0 --seed 0
```

This is the deployed flagship (goal-conditioned GFlowNet, MLP head, ensemble-pessimism `λ = 0.25`).
Full hyperparameters, the shared-`log Z` ablation, and the MPC / additive / random baselines are
documented in App. C–E of the paper and in the argument help (`--help`).

**3. Regenerate the figures.** `current_figures/fig*.py` read the precomputed metric JSON in the
same directory and emit `figN.pdf` / `figN.png`:

```bash
cd current_figures
python fig2.py   # held-out prediction vs additive
python fig5.py   # planner recovery / diversity / zero-shot HSPC
```

## Manuscript

The LaTeX source and compiled PDF are under [`paper_v2_current/paperv3/`](paper_v2_current/paperv3/)
(`main.tex`, `appendix_expanded.tex`, `references.bib`, `main.pdf`). LaTeX build artifacts are
not tracked.

## Citation

```bibtex
@inproceedings{goudarzi2026compositional,
  title     = {Compositional Perturbation Planning with a Generative Structural Causal World Model},
  author    = {Goudarzi, Shervin and Liu, Nuo and Goodarzi, Hani},
  booktitle = {NeurIPS 2026 Workshop},
  year      = {2026},
  url       = {https://openreview.net/pdf?id=N2ewxvaRBT}
}
```

## License

Released under the [MIT License](LICENSE). The public datasets used here remain under their
original licenses and terms of use.

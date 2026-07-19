# Relevant literature (2024–2026) for scCausalRL

Recovered from the deep-research sweep (25→33 sources, 5 angles, 116 claims extracted, adversarially verified). Each entry: citation · one-line finding · how it bears on our design. ★ = most actionable.

---

## 1. Causal-representation / latent-SCM VAEs (is the latent DAG load-bearing?)

- ★ **Discrepancy-VAE — "Identifiability Guarantees for Causal Disentanglement from Soft Interventions"** (Zhang, Squires, Uhler et al., NeurIPS 2023; arXiv 2307.06250 / OpenReview 3Fgylj4uqL). Proves a latent SCM under *soft* interventions is identifiable up to a CD-equivalence class; deep **linear** SCM (upper-triangular A) → mixing fn; predicts held-out Norman singles near-perfectly (R²≈0.99 all / 0.95 DE) and generalizes to 112 unseen doubles; collapsed latents recover literature edges (DUSP9→MAPK1, DUSP9→ETS2). **This is the direct ancestor of our architecture and the existence proof that a load-bearing latent A is achievable** — but note their identifiability needs a *stronger* "linear interventional faithfulness" assumption and a full-rank polynomial decoder. Their A is used (GRN read-out is interpretable), unlike ours currently.

- ★ **SENA-discrepancy-VAE** (arXiv 2506.22641, 2025). Imposes pathway/gene-program structure **during reconstruction** via a masked SENA-δ encoder (expression → biological-process activity → latent causal factors), not as a decorative read-out. Matches non-interpretable discrepancy-VAE on unseen-combo prediction while latent factors map to named processes. **Directly relevant to our "make A load-bearing" goal: structure enforced in the encoder/reconstruction path rather than only G=D·A·Dᵀ.** Identifiability inherited from discrepancy-VAE (perm+scale only).

- **SAMS-VAE** (Bereket & Karaletsos, arXiv 2311.02794). Perturbations = *sparse additive latent shifts*, each masked to a small perturbation-specific subspace, masks compose additively for combinations; beats CPA/SVAE+ in/out-of-dist incl. low-data combos; introduces an ATE + posterior-predictive-check eval framework (not raw recon/corr). **Validates our shift-encoder additive-composition design and our move away from raw-Pearson eval.** Caveat: PerturBench (below) found removing SAMS-VAE's sparsity constraint *helped* — argues against over-sparsifying.

- **GPO-VAE** (arXiv 2501.18973, 2025). Optimizes the latent perturbation-effect parameters to align with GRN structure; claims SOTA response prediction *and* "more meaningful" GRNs. **Same thesis as ours (load-bearing GRN-aligned latent), worth comparing against** — but abstract gives no drift-removed metric, so treat its prediction claim skeptically (see §3).

- **PerturbedVAE / "Perturbation Suppression Hypothesis"** (biorxiv 2025.10.20.683304, 2025). Formalizes that perturbation-invariant background dominates and perturbation signal is sparse → separate them in latent space; gives identifiability (responsive latents up to perm/scale, invariant up to linear block; needs 2·d_ν environments); a **contrastive invariance loss** raises MCC 0.81→0.86, R² 0.93→0.95, demonstrating the latent structure is load-bearing not decorative. **A concrete recipe (contrastive alignment + environment count) for making latent causal structure measurably load-bearing — most transferable to our problem.**

- **Biolord** (Piran, Cohen, Hoshen, Nitzan, Nat. Biotech. 42(11):1678, 2024). Disentangles known attributes (encoded per-attribute) + one unknown-attribute code; predicts response to *unseen* drugs/genetic perts by shifting cells across states; beats CPA. **The "shift cells across attribute states" move = our shift-encoder do-operator; a current baseline we should cite/compare.**

## 2. World models + RL / planning / active design for cell-fate

- ★ **DiRL** ("RL enables single-cell foundation models to learn cellular differentiation", biorxiv 2025.12.09.693267, 2025). **The closest paper to our entire RL-over-world-model thesis.** Goal-conditioned actor–critic; actor maps (current state, target cell type)→genetic-perturbation actions; critic scores progress; SC foundation model is the predictive environment. Beats random-perturbation 69% win-rate on iPSC organoids, improves with planning horizon, recovers Wnt/Hedgehog sequential regulators, critic value ≈ pseudotime. **Validates set/seq-PPO over a frozen world model for fate design; their multi-step framing is an argument for our seq-PPO over single-shot MPC. Cite as concurrent prior art and benchmark target.**

- ★ **Wytock & Motter** (PNAS 121, 2312942121, 2024). Pre-trained transfer-learning model of transcriptional network *dynamics*; given initial+target state, computes the gene-perturbation combination to drive the transition; reproduces known reprogramming protocols at **AUROC 0.91**; works by **linearly superposing** single-perturbation responses; #perturbations needed grows as developmental distance grows (fewer to progress than regress). **A non-RL control-theoretic competitor to our planner, with a clean ground-truth-protocol benchmark (AUROC 0.91) we can target, and independent support for additive composition of shifts.**

- **CEFCON** (PMC10864347). GNN-inferred lineage GRN + network control theory (Min Feedback Vertex Set / Min Dominating Set) to nominate fate drivers; recovers NANOG/SOX2/POU5F1; +64% AUPRC over DeepSEM. **Control-theoretic complement to our RL; relevant master-TF recovery validation** — but depends on a *prior* network as input (the prior-memorization risk we're avoiding).

- **IterPert** (RECOMB 2024; biorxiv 2023.12.12.571389). Active learning for Perturb-seq design: multi-modal public priors → reach next-best accuracy with ~⅓ the perturbations. *Verification flagged this as NOT a direct match to our problem* (it selects which perts to assay, not which TF set reaches a fate) — relevant only to a future active-design extension.

- **NewLimit 2025 Year-in-Review** (blog). Industry: AI-designed TF reprogramming payloads. Context/motivation, not a method to compare.

- **"Planning with Latent Dynamics Models" / reward-free offline** (arXiv 2502.14819). Generic RL-over-learned-latent-dynamics methodology — relevant to how we plan over the frozen world model.

## 3. Perturbation-prediction eval pitfalls (our drift/discrimination framing)

- ★ **Ahlmann-Eltze, Huber, Anders** (Nature Methods 2025; preprint Oct 2024). DL/foundation models (scGPT, scFoundation, GEARS) **do not beat a simple linear baseline** on unseen singles; an **additive** model beats DL on doubles; a **no-change** model beats everything on synergistic doubles (FDR/ROC/recall); FM embeddings retain value *separated from the transformer*. **The canonical citation for why we judge on drift-removed Δ + discrimination, not raw Pearson, and why we keep additive/no-change/context baselines on every plot.**

- ★ **"Drift" critique** (Nature s41592-025-02772-6, 2025). High measured performance is driven by **shared reference shifts (drift)**, not biology (shown in silico + 2 real datasets); proposes DEG-aware **weighted MSE / weighted delta-R²** under which the mean baseline collapses to null; WMSE-as-loss reduces mode collapse. **This *is* our drift baseline, named and formalized — cite it as the basis of our drift-removed scoring, and consider WMSE as a phase-1 loss.**

- ★ **PerturBench** (arXiv 2408.10609, 2024). Latent-Additive / Decoder-Only baselines are competitive-to-best (Decoder-Only beat all complex models on the largest dataset); a perturbation-blind Decoder-Only gets near-SOTA cosine-LogFC yet **~0.47 (random) on a rank/discrimination metric**; Norman doubles mostly linear; removing CPA's adversary and SAMS-VAE's sparsity *helped*. **Direct support for (a) our discrimination/retrieval metric over fit metrics, and (b) not over-constraining the model with sparsity/adversaries.**

- **PerturbationDiscriminationScore distance/scaling study** (arXiv 2511.16954, 2025). Same predictions score ~0.5 under L1/L2 but ~0.78–0.89 under **cosine**; L1/L2 are scale-sensitive and norm-matching doesn't rescue them; recommends cosine-based PDS. Current benchmarks (Systema=L2, Virtual Cell Challenge=L1) are miscalibrated. **Validates our use of `discrimination_cos` specifically; warns against Euclidean retrieval metrics.**

- **Systematic Comparison of Perturbation-Response Models** (biorxiv 2024.12.23.630036, Dec 2024). 12 methods + 3 baselines × 25 datasets × 24 metrics; FMs predict the **population average** not cell-to-cell heterogeneity; **scGPT degenerates with more epochs** (over-compresses the response). Broad benchmark to position against.

- **"What Makes a Representation Good for Single-Cell Perturbation Prediction?"** (arXiv 2605.19343, 2026). FM embeddings (scFoundation, UCE, Geneformer, STATE) show **weaker linear decodability than plain PCA**; additive linear competitive on doubles. Another nail in the "big pretrained embeddings aren't buying perturbation signal" coffin.

## 4. TF-combination / master-regulator prediction (post-Joung)

- **REPROcode** (Cell Systems 2025, S2405-4712(25)00290-X). Wet-lab pooled combinatorial TF screening (408 immune-specific TFs; sets of 9/22/42/48) → optimal stoichiometry, TF hierarchy of fate-branch points, induced cDC1/NK. **A combinatorial-TF search-space + ground-truth analog of Joung for the immune lineage; potential external validation target.**

- **TFcomb / scDirect** (claims 13–16). Identifies TF *combinations* (not just ranks singles) from scRNA+scATAC via CellOracle GRN + GAT; recovers POU5F1/SOX2 (8th/780) and Yamanaka (50th/10,626); beats DE in 16/18 states. **A computational TF-cocktail competitor to our planner.**

- **DiReG** (Lauber & List, npj Syst. Biol. 2026, s41540-026-00652-z). Meta-platform integrating multiple reprogramming-factor tools + RAG. Key claim: **existing tools share critical limitations, fail to capture regulatory *dynamics*, and there are no robust benchmarking standards** — explicitly motivates a dynamics/world-model approach like ours.

- **"Redefining cellular reprogramming with advanced genomic technologies"** (Nature Reviews Genetics 2025, s41576-025-00899-y). Review: reprogrammed cells still immature/low-fidelity/heterogeneous; computational + single-cell + molecular recording enable mechanism-guided protocol design. Framing/motivation citation.

## 5. GRN-inference benchmarking & prior memorization (retiring CollecTRI from the loss)

- ★ **geneRNIB** (Nourisa et al., biorxiv 2025.02.25.640181, 2025). Living GRN benchmark (Replogle/Nakatake/Adamson/Norman + OPSCA). **Simple correlation / GRNBoost2 beat complex pipelines incl. prior-knowledge/pretrained ones across all metrics**, and **methods that integrate context-independent priors are *hindered* in producing context-specific networks.** **This is the strongest external justification for our decision to retire CollecTRI from the training loss (eval-only) — the prior actively hurts context-specificity.**

- ★ **InfoSEM** (arXiv 2503.04483, 2025). Supervised GRN methods **learn gene-specific biases (ground-truth class imbalance) rather than true mechanism**; their benchmark exposes this prior-memorization; InfoSEM instead uses *unsupervised* textual-gene-embedding priors (+38.5% over baselines). **Direct evidence for our "G=D·A·Dᵀ memorizes CollecTRI" finding and an alternative prior strategy (text embeddings, label-free) if we ever re-add a prior.**

- **CausalBench challenge** (arXiv 2308.15395). GRN inference from large-scale *interventional* single-cell data; winning solutions beat baselines (counter to "nothing beats simple"). Argues interventional data (which we have via the shift do-operator) is where learned GRNs *can* win.

- **"Leveraging prior knowledge to infer GRNs from scRNA-seq"** (PMC11876610) and **Karamveer & Uzun, benchmarking review** (Sage 2024). Supporting surveys on prior use and GRN benchmarking design.

---

## Bottom line — what to act on

1. **Eval framing is fully vindicated.** The drift critique (Nat Methods s41592-025-02772-6), Ahlmann-Eltze (Nat Methods 2025), and PerturBench independently say: judge on drift-removed Δ + a **cosine** discrimination/rank metric, keep additive/no-change baselines. Our scoring already does this — cite these three and consider adding WMSE as a phase-1 loss.
2. **Retiring CollecTRI from the loss is the right call** — geneRNIB shows context-independent priors *hurt* context-specificity and InfoSEM shows prior-supervised GRNs memorize label bias. Our prior-off arms are well-motivated.
3. **Load-bearing A is achievable and there's a recipe.** Discrepancy-VAE proves it under soft interventions; SENA-discrepancy-VAE and PerturbedVAE show *how* — enforce structure in the encoder/reconstruction (masked program encoder, contrastive invariance) rather than only reading out G=D·A·Dᵀ. Strongly consider a SENA-style masked encoder or PerturbedVAE contrastive loss in the next ablation round.
4. **We have a direct concurrent competitor: DiRL** (RL over a SC world model for differentiation) and a control-theoretic one (Wytock & Motter, AUROC 0.91 vs known protocols). Both are benchmark/positioning targets; Wytock & Motter's protocol-recovery AUROC is a concrete number to beat. DiRL's horizon-helps result argues for prioritizing seq-PPO.
5. **Don't over-sparsify.** PerturBench found removing SAMS-VAE sparsity / CPA adversary *improved* results — a caution for our sparsity sweep (consistent with our finding that product-mode sparsity was degenerate).

# Developmental annotation of the Joung TF atlas (Cao 2019 shift-alignment)

Annotate each overexpressed TF in the Joung TF atlas (hESC) with **which mouse
developmental cell-type direction it drives cells toward**, by aligning the TF's induced
expression *shift* to cell-type signatures from the Cao et al. 2019 mouse organogenesis
atlas (MOCA).

> Cao, Spielmann, Qiu, …, Shendure & Trapnell. *The single-cell transcriptional
> landscape of mammalian organogenesis.* Nature 2019. (~2M cells, E9.5–E13.5.)

## Method

1. **Induced shift per TF.** `Δ_TF = mean(OE cells) − mean(control)` — the *shift*, NOT the
   absolute profile (all cells are hESC-derived, so absolute profiles share one huge
   baseline that hides TF identity).
2. **Cell-type signatures.** `D_c = mean(type c) − mean over cell types` (deviation from
   the average developmental cell), `D ∈ R^{G×C}`.
3. **Score.** `cosine(Δ_TF, D_c)` per cell type (primary, robust); an NNLS mixture is also
   written. `specificity = peak cosine − mean cosine` = a per-TF **confidence score**.

### Three corrections that make it work (see the diagnostic report)

Naïve NNLS on absolute profiles is degenerate (every TF → the same answer). The fixes:

1. **Orthology via MGI 1:1 homology** (`mgi_homologene.rpt`), not symbol-uppercasing →
   ~15 k mapped genes (≈3× the ~5 k from uppercasing).
2. **Shared OE-axis removal.** Every lentiviral-OE population shares a dominant
   selection/stress/proliferation axis (TF-shift pairwise corr 0.9+). We centre per TF and
   strip the **top-4 SVD components** (a simple perturbation factor model). Without this,
   *every* estimator (OLS/ridge/NNLS/cosine) collapses to one answer — verified.
3. **Early-progenitor reference.** hESC + ~7 d OE reaches early progenitor states, not
   E9.5–13.5 terminal types, so we restrict to **~20 progenitor/early `Main_cell_type`s**
   (`EARLY_MAIN_TYPES`). "Didn't map to Myocytes/Hepatocytes" is then expected, not a bug.

Plus: report only TFs with **≥50 cells** (small-n pseudobulks are noise; e.g. NEUROG2 n=8
is dropped). 1,084 of 1,595 TFs survive.

### What works / how to use it

High-`specificity` calls are trustworthy master regulators — validated examples:
`MSGN1→mesoderm`, `SNAI1/SNAI2→mesenchyme (EMT)`, `GRHL3→epithelial`, `ATOH7/SCRT2→neural`,
`LMX1B→isthmic organizer`. Low-specificity calls (near the ~0.05 median) are *not*
confidently annotatable — treat `specificity` as a confidence threshold. Muscle/cardiac/
endoderm masters (MYOD1, GATA4, HNF4A) score low **by design** (their fates aren't in the
early-progenitor whitelist and aren't reached in 7 days).

### Estimator note (NNLS vs OLS vs cosine)

We tested OLS / ridge / NNLS / cosine head-to-head: on the raw shift they are
**indistinguishable** (all collapse on the shared OE-axis); after axis removal cosine is
the most stable (the 38 cell-type signatures are collinear, so unconstrained OLS is
ill-conditioned). The estimator is a red herring — the corrections above are what matter.

## Files

- `build_mouse_reference.R` — **(needs R; the only R step)** reads MOCA
  `gene_count_cleaned.RDS` (26 183 genes × 1.33 M cells sparse `dgCMatrix`) +
  `cell_annotate.csv` + `gene_annotate.csv`, drops doublets/unlabeled cells,
  size-factor log-normalises, and writes per-cell-type pseudobulks
  `mouse_ref_<col>.{RDS,csv.gz,_celltypes.csv}` (genes × cell types).
- `nnls_annotate.py` — pseudobulks the Joung singles (induced shifts, n≥50), maps genes
  via MGI 1:1 orthology, removes the top-4 shared OE PCs, restricts to early types, scores
  cosine + NNLS, writes `joung_devannot_<ref>_{annotation,cos,nnls}.csv`. Flags:
  `--remove_pcs` (default 4), `--all_types` (disable early-type whitelist).
- `mgi_homologene.rpt` — MGI mouse↔human homology (18.8 k 1:1 orthologs).
- `../../scripts/mouse_ref.sbatch` — runs both R tiers on a high-mem node (annotation step
  then run with the upgraded `nnls_annotate.py`).

## Cell-type tiers (named, interpretable)

MOCA's named annotation tiers (the unnamed numeric `Sub_cluster` ~600 tier is skipped —
no biological names):

| column | # types | coverage | use |
|---|---|---|---|
| `Main_cell_type` | 38 | 0.98 | robust primary reference |
| `Sub_trajectory_name` | ~56 | 0.63 | finer-grained named trajectories |

## Data

`/large_storage/goodarzilab/shervin/CausalBioRL/Yue_Mouse_Developmental/`
`gene_count_cleaned.RDS` was provided; `cell_annotate.csv` (827 MB) + `gene_annotate.csv`
were downloaded from the matching MOCA public directory (Shendure lab) to supply the
per-cell labels and Ensembl→symbol map (the count matrix alone has neither).

## Caveats

- **Cross-species + developmental stage**: human hESC + 7 d OE vs mouse E9.5–13.5 — this
  is a fate *direction*, not a quantitative cell-type proportion. Use `specificity` as a
  confidence gate (high = trustworthy; near the ~0.05 median = not annotatable).
- **Restricted scope**: only early/progenitor lineages are scored; terminal-fate masters
  (muscle/cardiac/endoderm) score low by design.

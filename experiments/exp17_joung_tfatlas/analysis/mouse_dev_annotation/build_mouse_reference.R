#!/usr/bin/env Rscript
# Build cell-type reference pseudobulks from the Cao et al. 2019 mouse organogenesis
# atlas (MOCA, gene_count_cleaned.RDS: 26183 genes x 1.33M cells, sparse dgCMatrix).
#
# This is the ONLY step that needs R (the count matrix is an S4 dgCMatrix that
# pyreadr/scanpy can't read). It emits a compact, Python-friendly reference:
#   X  ∈ R^{G x C}  mean log-normalised expression per developmental cell type
#   + the cell-type names (C, named) and mouse gene symbols (G).
# Downstream NNLS deconvolution of the Joung TF atlas happens in nnls_annotate.py.
#
# The interpretable named tiers in MOCA are Main_cell_type (38) and
# Sub_trajectory_name (~56); the unnamed numeric Sub_cluster (~600) is skipped.
#
# Run:  ~/miniconda/envs/renv/bin/Rscript build_mouse_reference.R <celltype_col>
#   e.g.  Main_cell_type   (38 named developmental cell types, 98% coverage)
#         Sub_trajectory_name  (~56 named trajectories, 63% coverage)
suppressMessages({library(Matrix)})

DIR  <- "/large_storage/goodarzilab/shervin/CausalBioRL/Yue_Mouse_Developmental"
ODIR <- "/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/mouse_dev_annotation"
args <- commandArgs(trailingOnly = TRUE)
ctype_col <- if (length(args) >= 1) args[1] else "Main_cell_type"
OUT <- file.path(ODIR, paste0("mouse_ref_", ctype_col))

cat("[1/5] reading counts (1.9GB, ~1min)...\n"); flush.console()
M <- readRDS(file.path(DIR, "gene_count_cleaned.RDS"))      # genes x cells, dgCMatrix
cat("   counts:", paste(dim(M), collapse=" x "), "\n")

cat("[2/5] reading annotations...\n"); flush.console()
cell <- read.csv(file.path(DIR, "cell_annotate.csv"), stringsAsFactors = FALSE)
gene <- read.csv(file.path(DIR, "gene_annotate.csv"), stringsAsFactors = FALSE)
cat("   cell_annotate:", nrow(cell), "rows; gene_annotate:", nrow(gene), "rows\n")

# --- align cell metadata to the count-matrix columns (key = 'sample') -------
bc <- colnames(M)
rownames(cell) <- cell$sample
stopifnot(mean(bc %in% rownames(cell)) > 0.9)
cell <- cell[bc, ]

# --- filter doublets + unlabeled cells --------------------------------------
lab <- as.character(cell[[ctype_col]])
is_dbl <- cell$detected_doublet %in% c("TRUE", TRUE, "True")
keep <- !is.na(lab) & lab != "" & lab != "NA" & !is_dbl
cat("[3/5] cells kept:", sum(keep), "of", length(keep),
    "(dropped", sum(is_dbl), "doublets +", sum(is.na(lab)|lab==""), "unlabeled)\n")
M <- M[, keep]; lab <- lab[keep]
types <- sort(unique(lab))
cat("   cell-type column:", ctype_col, "->", length(types), "named types\n")

# --- size-factor log-normalise (median library size → log1p), sparse --------
cat("[4/5] size-factor log-normalising + pseudobulking...\n"); flush.console()
sf <- Matrix::colSums(M); sf[sf == 0] <- 1
M <- M %*% Diagonal(x = median(sf) / sf)
M@x <- log1p(M@x)

# --- mean expression per cell type (X = G x C) via normalised indicator -----
Ind <- sparseMatrix(i = seq_along(lab), j = match(lab, types),
                    x = 1, dims = c(length(lab), length(types)))
Ind <- Ind %*% Diagonal(x = 1 / pmax(1, Matrix::colSums(Ind)))
X <- as.matrix(M %*% Ind)                          # G x C dense
colnames(X) <- types
sym <- gene$gene_short_name[match(rownames(M), gene$gene_id)]
sym[is.na(sym)] <- rownames(M)[is.na(sym)]
rownames(X) <- sym

cat("[5/5] writing reference (", paste(dim(X), collapse=" x "), ")...\n"); flush.console()
saveRDS(X, paste0(OUT, ".RDS"))
write.csv(data.frame(cell_type = types, n_cells = as.integer(table(lab)[types])),
          paste0(OUT, "_celltypes.csv"), row.names = FALSE)
z <- gzfile(paste0(OUT, ".csv.gz"), "w")
write.csv(round(X, 5), z, row.names = TRUE); close(z)
cat("DONE ->", OUT, ".{RDS,csv.gz,_celltypes.csv}\n")

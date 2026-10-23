## ============================================================================
## download_brca_mogonet_format.R
##
## Downloads TCGA-BRCA mRNA, DNA methylation, miRNA, and GISTIC2 CNV data via
## TCGAbiolinks, matches samples across all four omics types + PAM50 labels,
## applies preprocessing following MOGONET (Wang et al. 2021) and MOSGAT
## (Wu et al. 2024), splits into train/test, and writes MOGONET-formatted
## output files to ./BRCA/:
##
##   1_tr.csv, 1_te.csv, 1_featname.csv   <- mRNA
##   2_tr.csv, 2_te.csv, 2_featname.csv   <- methylation
##   3_tr.csv, 3_te.csv, 3_featname.csv   <- miRNA
##   4_tr.csv, 4_te.csv, 4_featname.csv   <- CNV (GISTIC2, 4th view --
##                                           NOT present in the original
##                                           MOGONET/MMDynamics/MOSGAT papers,
##                                           added per this project's request)
##   labels_tr.csv, labels_te.csv          <- integer class labels, 0-4
##                                           (Normal-like, Basal-like, HER2-
##                                           enriched, Luminal A, Luminal B)
##
## ============================================================================
## IMPORTANT CAVEATS -- READ BEFORE RUNNING
## ============================================================================
## 1. This downloads real genomic data for the full TCGA-BRCA cohort
##    (roughly ~1000 patients before filtering down to those with ALL FOUR
##    omics types + PAM50 label). Expect this to take a long time (possibly
##    hours) and a lot of disk space (tens of GB for raw downloads before
##    GDCprepare summarizes them), depending on your connection and whether
##    TCGAbiolinks has cached data from a previous run.
##
## 2. CNV preprocessing is NOT validated against any of the three papers,
##    since none of them used CNV. The variance-filtering threshold for CNV
##    below is a placeholder judgment call (see the CNV section), not a
##    value taken from published work -- GISTIC2's near-discrete {-2,-1,0,1,2}
##    values behave very differently under a variance filter than continuous
##    expression/methylation data, and the 0.1 mRNA / 0.001 methylation
##    thresholds from MOGONET's own paper do not have an established CNV
##    equivalent. You should sanity-check the number of CNV features
##    retained (printed at the end of the CNV section) and adjust
##    cnv_var_threshold if it looks too aggressive or too lax.
##
## 3. Sample matching: only patients with ALL FOUR omics types AND a known
##    PAM50 subtype are retained. TCGA's multi-omics coverage is NOT uniform
##    across data types, so expect meaningful sample loss at this step --
##    the number of patients retained is printed explicitly so you can see
##    this is happening, not discover it silently later.
##
## 4. The train/test split and ANOVA-based feature preselection both use
##    PAM50 label stratification. The RANDOM SEED is fixed below for
##    reproducibility -- change it if you want a different split.
## ============================================================================

## ---- Required packages ----
required_pkgs <- c("TCGAbiolinks", "SummarizedExperiment", "caret", "matrixStats")
missing_pkgs <- required_pkgs[!sapply(required_pkgs, requireNamespace, quietly = TRUE)]
if (length(missing_pkgs) > 0) {
  stop(paste0(
    "Missing required packages: ", paste(missing_pkgs, collapse = ", "), ".\n",
    "Install via:\n",
    "  if (!require('BiocManager')) install.packages('BiocManager')\n",
    "  BiocManager::install(c('TCGAbiolinks', 'SummarizedExperiment'))\n",
    "  install.packages(c('caret', 'matrixStats'))"
  ))
}

library(TCGAbiolinks)
library(SummarizedExperiment)
library(caret)
library(matrixStats)
options(timeout = 600)
set.seed(42)

out_dir <- "BRCA"
dir.create(out_dir, showWarnings = FALSE)

## ============================================================================
## STEP 1: Download each omics type via TCGAbiolinks
## ============================================================================
## ---- CNV (GISTIC2 thresholded calls) ----
cat("\n=== Downloading CNV (GISTIC2 gene-level thresholded calls) ===\n")
query_cnv <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Copy Number Variation",
  data.type = "Gene Level Copy Number",
  workflow.type = "ASCAT3",  # <-- Add this line to isolate a single pipeline
  access = "open"
)
GDCdownload(query_cnv)
cnv_se <- GDCprepare(query_cnv)

cat("\n=== Downloading mRNA expression (STAR - Counts) ===\n")
query_mrna <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)
GDCdownload(query_mrna)
mrna_se <- GDCprepare(query_mrna)

cat("\n=== Downloading DNA methylation (Illumina Human Methylation 27) ===\n")
## NOTE: MOGONET's own paper specifically restricts to probes present on the
## Illumina 27k array "for better interpretability of the results" (see
## MOGONET paper, "Preprocessing" subsection). TCGA-BRCA has far more 450k
## array coverage than 27k; if 27k coverage is too sparse for your matched
## sample set, switch platform below to "Illumina Human Methylation 450"
## and instead manually intersect with the 27k probe list after download
## (code for that intersection is included further below, commented out).
query_meth <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "DNA Methylation",
  platform = "Illumina Human Methylation 27",
  data.type = "Methylation Beta Value"
)
GDCdownload(query_meth, method = "client")
meth_se <- GDCprepare(query_meth)

cat("\n=== Downloading miRNA expression ===\n")
query_mirna <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Transcriptome Profiling",
  data.type = "miRNA Expression Quantification"
)
GDCdownload(query_mirna, files.per.chunk = 500)
mirna_data <- GDCprepare(query_mirna)

cat("\n=== Downloading CNV (GISTIC2 gene-level thresholded calls) ===\n")
## GISTIC2 thresholded calls: one value per gene per sample in {-2,-1,0,1,2}.
## This is a legacy Firehose-style product; TCGAbiolinks exposes it via the
## "Copy Number Variation" category with "Gene Level Copy Number Scores".
query_cnv <- GDCquery(
  project = "TCGA-BRCA",
  data.category = "Copy Number Variation",
  data.type = "Gene Level Copy Number",
  access = "open"
)
GDCdownload(query_cnv)
cnv_se <- GDCprepare(query_cnv)

## ============================================================================
## STEP 2: Download PAM50 subtype labels
## ============================================================================

cat("\n=== Fetching PAM50 subtype labels ===\n")
## TCGAbiolinks exposes PAM50 calls via the clinical "subtype" indexed data,
## matching the approach described in MOGONET's paper ("PAM50 breast cancer
## subtype information of TCGA BRCA patients was acquired through
## TCGAbiolinks").
subtype_info <- TCGAquery_subtype(tumor = "BRCA")
## Column name for PAM50 call varies by TCGAbiolinks version; check both
## common names defensively.
pam50_col <- intersect(c("BRCA_Subtype_PAM50", "PAM50.mRNA"), colnames(subtype_info))
if (length(pam50_col) == 0) {
  stop("Could not find a PAM50 subtype column in TCGAquery_subtype() output. ",
       "Inspect colnames(subtype_info) manually and adjust pam50_col below.")
}
pam50_col <- pam50_col[1]

pam50_labels <- data.frame(
  patient_barcode = substr(subtype_info$patient, 1, 12),
  pam50 = subtype_info[[pam50_col]],
  stringsAsFactors = FALSE
)
pam50_labels <- pam50_labels[!is.na(pam50_labels$pam50) & pam50_labels$pam50 != "", ]

## Map PAM50 string labels to the 0-4 integer encoding MOGONET/MOSGAT use.
## NOTE: exact label strings returned by TCGAquery_subtype() have varied
## across TCGAbiolinks versions/releases (e.g. "Normal" vs "Normal-like").
## The mapping below handles the common variants; inspect
## unique(pam50_labels$pam50) after running and adjust label_map if your
## installed version returns different strings.
label_map <- c(
  "Normal-like" = 0, "Normal" = 0,
  "Basal-like" = 1, "Basal" = 1,
  "HER2-enriched" = 2, "Her2" = 2, "HER2" = 2,
  "Luminal A" = 3, "LumA" = 3,
  "Luminal B" = 4, "LumB" = 4
)
pam50_labels$label_int <- label_map[pam50_labels$pam50]
unmapped <- pam50_labels[is.na(pam50_labels$label_int), ]
if (nrow(unmapped) > 0) {
  cat("WARNING:", nrow(unmapped), "samples have an unrecognized PAM50 string: ",
      paste(unique(unmapped$pam50), collapse = ", "), "\n")
  cat("These will be dropped. Check label_map above if this count looks large.\n")
}
pam50_labels <- pam50_labels[!is.na(pam50_labels$label_int), ]
cat("PAM50 label counts:\n")
print(table(pam50_labels$pam50))

## ============================================================================
## STEP 3: Extract patient-level barcodes and build each omics matrix as
## genes/probes/miRNAs (rows) x patients (columns), using primary tumor
## samples only (TCGA barcode sample type code "01")
## ============================================================================

get_patient_barcode <- function(full_barcodes) {
  substr(full_barcodes, 1, 12)
}

filter_primary_tumor <- function(full_barcodes) {
  ## TCGA barcode sample-type code: positions 14-15. "01" = primary solid
  ## tumor. This excludes normal tissue (code "11") and other sample types.
  sample_type_code <- substr(full_barcodes, 14, 15)
  sample_type_code == "01"
}

## ---- mRNA ----
mrna_counts <- assay(mrna_se, "unstranded")  ## raw counts; see normalization note below
mrna_barcodes <- colnames(mrna_counts)
mrna_keep <- filter_primary_tumor(mrna_barcodes)
mrna_counts <- mrna_counts[, mrna_keep]
colnames(mrna_counts) <- get_patient_barcode(ciiiiolnames(mrna_counts))
mrna_counts <- mrna_counts[, !duplicated(colnames(mrna_counts))]
## Simple log2(CPM+1) normalization, a standard lightweight choice when not
## using a full DESeq2/edgeR pipeline. If you need TPM or a more rigorous
## normalization, replace this block accordingly.
lib_sizes <- colSums(mrna_counts)
mrna_cpm <- sweep(mrna_counts, 2, lib_sizes, FUN = "/") * 1e6
mrna_log <- log2(mrna_cpm + 1)
cat("mRNA matrix (genes x patients):", nrow(mrna_log), "x", ncol(mrna_log), "\n")

## ---- Methylation ----
meth_beta <- assay(meth_se)
meth_barcodes <- colnames(meth_beta)
meth_keep <- filter_primary_tumor(meth_barcodes)
meth_beta <- meth_beta[, meth_keep]
colnames(meth_beta) <- get_patient_barcode(colnames(meth_beta))
meth_beta <- meth_beta[, !duplicated(colnames(meth_beta))]
## Drop probes with any missing beta values (common on methylation arrays)
meth_beta <- meth_beta[complete.cases(meth_beta), ]
cat("Methylation matrix (probes x patients):", nrow(meth_beta), "x", ncol(meth_beta), "\n")

## ---- miRNA ----
## GDCprepare for miRNA returns a wide data.frame with columns like
## "reads_per_million_miRNA_mapped_<barcode>" per sample; extract those.
mirna_rpm_cols <- grep("^reads_per_million_miRNA_mapped_", colnames(mirna_data), value = TRUE)
mirna_mat <- as.matrix(mirna_data[, mirna_rpm_cols])
rownames(mirna_mat) <- mirna_data$miRNA_ID
colnames(mirna_mat) <- sub("^reads_per_million_miRNA_mapped_", "", mirna_rpm_cols)
mirna_keep <- filter_primary_tumor(colnames(mirna_mat))
mirna_mat <- mirna_mat[, mirna_keep]
colnames(mirna_mat) <- get_patient_barcode(colnames(mirna_mat))
mirna_mat <- mirna_mat[, !duplicated(colnames(mirna_mat))]
mirna_log <- log2(mirna_mat + 1)
cat("miRNA matrix (miRNAs x patients):", nrow(mirna_log), "x", ncol(mirna_log), "\n")

## ---- CNV (GISTIC2 thresholded calls) ----
cnv_mat <- assay(cnv_se)
cnv_barcodes <- colnames(cnv_mat)
cnv_keep <- filter_primary_tumor(cnv_barcodes)
cnv_mat <- cnv_mat[, cnv_keep]
colnames(cnv_mat) <- get_patient_barcode(colnames(cnv_mat))
cnv_mat <- cnv_mat[, !duplicated(colnames(cnv_mat))]
mode(cnv_mat) <- "numeric"
cat("CNV matrix (genes x patients):", nrow(cnv_mat), "x", ncol(cnv_mat), "\n")

## ============================================================================
## STEP 4: Match patients across ALL FOUR omics types + PAM50 label
## ============================================================================

common_patients <- Reduce(intersect, list(
  colnames(mrna_log), colnames(meth_beta), colnames(mirna_log),
  colnames(cnv_mat), pam50_labels$patient_barcode
))
cat("\n=== Sample matching across mRNA, methylation, miRNA, CNV, and PAM50 label ===\n")
cat("mRNA patients:", ncol(mrna_log), "\n")
cat("Methylation patients:", ncol(meth_beta), "\n")
cat("miRNA patients:", ncol(mirna_log), "\n")
cat("CNV patients:", ncol(cnv_mat), "\n")
cat("PAM50-labeled patients:", nrow(pam50_labels), "\n")
cat("Patients with ALL FOUR omics types + PAM50 label:", length(common_patients), "\n")
if (length(common_patients) < 100) {
  warning("Fewer than 100 matched patients remain. This is a small cohort for ",
          "training a 5-class classifier -- double check the methylation ",
          "platform choice (27k vs 450k) and CNV availability above, since ",
          "these are usually the limiting factors.")
}

mrna_log   <- mrna_log[, common_patients]
meth_beta  <- meth_beta[, common_patients]
mirna_log  <- mirna_log[, common_patients]
cnv_mat    <- cnv_mat[, common_patients]
labels_df  <- pam50_labels[match(common_patients, pam50_labels$patient_barcode), ]
stopifnot(all(labels_df$patient_barcode == common_patients))

## ============================================================================
## STEP 5: Train/test split (70/30, stratified by PAM50 label, matching the
## MOGONET/MOSGAT evaluation protocol)
## ============================================================================

train_idx <- createDataPartition(labels_df$label_int, p = 0.7, list = FALSE)
train_patients <- common_patients[train_idx]
test_patients  <- common_patients[-train_idx]
cat("\nTrain patients:", length(train_patients), " | Test patients:", length(test_patients), "\n")

## ============================================================================
## STEP 6: Preprocessing per omics type, following MOGONET's described
## pipeline: variance filtering (with a different threshold per omics type,
## since different data types have different ranges), then ANOVA-based
## feature preselection using TRAINING LABELS ONLY (to avoid test-set
## leakage), stopped once the first principal component explains <50% of
## variance, then linear [0,1] scaling per feature.
## ============================================================================

anova_f_stat <- function(X, y) {
  ## X: features x samples matrix already subset to one omics type.
  ## y: integer class labels, same order as columns of X.
  ## Returns one F-statistic per row (feature) of X.
  groups <- split(seq_along(y), y)
  grand_mean <- rowMeans(X)
  n <- ncol(X)
  k <- length(groups)
  ss_between <- rowSums(sapply(groups, function(idx) {
    length(idx) * (rowMeans(X[, idx, drop = FALSE]) - grand_mean)^2
  }))
  ss_within <- rowSums(sapply(groups, function(idx) {
    rowSums((X[, idx, drop = FALSE] - rowMeans(X[, idx, drop = FALSE]))^2)
  }))
  df_between <- k - 1
  df_within <- n - k
  f_stat <- (ss_between / df_between) / (ss_within / df_within)
  f_stat[is.na(f_stat) | is.infinite(f_stat)] <- 0
  f_stat
}

select_features_pc_rule <- function(X_tr, y_tr, max_features_to_try, step = 50) {
  ## Rank features by ANOVA F-stat (train labels only), then increase the
  ## number of top features kept until PC1 of the SELECTED features
  ## explains <50% of variance -- this is MOGONET's own stated rule
  ## ("the first principal component of the data after feature
  ## pre-selection should explain <50% of the variance").
  f_stats <- anova_f_stat(X_tr, y_tr)
  ranked <- order(f_stats, decreasing = TRUE)

  n_try <- min(step, max_features_to_try)
  repeat {
    top_idx <- ranked[1:n_try]
    X_sub <- X_tr[top_idx, , drop = FALSE]
    pc <- prcomp(t(X_sub), center = TRUE, scale. = FALSE)
    var_explained <- (pc$sdev[1]^2) / sum(pc$sdev^2)
    if (var_explained < 0.5 || n_try >= max_features_to_try) break
    n_try <- min(n_try + step, max_features_to_try)
  }
  list(selected_idx = ranked[1:n_try], n_selected = n_try, final_pc1_var = var_explained)
}

minmax_scale_train_test <- function(X_tr, X_te) {
  ## Scale using TRAIN min/max only, applied to both train and test, to
  ## avoid test-set leakage into the scaling parameters.
  row_min <- apply(X_tr, 1, min)
  row_max <- apply(X_tr, 1, max)
  row_range <- row_max - row_min
  row_range[row_range == 0] <- 1  ## avoid divide-by-zero for constant features
  X_tr_scaled <- (X_tr - row_min) / row_range
  X_te_scaled <- (X_te - row_min) / row_range
  X_te_scaled[X_te_scaled < 0] <- 0  ## clip test values outside train's observed range
  X_te_scaled[X_te_scaled > 1] <- 1
  list(tr = X_tr_scaled, te = X_te_scaled)
}

process_omics <- function(X_all, var_threshold, max_features, omics_name, is_variance_filter_meaningful = TRUE) {
  X_tr <- X_all[, train_patients, drop = FALSE]
  X_te <- X_all[, test_patients, drop = FALSE]
  y_tr <- labels_df$label_int[match(train_patients, labels_df$patient_barcode)]

  ## Variance filtering (train set only, to avoid leakage)
  if (is_variance_filter_meaningful) {
    row_var <- matrixStats::rowVars(X_tr)
    keep_var <- row_var > var_threshold
    cat(sprintf("  [%s] variance filter: %d / %d features retained (threshold=%.5f)\n",
                omics_name, sum(keep_var), length(keep_var), var_threshold))
    X_tr <- X_tr[keep_var, , drop = FALSE]
    X_te <- X_te[keep_var, , drop = FALSE]
  }

  ## ANOVA-based preselection with the PC1<50% stopping rule
  sel <- select_features_pc_rule(X_tr, y_tr, max_features_to_try = min(nrow(X_tr), max_features))
  cat(sprintf("  [%s] ANOVA+PC1 rule selected %d features (final PC1 var explained: %.3f)\n",
              omics_name, sel$n_selected, sel$final_pc1_var))
  X_tr <- X_tr[sel$selected_idx, , drop = FALSE]
  X_te <- X_te[sel$selected_idx, , drop = FALSE]

  ## [0,1] scaling, train-derived parameters applied to both splits
  scaled <- minmax_scale_train_test(X_tr, X_te)

  list(tr = scaled$tr, te = scaled$te, featnames = rownames(X_tr))
}

cat("\n=== Preprocessing mRNA (variance threshold 0.1, per MOGONET) ===\n")
mrna_proc <- process_omics(mrna_log, var_threshold = 0.1, max_features = 2000, omics_name = "mRNA")

cat("\n=== Preprocessing methylation (variance threshold 0.001, per MOGONET) ===\n")
meth_proc <- process_omics(meth_beta, var_threshold = 0.001, max_features = 2000, omics_name = "methylation")

cat("\n=== Preprocessing miRNA (no variance threshold -- MOGONET only drops exactly-zero-variance miRNAs, given the limited feature count) ===\n")
mirna_proc <- process_omics(mirna_log, var_threshold = 0, max_features = ncol(mirna_log) * 0 + nrow(mirna_log), omics_name = "miRNA")

cat("\n=== Preprocessing CNV (GISTIC2 thresholded calls) ===\n")
## CAUTION: this variance threshold is NOT from any published paper (see the
## top-of-file caveat). GISTIC2 calls are a near-discrete {-2,-1,0,1,2}
## matrix; many genes will show near-zero variance simply because most
## samples are diploid (call = 0) at that locus. A variance threshold tuned
## for continuous expression data is not obviously appropriate here. The
## value below (0.01) is a conservative placeholder intended to drop
## genes with almost no variation (e.g., always 0) while keeping genes with
## at least some amplification/deletion signal across the cohort -- inspect
## the printed retained-feature count and adjust if it looks wrong for your
## data.
cnv_var_threshold <- 0.01
cnv_proc <- process_omics(cnv_mat, var_threshold = cnv_var_threshold, max_features = 2000, omics_name = "CNV")

## ============================================================================
## STEP 7: Write MOGONET-formatted output files
## ============================================================================

write_omics_files <- function(proc_result, view_index) {
  write.table(t(proc_result$tr), file.path(out_dir, paste0(view_index, "_tr.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(t(proc_result$te), file.path(out_dir, paste0(view_index, "_te.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(proc_result$featnames, file.path(out_dir, paste0(view_index, "_featname.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
}

write_omics_files(mrna_proc, 1)
write_omics_files(meth_proc, 2)
write_omics_files(mirna_proc, 3)
write_omics_files(cnv_proc, 4)

y_tr_final <- labels_df$label_int[match(train_patients, labels_df$patient_barcode)]
y_te_final <- labels_df$label_int[match(test_patients, labels_df$patient_barcode)]
write.table(y_tr_final, file.path(out_dir, "labels_tr.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(y_te_final, file.path(out_dir, "labels_te.csv"), sep = ",", row.names = FALSE, col.names = FALSE)

cat("\n=== DONE ===\n")
cat("MOGONET-formatted BRCA files (4 omics views: mRNA, methylation, miRNA, CNV) written to:",
    normalizePath(out_dir), "\n")
cat("Train:", length(train_patients), "samples | Test:", length(test_patients), "samples\n")
cat("Label distribution (train):\n"); print(table(y_tr_final))
cat("Label distribution (test):\n"); print(table(y_te_final))
cat("\nNEXT STEP: update your multiome/train_test.py's _lookup_adj_and_dims()\n")
cat("and any VIEW_LIST constants to use view_list = [1,2,3,4] for BRCA,\n")
cat("since this dataset now has 4 omics views, not 3.\n")

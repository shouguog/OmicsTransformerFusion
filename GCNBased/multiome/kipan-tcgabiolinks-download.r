## ============================================================================
## download_kipan_mogonet_format.R
##
## Downloads TCGA-KICH + TCGA-KIRC + TCGA-KIRP (pooled as "KIPAN", matching
## MOGONET/MOSGAT's construction), extracts mRNA, methylation, miRNA, and
## GISTIC2 CNV for each of the three projects, pools them into one cohort,
## matches samples across all FOUR omics types, applies MOGONET/MOSGAT-style
## preprocessing, splits into train/test, and writes MOGONET-formatted
## output files to ./KIPAN/:
##
##   1_tr.csv, 1_te.csv, 1_featname.csv   <- mRNA
##   2_tr.csv, 2_te.csv, 2_featname.csv   <- methylation
##   3_tr.csv, 3_te.csv, 3_featname.csv   <- miRNA
##   4_tr.csv, 4_te.csv, 4_featname.csv   <- CNV (GISTIC2, 4th view --
##                                           NOT present in the original
##                                           papers, added per this project's
##                                           request)
##   labels_tr.csv, labels_te.csv          <- integer class labels, 0-2
##                                           (0=KICH, 1=KIRC, 2=KIRP)
##
## This is now a 4-omics-type dataset (the original papers used only 3).
## ============================================================================
## CAVEATS -- READ BEFORE RUNNING
## ============================================================================
## 1. KIPAN is NOT a real TCGA project code -- it's a pooled cohort MOGONET
##    and MOSGAT construct themselves from three separate projects (KICH,
##    KIRC, KIRP). This script downloads and processes each project
##    SEPARATELY (each project's own GDCquery/GDCdownload/GDCprepare calls),
##    then concatenates the resulting matrices by COLUMN (patients) once
##    each project's own primary-tumor-filtering and barcode-deduplication
##    is done -- this means roughly 3x the download time/storage of a
##    single-project script, now across FOUR omics types (12 total
##    download/prepare calls), making this the most expensive of the three
##    download scripts in this project by a meaningful margin.
##
## 2. Per MOGONET's own description, "differences among chromophobe renal
##    cell carcinoma (KICH), clear renal cell carcinoma (KIRC), and
##    papillary renal cell carcinoma (KIRP) can be clearly observed in the
##    omics data" -- i.e. this is described as the EASIEST of the four
##    classification tasks in all three source papers, serving "more as a
##    proof-of-concept experiment for multi-class applications" than a
##    discriminating benchmark. This characterization does not change
##    anything about how this script downloads/processes data, but is worth
##    keeping in mind when interpreting results from this dataset later
##    (as this project's own paper draft already does extensively for the
##    MOGONET/MOSGAT-provided KIPAN data).
##
## 3. ANOVA-based feature preselection and the PC1<50% stopping rule are
##    applied ACROSS all three pooled classes together (not per sub-project)
##    using training labels only, consistent with how MOGONET/MOSGAT
##    describe feature preselection as happening once per omics type on the
##    full (pooled) training set, not separately per cancer subtype.
##
## 4. CNV preprocessing is NOT validated against any of the three papers --
##    see the identical caveat in download_brca_mogonet_format.R. Adding
##    CNV as a 4th REQUIRED omics type will likely shrink the matched
##    cohort further than the original 3-omics papers' sample counts.
## ============================================================================
rm(list = ls())
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
options(timeout = 1000)
set.seed(42)

out_dir <- "KIPAN"
dir.create(out_dir, showWarnings = FALSE)

get_patient_barcode <- function(full_barcodes) substr(full_barcodes, 1, 12)
filter_primary_tumor <- function(full_barcodes) substr(full_barcodes, 14, 15) == "01"

## ============================================================================
## STEP 1: Download + process each omics type for EACH of the three projects
## separately, then pool by concatenating columns (patients)
## ============================================================================

kidney_projects <- c("TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP")
project_label <- c("TCGA-KICH" = 0, "TCGA-KIRC" = 1, "TCGA-KIRP" = 2)  ## 0=KICH,1=KIRC,2=KIRP

download_and_process_mrna <- function(project) {
  cat("\n=== [", project, "] Downloading mRNA expression (STAR - Counts) ===\n")
  q <- GDCquery(project = project, data.category = "Transcriptome Profiling",
                data.type = "Gene Expression Quantification", workflow.type = "STAR - Counts")
  GDCdownload(q, method = "client")
  se <- GDCprepare(q)
  counts <- assay(se, "unstranded")
  keep <- filter_primary_tumor(colnames(counts))
  counts <- counts[, keep]
  colnames(counts) <- get_patient_barcode(colnames(counts))
  counts <- counts[, !duplicated(colnames(counts))]
  counts
}

download_and_process_meth <- function(project) {
  cat("\n=== [", project, "] Downloading methylation (Illumina 27k) ===\n")
  q <- GDCquery(project = project, data.category = "DNA Methylation",
                platform = "Illumina Human Methylation 450", data.type = "Methylation Beta Value")
  GDCdownload(q)
  se <- GDCprepare(q)
  beta <- assay(se)
  keep <- filter_primary_tumor(colnames(beta))
  beta <- beta[, keep]
  colnames(beta) <- get_patient_barcode(colnames(beta))
  beta <- beta[, !duplicated(colnames(beta))]
  beta
}

download_and_process_mirna <- function(project) {
  cat("\n=== [", project, "] Downloading miRNA expression ===\n")
  q <- GDCquery(project = project, data.category = "Transcriptome Profiling",
                data.type = "miRNA Expression Quantification")
  GDCdownload(q)
  d <- GDCprepare(q)
  rpm_cols <- grep("^reads_per_million_miRNA_mapped_", colnames(d), value = TRUE)
  mat <- as.matrix(d[, rpm_cols])
  rownames(mat) <- d$miRNA_ID
  colnames(mat) <- sub("^reads_per_million_miRNA_mapped_", "", rpm_cols)
  keep <- filter_primary_tumor(colnames(mat))
  mat <- mat[, keep]
  colnames(mat) <- get_patient_barcode(colnames(mat))
  mat <- mat[, !duplicated(colnames(mat))]
  mat
}

download_and_process_cnv <- function(project) {
  cat("\n=== [", project, "] Downloading CNV (GISTIC2 gene-level thresholded calls) ===\n")
  q <- GDCquery(project = project, data.category = "Copy Number Variation",workflow.type = "ASCAT3", 
                data.type = "Gene Level Copy Number", access = "open")
  GDCdownload(q)
  se <- GDCprepare(q)
  mat <- assay(se)
  keep <- filter_primary_tumor(colnames(mat))
  mat <- mat[, keep]
  colnames(mat) <- get_patient_barcode(colnames(mat))
  mat <- mat[, !duplicated(colnames(mat))]
  mode(mat) <- "numeric"
  mat
}

## Download each omics type for each of the 3 projects
mrna_per_project  <- lapply(kidney_projects, download_and_process_mrna)
meth_per_project  <- lapply(kidney_projects, download_and_process_meth)
mirna_per_project <- lapply(kidney_projects, download_and_process_mirna)
cnv_per_project   <- lapply(kidney_projects, download_and_process_cnv)
names(mrna_per_project) <- names(meth_per_project) <- names(mirna_per_project) <- names(cnv_per_project) <- kidney_projects

## ---- Pool mRNA across projects ----
## Only keep genes present in ALL THREE projects' mRNA output (gene sets
## should be identical since all three use the same STAR-Counts workflow
## against the same reference, but this guards against any mismatch).
common_genes <- Reduce(intersect, lapply(mrna_per_project, rownames))
mrna_pooled <- do.call(cbind, lapply(mrna_per_project, function(m) m[common_genes, , drop = FALSE]))
lib_sizes <- colSums(mrna_pooled)
mrna_cpm <- sweep(mrna_pooled, 2, lib_sizes, FUN = "/") * 1e6
mrna_log <- log2(mrna_cpm + 1)
cat("\nPooled mRNA matrix (genes x patients):", nrow(mrna_log), "x", ncol(mrna_log), "\n")

## ---- Pool methylation across projects ----
common_probes <- Reduce(intersect, lapply(meth_per_project, rownames))
meth_pooled <- do.call(cbind, lapply(meth_per_project, function(m) m[common_probes, , drop = FALSE]))
meth_pooled <- meth_pooled[complete.cases(meth_pooled), ]
cat("Pooled methylation matrix (probes x patients):", nrow(meth_pooled), "x", ncol(meth_pooled), "\n")

## ---- Pool miRNA across projects ----
## miRNA panels can differ slightly in composition across projects/batches;
## intersect to the common set, same reasoning as mRNA/methylation above.
common_mirnas <- Reduce(intersect, lapply(mirna_per_project, rownames))
mirna_pooled <- do.call(cbind, lapply(mirna_per_project, function(m) m[common_mirnas, , drop = FALSE]))
mirna_log <- log2(mirna_pooled + 1)
cat("Pooled miRNA matrix (miRNAs x patients):", nrow(mirna_log), "x", ncol(mirna_log), "\n")

## ---- Pool CNV across projects ----
## Same intersection logic -- GISTIC2 gene lists are typically consistent
## within a given GDC/Firehose data product version, but this guards
## against any project-specific gene set mismatch.
common_cnv_genes <- Reduce(intersect, lapply(cnv_per_project, rownames))
cnv_pooled <- do.call(cbind, lapply(cnv_per_project, function(m) m[common_cnv_genes, , drop = FALSE]))
cat("Pooled CNV matrix (genes x patients):", nrow(cnv_pooled), "x", ncol(cnv_pooled), "\n")

## ============================================================================
## STEP 2: Build pooled labels (which project each patient came from)
## ============================================================================

patient_project_map <- unlist(lapply(kidney_projects, function(p) {
  setNames(rep(p, ncol(mrna_per_project[[p]])), colnames(mrna_per_project[[p]]))
}))
## NOTE: patient_project_map is keyed by the FIRST occurrence of a patient
## barcode across the per-project mRNA matrices' original column names
## (before cross-omics intersection below), which is fine since a given
## patient barcode belongs to exactly one of the three projects by
## construction (KICH/KIRC/KIRP are disjoint patient cohorts).
labels_df_all <- data.frame(
  patient_barcode = names(patient_project_map),
  project = unname(patient_project_map),
  stringsAsFactors = FALSE
)
labels_df_all$label_int <- project_label[labels_df_all$project]
labels_df_all <- labels_df_all[!duplicated(labels_df_all$patient_barcode), ]
cat("\nPooled project/label counts:\n")
print(table(labels_df_all$project))

## ============================================================================
## STEP 3: Match patients across mRNA, methylation, miRNA, and label
## ============================================================================

common_patients <- Reduce(intersect, list(
  colnames(mrna_log), colnames(meth_pooled), colnames(mirna_log),
  colnames(cnv_pooled), labels_df_all$patient_barcode
))
cat("\n=== Sample matching ===\n")
cat("mRNA patients:", ncol(mrna_log), "\n")
cat("Methylation patients:", ncol(meth_pooled), "\n")
cat("miRNA patients:", ncol(mirna_log), "\n")
cat("CNV patients:", ncol(cnv_pooled), "\n")
cat("Labeled patients:", nrow(labels_df_all), "\n")
cat("Patients with all FOUR omics types + label:", length(common_patients), "\n")
if (length(common_patients) < 100) {
  warning("Fewer than 100 matched patients remain -- check methylation ",
          "platform coverage (27k vs 450k) and CNV availability per ",
          "project if this seems too low.")
}

mrna_log    <- mrna_log[, common_patients]
meth_pooled <- meth_pooled[, common_patients]
mirna_log   <- mirna_log[, common_patients]
cnv_pooled  <- cnv_pooled[, common_patients]
labels_df   <- labels_df_all[match(common_patients, labels_df_all$patient_barcode), ]
stopifnot(all(labels_df$patient_barcode == common_patients))

cat("\nFinal pooled KIPAN class distribution:\n")
print(table(labels_df$project))

## ============================================================================
## STEP 4: Train/test split (70/30, stratified across all 3 classes)
## ============================================================================

train_idx <- createDataPartition(labels_df$label_int, p = 0.7, list = FALSE)
train_patients <- common_patients[train_idx]
test_patients  <- common_patients[-train_idx]
cat("\nTrain patients:", length(train_patients), " | Test patients:", length(test_patients), "\n")

## ============================================================================
## STEP 5: Preprocessing (same ANOVA+PC1 rule and [0,1] scaling as the BRCA
## and LGG scripts; applied across the pooled 3-class training labels)
## ============================================================================

anova_f_stat <- function(X, y) {
  groups <- split(seq_along(y), y)
  grand_mean <- rowMeans(X)
  n <- ncol(X); k <- length(groups)
  ss_between <- rowSums(sapply(groups, function(idx) {
    length(idx) * (rowMeans(X[, idx, drop = FALSE]) - grand_mean)^2
  }))
  ss_within <- rowSums(sapply(groups, function(idx) {
    rowSums((X[, idx, drop = FALSE] - rowMeans(X[, idx, drop = FALSE]))^2)
  }))
  f_stat <- (ss_between / (k - 1)) / (ss_within / (n - k))
  f_stat[is.na(f_stat) | is.infinite(f_stat)] <- 0
  f_stat
}

select_features_pc_rule <- function(X_tr, y_tr, max_features_to_try, step = 50) {
  f_stats <- anova_f_stat(X_tr, y_tr)
  ranked <- order(f_stats, decreasing = TRUE)
  n_try <- min(step, max_features_to_try)
  repeat {
    top_idx <- ranked[1:n_try]
    pc <- prcomp(t(X_tr[top_idx, , drop = FALSE]), center = TRUE, scale. = FALSE)
    var_explained <- (pc$sdev[1]^2) / sum(pc$sdev^2)
    if (var_explained < 0.5 || n_try >= max_features_to_try) break
    n_try <- min(n_try + step, max_features_to_try)
  }
  list(selected_idx = ranked[1:n_try], n_selected = n_try, final_pc1_var = var_explained)
}

minmax_scale_train_test <- function(X_tr, X_te) {
  row_min <- apply(X_tr, 1, min); row_max <- apply(X_tr, 1, max)
  row_range <- row_max - row_min
  row_range[row_range == 0] <- 1
  X_tr_scaled <- (X_tr - row_min) / row_range
  X_te_scaled <- (X_te - row_min) / row_range
  X_te_scaled[X_te_scaled < 0] <- 0
  X_te_scaled[X_te_scaled > 1] <- 1
  list(tr = X_tr_scaled, te = X_te_scaled)
}

process_omics <- function(X_all, var_threshold, max_features, omics_name, is_variance_filter_meaningful = TRUE) {
  X_tr <- X_all[, train_patients, drop = FALSE]
  X_te <- X_all[, test_patients, drop = FALSE]
  y_tr <- labels_df$label_int[match(train_patients, labels_df$patient_barcode)]

  if (is_variance_filter_meaningful) {
    row_var <- matrixStats::rowVars(X_tr)
    keep_var <- row_var > var_threshold
    cat(sprintf("  [%s] variance filter: %d / %d features retained (threshold=%.5f)\n",
                omics_name, sum(keep_var), length(keep_var), var_threshold))
    X_tr <- X_tr[keep_var, , drop = FALSE]
    X_te <- X_te[keep_var, , drop = FALSE]
  }

  sel <- select_features_pc_rule(X_tr, y_tr, max_features_to_try = min(nrow(X_tr), max_features))
  cat(sprintf("  [%s] ANOVA+PC1 rule selected %d features (final PC1 var explained: %.3f)\n",
              omics_name, sel$n_selected, sel$final_pc1_var))
  X_tr <- X_tr[sel$selected_idx, , drop = FALSE]
  X_te <- X_te[sel$selected_idx, , drop = FALSE]

  scaled <- minmax_scale_train_test(X_tr, X_te)
  list(tr = scaled$tr, te = scaled$te, featnames = rownames(X_tr))
}

cat("\n=== Preprocessing mRNA (variance threshold 0.1) ===\n")
mrna_proc <- process_omics(mrna_log, var_threshold = 0.1, max_features = 2000, omics_name = "mRNA")

cat("\n=== Preprocessing methylation (variance threshold 0.001) ===\n")
meth_proc <- process_omics(meth_pooled, var_threshold = 0.001, max_features = 2000, omics_name = "methylation")

cat("\n=== Preprocessing miRNA ===\n")
mirna_proc <- process_omics(mirna_log, var_threshold = 0, max_features = nrow(mirna_log), omics_name = "miRNA")

cat("\n=== Preprocessing CNV (GISTIC2 thresholded calls) ===\n")
## CAUTION: see the top-of-file caveat (#4) and the identical discussion in
## download_brca_mogonet_format.R -- this threshold is an unvalidated
## placeholder, not a published value, since none of the source papers used
## CNV. Inspect the printed retained-feature count and adjust if needed.
cnv_var_threshold <- 0.01
cnv_proc <- process_omics(cnv_pooled, var_threshold = cnv_var_threshold, max_features = 2000, omics_name = "CNV")

## ============================================================================
## STEP 6: Write MOGONET-formatted output files
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
cat("MOGONET-formatted KIPAN files (4 omics views: mRNA, methylation, miRNA, CNV;",
    "pooled KICH+KIRC+KIRP) written to:", normalizePath(out_dir), "\n")
cat("Train:", length(train_patients), "samples | Test:", length(test_patients), "samples\n")
cat("Label distribution (train, 0=KICH/1=KIRC/2=KIRP):\n"); print(table(y_tr_final))
cat("Label distribution (test, 0=KICH/1=KIRC/2=KIRP):\n"); print(table(y_te_final))
cat("\nNEXT STEP: update multiome/train_test.py's _lookup_adj_and_dims() and\n")
cat("any VIEW_LIST constants to use view_list = [1,2,3,4] for KIPAN.\n")

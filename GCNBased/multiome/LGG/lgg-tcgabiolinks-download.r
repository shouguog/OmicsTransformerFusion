## ============================================================================
## download_lgg_mogonet_format.R
##
## Downloads TCGA-LGG mRNA, DNA methylation, miRNA, and GISTIC2 CNV data via
## TCGAbiolinks, matches samples across all FOUR omics types + tumor grade
## label, applies MOGONET/MOSGAT-style preprocessing, splits into
## train/test, and writes MOGONET-formatted output files to ./LGG/:
##
##   1_tr.csv, 1_te.csv, 1_featname.csv   <- mRNA
##   2_tr.csv, 2_te.csv, 2_featname.csv   <- methylation
##   3_tr.csv, 3_te.csv, 3_featname.csv   <- miRNA
##   4_tr.csv, 4_te.csv, 4_featname.csv   <- CNV (GISTIC2, 4th view --
##                                           NOT present in the original
##                                           MOGONET/MMDynamics/MOSGAT papers,
##                                           added per this project's request)
##   labels_tr.csv, labels_te.csv          <- integer class labels, 0-1
##                                           (0 = Grade 2, 1 = Grade 3)
##
## This is now a 4-omics-type dataset (the original papers used only 3 --
## mRNA, methylation, miRNA -- CNV is an addition specific to this project).
## ============================================================================
## CAVEATS -- READ BEFORE RUNNING
## ============================================================================
## 1. Expect a multi-hour run and tens of GB of temporary download storage,
##    same as the BRCA script.
##
## 2. Grade label source: TCGA-LGG grade (G2/G3) is a CLINICAL field, not a
##    molecular subtype call like BRCA's PAM50 -- it comes from GDC clinical
##    data via TCGAbiolinks' GDCquery_clinic(), not TCGAquery_subtype(). The
##    exact column name for histologic grade has varied across TCGA
##    clinical data releases/versions; this script checks a few known
##    candidates defensively and prints a warning + the available column
##    names if none match, rather than silently producing an empty/wrong
##    label set.
##
## 3. Sample matching: only patients with ALL FOUR omics types AND a known
##    G2/G3 grade are retained (G4/GX/missing are dropped, consistent with
##    MOGONET's binary-only LGG task). The matched count is printed
##    explicitly -- adding CNV as a 4th required omics type will likely
##    shrink the matched cohort further than the original 3-omics papers'
##    sample counts, since it adds one more "must be present" requirement.
##
## 4. CNV preprocessing is NOT validated against any of the three papers,
##    since none of them used CNV -- see the CNV section below and the
##    same caveat in download_brca_mogonet_format.R for full discussion of
##    why the variance-filtering threshold for CNV is a placeholder
##    judgment call, not a published value.
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

out_dir <- "LGG"
dir.create(out_dir, showWarnings = FALSE)

## ============================================================================
## STEP 1: Download each omics type
## ============================================================================
## ---- CNV (GISTIC2 thresholded calls) ----
cat("\n=== Downloading CNV (GISTIC2 gene-level thresholded calls) ===\n")
query_cnv <- GDCquery(
  project = "TCGA-LGG",
  data.category = "Copy Number Variation",
  data.type = "Gene Level Copy Number",
  workflow.type = "ASCAT3",  # <-- Add this line to isolate a single pipeline
  access = "open"
)
GDCdownload(query_cnv)
cnv_se <- GDCprepare(query_cnv)

cat("\n=== Downloading mRNA expression (STAR - Counts) ===\n")
query_mrna <- GDCquery(
  project = "TCGA-LGG",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts"
)
GDCdownload(query_mrna)
mrna_se <- GDCprepare(query_mrna)

cat("\n=== Downloading DNA methylation (Illumina Human Methylation 450) ===\n")
## As in the BRCA script: MOGONET restricts to 27k-array probes for
## interpretability. If 27k coverage for TCGA-LGG is too sparse once
## intersected with mRNA/miRNA/grade, switch to "Illumina Human Methylation
## 450" and intersect with the 27k probe list manually.
query_meth <- GDCquery(
  project = "TCGA-LGG",
  data.category = "DNA Methylation",
  platform = "Illumina Human Methylation 450",
  data.type = "Methylation Beta Value"
)
GDCdownload(query_meth, method = "client")
meth_se <- GDCprepare(query_meth)

cat("\n=== Downloading miRNA expression ===\n")
query_mirna <- GDCquery(
  project = "TCGA-LGG",
  data.category = "Transcriptome Profiling",
  data.type = "miRNA Expression Quantification"
)
GDCdownload(query_mirna)
mirna_data <- GDCprepare(query_mirna)

## ============================================================================
## STEP 2: Download clinical data and extract tumor grade
## ============================================================================

cat("\n=== Fetching clinical data for tumor grade ===\n")
clinical <- GDCquery_clinic(project = "TCGA-LGG", type = "clinical")

## Column name for histologic/WHO grade has varied across TCGA clinical
## data releases. Check known candidates defensively.
grade_col_candidates <- c("neoplasm_histologic_grade", "tumor_grade", "grade")
grade_col <- intersect(grade_col_candidates, colnames(clinical))
if (length(grade_col) == 0) {
  cat("WARNING: none of the expected grade column names were found.\n")
  cat("Available clinical columns are:\n")
  print(colnames(clinical))
  stop("Inspect the column list above, set grade_col manually to the correct ",
       "column name, and rerun from this point.")
}
grade_col <- grade_col[1]
cat("Using clinical column:", grade_col, "\n")

grade_labels <- data.frame(
  patient_barcode = clinical$submitter_id,
  grade_raw = clinical[[grade_col]],
  stringsAsFactors = FALSE
)
cat("Raw grade value counts:\n")
print(table(grade_labels$grade_raw, useNA = "ifany"))

## Map to MOGONET's binary encoding: Grade 2 = 0, Grade 3 = 1. Drop G4, GX,
## and anything else (e.g. "Not Available"), consistent with MOGONET's
## binary-only LGG grading task.
grade_map <- c("G2" = 0, "Grade II" = 0, "GRADE II" = 0,
               "G3" = 1, "Grade III" = 1, "GRADE III" = 1)
grade_labels$label_int <- grade_map[grade_labels$grade_raw]
n_dropped <- sum(is.na(grade_labels$label_int))
cat(n_dropped, "patients dropped due to non-G2/G3 grade (G4, GX, missing, etc.)\n")
grade_labels <- grade_labels[!is.na(grade_labels$label_int), ]
cat("Final grade label counts (0=G2, 1=G3):\n")
print(table(grade_labels$label_int))

## ============================================================================
## STEP 3: Extract patient-level barcodes, primary tumor samples only
## ============================================================================

get_patient_barcode <- function(full_barcodes) substr(full_barcodes, 1, 12)
filter_primary_tumor <- function(full_barcodes) substr(full_barcodes, 14, 15) == "01"

## ---- mRNA ----
mrna_counts <- assay(mrna_se, "unstranded")
mrna_keep <- filter_primary_tumor(colnames(mrna_counts))
mrna_counts <- mrna_counts[, mrna_keep]
colnames(mrna_counts) <- get_patient_barcode(colnames(mrna_counts))
mrna_counts <- mrna_counts[, !duplicated(colnames(mrna_counts))]
lib_sizes <- colSums(mrna_counts)
mrna_cpm <- sweep(mrna_counts, 2, lib_sizes, FUN = "/") * 1e6
mrna_log <- log2(mrna_cpm + 1)
cat("mRNA matrix (genes x patients):", nrow(mrna_log), "x", ncol(mrna_log), "\n")

## ---- Methylation ----
meth_beta <- assay(meth_se)
meth_keep <- filter_primary_tumor(colnames(meth_beta))
meth_beta <- meth_beta[, meth_keep]
colnames(meth_beta) <- get_patient_barcode(colnames(meth_beta))
meth_beta <- meth_beta[, !duplicated(colnames(meth_beta))]
meth_beta <- meth_beta[complete.cases(meth_beta), ]
cat("Methylation matrix (probes x patients):", nrow(meth_beta), "x", ncol(meth_beta), "\n")

## ---- miRNA ----
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

## ---- CNV ----
cnv_mat <- assay(cnv_se)
cnv_keep <- filter_primary_tumor(colnames(cnv_mat))
cnv_mat <- cnv_mat[, cnv_keep]
colnames(cnv_mat) <- get_patient_barcode(colnames(cnv_mat))
cnv_mat <- cnv_mat[, !duplicated(colnames(cnv_mat))]
mode(cnv_mat) <- "numeric"
cat("CNV matrix (genes x patients):", nrow(cnv_mat), "x", ncol(cnv_mat), "\n")

## ============================================================================
## STEP 4: Match patients across mRNA, methylation, miRNA, CNV, and grade label
## ============================================================================

common_patients <- Reduce(intersect, list(
  colnames(mrna_log), colnames(meth_beta), colnames(mirna_log),
  colnames(cnv_mat), grade_labels$patient_barcode
))
cat("\n=== Sample matching ===\n")
cat("mRNA patients:", ncol(mrna_log), "\n")
cat("Methylation patients:", ncol(meth_beta), "\n")
cat("miRNA patients:", ncol(mirna_log), "\n")
cat("CNV patients:", ncol(cnv_mat), "\n")
cat("Grade-labeled (G2/G3) patients:", nrow(grade_labels), "\n")
cat("Patients with all FOUR omics types + grade label:", length(common_patients), "\n")
if (length(common_patients) < 100) {
  warning("Fewer than 100 matched patients remain -- check methylation ",
          "platform coverage (27k vs 450k) and CNV availability if this ",
          "seems too low.")
}

mrna_log  <- mrna_log[, common_patients]
meth_beta <- meth_beta[, common_patients]
mirna_log <- mirna_log[, common_patients]
cnv_mat   <- cnv_mat[, common_patients]
labels_df <- grade_labels[match(common_patients, grade_labels$patient_barcode), ]
stopifnot(all(labels_df$patient_barcode == common_patients))

## ============================================================================
## STEP 5: Train/test split (70/30, stratified)
## ============================================================================

train_idx <- createDataPartition(labels_df$label_int, p = 0.7, list = FALSE)
train_patients <- common_patients[train_idx]
test_patients  <- common_patients[-train_idx]
cat("\nTrain patients:", length(train_patients), " | Test patients:", length(test_patients), "\n")

## ============================================================================
## STEP 6: Preprocessing (same ANOVA+PC1 rule and [0,1] scaling as the BRCA
## script -- see that script's comments for the full rationale)
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
meth_proc <- process_omics(meth_beta, var_threshold = 0.001, max_features = 2000, omics_name = "methylation")

cat("\n=== Preprocessing miRNA ===\n")
mirna_proc <- process_omics(mirna_log, var_threshold = 0, max_features = nrow(mirna_log), omics_name = "miRNA")

cat("\n=== Preprocessing CNV (GISTIC2 thresholded calls) ===\n")
## CAUTION: see the top-of-file caveat (#4) and the identical discussion in
## download_brca_mogonet_format.R -- this threshold is an unvalidated
## placeholder, not a published value, since none of the source papers used
## CNV. Inspect the printed retained-feature count and adjust if needed.
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
cat("MOGONET-formatted LGG files (4 omics views: mRNA, methylation, miRNA, CNV) written to:",
    normalizePath(out_dir), "\n")
cat("Train:", length(train_patients), "samples | Test:", length(test_patients), "samples\n")
cat("Label distribution (train, 0=G2/1=G3):\n"); print(table(y_tr_final))
cat("Label distribution (test, 0=G2/1=G3):\n"); print(table(y_te_final))
cat("\nNEXT STEP: update multiome/train_test.py's _lookup_adj_and_dims() and\n")
cat("any VIEW_LIST constants to use view_list = [1,2,3,4] for LGG.\n")

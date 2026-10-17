"""
Baseline comparison methods for multi-omics classification, matching the
MOSGAT paper's experimental protocol (Table II/III style evaluation):
multiple repeated runs, reporting mean +/- std of ACC, F1 (binary) or
F1_weighted/F1_macro (multiclass).

Data layout expected per dataset folder (same as MOGONET/MOSGAT):
    1_tr.csv, 1_te.csv   -> view 1 (mRNA)
    2_tr.csv, 2_te.csv   -> view 2 (methylation)
    3_tr.csv, 3_te.csv   -> view 3 (miRNA)
    labels_tr.csv, labels_te.csv

======================= IMPORTANT SCOPE NOTE =======================
Of the 9 methods requested, 6 are implemented properly below using
sklearn/xgboost on simply-concatenated multi-omics features (the same
"early integration" strategy the paper itself uses for these baselines):

    SVM, KNN, Lasso, Random Forest (RF), FCNN, XGBoost

The other 4 do NOT have faithful, standard Python implementations:

  - BPLSDA / BSPLSDA: these are specific multi-BLOCK PLS-DA methods from
    the DIABLO framework in R's `mixOmics` package. They model each omics
    view as a separate "block" with inter-block weight matrices, which is
    structurally different from running PLS on concatenated features.
    There is no mature Python port. What's implemented below
    (`approx_BPLSDA`, `approx_BSPLSDA`) is ordinary PLS-DA /
    sparse-PLS-DA on CONCATENATED data -- i.e. single-block, not
    multi-block. This is a genuine approximation, not the real method,
    and will likely understate the real BPLSDA/BSPLSDA's ability to use
    between-block structure. Treat these results as a rough lower bound,
    not a faithful reproduction.

  - GRidge: Van de Wiel et al.'s group-regularized ridge regression with
    co-data (R package `GRridge`) uses external "co-data" (e.g. which
    omics view each feature came from) to set group-specific penalties
    during ridge regression. What's implemented below (`approx_GRidge`)
    is plain ridge-penalized logistic regression with NO group-specific
    penalty weighting -- i.e. it ignores the omics-view grouping
    entirely. This is a weaker approximation; it will not capture
    GRidge's main advantage.

  - CF (Cascade of Final representations): no canonical public
    implementation exists; the paper cites general multimodal-cascading
    ideas [40],[41], not one specific released method. What's implemented
    below (`approx_CF`) is a simple late-cascade: train a small classifier
    per omics view, concatenate their output probabilities, and train a
    final logistic regression on top. This is a reasonable reading of
    "cascade of final representations" but is a reconstruction, not a
    citation-backed implementation.

If you need the real BPLSDA/BSPLSDA/GRidge numbers for a paper or
comparison that will be scrutinized, those need to be run in R via
`mixOmics::block.plsda` / `block.splsda` and `GRridge::grridge`,
ideally called from Python via `rpy2` -- that's a different, heavier
piece of work and not what's below.
=====================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------
def load_dataset(data_folder, view_list=(1, 2, 3)):
    """Load and concatenate views; returns train/test splits + labels."""
    def _load(name):
        return pd.read_csv(os.path.join(data_folder, name), header=None).values

    X_tr_views = [_load(f"{v}_tr.csv") for v in view_list]
    X_te_views = [_load(f"{v}_te.csv") for v in view_list]
    y_tr = _load("labels_tr.csv").ravel().astype(int)
    y_te = _load("labels_te.csv").ravel().astype(int)

    X_tr = np.concatenate(X_tr_views, axis=1)
    X_te = np.concatenate(X_te_views, axis=1)
    return X_tr, X_te, y_tr, y_te


def merge_all(data_folder, view_list=(1, 2, 3)):
    """Merge original tr+te into one pool, for repeated re-splitting."""
    X_tr, X_te, y_tr, y_te = load_dataset(data_folder, view_list)
    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    return X_all, y_all


# --------------------------------------------------------------------
# Metrics, matching the paper's reporting (ACC/F1/AUC for binary,
# ACC/F1_weighted/F1_macro for multiclass)
# --------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba, num_class):
    out = {"ACC": accuracy_score(y_true, y_pred)}
    if num_class == 2:
        out["F1"] = f1_score(y_true, y_pred)
        if y_proba is not None:
            out["AUC"] = roc_auc_score(y_true, y_proba[:, 1])
    else:
        out["F1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        out["F1_macro"] = f1_score(y_true, y_pred, average="macro")
    return out


# --------------------------------------------------------------------
# The 6 properly-implemented baselines
# --------------------------------------------------------------------
def run_svm(X_tr, y_tr, X_te, num_class):
    clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te), clf.predict_proba(X_te)


def run_knn(X_tr, y_tr, X_te, num_class, k=5):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    return clf.predict(X_te), proba


def run_lasso(X_tr, y_tr, X_te, num_class):
    # L1-penalized logistic regression == "Lasso" for classification
    clf = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000,
        class_weight="balanced", multi_class="auto"
    )
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te), clf.predict_proba(X_te)


def run_rf(X_tr, y_tr, X_te, num_class, n_estimators=500):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=0
    )
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te), clf.predict_proba(X_te)


def run_fcnn(X_tr, y_tr, X_te, num_class, hidden=(128, 64)):
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden, max_iter=500, early_stopping=True,
        random_state=0
    )
    clf.fit(X_tr_s, y_tr)
    return clf.predict(X_te_s), clf.predict_proba(X_te_s)


def run_xgboost(X_tr, y_tr, X_te, num_class):
    objective = "binary:logistic" if num_class == 2 else "multi:softprob"
    params = dict(objective=objective, eval_metric="mlogloss",
                  n_estimators=300, max_depth=4, learning_rate=0.05,
                  use_label_encoder=False, verbosity=0)
    if num_class > 2:
        params["num_class"] = num_class
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te), clf.predict_proba(X_te)


# --------------------------------------------------------------------
# The 4 APPROXIMATIONS -- see module docstring for what's missing
# --------------------------------------------------------------------
def approx_BPLSDA(X_tr, y_tr, X_te, num_class, n_components=10):
    """APPROXIMATION of BPLSDA: ordinary (single-block) PLS-DA on
    concatenated features. Real BPLSDA models each omics view as a
    separate block with inter-block weights (mixOmics::block.plsda)."""
    Y_tr_bin = label_binarize(y_tr, classes=np.arange(num_class))
    if num_class == 2:
        Y_tr_bin = np.hstack([1 - Y_tr_bin, Y_tr_bin])
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    pls = PLSRegression(n_components=min(n_components, X_tr_s.shape[1]))
    pls.fit(X_tr_s, Y_tr_bin)
    scores = pls.predict(X_te_s)
    y_pred = np.argmax(scores, axis=1)
    proba = np.clip(scores, 0, None)
    proba = proba / (proba.sum(axis=1, keepdims=True) + 1e-9)
    return y_pred, proba


def approx_BSPLSDA(X_tr, y_tr, X_te, num_class, n_components=10, top_k=200):
    """APPROXIMATION of BSPLSDA: PLS-DA on concatenated features after a
    crude univariate feature-selection step to mimic "sparse" selection.
    Real BSPLSDA applies block-wise sparsity constraints jointly across
    omics blocks during PLS fitting (mixOmics::block.splsda), which is
    structurally different from pre-filtering features independently."""
    from sklearn.feature_selection import f_classif
    k = min(top_k, X_tr.shape[1])
    f_vals, _ = f_classif(X_tr, y_tr)
    f_vals = np.nan_to_num(f_vals)
    top_idx = np.argsort(f_vals)[-k:]
    return approx_BPLSDA(X_tr[:, top_idx], y_tr, X_te[:, top_idx],
                          num_class, n_components=n_components)


def approx_GRidge(X_tr, y_tr, X_te, num_class, alpha=1.0):
    """APPROXIMATION of GRidge: plain L2-penalized logistic regression
    with a single global penalty. Real GRidge (Van de Wiel et al.) uses
    "co-data" -- here, which omics view each feature belongs to -- to
    assign DIFFERENT penalty weights per group. This version ignores
    the view-grouping entirely, so it cannot show GRidge's main benefit
    of down-weighting less-informative omics blocks."""
    clf = LogisticRegression(
        penalty="l2", C=1.0 / alpha, solver="lbfgs", max_iter=5000,
        class_weight="balanced", multi_class="auto"
    )
    clf.fit(X_tr, y_tr)
    return clf.predict(X_te), clf.predict_proba(X_te)


def approx_CF(X_tr_views, y_tr, X_te_views, num_class):
    """APPROXIMATION of CF (Cascade of Final representations): train one
    classifier per omics view, concatenate their predicted probabilities
    as a new feature set, then train a final classifier on top. This is
    a reasonable reading of "cascading final multimodal representations"
    but is not a citation-backed reproduction of [40]/[41]."""
    view_proba_tr, view_proba_te = [], []
    for X_tr_v, X_te_v in zip(X_tr_views, X_te_views):
        scaler = StandardScaler().fit(X_tr_v)
        X_tr_s, X_te_s = scaler.transform(X_tr_v), scaler.transform(X_te_v)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                  multi_class="auto")
        clf.fit(X_tr_s, y_tr)
        view_proba_tr.append(clf.predict_proba(X_tr_s))
        view_proba_te.append(clf.predict_proba(X_te_s))

    cascade_tr = np.concatenate(view_proba_tr, axis=1)
    cascade_te = np.concatenate(view_proba_te, axis=1)

    final_clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                    multi_class="auto")
    final_clf.fit(cascade_tr, y_tr)
    return final_clf.predict(cascade_te), final_clf.predict_proba(cascade_te)


# --------------------------------------------------------------------
# Driver: run all methods, repeated splits, report mean +/- std
# --------------------------------------------------------------------
METHOD_REGISTRY = {
    "SVM": ("concat", run_svm),
    "KNN": ("concat", run_knn),
    "Lasso": ("concat", run_lasso),
    "RF": ("concat", run_rf),
    "FCNN": ("concat", run_fcnn),
    "XGBoost": ("concat", run_xgboost),
    "BPLSDA_approx": ("concat", approx_BPLSDA),
    "BSPLSDA_approx": ("concat", approx_BSPLSDA),
    "GRidge_approx": ("concat", approx_GRidge),
    "CF_approx": ("views", approx_CF),
}

DATASET_NUM_CLASS = {
    "ROSMAP": 2,
    "LGG": 2,
    "KIPAN": 3,
    "BRCA": 5,
}


def run_one_dataset(data_folder, dataset_name, view_list=(1, 2, 3),
                     n_repeats=20, test_size=0.30, base_seed=0):
    num_class = DATASET_NUM_CLASS[dataset_name]

    X_views_all = []
    for v in view_list:
        df_tr = pd.read_csv(os.path.join(data_folder, f"{v}_tr.csv"), header=None).values
        df_te = pd.read_csv(os.path.join(data_folder, f"{v}_te.csv"), header=None).values
        X_views_all.append(np.concatenate([df_tr, df_te], axis=0))
    y_tr0 = pd.read_csv(os.path.join(data_folder, "labels_tr.csv"), header=None).values.ravel()
    y_te0 = pd.read_csv(os.path.join(data_folder, "labels_te.csv"), header=None).values.ravel()
    y_all = np.concatenate([y_tr0, y_te0]).astype(int)
    X_concat_all = np.concatenate(X_views_all, axis=1)

    n_samples = X_concat_all.shape[0]
    results = {name: [] for name in METHOD_REGISTRY}

    for rep in range(n_repeats):
        seed = base_seed + rep
        idx_tr, idx_te = train_test_split(
            np.arange(n_samples), test_size=test_size,
            random_state=seed, stratify=y_all
        )
        y_tr, y_te = y_all[idx_tr], y_all[idx_te]
        X_concat_tr, X_concat_te = X_concat_all[idx_tr], X_concat_all[idx_te]
        X_views_tr = [Xv[idx_tr] for Xv in X_views_all]
        X_views_te = [Xv[idx_te] for Xv in X_views_all]

        for name, (mode, fn) in METHOD_REGISTRY.items():
            try:
                if mode == "concat":
                    y_pred, y_proba = fn(X_concat_tr, y_tr, X_concat_te, num_class)
                else:  # "views"
                    y_pred, y_proba = fn(X_views_tr, y_tr, X_views_te, num_class)
                m = compute_metrics(y_te, y_pred, y_proba, num_class)
                results[name].append(m)
            except Exception as e:
                print(f"  [{dataset_name}] {name} failed on rep {rep}: {e}")

    # Summarize mean +/- std
    summary_rows = []
    for name, reps in results.items():
        if not reps:
            continue
        keys = reps[0].keys()
        row = {"Method": name}
        for k in keys:
            vals = [r[k] for r in reps if k in r]
            row[f"{k}_mean"] = np.mean(vals)
            row[f"{k}_std"] = np.std(vals)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def fmt_mean_std(mean, std, decimals=3):
    """'0.853 +/- 0.004' style formatting, matching the paper's tables."""
    return f"{mean:.{decimals}f}\u00b1{std:.{decimals}f}"


def build_paper_style_table(all_summaries, dataset_names, metric_cols,
                             label_map=None, method_order=None):
    """
    Build a single wide table: rows = methods, columns = one block of
    metrics per dataset, each cell 'mean+/-std'. Mirrors the layout of
    Table II (binary: ROSMAP/LGG) and Table III (multiclass: BRCA/KIPAN)
    in the paper, and is reusable for the omics-combination style table
    (Table VI) if you pass per-omics-combination summaries instead.

    all_summaries: dict {dataset_name: summary_df} from run_one_dataset()
    dataset_names: list of dataset names to include, sets column grouping
    metric_cols: list of metric base-names to show, e.g. ['ACC','F1','AUC']
                 or ['ACC','F1_weighted','F1_macro']
    label_map: optional dict to rename methods for display
    method_order: optional explicit row order (method names as they
        appear in the summary 'Method' column, pre-label_map). If None,
        defaults to METHOD_REGISTRY order.
    """
    if method_order is None:
        method_order = list(METHOD_REGISTRY.keys())

    rows = []
    for method in method_order:
        display_name = label_map.get(method, method) if label_map else method
        row = {"Method": display_name}
        for ds in dataset_names:
            df = all_summaries.get(ds)
            if df is None or df.empty:
                for metric in metric_cols:
                    row[f"{ds}_{metric}"] = "--"
                continue
            match = df[df["Method"] == method]
            if match.empty:
                for metric in metric_cols:
                    row[f"{ds}_{metric}"] = "--"
                continue
            r = match.iloc[0]
            for metric in metric_cols:
                mean_col, std_col = f"{metric}_mean", f"{metric}_std"
                if mean_col in r and std_col in r and not pd.isna(r[mean_col]):
                    row[f"{ds}_{metric}"] = fmt_mean_std(r[mean_col], r[std_col])
                else:
                    row[f"{ds}_{metric}"] = "--"
        rows.append(row)

    table = pd.DataFrame(rows)

    # Build a readable multi-level column header: (Dataset, Metric)
    col_tuples = [("", "Method")]
    for ds in dataset_names:
        for metric in metric_cols:
            col_tuples.append((ds, metric))
    table.columns = pd.MultiIndex.from_tuples(col_tuples)
    return table


def mark_best_per_column(table, higher_is_better=True):
    """Bold the best mean value per (dataset, metric) column, matching the
    paper's convention of bolding MOSGAT's wins. Returns a copy with
    '**value**' markers around the best entry in each metric column."""
    out = table.copy()
    for col in table.columns:
        if col[1] == "Method":
            continue
        means = []
        for cell in table[col]:
            if cell == "--":
                means.append(-np.inf if higher_is_better else np.inf)
            else:
                means.append(float(cell.split("\u00b1")[0]))
        means = np.array(means)
        best_idx = np.argmax(means) if higher_is_better else np.argmin(means)
        if table[col].iloc[best_idx] != "--":
            out.loc[out.index[best_idx], col] = f"**{table[col].iloc[best_idx]}**"
    return out


if __name__ == "__main__":
    # Point this at the parent folder containing ROSMAP/, LGG/, KIPAN/, BRCA/
    # subfolders, each with 1_tr.csv ... labels_te.csv as described above.
    base_dir = "../"
    datasets = ["ROSMAP", "LGG", "KIPAN", "BRCA"]

    all_summaries = {}
    for ds in datasets:
        folder = os.path.join(base_dir, ds)
        if not os.path.isdir(folder):
            print(f"Skipping {ds}: folder not found at {folder}")
            continue
        print(f"\n=== Running baselines on {ds} ===")
        summary = run_one_dataset(folder, ds, n_repeats=20)
        all_summaries[ds] = summary
        print(summary.round(4).to_string(index=False))
        summary.to_csv(f"baseline_results_{ds}.csv", index=False)

    # ---------------- Aggregated, paper-style tables ----------------
    display_names = {
        "BPLSDA_approx": "BPLSDA (approx.)",
        "BSPLSDA_approx": "BSPLSDA (approx.)",
        "GRidge_approx": "GRidge (approx.)",
        "CF_approx": "CF (approx.)",
    }

    binary_datasets = [d for d in ["ROSMAP", "LGG"] if d in all_summaries and not all_summaries[d].empty]
    multiclass_datasets = [d for d in ["BRCA", "KIPAN"] if d in all_summaries and not all_summaries[d].empty]

    if binary_datasets:
        print("\n\n=== Table II style: Binary classification (ROSMAP, LGG) ===")
        table_bin = build_paper_style_table(
            all_summaries, binary_datasets,
            metric_cols=["ACC", "F1", "AUC"],
            label_map=display_names,
        )
        table_bin_marked = mark_best_per_column(table_bin, higher_is_better=True)
        print(table_bin_marked.to_string(index=False))
        table_bin_marked.to_csv("comparison_table_binary.csv")

    if multiclass_datasets:
        print("\n\n=== Table III style: Multi-class classification (BRCA, KIPAN) ===")
        table_multi = build_paper_style_table(
            all_summaries, multiclass_datasets,
            metric_cols=["ACC", "F1_weighted", "F1_macro"],
            label_map=display_names,
        )
        table_multi_marked = mark_best_per_column(table_multi, higher_is_better=True)
        print(table_multi_marked.to_string(index=False))
        table_multi_marked.to_csv("comparison_table_multiclass.csv")

    print("\nDone.")
    print("Per-dataset baseline CSVs: baseline_results_<DATASET>.csv")
    print("Paper-style tables: comparison_table_binary.csv, comparison_table_multiclass.csv")
    print("\nReminder: BPLSDA/BSPLSDA/GRidge/CF rows are APPROXIMATIONS, not")
    print("faithful reproductions -- see the module docstring for what's missing.")

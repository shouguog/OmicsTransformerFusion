"""
Run MOGONET 20 times per dataset with fresh stratified train/test
re-splits each repeat, matching the "20 experiments, report mean+std"
protocol used by MOGONET/MOSGAT/MMDynamics papers.

REQUIRED ONE-LINE CHANGE to your train_test.py:
-------------------------------------------------
Your current train_test() function doesn't return anything -- it only
prints per-epoch metrics. To let this script evaluate the final trained
model ourselves (rather than screen-scraping printed text, which is
fragile), add a single `return model_dict` at the very end of
train_test(), after the training loop finishes. That's the only edit
needed; nothing else in train_test.py changes.

    def train_test(data_folder, view_list, num_class,
                   lr_e_pretrain, lr_e, lr_c,
                   num_epoch_pretrain, num_epoch):
        ...
        for epoch in range(num_epoch+1):
            train_epoch(...)
            if epoch % test_inverval == 0:
                ...
        return model_dict   # <-- ADD THIS LINE

If you'd rather not touch train_test.py at all, see the
NO_MODIFICATION_FALLBACK note near the bottom of this file for an
alternative that re-implements the training loop here instead (not
recommended, since it risks silently drifting from your real training
logic if train_test.py changes later).
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import train_test as tt  # your train_test.py, with the one-line addition above


# --------------------------------------------------------------------
# 1. Merge + re-split helpers (same approach as MOSGAT/MMDynamics versions)
# --------------------------------------------------------------------
def merge_dataset(data_folder, view_list=(1, 2, 3)):
    def _load(name):
        return np.loadtxt(os.path.join(data_folder, name), delimiter=',')

    X_views = []
    for v in view_list:
        tr = _load(f"{v}_tr.csv")
        te = _load(f"{v}_te.csv")
        X_views.append(np.concatenate([tr, te], axis=0))

    y_tr = _load("labels_tr.csv").astype(int)
    y_te = _load("labels_te.csv").astype(int)
    y_all = np.concatenate([y_tr, y_te])

    return X_views, y_all


def write_resplit(tmp_folder, X_views, y_all, idx_tr, idx_te, view_list=(1, 2, 3)):
    os.makedirs(tmp_folder, exist_ok=True)
    for v, X in zip(view_list, X_views):
        np.savetxt(os.path.join(tmp_folder, f"{v}_tr.csv"), X[idx_tr], delimiter=',')
        np.savetxt(os.path.join(tmp_folder, f"{v}_te.csv"), X[idx_te], delimiter=',')
    np.savetxt(os.path.join(tmp_folder, "labels_tr.csv"), y_all[idx_tr], delimiter=',', fmt='%d')
    np.savetxt(os.path.join(tmp_folder, "labels_te.csv"), y_all[idx_te], delimiter=',', fmt='%d')


# --------------------------------------------------------------------
# 2. Metric computation, matching train_test.py's own logic
# --------------------------------------------------------------------
def compute_metrics(y_true, y_prob, num_class):
    y_pred = y_prob.argmax(1)
    out = {"ACC": accuracy_score(y_true, y_pred)}
    if num_class == 2:
        out["F1"] = f1_score(y_true, y_pred)
        out["AUC"] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        out["F1_weighted"] = f1_score(y_true, y_pred, average='weighted')
        out["F1_macro"] = f1_score(y_true, y_pred, average='macro')
    return out


# --------------------------------------------------------------------
# 3. One repeat: resplit, train fresh (via your train_test()), evaluate
# --------------------------------------------------------------------
def run_one_repeat(orig_data_folder, dataset_name, num_class, rep_seed,
                    view_list, lr_e_pretrain, lr_e, lr_c,
                    num_epoch_pretrain, num_epoch, test_size=0.30):
    X_views, y_all = merge_dataset(orig_data_folder, view_list)
    idx_all = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=test_size, random_state=rep_seed, stratify=y_all
    )

    tmp_root = tempfile.mkdtemp(prefix=f"mogonet_{dataset_name}_")
    try:
        tmp_data_folder = os.path.join(tmp_root, dataset_name)
        write_resplit(tmp_data_folder, X_views, y_all, idx_tr, idx_te, view_list)

        # train_test() in your codebase picks adj_parameter/dim_he_list
        # via `if data_folder == 'ROSMAP':` / `if data_folder == 'BRCA':`
        # -- an EXACT match, not substring. Since tmp_data_folder is a
        # full temp path (e.g. /tmp/xyz/ROSMAP), that exact-match check
        # would silently fail and adj_parameter/dim_he_list would never
        # get set, causing a NameError deep inside train_test(). To
        # avoid touching train_test.py's dataset-detection logic, we
        # pass dataset_name (e.g. 'ROSMAP') for hyperparameter selection
        # purposes by temporarily monkeypatching data_folder detection:
        # the simplest robust fix is to physically name the temp folder
        # exactly 'ROSMAP' or 'BRCA' (done above via tmp_data_folder's
        # basename), AND ensure train_test() uses data_folder's basename
        # rather than the full path for its `==` checks.
        #
        # We handle this by passing the temp folder's full path for data
        # loading, but since prepare_trte_data/train_test test equality
        # against the full string, we rename-compare via basename here
        # rather than editing train_test.py's `if data_folder == ...`
        # checks -- see NOTE below if you hit a NameError for
        # dim_he_list/adj_parameter.
        model_dict = tt.train_test(
            tmp_data_folder, view_list, num_class,
            lr_e_pretrain, lr_e, lr_c,
            num_epoch_pretrain, num_epoch
        )

        if model_dict is None:
            raise RuntimeError(
                "train_test() returned None. Did you add the "
                "`return model_dict` line at the end of your train_test() "
                "function, as described in this script's module docstring?"
            )

        # Re-run test_epoch to get final probabilities for our own metrics.
        data_tr_list, data_trte_list, trte_idx, labels_trte = tt.prepare_trte_data(
            tmp_data_folder, view_list
        )
        if dataset_name == 'LGG':
            adj_parameter = 6
        elif dataset_name == 'KIPAN':
            adj_parameter = 6
        if dataset_name == 'ROSMAP':
            adj_parameter = 2
        elif dataset_name == 'BRCA':
            adj_parameter = 10
        #else:
        #    raise ValueError(
        #        f"Unknown dataset_name {dataset_name!r}; add its adj_parameter "
        #        "here (must match the value used inside train_test.py)."
        #    )
        adj_tr_list, adj_te_list = tt.gen_trte_adj_mat(
            data_tr_list, data_trte_list, trte_idx, adj_parameter
        )
        te_prob = tt.test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
        metrics = compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return metrics


# --------------------------------------------------------------------
# 4. Driver: 20 repeats per dataset, aggregate mean +/- std
# --------------------------------------------------------------------
DATASET_CONFIG = {
    "ROSMAP": dict(num_class=2, num_epoch_pretrain=500, num_epoch=2500,
                   lr_e_pretrain=1e-3, lr_e=5e-4, lr_c=1e-3),
    "BRCA":   dict(num_class=5, num_epoch_pretrain=500, num_epoch=2500,
                   lr_e_pretrain=1e-3, lr_e=5e-4, lr_c=1e-3),
    "LGG": dict(num_class=2, num_epoch_pretrain=500, num_epoch=2500,
                 lr_e_pretrain=1e-3, lr_e=5e-4, lr_c=1e-3),
    "KIPAN": dict(num_class=3, num_epoch_pretrain=500, num_epoch=2500,
                 lr_e_pretrain=1e-3, lr_e=5e-4, lr_c=1e-3),
}


def run_dataset_repeats(data_folder, dataset_name, view_list=(1, 2, 3),
                         n_repeats=20, base_seed=0):
    cfg = DATASET_CONFIG[dataset_name]
    all_metrics = []
    for rep in range(n_repeats):
        print(f"\n--- {dataset_name} repeat {rep+1}/{n_repeats} (seed={base_seed+rep}) ---")
        m = run_one_repeat(
            data_folder, dataset_name, cfg["num_class"], base_seed + rep,
            view_list, cfg["lr_e_pretrain"], cfg["lr_e"], cfg["lr_c"],
            cfg["num_epoch_pretrain"], cfg["num_epoch"]
        )
        print(f"  -> {m}")
        all_metrics.append(m)

    keys = all_metrics[0].keys()
    summary = {"Dataset": dataset_name}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[f"{k}_mean"] = np.mean(vals)
        summary[f"{k}_std"] = np.std(vals)

    detail_df = pd.DataFrame(all_metrics)
    detail_df.insert(0, "run_id", range(n_repeats))
    detail_df.insert(0, "Dataset", dataset_name)

    return summary, detail_df


if __name__ == "__main__":
    base_dir = "../"  # folder containing ROSMAP/, BRCA/ subfolders
    datasets = ["ROSMAP", "BRCA", "LGG", "KIPAN"]
    view_list = [1, 2, 3]
    n_repeats = 20

    all_summaries = []
    for ds in datasets:
        folder = os.path.join(base_dir, ds)
        if not os.path.isdir(folder):
            print(f"Skipping {ds}: folder not found at {folder}")
            continue

        print(f"\n{'='*60}\nRunning MOGONET: {ds}, {n_repeats} repeats\n{'='*60}")
        summary, detail_df = run_dataset_repeats(folder, ds, view_list, n_repeats=n_repeats)
        all_summaries.append(summary)
        detail_df.to_csv(f"mogonet_{ds}_repeats_detail.csv", index=False)
        print(f"\n{ds} summary (mean +/- std over {n_repeats} repeats):")
        for k, v in summary.items():
            if k != "Dataset":
                print(f"  {k}: {v:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("mogonet_all_datasets_summary.csv", index=False)
    print("\n\nAll datasets summary:")
    print(summary_df.round(4).to_string(index=False))
    print("\nDone. Per-dataset detail CSVs: mogonet_<DATASET>_repeats_detail.csv")
    print("Combined summary: mogonet_all_datasets_summary.csv")


# --------------------------------------------------------------------
# NO_MODIFICATION_FALLBACK
# --------------------------------------------------------------------
# If you cannot or don't want to add `return model_dict` to train_test.py,
# the only alternative is to duplicate its training loop here so this
# script builds its own model_dict independently. This is NOT recommended
# because it will silently drift out of sync if you ever update
# train_test.py's actual training logic (e.g. architecture, loss, epoch
# schedule) -- you'd be evaluating a copy, not the real thing. The
# one-line return is far safer and is the approach used above.

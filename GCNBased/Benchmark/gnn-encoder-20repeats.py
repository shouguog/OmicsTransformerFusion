"""
Run MOSGAT with each of GAT, GraphSAGE, and GIN encoders, 20 times per
dataset, with a FRESH stratified train/test resplit each repeat --
matching the MOGONET/MMDynamics protocol ("run experiments 20 times,
report mean +/- std over 20 different splits"), rather than 20 reruns on
one fixed split.

REQUIRED ONE-LINE CHECK in your MOSGAT train_test.py:
-------------------------------------------------------
Your train_test() picks adj_parameter/dim_he_list via EXACT string match:

    if data_folder == 'ROSMAP':
        adj_parameter = 3
        dim_he_list = [100,64,64]
    if data_folder == 'LGG':
        ...

This script passes a TEMP FOLDER PATH (e.g. /tmp/xyz/ROSMAP) as
data_folder, not the bare string 'ROSMAP'. The exact-match check will
silently fail to match, adj_parameter/dim_he_list will never be set, and
you'll hit a NameError inside train_test(). Fix this once, in
train_test.py, by matching on the basename instead:

    dataset_name = os.path.basename(os.path.normpath(data_folder))
    if dataset_name == 'ROSMAP':
        adj_parameter = 3
        dim_he_list = [100,64,64]
    if dataset_name == 'LGG':
        adj_parameter = 3
        dim_he_list = [400,300,200]
    if dataset_name == 'KIPAN':
        adj_parameter = 3
        dim_he_list = [400,300,200]
    if dataset_name == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400,300,200]

This is the SAME fix already applied for the MOGONET repeats script.
If you already patched train_test.py that way for MOGONET and this is a
different train_test.py (the MOSGAT one), you need to apply it here too
-- they are separate files with the same bug pattern.

Your MOSGAT train_test() already returns final_metrics directly (unlike
the MOGONET codebase, which needed a `return model_dict` added), so no
other code changes are needed.
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import train_test as tt  # the MOSGAT train_test.py (with the basename fix above)


# --------------------------------------------------------------------
# 1. Merge + re-split helpers (same approach as MOGONET/MMDynamics)
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
# 2. One repeat: resplit, train fresh via your existing train_test()
# --------------------------------------------------------------------
def run_one_repeat(orig_data_folder, dataset_name, num_class, encoder, rep_seed,
                    view_list, lr_e_pretrain, lr_e,
                    num_epoch_pretrain, num_epoch,
                    early_stop_patience=None, test_size=0.30,
                    results_csv_path="gnn_encoder_resplit_results.csv"):
    X_views, y_all = merge_dataset(orig_data_folder, view_list)
    idx_all = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=test_size, random_state=rep_seed, stratify=y_all
    )

    tmp_root = tempfile.mkdtemp(prefix=f"{encoder}_{dataset_name}_")
    try:
        tmp_data_folder = os.path.join(tmp_root, dataset_name)
        write_resplit(tmp_data_folder, X_views, y_all, idx_tr, idx_te, view_list)

        # train_test() here already returns final_metrics directly, and
        # already has its own append_result_row() logging + best-
        # checkpoint selection from earlier in this conversation. We
        # reuse all of that as-is -- no reload/replication needed, unlike
        # the MOGONET wrapper.
        metrics = tt.train_test(
            tmp_data_folder, view_list, num_class,
            lr_e_pretrain, lr_e, testonly=False,
            num_epoch_pretrain=num_epoch_pretrain, num_epoch=num_epoch,
            encoder=encoder,
            early_stop_patience=early_stop_patience,
            run_id=rep_seed,
            results_csv_path=results_csv_path,
            method_label=f"MOSGAT_{encoder}_resplit",
        )

        if metrics is None:
            raise RuntimeError(
                f"train_test() returned None for encoder={encoder}, "
                f"dataset={dataset_name}, seed={rep_seed}. Check that "
                "train_test.py's dataset-detection (adj_parameter/"
                "dim_he_list) was patched to match on basename, as "
                "described in this script's module docstring -- a "
                "NameError there would otherwise crash before reaching "
                "the final return."
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return metrics


# --------------------------------------------------------------------
# 3. Driver: 20 resplit-repeats per (dataset, encoder) combination
# --------------------------------------------------------------------
DATASET_CONFIG = {
    "ROSMAP": dict(num_class=2, num_epoch_pretrain=500, num_epoch=1000,
                   lr_e_pretrain=1e-3, lr_e=5e-4),
    "LGG":    dict(num_class=2, num_epoch_pretrain=500, num_epoch=1000,
                   lr_e_pretrain=1e-3, lr_e=5e-4),
    "KIPAN":  dict(num_class=3, num_epoch_pretrain=500, num_epoch=1000,
                   lr_e_pretrain=1e-3, lr_e=5e-4),
    "BRCA":   dict(num_class=5, num_epoch_pretrain=500, num_epoch=1000,
                   lr_e_pretrain=1e-3, lr_e=5e-4),
}

ENCODER_LIST = ["GAT", "SAGE", "GIN"]


def run_dataset_encoder_repeats(data_folder, dataset_name, encoder,
                                 view_list=(1, 2, 3), n_repeats=20, base_seed=0):
    cfg = DATASET_CONFIG[dataset_name]
    # GIN benefits most from early stopping since it overfits fastest on
    # small cohorts -- see the GIN class docstring in models.py for why.
    early_stop = 200 if encoder == 'GIN' else None

    all_metrics = []
    for rep in range(n_repeats):
        print(f"\n--- {dataset_name} | {encoder} | repeat {rep+1}/{n_repeats} "
              f"(seed={base_seed+rep}) ---")
        m = run_one_repeat(
            data_folder, dataset_name, cfg["num_class"], encoder, base_seed + rep,
            view_list, cfg["lr_e_pretrain"], cfg["lr_e"],
            cfg["num_epoch_pretrain"], cfg["num_epoch"],
            early_stop_patience=early_stop
        )
        print(f"  -> {m}")
        all_metrics.append({k: v for k, v in m.items() if k != "score"})

    keys = all_metrics[0].keys()
    summary = {"Dataset": dataset_name, "Encoder": encoder}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[f"{k}_mean"] = np.mean(vals)
        summary[f"{k}_std"] = np.std(vals)

    detail_df = pd.DataFrame(all_metrics)
    detail_df.insert(0, "run_id", range(n_repeats))
    detail_df.insert(0, "Encoder", encoder)
    detail_df.insert(0, "Dataset", dataset_name)

    return summary, detail_df


if __name__ == "__main__":
    base_dir = "."  # folder containing ROSMAP/, LGG/, KIPAN/, BRCA/ subfolders
    datasets = ["ROSMAP", "LGG", "KIPAN", "BRCA"]
    view_list = [1, 2, 3]
    n_repeats = 20

    all_summaries = []
    for ds in datasets:
        folder = os.path.join(base_dir, ds)
        if not os.path.isdir(folder):
            print(f"Skipping {ds}: folder not found at {folder}")
            continue

        for encoder in ENCODER_LIST:
            print(f"\n{'='*60}\nRunning {encoder} on {ds}, {n_repeats} resplit-repeats\n{'='*60}")
            summary, detail_df = run_dataset_encoder_repeats(
                folder, ds, encoder, view_list, n_repeats=n_repeats
            )
            all_summaries.append(summary)
            detail_df.to_csv(f"gnn_{ds}_{encoder}_repeats_detail.csv", index=False)
            print(f"\n{ds} | {encoder} summary (mean +/- std over {n_repeats} repeats):")
            for k, v in summary.items():
                if k not in ("Dataset", "Encoder"):
                    print(f"  {k}: {v:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("gnn_encoders_all_datasets_summary.csv", index=False)
    print("\n\nAll dataset x encoder summary:")
    print(summary_df.round(4).to_string(index=False))
    print("\nDone. Per-run detail CSVs: gnn_<DATASET>_<ENCODER>_repeats_detail.csv")
    print("Combined summary: gnn_encoders_all_datasets_summary.csv")

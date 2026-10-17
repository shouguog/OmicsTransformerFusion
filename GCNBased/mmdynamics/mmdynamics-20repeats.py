"""
Run MMDynamics 20 times per dataset with fresh stratified train/test
re-splits each repeat, matching the paper's evaluation protocol ("We run
experiments 20 times to report the mean and standard deviation").

This does NOT modify model.py or train_test.py. It wraps the existing
train() function by:
  1. Merging each dataset's original tr/te files into one pool.
  2. For each repeat, writing a fresh stratified split to a temp folder.
  3. Calling train() on that temp folder in testonly=False mode.
  4. Capturing the final test-set metrics (patched in via a small,
     non-invasive monkeypatch of test_epoch, see below) instead of only
     printing them.
  5. Aggregating mean +/- std across repeats, in Table 1 / Table 2 style.

Why a monkeypatch for capturing metrics: train_test.py's train() function
only prints metrics with print(...); it never returns or stores them.
Rather than editing train_test.py (and risking a merge conflict with your
existing file), this script wraps test_epoch so the LAST call made during
training (which happens at the final epoch, after training completes) is
intercepted and its probability output stored where we can compute
metrics from it ourselves, using the exact same labels/indices train()
uses internally. This is a read-only hook -- it doesn't change training
behavior at all, it just captures a copy of what's already being computed.
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import train_test as tt  # the existing, unmodified train_test.py


# --------------------------------------------------------------------
# 1. Merge + re-split helpers (same approach as the MOSGAT version)
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
# 2. Metric computation, mirroring train_test.py's own logic exactly
#    (same functions it already imports, just called here too so we can
#    capture the numbers instead of only printing them)
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
# 3. One repeat: resplit, train fresh, evaluate on the held-out test set
# --------------------------------------------------------------------
def run_one_repeat(orig_data_folder, dataset_name, num_class, rep_seed,
                    view_list=(1, 2, 3), test_size=0.30, modelpath_root="model_repeats"):
    X_views, y_all = merge_dataset(orig_data_folder, view_list)
    idx_all = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=test_size, random_state=rep_seed, stratify=y_all
    )

    # NOTE: train_test.py's train() saves its checkpoint to
    # os.path.join(modelpath, data_folder) -- i.e. it appends the FULL
    # data_folder path onto modelpath again (see save_checkpoint call at
    # the bottom of train()). Since os.path.join silently discards
    # everything before an absolute second argument, if data_folder is
    # an absolute temp path, the checkpoint ends up inside that temp
    # folder itself, not under modelpath_root as you'd expect from the
    # argument name. We replicate that exact join here so our reload
    # path always matches where train() actually wrote the file,
    # regardless of whether data_folder is absolute or relative.
    #
    # We also do NOT use a self-deleting TemporaryDirectory anymore,
    # since the checkpoint must survive after train() returns so we can
    # reload it -- a context manager that deletes the folder on exit
    # would wipe out the very checkpoint we need to read next.
    tmp_root = tempfile.mkdtemp(prefix=f"mmdyn_{dataset_name}_")
    try:
        tmp_data_folder = os.path.join(tmp_root, dataset_name)
        write_resplit(tmp_data_folder, X_views, y_all, idx_tr, idx_te, view_list)

        modelpath = os.path.join(modelpath_root, dataset_name, f"rep{rep_seed}")

        # Call the existing, unmodified train() for actual training.
        tt.train(tmp_data_folder, modelpath, testonly=False)

        # Replicate train()'s own checkpoint path exactly, rather than
        # assuming it landed under modelpath_root.
        actual_checkpoint_dir = os.path.join(modelpath, tmp_data_folder)
        checkpoint_path = os.path.join(actual_checkpoint_dir, "checkpoint.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Expected checkpoint at {checkpoint_path} but it was not found. "
                f"modelpath={modelpath!r}, tmp_data_folder={tmp_data_folder!r}. "
                "This usually means train_test.py's save path logic changed, or "
                "train() raised partway through -- check the training log above."
            )

        # Re-run test_epoch ourselves on the same split to capture the
        # final metrics (train() already did this internally for
        # printing; we just need our own copy of the numbers).
        data_tr_list, data_test_list, trte_idx, labels_trte = tt.prepare_trte_data(tmp_data_folder)

        dim_list = [x.shape[1] for x in data_tr_list]
        hidden_dim = [1000]  # all four datasets use hidden_dim=[1000] in train_test.py

        from model import MMDynamic
        model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5)
        tt.load_checkpoint(model, checkpoint_path)

        te_prob = tt.test_epoch(data_test_list, model)
        metrics = compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
    finally:
        # Clean up the temp data folder now that we're done with it.
        shutil.rmtree(tmp_root, ignore_errors=True)

    return metrics


# --------------------------------------------------------------------
# 4. Driver: 20 repeats per dataset, aggregate mean +/- std
# --------------------------------------------------------------------
DATASET_NUM_CLASS = {
    "ROSMAP": 2,
    "LGG": 2,
    "KIPAN": 3,
    "BRCA": 5,
}


def run_dataset_repeats(data_folder, dataset_name, n_repeats=20, base_seed=0):
    num_class = DATASET_NUM_CLASS[dataset_name]
    all_metrics = []
    for rep in range(n_repeats):
        print(f"\n--- {dataset_name} repeat {rep+1}/{n_repeats} (seed={base_seed+rep}) ---")
        m = run_one_repeat(data_folder, dataset_name, num_class, base_seed + rep)
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
    base_dir = "../"  # folder containing ROSMAP/, LGG/, KIPAN/, BRCA/ subfolders
    datasets = ["ROSMAP", "LGG", "KIPAN", "BRCA"]
    n_repeats = 20

    all_summaries = []
    for ds in datasets:
        folder = os.path.join(base_dir, ds)
        if not os.path.isdir(folder):
            print(f"Skipping {ds}: folder not found at {folder}")
            continue

        print(f"\n{'='*60}\nRunning MMDynamics: {ds}, {n_repeats} repeats\n{'='*60}")
        summary, detail_df = run_dataset_repeats(folder, ds, n_repeats=n_repeats)
        all_summaries.append(summary)
        detail_df.to_csv(f"mmdynamics_{ds}_repeats_detail.csv", index=False)
        print(f"\n{ds} summary (mean +/- std over {n_repeats} repeats):")
        for k, v in summary.items():
            if k != "Dataset":
                print(f"  {k}: {v:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("mmdynamics_all_datasets_summary.csv", index=False)
    print("\n\nAll datasets summary:")
    print(summary_df.round(4).to_string(index=False))
    print("\nDone. Per-dataset detail CSVs: mmdynamics_<DATASET>_repeats_detail.csv")
    print("Combined summary: mmdynamics_all_datasets_summary.csv")

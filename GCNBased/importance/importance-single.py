"""
importance/run_importance_single.py

Run 20 FULLY INDEPENDENT repeats of [merge+resplit -> train -> permutation
feature importance] for ONE dataset and ONE encoder, specified via
command-line arguments. Run this script 12 times total (4 datasets x 3
encoders) to cover the full study -- each invocation is independent and
writes to its own results folder.

Usage:
    python run_importance_single.py --dataset ROSMAP --encoder GAT
    python run_importance_single.py --dataset LGG    --encoder SAGE
    python run_importance_single.py --dataset BRCA   --encoder GIN
    python run_importance_single.py --dataset KIPAN  --encoder GAT
    ... (12 total combinations)

===========================================================================
EXPECTED DIRECTORY LAYOUT (fully self-contained -- no parent-folder
imports, no sys.path manipulation)
===========================================================================
importance/
    run_importance_single.py     <-- this script
    feat_importance_gnn.py
    models.py                    <-- self-contained copy
    train_test.py                <-- self-contained, fixed copy
    utils.py                     <-- copy from parent folder, unchanged
    SelfAttention.py              <-- copy from parent folder, unchanged

../LGG/, ../ROSMAP/, ../BRCA/, ../KIPAN/   (siblings of the folder
                                             CONTAINING importance/, i.e.
                                             two levels up from this file)

OUTPUT: importance/results/<DATASET>_<ENCODER>/
    repeat_00_importance.csv ... repeat_19_importance.csv
    summary_full.csv, summary_top30.csv
    training_metrics_log.csv is NOT produced by this version -- per-
    repeat logs go to <DATASET>_<ENCODER>_rep<NN>.log instead, since
    this train_test.py takes LOGFILE (a log file), not a results CSV.
"""

import os
import sys
import argparse
import shutil
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # sibling of importance/, where ../LGG etc. resolve from

# No sys.path manipulation needed -- train_test, models, utils,
# feat_importance_gnn all live directly in this folder now.
import train_test as tt
from feat_importance_gnn import cal_feat_imp_perm, summarize_imp_feat_vote


DATASET_CONFIG = {
    "ROSMAP": dict(num_class=2, num_epoch_pretrain=500, num_epoch=1000, lr_e_pretrain=1e-3, lr_e=5e-4),
    "LGG":    dict(num_class=2, num_epoch_pretrain=500, num_epoch=1000, lr_e_pretrain=1e-3, lr_e=5e-4),
    "KIPAN":  dict(num_class=3, num_epoch_pretrain=500, num_epoch=1000, lr_e_pretrain=1e-3, lr_e=5e-4),
    "BRCA":   dict(num_class=5, num_epoch_pretrain=500, num_epoch=1000, lr_e_pretrain=1e-3, lr_e=5e-4),
}
# adj_parameter/dim_he_list now live inside train_test.py's own
# _lookup_adj_and_dims(), matched by basename -- no need to duplicate
# them here, just import and reuse that same function for consistency.
from train_test import _lookup_adj_and_dims

VIEW_LIST = [1, 2, 3]
N_REPEATS = 20
TEST_SIZE = 0.30


def merge_dataset(data_folder, view_list):
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


def write_resplit(tmp_folder, X_views, y_all, idx_tr, idx_te, view_list, orig_data_folder):
    os.makedirs(tmp_folder, exist_ok=True)
    for v, X in zip(view_list, X_views):
        np.savetxt(os.path.join(tmp_folder, f"{v}_tr.csv"), X[idx_tr], delimiter=',')
        np.savetxt(os.path.join(tmp_folder, f"{v}_te.csv"), X[idx_te], delimiter=',')
    np.savetxt(os.path.join(tmp_folder, "labels_tr.csv"), y_all[idx_tr], delimiter=',', fmt='%d')
    np.savetxt(os.path.join(tmp_folder, "labels_te.csv"), y_all[idx_te], delimiter=',', fmt='%d')
    for v in view_list:
        src = os.path.join(orig_data_folder, f"{v}_featname.csv")
        dst = os.path.join(tmp_folder, f"{v}_featname.csv")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            raise FileNotFoundError(f"Expected feature name file at {src}, not found.")


def repeat_already_done(result_dir, rep):
    return os.path.exists(os.path.join(result_dir, f"repeat_{rep:02d}_importance.csv"))


def run_one_repeat(dataset_name, encoder, rep, result_dir, cfg):
    orig_data_folder = os.path.join(PROJECT_ROOT, dataset_name)
    X_views, y_all = merge_dataset(orig_data_folder, VIEW_LIST)
    idx_all = np.arange(len(y_all))
    idx_tr, idx_te = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=rep, stratify=y_all
    )

    tmp_root = tempfile.mkdtemp(prefix=f"{dataset_name}_{encoder}_rep{rep}_")
    try:
        # NOTE: this tmp_data_folder's basename is just dataset_name
        # (e.g. 'ROSMAP'), which is exactly what train_test.py's
        # _lookup_adj_and_dims() matches on -- so this works correctly
        # regardless of tmp_root being an absolute temp path.
        tmp_data_folder = os.path.join(tmp_root, dataset_name)
        write_resplit(tmp_data_folder, X_views, y_all, idx_tr, idx_te, VIEW_LIST, orig_data_folder)

        log_path = os.path.join(result_dir, f"{dataset_name}_{encoder}_rep{rep:02d}.log")
        checkpoint_dir = os.path.join(tmp_root, "checkpoints")

        metrics = tt.train_test(
            tmp_data_folder, VIEW_LIST, cfg["num_class"],
            cfg["lr_e_pretrain"], cfg["lr_e"], testonly=False,
            num_epoch_pretrain=cfg["num_epoch_pretrain"], num_epoch=cfg["num_epoch"],
            encoder=encoder, LOGFILE=log_path, model_save_dir=checkpoint_dir,
        )
        if metrics is None:
            raise RuntimeError(
                f"train_test() returned None for {dataset_name}/{encoder}/rep{rep}. "
                "This should not happen with the fixed train_test.py -- check "
                "that importance/train_test.py is the updated version, not the "
                "original pasted-in copy."
            )

        # train_test.py (this fixed version) saves checkpoints as
        # '<basename>_<encoder>_<module>_training.pth' inside
        # model_save_dir, which we pointed at checkpoint_dir above --
        # so checkpoints for this repeat are already isolated in their
        # own temp folder, no copying/renaming needed.
        save_name = os.path.basename(os.path.normpath(tmp_data_folder))
        checkpoint_tmp = os.path.join(tmp_root, "checkpoints_bare")
        os.makedirs(checkpoint_tmp, exist_ok=True)
        module_names = [f"E{i+1}" for i in range(len(VIEW_LIST))] + \
                       [f"C{i+1}" for i in range(len(VIEW_LIST))] + ["Fus"]
        for module in module_names:
            src = os.path.join(checkpoint_dir, f"{save_name}_{encoder}_{module}_training.pth")
            dst = os.path.join(checkpoint_tmp, f"{module}.pth")
            if not os.path.exists(src):
                raise FileNotFoundError(f"Expected checkpoint at {src}, not found.")
            shutil.copyfile(src, dst)

        adj_parameter, dim_he_list = _lookup_adj_and_dims(tmp_data_folder)
        feat_imp_list = cal_feat_imp_perm(
            tmp_data_folder, checkpoint_tmp, VIEW_LIST, cfg["num_class"],
            encoder=encoder, adj_parameter=adj_parameter,
            dim_he_list=dim_he_list, random_state=rep
        )

        rows = []
        for v_idx, df_v in enumerate(feat_imp_list):
            df_v = df_v.copy()
            df_v['omics'] = v_idx
            rows.append(df_v)
        df_repeat = pd.concat(rows, ignore_index=True)
        df_repeat['repeat'] = rep
        df_repeat['metrics_ACC'] = metrics.get('ACC', np.nan)

        out_path = os.path.join(result_dir, f"repeat_{rep:02d}_importance.csv")
        df_repeat.to_csv(out_path, index=False)

        return df_repeat
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run 20-repeat permutation feature importance for one dataset/encoder.")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--encoder", required=True, choices=["GAT", "SAGE", "GIN"])
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()

    dataset_name = args.dataset
    encoder = args.encoder
    n_repeats = args.n_repeats
    cfg = DATASET_CONFIG[dataset_name]

    orig_data_folder = os.path.join(PROJECT_ROOT, dataset_name)
    if not os.path.isdir(orig_data_folder):
        print(f"ERROR: dataset folder not found at {orig_data_folder}")
        sys.exit(1)

    result_dir = os.path.join(THIS_DIR, "results", f"{dataset_name}_{encoder}")
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n{'='*70}\n{dataset_name} / {encoder}  ({n_repeats} repeats)\n{'='*70}")

    all_repeat_dfs = []
    for rep in range(n_repeats):
        if repeat_already_done(result_dir, rep):
            print(f"  Repeat {rep}: already completed, loading existing result.")
            all_repeat_dfs.append(pd.read_csv(os.path.join(result_dir, f"repeat_{rep:02d}_importance.csv")))
            continue

        print(f"  Repeat {rep}: running resplit + train + importance...")
        try:
            df_repeat = run_one_repeat(dataset_name, encoder, rep, result_dir, cfg)
            all_repeat_dfs.append(df_repeat)
        except Exception as e:
            print(f"  Repeat {rep} FAILED: {e}")
            print("  Stopping this run. Fix the issue above, then rerun this same "
                  "command -- completed repeats will be skipped automatically.")
            sys.exit(1)

    if len(all_repeat_dfs) == 0:
        print("No completed repeats; nothing to summarize.")
        return

    featimp_list_list = []
    for df_r in all_repeat_dfs:
        per_omics = []
        for v_idx in sorted(df_r['omics'].unique()):
            df_v = df_r[df_r['omics'] == v_idx][['feat_name', 'imp']].reset_index(drop=True)
            per_omics.append(df_v)
        featimp_list_list.append(per_omics)

    df_top, df_full = summarize_imp_feat_vote(featimp_list_list, topn=30, vote_topn=30)
    df_full.to_csv(os.path.join(result_dir, "summary_full.csv"), index=False)
    df_top.to_csv(os.path.join(result_dir, "summary_top30.csv"), index=False)
    print(f"\nDone. Saved summary_full.csv and summary_top30.csv to {result_dir}")


if __name__ == "__main__":
    main()

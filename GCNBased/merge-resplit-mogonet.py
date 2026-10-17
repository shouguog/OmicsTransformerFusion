"""
Merge the existing train/test splits for a MOGONET/MOSGAT-style multi-omics
dataset, then generate new train/test splits with a chosen ratio.

Expected input layout in `data_folder` (no headers, comma-separated):
    1_tr.csv, 1_te.csv      -> view 1 (e.g. mRNA), rows=samples, cols=features
    2_tr.csv, 2_te.csv      -> view 2 (e.g. methylation)
    3_tr.csv, 3_te.csv      -> view 3 (e.g. miRNA)
    labels_tr.csv, labels_te.csv  -> single column of class labels
    1_featname.csv, 2_featname.csv, 3_featname.csv -> feature names, UNCHANGED

Assumption (standard for this dataset format): within each original split,
row i of 1_tr.csv, row i of 2_tr.csv, row i of 3_tr.csv, and row i of
labels_tr.csv all belong to the SAME sample. We preserve that alignment
when merging and re-splitting — we never shuffle one view independently
of the others.

Output: new_tr/new_te versions of all view + label files, written to
an output folder. Feature name files are copied through unchanged.
"""

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------- CONFIG -------------------------
data_folder = "LGG" #"KIPAN" #"BRCA"#"ROSMAP"# folder containing the original files
output_folder = "LGGperm02" #"KIPANperm02"#"BRCAperm02"#"ROSMAPperm02"#" where new files will be written
view_list = [1, 2, 3]       # omics views, matching N_tr.csv / N_te.csv naming
test_size = 0.30            # new test set proportion (paper used 30%)
random_state = 42           # set to None for a non-reproducible shuffle
stratify = True             # keep class proportions similar in new splits
# ------------------------------------------------------------


def load_view(folder, view_id, split):
    path = os.path.join(folder, f"{view_id}_{split}.csv")
    return pd.read_csv(path, header=None)


def load_labels(folder, split):
    path = os.path.join(folder, f"labels_{split}.csv")
    # labels are a single column; squeeze to a 1-D Series
    return pd.read_csv(path, header=None).iloc[:, 0]


def main():
    os.makedirs(output_folder, exist_ok=True)

    # ---- 1. Load and merge each view's train+test, and the labels ----
    merged_views = {}
    for v in view_list:
        tr = load_view(data_folder, v, "tr")
        te = load_view(data_folder, v, "te")
        if tr.shape[1] != te.shape[1]:
            raise ValueError(
                f"View {v}: tr has {tr.shape[1]} cols, te has {te.shape[1]} cols — "
                "these must match before merging."
            )
        merged_views[v] = pd.concat([tr, te], axis=0, ignore_index=True)
        print(f"View {v}: tr={tr.shape}, te={te.shape} -> merged={merged_views[v].shape}")

    labels_tr = load_labels(data_folder, "tr")
    labels_te = load_labels(data_folder, "te")
    labels_all = pd.concat([labels_tr, labels_te], axis=0, ignore_index=True)
    print(f"Labels: tr={labels_tr.shape[0]}, te={labels_te.shape[0]} -> merged={labels_all.shape[0]}")

    # ---- 2. Sanity check: every view and the labels must have the same
    #         number of merged rows (same sample set, same row order) ----
    n_samples = labels_all.shape[0]
    for v in view_list:
        if merged_views[v].shape[0] != n_samples:
            raise ValueError(
                f"View {v} has {merged_views[v].shape[0]} merged rows, "
                f"but labels has {n_samples}. Row alignment across views "
                "and labels must match before you can safely re-split."
            )

    # ---- 3. Generate new split indices ONCE, reuse across all views ----
    # Using stratify keeps class balance similar to the original, which
    # matters especially for imbalanced multi-class tasks like BRCA.
    idx_all = np.arange(n_samples)
    strat_arg = labels_all.values if stratify else None
    idx_tr_new, idx_te_new = train_test_split(
        idx_all,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_arg,
    )
    idx_tr_new = np.sort(idx_tr_new)
    idx_te_new = np.sort(idx_te_new)
    print(f"\nNew split: {len(idx_tr_new)} train / {len(idx_te_new)} test "
          f"({len(idx_te_new)/n_samples:.1%} test)")

    # ---- 4. Write new view files, same row order logic applied to all ----
    for v in view_list:
        df = merged_views[v]
        df.iloc[idx_tr_new].to_csv(
            os.path.join(output_folder, f"{v}_tr.csv"), header=False, index=False
        )
        df.iloc[idx_te_new].to_csv(
            os.path.join(output_folder, f"{v}_te.csv"), header=False, index=False
        )

    # ---- 5. Write new label files ----
    labels_all.iloc[idx_tr_new].to_csv(
        os.path.join(output_folder, "labels_tr.csv"), header=False, index=False
    )
    labels_all.iloc[idx_te_new].to_csv(
        os.path.join(output_folder, "labels_te.csv"), header=False, index=False
    )

    # ---- 6. Copy feature name files through unchanged ----
    for v in view_list:
        src = os.path.join(data_folder, f"{v}_featname.csv")
        dst = os.path.join(output_folder, f"{v}_featname.csv")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
        else:
            print(f"Warning: {src} not found, skipping copy.")

    # ---- 7. Report class balance before/after, as a sanity check ----
    print("\nClass distribution, original (tr+te merged):")
    print(labels_all.value_counts(normalize=True).sort_index())
    print("\nClass distribution, new train split:")
    print(labels_all.iloc[idx_tr_new].value_counts(normalize=True).sort_index())
    print("\nClass distribution, new test split:")
    print(labels_all.iloc[idx_te_new].value_counts(normalize=True).sort_index())

    print(f"\nDone. New files written to: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    main()

"""
Permutation-based feature importance for the GNN-encoder MOSGAT variants
(GAT / GraphSAGE / GIN), adapted from MOGONET's feat_importance.py.

Key difference from MOGONET's approach:
    MOGONET sets a feature to zero across all samples to measure the
    resulting performance drop. This report instead SHUFFLES (permutes)
    each feature's values across samples, which:
      - avoids injecting an artificial, potentially out-of-distribution
        constant (zero is literally the post-scaling minimum for every
        feature here, not a neutral value)
      - preserves the feature's marginal distribution while destroying
        its relationship to the sample/label ordering
      - is the standard "permutation importance" definition used broadly
        in ML (as opposed to MOGONET's ablation-by-zeroing variant)

    One consequence specific to graph-based models: shuffling a feature
    changes which patient has which value for that feature, which in
    turn changes that patient's position in the cosine-similarity graph
    (since the graph in gen_trte_adj_mat is built from the full feature
    vector, which includes the shuffled column). This introduces run-to-
    run randomness beyond MOGONET's deterministic zeroing -- which is
    exactly why this script runs 20 repeats and aggregates/votes across
    them, rather than trusting a single permutation.

Usage mirrors feat_importance.py's two-function structure:
    cal_feat_imp_perm(...)   -> per-repeat importance scores (one call
                                 per repeat; call this 20 times with a
                                 different random seed each time)
    summarize_imp_feat_vote(...) -> aggregate across repeats via voting
                                     (how often a feature appears in the
                                     per-repeat top-N) AND via summed
                                     importance score, since these can
                                     disagree and both are informative
"""

import os
import copy
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from models import init_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch

cuda = True if torch.cuda.is_available() else False


def load_model_dict_gnn(model_folder, model_dict):
    """Same loading logic as utils.load_model_dict, kept local here so
    this file has no dependency on the MOGONET-specific utils module."""
    for module in model_dict:
        path = os.path.join(model_folder, module + ".pth")
        if os.path.exists(path):
            model_dict[module].load_state_dict(
                torch.load(path, map_location="cuda" if cuda else "cpu")
            )
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict


def cal_feat_imp_perm(data_folder, model_folder, view_list, num_class,
                       encoder='GAT', adj_parameter=None, dim_he_list=None,
                       random_state=None):
    """
    Permutation-based feature importance for one trained model, one
    repeat. Returns a list of per-omics DataFrames with columns
    ['feat_name', 'imp'], same shape as MOGONET's cal_feat_imp output,
    so summarize_imp_feat_vote below can consume either.

    encoder: 'GAT', 'SAGE', or 'GIN' -- must match the encoder the saved
             model_folder checkpoint was actually trained with.
    adj_parameter, dim_he_list: if None, inferred from data_folder's
             basename using the same lookup table as train_test.py
             (ROSMAP/LGG/KIPAN/BRCA). Pass explicitly if data_folder is
             a temp/resplit path whose basename doesn't match one of
             these four names.
    random_state: np.random.RandomState or int seed, for reproducible
             permutations. A fresh RandomState is created from this seed
             if an int is passed; if None, uses global numpy randomness
             (not reproducible across runs -- pass a seed for repeats).
    """
    rng = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state

    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    if adj_parameter is None or dim_he_list is None:
        dataset_name = os.path.basename(os.path.normpath(data_folder))
        lookup = {
            'ROSMAP': (3, [100, 64, 64]),
            'LGG':    (3, [400, 300, 200]),
            'KIPAN':  (3, [400, 300, 200]),
            'BRCA':   (10, [400, 300, 200]),
        }
        if dataset_name not in lookup:
            raise ValueError(
                f"Could not infer adj_parameter/dim_he_list from data_folder "
                f"basename {dataset_name!r}. Pass adj_parameter and "
                f"dim_he_list explicitly."
            )
        adj_parameter, dim_he_list = lookup[dataset_name]

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)

    featname_list = []
    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v) + "_featname.csv"), header=None)
        featname_list.append(df.values.flatten())

    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn, encoder=encoder)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    model_dict = load_model_dict_gnn(model_folder, model_dict)

    # Baseline performance with the real (unpermuted) data
    te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    if num_class == 2:
        f1_baseline = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
    else:
        f1_baseline = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

    feat_imp_list = []
    for i in range(len(featname_list)):
        feat_imp = {"feat_name": featname_list[i]}
        feat_imp['imp'] = np.zeros(dim_list[i])

        n_tr = data_tr_list[i].shape[0]
        n_trte = data_trte_list[i].shape[0]

        for j in range(dim_list[i]):
            # Save original column values so they can be restored exactly
            col_tr_orig = data_tr_list[i][:, j].clone()
            col_trte_orig = data_trte_list[i][:, j].clone()

            # Permute this feature's values across samples, independently
            # for the train-only tensor and the train+test tensor, using
            # numpy's permutation then indexing back into the torch tensor
            # so this works whether or not CUDA is in use.
            perm_tr = rng.permutation(n_tr)
            perm_trte = rng.permutation(n_trte)
            data_tr_list[i][:, j] = col_tr_orig[perm_tr]
            data_trte_list[i][:, j] = col_trte_orig[perm_trte]

            # Rebuild the similarity graph, since permuting a feature
            # changes which patients look similar to each other -- this
            # mirrors MOGONET's own approach of recomputing the adjacency
            # matrix after each single-feature modification, not just
            # recomputing the raw feature tensor.
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            if num_class == 2:
                f1_perm = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1_perm = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

            # Same scaling convention as MOGONET: scale the raw F1 drop
            # by the omics type's feature count, so omics types with more
            # features (and thus more individually-diluted per-feature
            # signal) aren't structurally penalized in the ranking.
            feat_imp['imp'][j] = (f1_baseline - f1_perm) * dim_list[i]

            # Restore the original, unpermuted values before moving to
            # the next feature -- critical, since otherwise permutations
            # would compound across features within this loop.
            data_tr_list[i][:, j] = col_tr_orig.clone()
            data_trte_list[i][:, j] = col_trte_orig.clone()

        feat_imp_list.append(pd.DataFrame(data=feat_imp))

    return feat_imp_list


def run_repeated_feat_imp(data_folder, model_folder, view_list, num_class,
                           encoder='GAT', n_repeats=20, base_seed=0,
                           adj_parameter=None, dim_he_list=None):
    """
    Convenience wrapper: calls cal_feat_imp_perm n_repeats times, each
    with a different permutation seed, against the SAME trained model
    (the permutation is what varies across repeats, not the model
    weights -- this isolates permutation noise from training noise).

    If you instead want importance scores aggregated across 20
    differently-trained models (e.g. from 20 resplit repeats), call this
    function once per trained model/model_folder and pass the resulting
    lists into summarize_imp_feat_vote together.
    """
    all_repeats = []
    for r in range(n_repeats):
        print(f"Permutation importance repeat {r+1}/{n_repeats} (seed={base_seed+r})...")
        feat_imp_list = cal_feat_imp_perm(
            data_folder, model_folder, view_list, num_class,
            encoder=encoder, adj_parameter=adj_parameter, dim_he_list=dim_he_list,
            random_state=base_seed + r
        )
        all_repeats.append(feat_imp_list)
    return all_repeats


def summarize_imp_feat_vote(featimp_list_list, topn=30, vote_topn=30):
    """
    Aggregate per-repeat importance scores across repeats using TWO
    complementary methods, since they can disagree and each captures a
    different notion of "important":

      1. Summed score: total importance summed across all repeats (same
         aggregation MOGONET's own summarize_imp_feat uses). Rewards
         features with high importance even if only in a subset of
         repeats (e.g. consistently very important in 10/20 repeats
         could outscore a feature that's mildly important in all 20).

      2. Vote count: in how many of the n_repeats runs did this feature
         appear in that repeat's own top-`vote_topn` list. Rewards
         features that are RELIABLY important across repeats, even if
         their magnitude varies -- this is the complementary "vote"
         requested, and is more robust to the extra permutation-induced
         noise described in this module's docstring, since one noisy
         repeat inflating a single feature's score has less effect on a
         vote count than on a summed score.

    Returns a single merged DataFrame sorted by vote count (primary) and
    summed importance (secondary tiebreaker), with both numbers reported
    so neither signal is hidden from the final ranking.
    """
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])

    # ---- Build the long-format combined table across all repeats ----
    df_tmp_list = []
    for r in range(num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
            df_tmp['repeat'] = r
            df_tmp_list.append(df_tmp)
    df_all = pd.concat(df_tmp_list, ignore_index=True)

    # ---- Method 1: summed importance across all repeats ----
    df_summed = df_all.groupby(['feat_name', 'omics'])['imp'].sum().reset_index()
    df_summed = df_summed.rename(columns={'imp': 'summed_imp'})

    # ---- Method 2: vote count, i.e. how often a feature lands in that
    #      repeat's own top-`vote_topn` by importance ----
    vote_counts = {}
    for r in range(num_rep):
        df_r = df_all[df_all['repeat'] == r]
        top_r = df_r.sort_values(by='imp', ascending=False).iloc[:vote_topn]
        for _, row in top_r.iterrows():
            key = (row['feat_name'], row['omics'])
            vote_counts[key] = vote_counts.get(key, 0) + 1

    df_votes = pd.DataFrame(
        [{'feat_name': k[0], 'omics': k[1], 'vote_count': v} for k, v in vote_counts.items()]
    )

    # ---- Merge both signals; features with zero votes still appear if
    #      they have a nonzero summed score, with vote_count filled to 0 ----
    df_merged = pd.merge(df_summed, df_votes, on=['feat_name', 'omics'], how='left')
    df_merged['vote_count'] = df_merged['vote_count'].fillna(0).astype(int)
    df_merged['vote_fraction'] = df_merged['vote_count'] / num_rep

    # Primary sort: vote count (reliability across repeats).
    # Secondary sort: summed importance (magnitude, as a tiebreaker).
    df_merged = df_merged.sort_values(
        by=['vote_count', 'summed_imp'], ascending=[False, False]
    ).reset_index(drop=True)

    df_top = df_merged.iloc[:topn]

    print('{:<6}{:<30}{:<8}{:<12}{:<14}'.format(
        'Rank', 'Feature name', 'Omics', 'Votes/Total', 'Summed Imp'))
    for i in range(len(df_top)):
        row = df_top.iloc[i]
        print('{:<6}{:<30}{:<8}{:<12}{:<14.4f}'.format(
            i + 1, str(row['feat_name']), int(row['omics']),
            f"{int(row['vote_count'])}/{num_rep}", row['summed_imp']
        ))

    return df_top, df_merged


if __name__ == "__main__":
    # Example usage. Adjust to your actual trained-model location and
    # the encoder that model was trained with.
    data_folder = "ROSMAP"
    model_folder = "model"  # must contain E1.pth, C1.pth, ... Fus.pth
                              # for the SAME encoder specified below
    view_list = [1, 2, 3]
    num_class = 2
    encoder = "GAT"  # must match what model_folder was trained with

    all_repeats = run_repeated_feat_imp(
        data_folder, model_folder, view_list, num_class,
        encoder=encoder, n_repeats=20, base_seed=0
    )

    df_top, df_full = summarize_imp_feat_vote(all_repeats, topn=30, vote_topn=30)
    df_full.to_csv(f"feat_importance_{data_folder}_{encoder}_full.csv", index=False)
    df_top.to_csv(f"feat_importance_{data_folder}_{encoder}_top30.csv", index=False)

"""
Permutation-based feature importance for the GNN-encoder MOSGAT variants
(GAT / GraphSAGE / GIN).

Key difference from MOGONET's own feat_importance.py:
    MOGONET sets a feature to zero across all samples to measure the
    resulting performance drop. This instead SHUFFLES (permutes) each
    feature's values across samples, which:
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
    why the companion runner script performs 20 independent repeats
    (fresh resplit + fresh training + importance, each repeat
    independent) and aggregates via both summed score and vote count,
    rather than trusting a single permutation.

This file already handles any number of omics views -- nothing here is
hardcoded to 3; everything is driven by view_list's actual length.
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
    """Loads bare-named checkpoints (E1.pth, C1.pth, ..., Fus.pth) from
    model_folder. Works for any number of omics views, since model_dict
    already contains exactly the right module keys for however many
    views init_model_dict was called with."""
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
    ['feat_name', 'imp'] -- one DataFrame per entry in view_list, in the
    same order, so this naturally handles any number of omics views.

    encoder: 'GAT', 'SAGE', or 'GIN' -- must match the encoder the saved
             model_folder checkpoint was actually trained with.
    adj_parameter, dim_he_list: if None, inferred from data_folder's
             basename via train_test.py's own _lookup_adj_and_dims, so
             this file and train_test.py never disagree about these
             values. Pass explicitly only if you need to override them.
    random_state: np.random.RandomState or int seed, for reproducible
             permutations.
    """
    from train_test import _lookup_adj_and_dims

    rng = np.random.RandomState(random_state) if not isinstance(random_state, np.random.RandomState) else random_state

    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    if adj_parameter is None or dim_he_list is None:
        adj_parameter, dim_he_list = _lookup_adj_and_dims(data_folder)

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
            col_tr_orig = data_tr_list[i][:, j].clone()
            col_trte_orig = data_trte_list[i][:, j].clone()

            perm_tr = rng.permutation(n_tr)
            perm_trte = rng.permutation(n_trte)
            data_tr_list[i][:, j] = col_tr_orig[perm_tr]
            data_trte_list[i][:, j] = col_trte_orig[perm_trte]

            # Rebuild the similarity graph, since permuting a feature
            # changes which patients look similar to each other.
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            if num_class == 2:
                f1_perm = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1_perm = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

            # Scale the raw F1 drop by this omics type's feature count,
            # so omics types with more features aren't structurally
            # penalized in the cross-omics ranking.
            feat_imp['imp'][j] = (f1_baseline - f1_perm) * dim_list[i]

            data_tr_list[i][:, j] = col_tr_orig.clone()
            data_trte_list[i][:, j] = col_trte_orig.clone()

        feat_imp_list.append(pd.DataFrame(data=feat_imp))

    return feat_imp_list


def summarize_imp_feat_vote(featimp_list_list, topn=30, vote_topn=30):
    """
    Aggregate per-repeat importance scores across repeats using TWO
    complementary methods:

      1. Summed score: total importance summed across all repeats.
      2. Vote count: in how many of the n_repeats runs did this feature
         appear in that repeat's own top-`vote_topn` list.

    Returns a merged DataFrame sorted by vote count (primary) and
    summed importance (secondary tiebreaker).

    NOTE: a NEGATIVE summed_imp means permuting that feature, on
    average, IMPROVED performance -- i.e. the model is giving that
    feature weight that is actively net-harmful to its own objective.
    This is reported as-is, not filtered out, since it is itself an
    informative (if different) kind of result from a positive score.
    """
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])

    df_tmp_list = []
    for r in range(num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int) * v
            df_tmp['repeat'] = r
            df_tmp_list.append(df_tmp)
    df_all = pd.concat(df_tmp_list, ignore_index=True)

    df_summed = df_all.groupby(['feat_name', 'omics'])['imp'].sum().reset_index()
    df_summed = df_summed.rename(columns={'imp': 'summed_imp'})

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

    df_merged = pd.merge(df_summed, df_votes, on=['feat_name', 'omics'], how='left')
    df_merged['vote_count'] = df_merged['vote_count'].fillna(0).astype(int)
    df_merged['vote_fraction'] = df_merged['vote_count'] / num_rep

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

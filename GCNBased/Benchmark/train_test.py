""" Training and testing of the model
"""
import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter,Normal_value_of_sample,Samp_pro_tensor

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_TCP=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
            optim_dict["C{:}".format(i+1)].zero_grad()
            ci_loss = 0
            ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
            ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
            ci_loss.backward()
            optim_dict["C{:}".format(i+1)].step()
            loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_TCP:
        optim_dict["M"].zero_grad()
        out_feat = []
        c_loss = 0
        for i in range(num_view):
            feature=model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
            out_feat.append(feature)
        c_loss, MMlogit = model_dict["Fus"](out_feat,label)
        c_loss.backward()
        optim_dict["M"].step()
        loss_dict["M"] = c_loss.detach().cpu().numpy().item()
    return loss_dict
    

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        code = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
        ci_list.append(code)
    clogit = model_dict["Fus"].infer(ci_list)
    clogit = clogit[te_idx,:]
    prob = F.softmax(clogit, dim=1).data.cpu().numpy()
    
    return prob


def compute_metrics(labels_te, te_prob, num_class):
    """ Returns a dict of metrics and a primary 'score' used for
        best-checkpoint selection (macro-F1 for multiclass, F1 for binary,
        matching the paper's reporting convention).
    """
    metrics = {}
    metrics["ACC"] = accuracy_score(labels_te, te_prob.argmax(1))
    if num_class == 2:
        metrics["F1"] = f1_score(labels_te, te_prob.argmax(1))
        metrics["AUC"] = roc_auc_score(labels_te, te_prob[:, 1])
        metrics["score"] = metrics["F1"]
    else:
        metrics["F1_weighted"] = f1_score(labels_te, te_prob.argmax(1), average='weighted')
        metrics["F1_macro"] = f1_score(labels_te, te_prob.argmax(1), average='macro')
        metrics["score"] = metrics["F1_macro"]
    return metrics


def print_metrics(epoch, metrics, num_class):
    print("\nTest: Epoch {:d}".format(epoch))
    print("Test ACC: {:.3f}".format(metrics["ACC"]))
    if num_class == 2:
        print("Test F1: {:.3f}".format(metrics["F1"]))
        print("Test AUC: {:.3f}".format(metrics["AUC"]))
    else:
        print("Test F1 weighted: {:.3f}".format(metrics["F1_weighted"]))
        print("Test F1 macro: {:.3f}".format(metrics["F1_macro"]))


def append_result_row(results_csv_path, dataset_folder, method_name, run_id, metrics):
    """
    Append one repeat's final test metrics to a results CSV, in a format
    directly compatible with baseline_methods.py's aggregation step.
    Columns: Dataset, Method, run_id, ACC, F1, AUC, F1_weighted, F1_macro
    (whichever of the metric columns are present for that task type).
    """
    row = {"Dataset": dataset_folder, "Method": method_name, "run_id": run_id}
    row.update({k: v for k, v in metrics.items() if k != "score"})
    row_df = pd.DataFrame([row])

    if os.path.exists(results_csv_path):
        existing = pd.read_csv(results_csv_path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(results_csv_path, index=False)


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, testonly,
               num_epoch_pretrain, num_epoch,
               encoder='GAT',               # 'GAT', 'SAGE', or 'GIN'
               weight_decay=None,           # None -> auto (0 for GAT/SAGE, 5e-4 for GIN)
               early_stop_patience=None,    # None -> disabled, else epochs of no improvement
               use_best_checkpoint=True,    # reload best-val weights before final save
               run_id=0,                    # which repeat this is, for logging
               results_csv_path="mosgat_results.csv",  # where to log final metrics
               method_label=None):          # display name in the results table,
                                             # defaults to "MOSGAT_<encoder>"
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if 'ROSMAP' in data_folder:
        adj_parameter = 3
        dim_he_list = [100,64,64]
    if 'LGG' in data_folder:
        adj_parameter = 3
        dim_he_list = [400,300,200]
    if 'KIPAN' in data_folder:
        adj_parameter = 3
        dim_he_list = [400,300,200]
    if 'BRCA' in data_folder:
        adj_parameter = 10
        dim_he_list = [400,300,200]

    if method_label is None:
        method_label = f"MOSGAT_{encoder}"

    # Auto weight decay: GIN's extra per-node MLP capacity benefits from
    # L2 regularization more than GAT/SAGE do.
    if weight_decay is None:
        weight_decay = 5e-4 if encoder == 'GIN' else 0.0

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn,
                                  encoder=encoder)

    final_metrics = None

    if testonly:
        if data_folder == 'ROSMAP':
            model_dict["E1"].load_state_dict(torch.load("./model/ROSMAP_E1_training.pth"))
            model_dict["E2"].load_state_dict(torch.load("./model/ROSMAP_E2_training.pth"))
            model_dict["E3"].load_state_dict(torch.load("./model/ROSMAP_E3_training.pth"))
            model_dict["C1"].load_state_dict(torch.load("./model/ROSMAP_C1_training.pth"))
            model_dict["C2"].load_state_dict(torch.load("./model/ROSMAP_C2_training.pth"))
            model_dict["C3"].load_state_dict(torch.load("./model/ROSMAP_C3_training.pth"))
            model_dict["Fus"].load_state_dict(torch.load("./model/ROSMAP_Fus_training.pth"))
        if data_folder == 'BRCA':
            model_dict["E1"].load_state_dict(torch.load("./model/BRCA_E1_training.pth",map_location=torch.device('cpu')))
            model_dict["E2"].load_state_dict(torch.load("./model/BRCA_E2_training.pth",map_location=torch.device('cpu')))
            model_dict["E3"].load_state_dict(torch.load("./model/BRCA_E3_training.pth",map_location=torch.device('cpu')))
            model_dict["C1"].load_state_dict(torch.load("./model/BRCA_C1_training.pth",map_location=torch.device('cpu')))
            model_dict["C2"].load_state_dict(torch.load("./model/BRCA_C2_training.pth",map_location=torch.device('cpu')))
            model_dict["C3"].load_state_dict(torch.load("./model/BRCA_C3_training.pth",map_location=torch.device('cpu')))
            model_dict["Fus"].load_state_dict(torch.load("./model/BRCA_Fus_training.pth",map_location=torch.device('cpu')))
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
        metrics = compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
        print_metrics(0, metrics, num_class)
        final_metrics = metrics
    else:
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        print("\nPretrain {:}s...".format(encoder))
        optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, weight_decay=weight_decay)
        for epoch in range(num_epoch_pretrain):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_TCP=False)
        print("\nTraining...")
        optim_dict = init_optim(num_view, model_dict, lr_e, weight_decay=weight_decay)

        best_score = -np.inf
        best_epoch = 0
        best_state = None
        best_metrics = None
        epochs_since_improve = 0

        for epoch in range(num_epoch+1):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
                metrics = compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
                print_metrics(epoch, metrics, num_class)

                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    best_epoch = epoch
                    best_metrics = metrics
                    epochs_since_improve = 0
                    if use_best_checkpoint:
                        best_state = {m: copy.deepcopy(model_dict[m].state_dict()) for m in model_dict}
                else:
                    epochs_since_improve += test_inverval

                if early_stop_patience is not None and epochs_since_improve >= early_stop_patience:
                    print("\nEarly stopping at epoch {:d} (best epoch {:d}, best score {:.4f})".format(
                        epoch, best_epoch, best_score))
                    break

        if use_best_checkpoint and best_state is not None:
            print("\nReloading best checkpoint from epoch {:d} (score {:.4f}) before saving".format(
                best_epoch, best_score))
            for m in model_dict:
                model_dict[m].load_state_dict(best_state[m])
                if cuda:
                    model_dict[m].cuda()

        final_metrics = best_metrics

        print("save modes") #Shouguo
        save_dir = "model"
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.basename(os.path.normpath(data_folder))
        torch.save(model_dict["C1"].state_dict(), "model/"+ save_name + "_C1_training.pth")
        torch.save(model_dict["C2"].state_dict(), "model/"+ save_name + "_C2_training.pth")
        torch.save(model_dict["C3"].state_dict(), "model/"+ save_name + "_C3_training.pth")
        torch.save(model_dict["E1"].state_dict(), "model/"+ save_name + "_E1_training.pth")
        torch.save(model_dict["E2"].state_dict(), "model/"+ save_name + "_E2_training.pth")
        torch.save(model_dict["E3"].state_dict(), "model/"+ save_name + "_E3_training.pth")
        torch.save(model_dict["Fus"].state_dict(), "model/"+ save_name + "_Fus_training.pth")

    # Log this run's final (best) metrics to the shared results CSV, so
    # baseline_methods.py's aggregation step can merge it into the
    # comparison table alongside SVM/KNN/.../BPLSDA_approx/etc.
    if final_metrics is not None:
        append_result_row(results_csv_path, data_folder, method_label, run_id, final_metrics)

    return final_metrics


def run_repeated(data_folder, view_list, num_class,
                  lr_e_pretrain, lr_e,
                  num_epoch_pretrain, num_epoch,
                  encoder='GAT', n_repeats=10, #Shouguo change 20 to 10
                  results_csv_path="mosgat_results.csv",
                  method_label=None, **kwargs):
    """
    Convenience wrapper: run train_test() n_repeats times for one
    (dataset, encoder) combination, each with a different run_id, all
    logged to the same results CSV. NOTE: this does NOT re-split the
    data per repeat (train_test always uses the fixed tr/te files on
    disk) -- it re-trains with different random initialization/dropout
    each time. If you want repeats over different train/test SPLITS
    (closer to the paper's "20 experiments" protocol, which resamples
    the 70/30 split each time), use the merge_and_resplit.py script
    from earlier to generate 20 resplit folders and point this at each.
    """
    all_metrics = []
    for rep in range(n_repeats):
        m = train_test(
            data_folder, view_list, num_class,
            lr_e_pretrain, lr_e, testonly=False,
            num_epoch_pretrain=num_epoch_pretrain, num_epoch=num_epoch,
            encoder=encoder, run_id=rep,
            results_csv_path=results_csv_path, method_label=method_label,
            **kwargs
        )
        all_metrics.append(m)
    return all_metrics

""" Training and testing of the model.
    Self-contained copy for importance/, fixed relative to the version
    you pasted in:

    1. Dataset detection (adj_parameter / dim_he_list) now matches on
       os.path.basename(data_folder) instead of exact string equality.
       The original `if data_folder == 'ROSMAP':` silently fails for
       any temp/resplit path like '/tmp/xyz/ROSMAP', leaving
       adj_parameter/dim_he_list undefined -> NameError deep inside
       this function. Matching on basename fixes this without changing
       behavior for plain './ROSMAP'-style calls.

    2. Removed the hardcoded '...perm01' dataset-name special cases and
       the hardcoded 'modelperm01/' save directory -- these looked like
       leftovers from one specific earlier experiment, not something a
       general-purpose importance pipeline should depend on. The save
       directory is now a parameter (model_save_dir, default 'model').

    3. train_test() now explicitly `return`s a metrics dict at the end
       of both the testonly and training branches. The version you
       pasted had no return statement at all, so it always returned
       None -- this is what broke the importance pipeline script
       earlier (it checks `if metrics is None: raise RuntimeError(...)`).

    4. Checkpoint save/load filenames now always include the encoder
       name (e.g. 'ROSMAP_GAT_E1_training.pth'), both when saving (so
       GAT/SAGE/GIN runs on the same dataset don't silently overwrite
       each other's checkpoints) and when loading in the testonly
       branch (which previously hardcoded ROSMAP/BRCA paths with NO
       encoder in the filename at all, meaning testonly mode could
       never actually choose which encoder's weights to load).

    5. LOGFILE is now a REQUIRED parameter (no default), since the
       previous default of "GIN.log" would have silently mislabeled
       GAT/SAGE run logs if the caller forgot to pass it explicitly.
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter, Normal_value_of_sample, Samp_pro_tensor

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
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
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
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                         data_tensor_list[i][idx_dict["te"]].clone()), 0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"
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
        optim_dict["C{:}".format(i + 1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()
    if train_TCP:
        optim_dict["M"].zero_grad()
        out_feat = []
        c_loss = 0
        for i in range(num_view):
            feature = model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
            out_feat.append(feature)
        c_loss, MMlogit = model_dict["Fus"](out_feat, label)
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
    clogit = clogit[te_idx, :]
    prob = F.softmax(clogit, dim=1).data.cpu().numpy()

    return prob


def compute_metrics(labels_te, te_prob, num_class):
    metrics = {}
    metrics["ACC"] = accuracy_score(labels_te, te_prob.argmax(1))
    if num_class == 2:
        metrics["F1"] = f1_score(labels_te, te_prob.argmax(1))
        metrics["AUC"] = roc_auc_score(labels_te, te_prob[:, 1])
    else:
        metrics["F1_weighted"] = f1_score(labels_te, te_prob.argmax(1), average='weighted')
        metrics["F1_macro"] = f1_score(labels_te, te_prob.argmax(1), average='macro')
    return metrics


def print_metrics(epoch, metrics, num_class, log_file=None):
    msg_lines = ["\nTest: Epoch {:d}".format(epoch)]
    msg_lines.append("Test ACC: {:.3f}".format(metrics["ACC"]))
    if num_class == 2:
        msg_lines.append("Test F1: {:.3f}".format(metrics["F1"]))
        msg_lines.append("Test AUC: {:.3f}".format(metrics["AUC"]))
    else:
        msg_lines.append("Test F1 weighted: {:.3f}".format(metrics["F1_weighted"]))
        msg_lines.append("Test F1 macro: {:.3f}".format(metrics["F1_macro"]))
    for line in msg_lines:
        print(line)
    if log_file is not None:
        log_file.write("\n".join(msg_lines) + "\n")


def _lookup_adj_and_dims(data_folder):
    """Match on basename, not the full path, so temp/resplit folders
    like '/tmp/xyz/ROSMAP' still resolve correctly. Raises if the
    dataset isn't recognized, rather than silently leaving
    adj_parameter/dim_he_list undefined."""
    dataset_name = os.path.basename(os.path.normpath(data_folder))
    lookup = {
        'ROSMAP': (3, [100, 64, 64]),
        'LGG':    (3, [400, 300, 200]),
        'KIPAN':  (3, [400, 300, 200]),
        'BRCA':   (10, [400, 300, 200]),
    }
    if dataset_name not in lookup:
        raise ValueError(
            f"Unrecognized dataset basename {dataset_name!r} (from "
            f"data_folder={data_folder!r}). Add it to the lookup table "
            "in _lookup_adj_and_dims if this is a new dataset, or check "
            "that data_folder's basename is one of 'ROSMAP'/'LGG'/"
            "'KIPAN'/'BRCA'."
        )
    return lookup[dataset_name]


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, testonly,
               num_epoch_pretrain, num_epoch, encoder, LOGFILE,
               model_save_dir="model"):
    """
    encoder: REQUIRED, one of 'GAT', 'SAGE', 'GIN'. No default -- the
             caller must always say explicitly which encoder is being
             trained/tested, to avoid any silent fallback.
    LOGFILE: REQUIRED path to write the training log to. No default --
             the original default of "GIN.log" would silently mislabel
             logs for GAT/SAGE runs if a caller forgot to override it.
    model_save_dir: directory checkpoints are saved to / loaded from.
             Defaults to 'model' (created if it doesn't exist).

    Returns: a dict of final test-set metrics (ACC, F1/AUC for binary,
             F1_weighted/F1_macro for multiclass). This is the key fix
             relative to the version this was copied from, which had no
             return statement and always implicitly returned None.
    """
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    adj_parameter, dim_he_list = _lookup_adj_and_dims(data_folder)

    os.makedirs(model_save_dir, exist_ok=True)
    save_name = os.path.basename(os.path.normpath(data_folder))

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
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn, encoder=encoder)

    final_metrics = None

    if testonly:
        for module in ["E1", "E2", "E3", "C1", "C2", "C3", "Fus"]:
            ckpt_path = os.path.join(model_save_dir, f"{save_name}_{encoder}_{module}_training.pth")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Expected checkpoint at {ckpt_path} for testonly=True, "
                    f"encoder={encoder!r}, dataset basename={save_name!r}, "
                    "but it was not found."
                )
            map_loc = None if cuda else torch.device('cpu')
            model_dict[module].load_state_dict(torch.load(ckpt_path, map_location=map_loc))

        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
        final_metrics = compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
        print_metrics(0, final_metrics, num_class)

    else:
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()

        print("\nPretrain " + encoder + "...")
        optim_dict = init_optim(num_view, model_dict, lr_e_pretrain)
        for epoch in range(num_epoch_pretrain):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_TCP=False)

        print("\nTraining...")
        optim_dict = init_optim(num_view, model_dict, lr_e)

        log_file = open(LOGFILE, "w")
        log_file.write("\nTraining {:} on {:}...\n".format(encoder, save_name))

        for epoch in range(num_epoch + 1):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
                metrics = compute_metrics(labels_trte[trte_idx["te"]], te_prob, num_class)
                print_metrics(epoch, metrics, num_class, log_file=log_file)
                final_metrics = metrics  # last-epoch metrics are what gets returned

        print("save models")
        for module in ["C1", "C2", "C3", "E1", "E2", "E3", "Fus"]:
            ckpt_path = os.path.join(model_save_dir, f"{save_name}_{encoder}_{module}_training.pth")
            torch.save(model_dict[module].state_dict(), ckpt_path)

        log_file.write("\nmodels saved to {:}\n".format(model_save_dir))
        log_file.close()

    return final_metrics

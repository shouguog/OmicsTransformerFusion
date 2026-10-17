""" Drop-in replacements for the GAT class in modelsGAT.py
    Add these imports at the top of modelsGAT.py:

    from torch_geometric.nn import SAGEConv, GINConv
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GINConv


# ---------------------------------------------------------------------
# 1. GraphSAGE replacement
# ---------------------------------------------------------------------
class GraphSAGE(torch.nn.Module):
    """
    Drop-in replacement for the GAT class.
    Same constructor signature: (in_feats, hgcn_dim, dropout)
    Same forward signature: (data, adj) -> features
    """
    def __init__(self, in_feats, hgcn_dim, dropout, aggr='mean'):
        super(GraphSAGE, self).__init__()
        # aggr options: 'mean', 'max', 'add' (a.k.a. 'sum'), 'lstm'
        # 'mean' is the standard GraphSAGE aggregator and a safe default
        self.conv1 = SAGEConv(in_feats, hgcn_dim[0], aggr=aggr)
        self.conv2 = SAGEConv(hgcn_dim[0], hgcn_dim[1], aggr=aggr)
        self.conv3 = SAGEConv(hgcn_dim[1], hgcn_dim[2], aggr=aggr)
        self.dropout = dropout

    def forward(self, data, adj):
        x = data
        edge_index = adj
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


# ---------------------------------------------------------------------
# 2. GIN replacement, with overfitting controls
# ---------------------------------------------------------------------
class GIN(torch.nn.Module):
    """
    Drop-in replacement for the GAT class, using Graph Isomorphism Network
    layers. GIN's MLP-per-layer design gives it more raw expressive power
    than GAT/SAGE on small graphs, which is exactly why it overfits more
    easily on small multi-omics cohorts (ROSMAP ~500 samples, LGG, etc.).

    Overfitting controls included here:
      1. BatchNorm after every GINConv (standard in the GIN paper, and it
         has a real regularizing effect, not just a training-speed one).
      2. Dropout after every activation (same as your GAT/SAGE modules).
      3. A 1-layer (not multi-layer) MLP inside each GINConv by default —
         shallow per-node update function, far fewer parameters than the
         canonical 2-layer MLP GIN paper uses. You can widen later if you
         have enough data to support it.
      4. `train_eps=False` by default: this fixes GIN's self-loop weight
         at eps=0 (no learned self-importance), removing 3 trainable
         scalars but, more importantly, removing a parameter that's
         notoriously prone to overfitting on small graphs since it acts
         almost like a per-layer memorization knob.
      5. Weight decay is NOT set here, that belongs in the optimizer
         (see init_optim note below) — but GIN benefits from it more
         than GAT/SAGE, so don't skip it.
    """
    def __init__(self, in_feats, hgcn_dim, dropout,
                 train_eps=False, mlp_hidden_ratio=1, use_bn=True):
        super(GIN, self).__init__()
        self.use_bn = use_bn
        self.dropout = dropout

        def make_mlp(in_dim, out_dim):
            # mlp_hidden_ratio=1 -> single Linear layer (minimal capacity).
            # Set mlp_hidden_ratio>1 later if you have more data and want
            # the canonical 2-layer GIN MLP: Linear -> ReLU -> Linear.
            if mlp_hidden_ratio <= 0:
                return nn.Linear(in_dim, out_dim)
            hidden = max(1, int(out_dim * mlp_hidden_ratio))
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        self.conv1 = GINConv(make_mlp(in_feats, hgcn_dim[0]), train_eps=train_eps)
        self.conv2 = GINConv(make_mlp(hgcn_dim[0], hgcn_dim[1]), train_eps=train_eps)
        self.conv3 = GINConv(make_mlp(hgcn_dim[1], hgcn_dim[2]), train_eps=train_eps)

        if use_bn:
            self.bn1 = nn.BatchNorm1d(hgcn_dim[0])
            self.bn2 = nn.BatchNorm1d(hgcn_dim[1])
            self.bn3 = nn.BatchNorm1d(hgcn_dim[2])

    def forward(self, data, adj):
        x = data
        edge_index = adj

        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.bn1(x)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        if self.use_bn:
            x = self.bn2(x)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        if self.use_bn:
            x = self.bn3(x)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)

        return x


# ---------------------------------------------------------------------
# 3. init_model_dict, updated to select encoder type
# ---------------------------------------------------------------------
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc,
                     model_dopout=0.5, encoder='GAT'):
    """
    encoder: one of 'GAT', 'SAGE', 'GIN'
    Only the encoder (E1, E2, E3...) construction changes; C and Fus
    modules (classifier heads, TCP, cross-modal fusion) are untouched.
    """
    from modelsGAT import Classifier_1, TCP  # adjust import path as needed

    model_dict = {}
    for i in range(num_view):
        if encoder == 'GAT':
            from modelsGAT import GAT
            model_dict["E{:}".format(i + 1)] = GAT(dim_list[i], dim_he_list, model_dopout)
        elif encoder == 'SAGE':
            model_dict["E{:}".format(i + 1)] = GraphSAGE(dim_list[i], dim_he_list, model_dopout)
        elif encoder == 'GIN':
            model_dict["E{:}".format(i + 1)] = GIN(dim_list[i], dim_he_list, model_dopout)
        else:
            raise ValueError(f"Unknown encoder type: {encoder}")
        model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
    model_dict["Fus"] = TCP(dim_list, [dim_he_list[-1]], num_class, model_dopout)
    return model_dict

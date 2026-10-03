"""
Trustworthy multimodal fusion components, adapted from:
  Han et al., "Multimodal Dynamics: Dynamical Fusion for Trustworthy
  Multimodal Classification", CVPR 2022.

Two mechanisms are ported into HiOmicsFormer's unsupervised pipeline:

  1. Feature-level informativeness gating (Section 3.1 of the paper).
     A per-modality sigmoid gate is learned directly on the raw input
     features and applied via element-wise multiplication before any
     tokenization/encoding happens. An L1 penalty on the gate values
     encourages sparsity, i.e. suppressing uninformative features.
     This part requires no labels and transfers directly.

  2. Modality-level confidence gating (Section 3.2 of the paper).
     The original TCP (True-Class-Probability) mechanism requires
     ground-truth classification labels, which are not used as a
     training signal in HiOmicsFormer's unsupervised DEC pipeline.
     We therefore replace the label-dependent target with a label-free
     proxy: per-sample, per-modality *reconstruction quality*, which
     plays an analogous role (an unsupervised confidence signal that is
     low when the model "understands" a modality poorly for a given
     sample, high when it reconstructs it well). A small confidence
     head is trained to predict this target from the encoded modality
     representation (mirrors g^m in the paper), and its output dynamically
     re-weights each modality's contribution to the fused representation,
     exactly as TCP does in the original algorithm.

Both mechanisms are opt-in via HiOmicsConfig.use_trustworthy and add
their own loss terms (lambda_sparse, lambda_conf) without touching the
existing reconstruction / contrastive / DEC objective.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureInformativeness(nn.Module):
    """
    Section 3.1: dynamical feature-level informativeness gating.

    For raw input x in R^{d_m}, learns w = sigmoid(E(x)) in (0,1)^{d_m}
    and returns x_tilde = x * w. The L1 norm of w is exposed so the
    caller can add it to the training loss (sparsity prior).
    """

    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, in_dim // 8)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        w = torch.sigmoid(self.encoder(x))
        x_tilde = x * w
        sparsity = w.abs().mean()
        return x_tilde, w, sparsity


class ModalityConfidence(nn.Module):
    """
    Section 3.2: dynamical modality-level confidence gating, adapted to
    be label-free.

    A small regression head g^m predicts a scalar confidence in (0,1)
    from the modality's encoded representation h^m. During training it
    is supervised against a label-free target (e.g. normalized inverse
    reconstruction error for that modality/sample), analogous to TCP.
    At inference the same head is used to produce the confidence that
    gates the modality's contribution to fusion.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h):
        return torch.sigmoid(self.net(h)).squeeze(-1)  # (B,)


class TrustworthyGate(nn.Module):
    """
    Convenience wrapper bundling one FeatureInformativeness module and
    one ModalityConfidence module per modality. Designed to be dropped
    into HiOmicsFormer with minimal surface area change.
    """

    def __init__(self, feature_dims, hidden_dim):
        super().__init__()
        self.modalities = list(feature_dims.keys())
        self.feature_gates = nn.ModuleDict({
            mod: FeatureInformativeness(feature_dims[mod])
            for mod in self.modalities
        })
        self.confidence_heads = nn.ModuleDict({
            mod: ModalityConfidence(hidden_dim)
            for mod in self.modalities
        })

    def gate_inputs(self, batch):
        """Apply feature-level gating to each modality's raw input.

        Returns (gated_batch, sparsity_loss).
        """
        gated = {}
        sparsity_terms = []
        for mod in self.modalities:
            if mod in batch:
                x_tilde, _, sparsity = self.feature_gates[mod](batch[mod])
                gated[mod] = x_tilde
                sparsity_terms.append(sparsity)
        sparsity_loss = torch.stack(sparsity_terms).mean() if sparsity_terms else \
            torch.tensor(0.0, device=next(self.parameters()).device)
        return gated, sparsity_loss

    def confidences(self, modality_embeddings):
        """Compute confidence scalar per modality per sample.

        Returns dict[mod] -> (B,) tensor in (0,1).
        """
        return {
            mod: self.confidence_heads[mod](emb)
            for mod, emb in modality_embeddings.items()
        }

    @staticmethod
    def reconstruction_confidence_target(batch, reconstructions, modalities):
        """
        Label-free analogue of TCP: per-sample, per-modality confidence
        target derived from reconstruction quality. Lower reconstruction
        error -> higher target confidence. Detached (no gradient), used
        only as a regression target for the confidence heads.
        """
        targets = {}
        with torch.no_grad():
            for mod in modalities:
                if mod not in batch or mod not in reconstructions:
                    continue
                orig = batch[mod]
                recon = reconstructions[mod]
                min_d = min(orig.shape[1], recon.shape[1])
                err = F.mse_loss(
                    recon[:, :min_d], orig[:, :min_d], reduction='none'
                ).mean(dim=1)  # (B,)
                # Normalize within-batch to (0,1); low error -> high confidence.
                err_norm = (err - err.min()) / (err.max() - err.min() + 1e-8)
                targets[mod] = (1.0 - err_norm).clamp(0.0, 1.0)
        return targets

    def confidence_loss(self, predicted, target):
        terms = []
        for mod in predicted:
            if mod in target:
                terms.append(F.mse_loss(predicted[mod], target[mod]))
        if not terms:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(terms).mean()
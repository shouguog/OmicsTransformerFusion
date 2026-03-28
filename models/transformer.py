import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import *

class CrossOmicsContrastiveLearning(nn.Module):
    """COCL module (Section 2.3.3, Eq. 6). τ=0.07."""

    def __init__(self, hidden_dim, projection_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, projection_dim)
            ) for mod in ['mRNA', 'miRNA', 'Methy', 'CNV']
        })

    def forward(self, embeddings):
        projected = {mod: F.normalize(self.projectors[mod](emb), dim=-1)
                     for mod, emb in embeddings.items()}

        mods = list(projected.keys())
        total_loss = 0.0
        n_pairs = 0

        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                z1, z2 = projected[mods[i]], projected[mods[j]]
                B = z1.shape[0]
                logits = torch.mm(z1, z2.T) / self.temperature
                labels = torch.arange(B, device=z1.device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                total_loss += loss
                n_pairs += 1

        return total_loss / max(n_pairs, 1)


class GatedCrossModalAttention(nn.Module):
    """Gated fusion (Section 2.3.4, Eq. 7). d_a=64."""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.modalities = ['mRNA', 'miRNA', 'Methy', 'CNV']
        self.hidden_dim = hidden_dim

        self.cross_attn = nn.ModuleDict({
            mod: nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for mod in self.modalities
        })

        # Gating (Eq. 7)
        self.gate_net = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),  # d_a = 64
                nn.Tanh(),
                nn.Linear(64, 1),
            ) for mod in self.modalities
        })

        self.norms = nn.ModuleDict({mod: nn.LayerNorm(hidden_dim) for mod in self.modalities})
        self.ffs = nn.ModuleDict({mod: FeedForward(hidden_dim, dropout=dropout) for mod in self.modalities})

    def forward(self, embeddings, return_attention=False):
        mods = [m for m in self.modalities if m in embeddings]
        B = embeddings[mods[0]].shape[0]

        # Stack for cross-attention context
        context = torch.stack([embeddings[m] for m in mods], dim=1)  # (B, M, D)

        updated = {}
        attention_weights = {} if return_attention else None
        gate_values = {}

        for mod in mods:
            q = embeddings[mod].unsqueeze(1)  # (B, 1, D)
            attn_out, attn_w = self.cross_attn[mod](q, context, context)
            attn_out = attn_out.squeeze(1)

            if return_attention:
                attention_weights[mod] = attn_w

            combined = torch.cat([embeddings[mod], attn_out], dim=-1)
            gate = torch.sigmoid(self.gate_net[mod](combined))
            gate_values[mod] = gate

            updated[mod] = self.norms[mod](embeddings[mod] + gate * attn_out)
            updated[mod] = updated[mod] + self.ffs[mod](updated[mod])

        if return_attention:
            return updated, attention_weights, gate_values
        return updated, None, gate_values


class HiOmicsFormer(nn.Module):
    """
    Complete HiOmicsFormer (Section 2.3).
    Architecture: d=256, L=4, H=8, K=16, K_c=9.
    """

    def __init__(self, feature_dims, config):
        super().__init__()
        self.feature_dims = feature_dims
        self.config = config
        self.modalities = list(feature_dims.keys())
        self.hidden_dim = config.hidden_dim  # 256
        self.latent_dim = config.latent_dim  # 128
        self.num_clusters = config.num_clusters  # 9

        # 1. Hierarchical encoders (L=4 layers each)
        self.encoders = nn.ModuleDict({
            mod: HierarchicalOmicsEncoder(
                input_dim=feature_dims[mod],
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_encoder_layers,  # L=4
                group_sizes=config.group_sizes,
                dropout=config.dropout
            ) for mod in self.modalities
        })

        # 2. Cross-omics contrastive learning
        self.cocl = CrossOmicsContrastiveLearning(
            config.hidden_dim, config.projection_dim, config.contrastive_temperature
        )

        # 3. Gated cross-modal attention
        self.cross_modal_layers = nn.ModuleList([
            GatedCrossModalAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_cross_layers)
        ])

        # 4. Fusion → latent
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * len(self.modalities), config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.GELU(),
        )
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # 5. DEC (Section 2.3.5)
        self.cluster_centroids = nn.Parameter(
            torch.randn(config.num_clusters, config.latent_dim)
        )
        self.cluster_head = nn.Linear(config.latent_dim, config.num_clusters)

        # 6. Decoders
        self.decoders = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, feature_dims[mod])
            ) for mod in self.modalities
        })

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std) if self.training else mu

    def encode(self, batch, return_attention=False):
        mod_embeddings = {}
        encoder_attentions = {} if return_attention else None

        for mod in self.modalities:
            if mod in batch:
                out, scales, attn = self.encoders[mod](batch[mod], return_attention)
                mod_embeddings[mod] = out
                if return_attention and attn:
                    encoder_attentions[mod] = attn

        # Contrastive loss
        contrastive_loss = self.cocl(mod_embeddings)

        # Cross-modal attention
        cross_attentions = []
        gate_values_all = []
        for layer in self.cross_modal_layers:
            mod_embeddings, c_attn, gates = layer(mod_embeddings, return_attention)
            if return_attention and c_attn:
                cross_attentions.append(c_attn)
            gate_values_all.append(gates)

        # Fusion
        fused = self.fusion(torch.cat([mod_embeddings[m] for m in self.modalities], dim=-1))
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        z = self.reparameterize(mu, logvar)

        result = {
            'z': z, 'mu': mu, 'logvar': logvar,
            'modality_embeddings': mod_embeddings,
            'contrastive_loss': contrastive_loss
        }
        if return_attention:
            result['encoder_attentions'] = encoder_attentions
            result['cross_attentions'] = cross_attentions
            result['gate_values'] = gate_values_all
        return result

    def get_cluster_assignments(self, z):
        """Student's t-distribution soft assignments (Eq. 8)."""
        dist = torch.cdist(z, self.cluster_centroids)
        q = 1.0 / (1.0 + dist ** 2 / self.config.cluster_alpha)
        q = q ** ((self.config.cluster_alpha + 1) / 2)
        return q / q.sum(dim=1, keepdim=True)

    @staticmethod
    def get_target_distribution(q):
        """Target distribution (Eq. 8)."""
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        return (weight / weight.sum(dim=1, keepdim=True)).detach()

    def decode(self, z):
        return {mod: self.decoders[mod](z) for mod in self.modalities}

    def forward(self, batch, return_attention=False):
        enc = self.encode(batch, return_attention)
        z = enc['z']
        q = self.get_cluster_assignments(z)
        recon = self.decode(z)
        return {**enc, 'q': q, 'cluster_logits': self.cluster_head(z), 'reconstructions': recon}

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import  *
class HierarchicalOmicsEncoder(nn.Module):
    """
    Hierarchical multi-scale encoder (Section 2.3.2).
    d=256, L=4 Transformer layers, K=16 feature clusters.
    """

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers,
                 group_sizes=(16, 32, 64), dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = len(group_sizes)
        self.group_sizes = group_sizes

        # Input projection (Eq. 1)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Multi-scale local attention
        self.local_projections = nn.ModuleList()
        self.cls_tokens = nn.ParameterList()
        self.pos_encodings = nn.ModuleList()
        self.local_transformers = nn.ModuleList()

        for gs in group_sizes:
            actual_gs = min(gs, hidden_dim)
            n_groups = hidden_dim // actual_gs
            self.local_projections.append(nn.Sequential(
                nn.Linear(actual_gs, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU()
            ))
            self.cls_tokens.append(nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02))
            self.pos_encodings.append(PositionalEncoding(hidden_dim, max_len=n_groups + 1, dropout=dropout))
            self.local_transformers.append(nn.ModuleList([
                TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
            ]))

        # Multi-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]
        h = self.input_proj(x)  # (B, hidden_dim)

        attention_weights = [] if return_attention else None
        scale_cls = []

        for s, gs in enumerate(self.group_sizes):
            actual_gs = min(gs, self.hidden_dim)
            n_groups = self.hidden_dim // actual_gs

            x_scale = h[:, :n_groups * actual_gs]
            x_scale = x_scale.view(batch_size, n_groups, actual_gs)
            x_scale = self.local_projections[s](x_scale)

            cls = self.cls_tokens[s].expand(batch_size, -1, -1)
            x_scale = torch.cat([cls, x_scale], dim=1)
            x_scale = self.pos_encodings[s](x_scale)

            for block in self.local_transformers[s]:
                x_scale, attn_w = block(x_scale, return_attention=return_attention)
                if return_attention and attn_w is not None:
                    attention_weights.append(attn_w)

            scale_cls.append(x_scale[:, 0])

        combined = torch.cat(scale_cls, dim=-1)
        output = self.scale_fusion(combined)
        output = self.output_norm(output)

        if return_attention:
            return output, scale_cls, attention_weights
        return output, scale_cls, None



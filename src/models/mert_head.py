"""Model C head: learnable weighted sum over frozen MERT layers -> small MLP.

Input: precomputed embeddings [B, n_layers, 768] (see extract_mert.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import register


class MertHead(nn.Module):
    def __init__(self, n_classes: int, n_layers: int = 13, dim: int = 768, hidden: int = 256, dropout: float = 0.3,
                 layer_norm: bool = False):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))  # softmax -> layer weights
        self.norm = nn.LayerNorm(dim) if layer_norm else nn.Identity()  # off = brief's spec
        self.mlp_in = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def mix(self, x):
        w = torch.softmax(self.layer_logits, dim=0)
        return self.norm(torch.einsum("l,bld->bd", w, x))

    def forward(self, x):
        return self.out(self.mlp_in(self.mix(x)))

    def embed(self, x):
        return self.mlp_in(self.mix(x))

    def layer_weights(self):
        return torch.softmax(self.layer_logits.detach(), 0).cpu().tolist()


@register("mert_head")
def build(cfg: dict, n_classes: int) -> nn.Module:
    m = cfg["model"]
    return MertHead(n_classes, m.get("n_layers", 13), m.get("dim", 768),
                    m.get("hidden", 256), m.get("dropout", 0.3), m.get("layer_norm", False))

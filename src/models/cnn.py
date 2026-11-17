"""Model A: compact 2D CNN on log-mel input (1, 128, T), ~5M parameters."""
from __future__ import annotations

import torch
import torch.nn as nn

from . import register


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class GenreCNN(nn.Module):
    def __init__(self, n_classes: int, channels=(32, 64, 128, 256, 512), dropout=(0.5, 0.3)):
        super().__init__()
        chans = (1, *channels)
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        c = channels[-1]
        self.head = nn.Sequential(
            nn.Dropout(dropout[0]), nn.Linear(2 * c, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout[1]), nn.Linear(256, n_classes),
        )
        self.head_in = self.head[:3]  # up to the penultimate activation

    def pool(self, x):
        h = self.features(x)
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)  # avg + max -> 2c

    def forward(self, x):
        return self.head(self.pool(x))

    def embed(self, x):
        return self.head_in(self.pool(x))


@register("cnn")
def build(cfg: dict, n_classes: int) -> nn.Module:
    m = cfg["model"]
    return GenreCNN(n_classes, tuple(m.get("channels", (32, 64, 128, 256, 512))),
                    tuple(m.get("dropout", (0.5, 0.3))))

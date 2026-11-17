"""Pluggable model registry.

A model is registered by name and built from the YAML config:

    @register("my_model")
    def build(cfg: dict, n_classes: int) -> nn.Module: ...

Contract every model follows (so train.py / evaluate.py stay model-agnostic):
  forward(x) -> logits [B, n_classes], where x is one batch from the dataset
  named by cfg["data"]["kind"] (see datasets.DATASETS). Evaluation flattens the
  window axis, calls forward, and averages softmax probabilities per track.
Optional hooks:
  param_groups(cfg) -> list of optimizer param groups (e.g. backbone vs head lr)
  embed(x) -> penultimate features [B, D] (for UMAP plots)

To add AST later: create models/ast_model.py with @register("ast"), import it
below, and set data.kind: wave in configs/ast.yaml.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, Callable[[dict, int], nn.Module]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def build_model(cfg: dict, n_classes: int) -> nn.Module:
    name = cfg["model"]["name"]
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"model {name!r} is not implemented/registered; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, n_classes)


# Import model modules so they register themselves.
from . import cnn, mert_head  # noqa: E402,F401

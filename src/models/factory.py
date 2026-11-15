"""
Fábrica de modelos do PA3.

- ``unet``     : CNN pequena do zero (encoder da U-Net + GAP + linear).
- ``resnet18`` / ``resnet34`` / ``resnet50``
- ``mobilenetv3`` / ``mobilenet_v3_large``
- ``efficientnet_b0``
Todos aceitam ``pretrained=True`` (pesos ImageNet) exceto ``unet``.

Também expõe ``get_param_groups(model, lr, lr_head)`` para separar a
cabeça do backbone — útil no fine-tuning dos pré-treinados.
"""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn
import torchvision.models as tvm

from .unet_classifier import UNetEncoderClassifier


def _tv_load(fn, pretrained: bool) -> nn.Module:
    """Compatível com torchvision >= 0.13 (``weights=``) e < 0.13 (``pretrained=``)."""
    if pretrained:
        try:
            return fn(weights="DEFAULT")
        except TypeError:
            return fn(pretrained=True)  # type: ignore[call-arg]
    try:
        return fn(weights=None)
    except TypeError:
        return fn(pretrained=False)  # type: ignore[call-arg]


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Instancia o backbone com a cabeça de ``num_classes`` saídas."""
    key = name.lower()
    if key in ("unet", "unet_encoder", "unet_scratch", "cnn"):
        return UNetEncoderClassifier(num_classes=num_classes)

    if key == "resnet18":
        m = _tv_load(tvm.resnet18, pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if key == "resnet34":
        m = _tv_load(tvm.resnet34, pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if key == "resnet50":
        m = _tv_load(tvm.resnet50, pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if key in ("mobilenetv3", "mobilenet_v3_large"):
        m = _tv_load(tvm.mobilenet_v3_large, pretrained)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if key in ("efficientnet_b0", "efficientnet-b0"):
        m = _tv_load(tvm.efficientnet_b0, pretrained)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    raise ValueError(f"Backbone não suportado: {name}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def get_param_groups(
    model: nn.Module,
    lr: float,
    lr_head: float | None = None,
) -> list[dict]:
    """
    Separa parâmetros do backbone e da cabeça para usar LRs diferentes.

    Se ``lr_head`` é ``None``, devolve um grupo só (LR uniforme).
    A detecção da cabeça usa os nomes comuns: ``head``, ``fc``, ``classifier``.
    """
    if lr_head is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    head_params: list[nn.Parameter] = []
    backbone_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(tok in name for tok in ("head", "fc", "classifier")):
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups: list[dict] = [{"params": backbone_params, "lr": lr}]
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    return groups
"""
Classificador baseado no encoder da U-Net.

Reaproveita o bloco convolucional da U-Net (double 3x3 conv + BN + ReLU),
empilha alguns níveis com max-pooling entre eles (caminho contrátil) e,
no final, faz global average pooling + uma camada linear para as classes.
Não inclui o decoder (irrelevante para classificação).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class UNetConvBlock(nn.Module):
    """Bloco da U-Net: (Conv 3x3 -> BN -> ReLU) x 2, opcionalmente com dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoderClassifier(nn.Module):
    """
    Encoder estilo U-Net + GAP + classificador linear.

    Parameters
    ----------
    num_classes : número de classes de saída.
    in_channels : canais de entrada (3 para RGB).
    widths : canais de saída de cada nível do encoder. O comprimento define
        quantos níveis (e portanto quantos max-poolings) existem.
        - ``(32, 64, 128, 256)``  -> ~1.2M parâmetros, 3 poolings (224 -> 28)
        - ``(64, 128, 256, 512)`` -> ~4.7M parâmetros, 3 poolings (224 -> 28)
        - ``(64, 128, 256, 512, 1024)`` -> ~12.5M, 4 poolings (224 -> 14)
    dropout : taxa de dropout aplicada no bloco mais profundo.
    head_dropout : dropout aplicado nas features depois do GAP, antes da linear.

    Entrada esperada: ``(B, 3, H, W)`` com H, W >= 32 (por causa dos poolings).
    Saída: logits ``(B, num_classes)``.
    """

    def __init__(
        self,
        num_classes: int = 21,
        in_channels: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256),
        dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.widths = tuple(widths)

        blocks: list[nn.Module] = []
        prev = in_channels
        for i, w in enumerate(self.widths):
            # Dropout só no bloco mais profundo (como na U-Net original).
            d = dropout if i == len(self.widths) - 1 else 0.0
            blocks.append(UNetConvBlock(prev, w, dropout=d))
            prev = w

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.widths[-1], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming nas convoluções, 1/0 nas BN — padrão U-Net."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = self.pool(x)
        x = self.gap(x).flatten(1)       # (B, C)
        x = self.head_dropout(x)
        return self.head(x)              # (B, num_classes)

    # Atalho útil para avaliar/contar sem instanciar de novo
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
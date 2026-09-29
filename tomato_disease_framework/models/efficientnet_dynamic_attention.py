# Placeholder for: models/efficientnet_dynamic_attention.py\n
"""EfficientNetB0 + Dynamic Self-Attention classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from models.dynamic_attention import DynamicSelfAttention


class EfficientNetB0DynamicAttention(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()

        if num_classes != 10:
            raise ValueError(
                "The tomato classifier is defined for exactly 10 classes."
            )

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = efficientnet_b0(weights=weights)

        # EfficientNet-B0 final feature map has 1280 channels.
        self.attention = DynamicSelfAttention(
            channels=1280,
            reduction=8,
        )

        # Reuse EfficientNet-B0's original classifier.
        in_features = self.backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            self.backbone.classifier[0],
            self.backbone.classifier[1],
        )

        self.classifier[1] = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(x)

        x = self.attention(x)

        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
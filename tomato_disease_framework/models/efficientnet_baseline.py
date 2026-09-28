"""EfficientNetB0 classifier used for the Phase 3 baseline."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(
    num_classes: int = 10,
    pretrained: bool = True,
) -> nn.Module:
    if num_classes != 10:
        raise ValueError("The tomato baseline is defined for exactly 10 classes.")
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

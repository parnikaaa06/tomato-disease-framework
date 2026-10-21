"""Grad-CAM explanations for the trained EfficientNet-B0 baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Return a normalized Grad-CAM heatmap and target class index.

    ``input_tensor`` must have shape ``[1, 3, H, W]`` and use the same
    preprocessing as the trained model.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor must have shape [1, 3, H, W].")
    if input_tensor.shape[1] != 3:
        raise ValueError("input_tensor must contain exactly 3 image channels.")
    if input_tensor.shape[2] <= 0 or input_tensor.shape[3] <= 0:
        raise ValueError("input_tensor must have non-empty spatial dimensions.")
    if class_idx is not None and (
        isinstance(class_idx, bool) or not isinstance(class_idx, int)
    ):
        raise TypeError("class_idx must be an integer or None.")

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_gradient(gradient: torch.Tensor) -> None:
        gradients.append(gradient)

    def save_activations(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("target_layer must return a tensor.")
        activations.append(output)
        if output.requires_grad:
            output.register_hook(save_gradient)

    forward_handle = target_layer.register_forward_hook(save_activations)
    was_training = model.training

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(input_tensor)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Model output must be a tensor of class logits.")
            if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] == 0:
                raise ValueError("Model output must have shape [1, num_classes].")

            predicted_idx = int(logits.argmax(dim=1).item())
            target_idx = predicted_idx if class_idx is None else class_idx
            if not 0 <= target_idx < logits.shape[1]:
                raise ValueError("class_idx is outside the model's class range.")

            logits[0, target_idx].backward()
            if len(activations) != 1 or len(gradients) != 1:
                raise RuntimeError(
                    "Could not capture one activation and its gradient from target_layer."
                )

            feature_maps = activations[0].detach()
            grads = gradients[0].detach()
            if (
                feature_maps.ndim != 4
                or feature_maps.shape[0] != 1
                or grads.shape != feature_maps.shape
            ):
                raise RuntimeError(
                    "target_layer must produce matching [1, channels, height, width] "
                    "activations and gradients."
                )

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)
            if not torch.isfinite(cam).all():
                raise RuntimeError("Grad-CAM produced non-finite heatmap values.")

            return cam.cpu(), target_idx
    finally:
        forward_handle.remove()
        try:
            model.zero_grad(set_to_none=True)
        finally:
            model.train(was_training)

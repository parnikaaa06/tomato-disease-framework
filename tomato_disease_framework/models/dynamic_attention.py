# Placeholder for: models/dynamic_attention.py\n
import torch
from torch import nn


class DynamicSelfAttention(nn.Module):
    """
    Lightweight spatial self-attention module.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ):
        super().__init__()

        reduced_channels = max(
            channels // reduction,
            1,
        )

        self.query = nn.Conv2d(
            channels,
            reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.key = nn.Conv2d(
            channels,
            reduced_channels,
            kernel_size=1,
            bias=False,
        )

        self.value = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

        self.softmax = nn.Softmax(
            dim=-1
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(
            batch_size,
            -1,
            height * width,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            -1,
            height * width,
        )

        attention = torch.bmm(
            query,
            key,
        )

        attention = self.softmax(
            attention
        )

        value = value.view(
            batch_size,
            channels,
            height * width,
        )

        attended = torch.bmm(
            value,
            attention.transpose(1, 2),
        )

        attended = attended.view(
            batch_size,
            channels,
            height,
            width,
        )

        return x + self.gamma * attended
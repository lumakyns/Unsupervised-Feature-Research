"""Compact convolutional classifier used by the separation experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.separation.datasets import Dataset, get_dataset_spec


class SeparationConvNet(nn.Module):
    """A small three-layer CNN with named convolution feature taps."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SeparationConvNet expects exactly three channel widths")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd so spatial shapes are preserved")

        spec = get_dataset_spec(dataset)
        padding = kernel_size // 2
        c1, c2, c3 = channels
        self.dataset = dataset
        self.spec = spec
        self.conv1 = nn.Conv2d(spec.channels, c1, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

        pooled_height = spec.height // 4
        pooled_width = spec.width // 4
        self.classifier = nn.Linear(c3 * pooled_height * pooled_width, spec.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits."""

        self._validate_inputs(inputs)
        x = F.relu(self.conv1(inputs))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        return self.classifier(self.dropout(x))

    def conv_outputs(
        self,
        inputs: torch.Tensor,
        *,
        detach_layer_inputs: bool = False,
    ) -> list[torch.Tensor]:
        """Return pre-activation outputs for conv1, conv2, and conv3.

        Separation pretraining treats each layer's feature patches as inputs to
        that layer, so it can detach those inputs and update only the selected
        layer's filters.
        """

        self._validate_inputs(inputs)
        out1 = self.conv1(inputs.detach() if detach_layer_inputs else inputs)
        x = F.relu(out1)
        if detach_layer_inputs:
            x = x.detach()
        out2 = self.conv2(x)
        x = self.pool(F.relu(out2))
        if detach_layer_inputs:
            x = x.detach()
        out3 = self.conv3(x)
        return [out1, out2, out3]

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        expected = (self.spec.channels, self.spec.height, self.spec.width)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"Expected inputs shaped [N, {expected[0]}, {expected[1]}, "
                f"{expected[2]}], got {tuple(inputs.shape)}"
            )

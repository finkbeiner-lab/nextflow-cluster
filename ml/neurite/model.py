#!/usr/bin/env python3
"""Compact 2D U-Net for single-channel neurite segmentation.

A small, dependency-light U-Net (Ronneberger et al., 2015) sized for CPU
dry-runs on tiny crops and GPU training on the full annotation set. One input
channel (morphology: RGEDI / FITC=EGFP fill), one output channel of raw logits
(apply ``sigmoid`` downstream for a probability map).

Design choices:

* ``depth`` and ``base_channels`` are configurable so the same class scales from
  a toy 2-level net (fast CPU smoke test) to a 4-level net for real training.
* Padded 3x3 convolutions keep spatial size within a block, so output HxW equals
  input HxW — convenient for tiled inference on 2048x2048 fields.
* Input HxW must be divisible by ``2 ** (depth - 1)`` for clean skip
  concatenation; ``valid_multiple`` reports that constraint and the inference
  helper in ``infer.py`` pads to satisfy it.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> norm -> ReLU) x2, spatial size preserved."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialise the double-conv block.

        Args:
            in_ch: Input channel count.
            out_ch: Output channel count.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Input tensor (N, in_ch, H, W).

        Returns:
            Output tensor (N, out_ch, H, W).
        """
        return self.block(x)


class UNet(nn.Module):
    """Configurable 2D U-Net, single-channel in, single-logit-channel out."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        base_channels: int = 16,
    ) -> None:
        """Initialise the U-Net.

        Args:
            in_channels: Number of input channels (1 for morphology).
            out_channels: Number of output channels (1 logit channel).
            depth: Number of resolution levels (encoder blocks). ``depth=4``
                gives 3 downsampling steps.
            base_channels: Channel count at the top level; doubles each level.
        """
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        chans = [base_channels * (2 ** i) for i in range(depth)]

        # Encoder.
        self.downs = nn.ModuleList()
        prev = in_channels
        for c in chans[:-1]:
            self.downs.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)

        # Bottleneck.
        self.bottleneck = DoubleConv(chans[-2] if depth > 1 else in_channels,
                                     chans[-1])

        # Decoder.
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(
                nn.ConvTranspose2d(chans[i], chans[i - 1], kernel_size=2, stride=2)
            )
            self.up_convs.append(DoubleConv(chans[i], chans[i - 1]))

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def valid_multiple(self) -> int:
        """Return the required spatial divisor for input HxW.

        Returns:
            ``2 ** (depth - 1)``; input height and width must be multiples of
            this for skip connections to align.
        """
        return 2 ** (self.depth - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor (N, in_channels, H, W), with H, W divisible by
                ``valid_multiple()``.

        Returns:
            Logit tensor (N, out_channels, H, W).
        """
        skips: List[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = up_conv(x)
        return self.head(x)


def build_model(
    depth: int = 4, base_channels: int = 16, in_channels: int = 1
) -> UNet:
    """Convenience factory for a :class:`UNet`.

    Args:
        depth: Number of resolution levels.
        base_channels: Top-level channel count.
        in_channels: Number of input channels.

    Returns:
        A configured :class:`UNet`.
    """
    return UNet(
        in_channels=in_channels,
        out_channels=1,
        depth=depth,
        base_channels=base_channels,
    )


if __name__ == "__main__":
    # Smoke test: tiny net, tiny input, on CPU.
    net = build_model(depth=3, base_channels=8)
    n_params = sum(p.numel() for p in net.parameters())
    m = net.valid_multiple()
    x = torch.randn(2, 1, 4 * m, 4 * m)
    y = net(x)
    print(f"UNet depth=3 base=8 params={n_params:,}")
    print(f"in {tuple(x.shape)} -> out {tuple(y.shape)}  (valid_multiple={m})")
    assert y.shape == (2, 1, 4 * m, 4 * m)
    print("model.py smoke test PASSED")

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Self-contained (vendored) torch reference for ConvNeXt-Small.

The module/parameter naming mirrors ``torchvision.models.convnext_small`` so the
official checkpoint (``convnext_small-0c510722.pth``) loads directly with
``load_state_dict``. Keeping the architecture vendored lets the reference run on
CPU without a torchvision dependency.
"""

from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ConvNeXt-Small configuration.
CONVNEXT_SMALL_DEPTHS = [3, 3, 27, 3]
CONVNEXT_SMALL_DIMS = [96, 192, 384, 768]


class Permute(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dim of an NCHW tensor (channels-first)."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class StochasticDepth(nn.Module):
    """Row-wise stochastic depth. Identity at inference (eval / p == 0)."""

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        survival = 1.0 - self.p
        size = [x.shape[0]] + [1] * (x.ndim - 1)
        noise = torch.empty(size, dtype=x.dtype, device=x.device).bernoulli_(survival)
        if survival > 0.0:
            noise.div_(survival)
        return x * noise


class CNBlock(nn.Module):
    def __init__(self, dim: int, layer_scale: float, stochastic_depth_prob: float) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x: Tensor) -> Tensor:
        result = self.layer_scale * self.block(x)
        result = self.stochastic_depth(result)
        result += x
        return result


class ConvNeXt(nn.Module):
    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        stochastic_depth_prob: float = 0.4,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem: patchify with a 4x4 stride-4 conv, then channels-first LayerNorm.
        layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                norm_layer(dims[0]),
            )
        )

        total_blocks = sum(depths)
        block_id = 0
        for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
            stage: List[nn.Module] = []
            for _ in range(depth):
                sd_prob = stochastic_depth_prob * block_id / (total_blocks - 1.0)
                stage.append(CNBlock(dim, layer_scale, sd_prob))
                block_id += 1
            layers.append(nn.Sequential(*stage))

            # Downsample between stages: channels-first LayerNorm + 2x2 stride-2 conv.
            if stage_idx < len(depths) - 1:
                layers.append(
                    nn.Sequential(
                        norm_layer(dim),
                        nn.Conv2d(dim, dims[stage_idx + 1], kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            norm_layer(dims[-1]),
            nn.Flatten(1),
            nn.Linear(dims[-1], num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# Per-variant architecture (depths/dims/stochastic_depth_prob). This is the single
# source of truth for ConvNeXt geometry; stochastic depth is identity at inference
# and does not affect loaded weights — it only mirrors torchvision faithfully.
VARIANT_CONFIG = {
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768], "stochastic_depth_prob": 0.1},
    "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768], "stochastic_depth_prob": 0.4},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024], "stochastic_depth_prob": 0.5},
    "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536], "stochastic_depth_prob": 0.5},
}


def convnext(variant: str = "small", num_classes: int = 1000, **kwargs) -> ConvNeXt:
    cfg = VARIANT_CONFIG[variant]
    return ConvNeXt(
        depths=cfg["depths"],
        dims=cfg["dims"],
        stochastic_depth_prob=cfg["stochastic_depth_prob"],
        num_classes=num_classes,
        **kwargs,
    )


def convnext_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    return convnext("tiny", num_classes, **kwargs)


def convnext_small(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    return convnext("small", num_classes, **kwargs)


def convnext_base(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    return convnext("base", num_classes, **kwargs)


def convnext_large(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    return convnext("large", num_classes, **kwargs)

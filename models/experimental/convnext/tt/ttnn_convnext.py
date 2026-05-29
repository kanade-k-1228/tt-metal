# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN ConvNeXt.

Implemented against established tt-metal CNN patterns (tt_cnn `TtConv2d`,
ttnn layer_norm / linear / gelu / adaptive_avg_pool2d). See tt/common.py for the
layout convention (flattened `(1, 1, N*H*W, C)`).

The model is built directly from the vendored torch reference (whose module
layout mirrors torchvision.convnext_small): ``features`` is
``[stem, stage0, down0, stage1, down1, stage2, down2, stage3]`` and
``classifier`` is ``[LayerNorm2d, Flatten, Linear]``.
"""

import ttnn
from models.experimental.convnext.reference.convnext import VARIANT_CONFIG
from models.experimental.convnext.tt.common import (
    LAYER_NORM_EPS,
    TtConvNeXtBlock,
    TtConvNeXtDownsample,
    _compute_kernel_config,
)
from models.experimental.convnext.tt.model_preprocessing import (
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    preprocess_linear_weight,
)
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d


class TtConvNeXt:
    def __init__(self, torch_model, device, batch_size=1, resolution=224, variant="small"):
        self.device = device
        self.batch_size = batch_size
        self.depths = VARIANT_CONFIG[variant]["depths"]
        self.dims = VARIANT_CONFIG[variant]["dims"]
        self.compute_kernel_config = _compute_kernel_config(device)

        features = torch_model.features
        height = width = resolution

        # Stem: Sequential(Conv2d(3, dims[0], k=4, s=4), LayerNorm2d).
        stem_conv = features[0][0]
        stem_norm = features[0][1]
        self.stem_conv = TtConv2d(
            Conv2dConfiguration.from_torch(
                stem_conv, input_height=height, input_width=width, batch_size=batch_size
            ),
            device,
        )
        height //= stem_conv.stride[0]
        width //= stem_conv.stride[1]
        self.stem_norm_weight = preprocess_layernorm_parameter(stem_norm.weight.data, device=device)
        self.stem_norm_bias = preprocess_layernorm_parameter(stem_norm.bias.data, device=device)

        # Stages and the downsample layers that sit between them.
        self.stages = []
        self.downsamples = []
        feat_idx = 1
        for stage_idx, depth in enumerate(self.depths):
            stage_seq = features[feat_idx]
            self.stages.append(
                [TtConvNeXtBlock(stage_seq[b], device, batch_size, height, width) for b in range(depth)]
            )
            feat_idx += 1

            if stage_idx < len(self.depths) - 1:
                down_seq = features[feat_idx]  # Sequential(LayerNorm2d, Conv2d(k=2, s=2))
                self.downsamples.append(
                    TtConvNeXtDownsample(down_seq[0], down_seq[1], device, batch_size, height, width)
                )
                height //= down_seq[1].stride[0]
                width //= down_seq[1].stride[1]
                feat_idx += 1
            else:
                self.downsamples.append(None)

        self.final_height = height
        self.final_width = width

        # Head: Sequential(LayerNorm2d, Flatten, Linear). Applied after global avg pool.
        classifier = torch_model.classifier
        head_norm = classifier[0]
        head_fc = classifier[2]
        self.head_norm_weight = preprocess_layernorm_parameter(head_norm.weight.data, device=device)
        self.head_norm_bias = preprocess_layernorm_parameter(head_norm.bias.data, device=device)
        self.head_fc_weight = preprocess_linear_weight(head_fc.weight.data, device=device)
        self.head_fc_bias = preprocess_linear_bias(head_fc.bias.data, device=device)
        self.num_classes = head_fc.out_features

    def __call__(self, input_tensor):
        x = self.stem_conv(input_tensor, return_output_dim=False)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.layer_norm(
            x,
            weight=self.stem_norm_weight,
            bias=self.stem_norm_bias,
            epsilon=LAYER_NORM_EPS,
            compute_kernel_config=self.compute_kernel_config,
        )

        for stage_idx, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x)
            downsample = self.downsamples[stage_idx]
            if downsample is not None:
                x, _ = downsample(x)

        # Global average pool over the final feature map -> (1, 1, batch, C).
        x = ttnn.adaptive_avg_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=self.final_height,
            input_w=self.final_width,
            channels=self.dims[-1],
            output_size=[1, 1],
        )
        x = ttnn.reshape(x, (self.batch_size, self.dims[-1]))
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.layer_norm(
            x,
            weight=self.head_norm_weight,
            bias=self.head_norm_bias,
            epsilon=LAYER_NORM_EPS,
            compute_kernel_config=self.compute_kernel_config,
        )
        x = ttnn.linear(
            x, self.head_fc_weight, bias=self.head_fc_bias, compute_kernel_config=self.compute_kernel_config
        )
        x = ttnn.reshape(x, (self.batch_size, self.num_classes))
        return x

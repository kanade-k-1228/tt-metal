# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN building blocks for ConvNeXt.

This forward pass is implemented against the established tt-metal CNN patterns
(tt_cnn `TtConv2d`, ttnn.layer_norm / linear / gelu).

Activations flow in the flattened conv layout `(1, 1, N*H*W, C)` throughout, so
the channel axis is always last — this lets LayerNorm / Linear / GELU operate
directly on it without spatial reshapes.
"""

import ttnn
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d
from models.experimental.convnext.tt.model_preprocessing import (
    preprocess_layer_scale,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    preprocess_linear_weight,
)

LAYER_NORM_EPS = 1e-6


def _compute_kernel_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


class TtConvNeXtBlock:
    """ConvNeXt block: dwconv(7x7) -> LN -> pwconv1 -> GELU -> pwconv2 -> layer_scale -> residual.

    Built directly from the torch reference ``CNBlock`` (``block`` Sequential is
    [dwconv, Permute, LayerNorm, Linear, GELU, Linear, Permute] and ``layer_scale``).
    Spatial dims are unchanged, so the activation layout is preserved end to end.
    """

    def __init__(self, torch_block, device, batch_size, input_height, input_width):
        self.device = device
        self.compute_kernel_config = _compute_kernel_config(device)

        dwconv = torch_block.block[0]  # depthwise Conv2d (groups == dim)
        norm = torch_block.block[2]  # nn.LayerNorm
        fc1 = torch_block.block[3]  # nn.Linear dim -> 4*dim
        fc2 = torch_block.block[5]  # nn.Linear 4*dim -> dim

        self.dwconv = TtConv2d(
            Conv2dConfiguration.from_torch(
                dwconv,
                input_height=input_height,
                input_width=input_width,
                batch_size=batch_size,
            ),
            device,
        )
        self.norm_weight = preprocess_layernorm_parameter(norm.weight.data, device=device)
        self.norm_bias = preprocess_layernorm_parameter(norm.bias.data, device=device)
        self.fc1_weight = preprocess_linear_weight(fc1.weight.data, device=device)
        self.fc1_bias = preprocess_linear_bias(fc1.bias.data, device=device)
        self.fc2_weight = preprocess_linear_weight(fc2.weight.data, device=device)
        self.fc2_bias = preprocess_linear_bias(fc2.bias.data, device=device)
        self.layer_scale = preprocess_layer_scale(torch_block.layer_scale.data, device=device)

    def __call__(self, x):
        identity = x

        out = self.dwconv(x, return_output_dim=False)
        out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)

        out = ttnn.layer_norm(
            out,
            weight=self.norm_weight,
            bias=self.norm_bias,
            epsilon=LAYER_NORM_EPS,
            compute_kernel_config=self.compute_kernel_config,
        )
        out = ttnn.linear(
            out,
            self.fc1_weight,
            bias=self.fc1_bias,
            activation="gelu",  # fused GELU saves a separate op per block
            compute_kernel_config=self.compute_kernel_config,
        )
        out = ttnn.linear(out, self.fc2_weight, bias=self.fc2_bias, compute_kernel_config=self.compute_kernel_config)

        out = ttnn.multiply(out, self.layer_scale)
        out = ttnn.add(out, identity)
        return out


class TtConvNeXtDownsample:
    """Stage transition: channels-first LayerNorm -> 2x2 stride-2 conv.

    Built from the torch downsample ``Sequential(LayerNorm2d, Conv2d)``.
    Halves spatial dims and changes the channel count, so it returns the new
    (height, width).
    """

    def __init__(self, torch_norm, torch_conv, device, batch_size, input_height, input_width):
        self.device = device
        self.compute_kernel_config = _compute_kernel_config(device)

        self.norm_weight = preprocess_layernorm_parameter(torch_norm.weight.data, device=device)
        self.norm_bias = preprocess_layernorm_parameter(torch_norm.bias.data, device=device)
        self.conv = TtConv2d(
            Conv2dConfiguration.from_torch(
                torch_conv,
                input_height=input_height,
                input_width=input_width,
                batch_size=batch_size,
            ),
            device,
        )

    def __call__(self, x):
        x = ttnn.layer_norm(
            x,
            weight=self.norm_weight,
            bias=self.norm_bias,
            epsilon=LAYER_NORM_EPS,
            compute_kernel_config=self.compute_kernel_config,
        )
        x, (out_height, out_width) = self.conv(x, return_output_dim=True)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        return x, (out_height, out_width)

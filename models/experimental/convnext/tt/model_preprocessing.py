# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def preprocess_linear_weight(weight, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=None):
    # torch Linear stores (out, in); ttnn.linear computes x @ w so transpose to (in, out).
    weight = weight.T.contiguous()
    return ttnn.from_torch(weight, dtype=dtype, layout=layout, device=device, mesh_mapper=mesh_mapper)


def preprocess_linear_bias(bias, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    return ttnn.from_torch(bias, dtype=dtype, layout=layout, device=device, mesh_mapper=mesh_mapper)


def preprocess_layernorm_parameter(parameter, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=None):
    # gamma/beta for ttnn.layer_norm are reshaped to (1, normalized channels); all
    # ConvNeXt channel counts are multiples of the tile width, so TILE_LAYOUT is valid.
    parameter = parameter.reshape((1, -1))
    return ttnn.from_torch(parameter, dtype=dtype, layout=layout, device=device, mesh_mapper=mesh_mapper)


def preprocess_layer_scale(parameter, *, dtype=ttnn.bfloat16, device=None, mesh_mapper=None):
    # ConvNeXt layer scale is (dim, 1, 1); reshape to (1, 1, 1, dim) so it broadcasts
    # over the flattened (1, 1, N*H*W, dim) activation along the channel axis.
    parameter = parameter.reshape(1, 1, 1, -1)
    return ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper)


def create_convnext_input_tensors(
    batch=1, input_channels=3, input_height=224, input_width=224, pad_channels=None, mesh_mapper=None
):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    if pad_channels:
        ttnn_input_tensor = torch.nn.functional.pad(
            ttnn_input_tensor, (0, pad_channels - ttnn_input_tensor.shape[-1]), value=0
        )
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper
    )
    ttnn_input_tensor = ttnn.reshape(
        ttnn_input_tensor,
        (
            1,
            1,
            ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
            ttnn_input_tensor.shape[3],
        ),
    )
    return torch_input_tensor, ttnn_input_tensor

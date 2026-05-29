# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.experimental.convnext.common import load_torch_model
from models.experimental.convnext.reference.convnext import convnext
from models.experimental.convnext.tt import ttnn_convnext
from models.experimental.convnext.tt.model_preprocessing import create_convnext_input_tensors
from tests.ttnn.utils_for_testing import assert_with_pcc

# Matches the PCC threshold in tests/pcc/test_convnext.py.
CONVNEXT_PCC = 0.98


class ConvNeXtPerformanceRunnerInfra:
    """Holds the ConvNeXt model, its persistent input/output tensors and the torch
    reference, exposing ``run`` / ``validate`` for the trace runner.

    The model reads its flattened ``(1, 1, N*H*W, C)`` NHWC input directly from a
    persistent DRAM tensor (the stem conv does not deallocate it), so the captured
    trace can replay against a fixed address that is refilled per inference.
    """

    def __init__(
        self,
        device,
        batch_size=1,
        variant="small",
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        model_location_generator=None,
        resolution=224,
    ):
        torch.manual_seed(0)
        self.device = device
        self.batch_size = batch_size
        self.variant = variant
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.resolution = resolution
        self.num_devices = device.get_num_devices()

        torch_model = convnext(variant, num_classes=1000)
        torch_model = load_torch_model(torch_model, variant=variant, model_location_generator=model_location_generator)
        torch_model.eval()

        self.torch_input_tensor, self.tt_inputs_host = create_convnext_input_tensors(
            batch=batch_size * self.num_devices, input_height=resolution, input_width=resolution
        )
        with torch.no_grad():
            self.torch_output_tensor = torch_model(self.torch_input_tensor)

        self.ttnn_model = ttnn_convnext.TtConvNeXt(
            torch_model, device, batch_size=batch_size, resolution=resolution, variant=variant
        )
        self.input_tensor = None
        self.output_tensor = None

    def setup_inputs(self, device):
        """Return the host input tensor and the (DRAM, interleaved) config it lives in."""
        return self.tt_inputs_host, ttnn.DRAM_MEMORY_CONFIG

    def run(self):
        self.output_tensor = self.ttnn_model(self.input_tensor)

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensor if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor)
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=CONVNEXT_PCC)
        logger.info(
            f"ConvNeXt-{self.variant} batch_size={self.batch_size * self.num_devices} "
            f"act_dtype={self.act_dtype} weight_dtype={self.weight_dtype} PCC={self.pcc_message}"
        )

    def dealloc_output(self):
        if self.output_tensor is not None:
            ttnn.deallocate(self.output_tensor)
            self.output_tensor = None

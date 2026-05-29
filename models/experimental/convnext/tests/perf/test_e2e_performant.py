# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.experimental.convnext.common import CONVNEXT_VARIANTS
from models.experimental.convnext.runner.performant_runner import ConvNeXtPerformantRunner

# ConvNeXt issues many small ops per inference, so the captured trace needs a generous
# DRAM trace region.
TRACE_REGION_SIZE = 23887872


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("batch_size, act_dtype, weight_dtype", [(1, ttnn.bfloat16, ttnn.bfloat16)])
@pytest.mark.parametrize("resolution", [224])
@pytest.mark.parametrize(
    "device_params, variant",
    [
        ({"l1_small_size": cfg["l1_small_size"], "trace_region_size": TRACE_REGION_SIZE}, variant)
        for variant, cfg in CONVNEXT_VARIANTS.items()
    ],
    indirect=["device_params"],
    ids=list(CONVNEXT_VARIANTS.keys()),
)
def test_e2e_performant(device, batch_size, act_dtype, weight_dtype, resolution, variant, model_location_generator):
    performant_runner = ConvNeXtPerformantRunner(
        device,
        batch_size,
        variant,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
        resolution=resolution,
    )

    total_batch = batch_size * device.get_num_devices()
    iterations = 50
    t0 = time.time()
    for _ in range(iterations):
        performant_runner.run()
    ttnn.synchronize_device(device)
    elapsed = time.time() - t0

    performant_runner.release()

    inference_time_avg = elapsed / iterations
    logger.info(
        f"ConvNeXt-{variant} batch_size={total_batch} resolution={resolution}: "
        f"latency={inference_time_avg * 1e3:.2f} ms, FPS={total_batch / inference_time_avg:.1f}"
    )

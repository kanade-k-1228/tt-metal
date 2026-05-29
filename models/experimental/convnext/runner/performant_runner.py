# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.convnext.runner.performant_runner_infra import ConvNeXtPerformanceRunnerInfra


class ConvNeXtPerformantRunner:
    """Trace runner for ConvNeXt.

    ConvNeXt issues ~300 small ops per inference at batch=1, so wall-clock is dominated
    by host-side op dispatch. Capturing the forward pass once and replaying it with
    ``ttnn.execute_trace`` removes that dispatch overhead (~3x lower latency on Blackhole).
    The input lives in a persistent DRAM tensor that is refilled per inference; the trace
    reads it by address.
    """

    def __init__(
        self,
        device,
        device_batch_size=1,
        variant="small",
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        model_location_generator=None,
        resolution=224,
    ):
        self.device = device
        self.runner_infra = ConvNeXtPerformanceRunnerInfra(
            device,
            device_batch_size,
            variant,
            act_dtype,
            weight_dtype,
            model_location_generator,
            resolution=resolution,
        )
        self.tt_inputs_host, self.input_mem_config = self.runner_infra.setup_inputs(device)
        # Persistent DRAM input; the trace reads it by address on every replay.
        self.input_tensor = self.tt_inputs_host.to(device, self.input_mem_config)
        self.runner_infra.input_tensor = self.input_tensor
        self._capture_trace()

    def _capture_trace(self):
        # Warm-up run compiles the conv kernels and prepares weights (writes are only
        # allowed outside trace capture). Its output is discarded.
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Capture the forward pass on CQ0.
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

    def _execute_trace_inference(self, tt_inputs_host=None):
        if tt_inputs_host is not None:
            ttnn.copy_host_to_device_tensor(tt_inputs_host, self.input_tensor, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.output_tensor

    def run(self, tt_inputs_host=None):
        return self._execute_trace_inference(tt_inputs_host)

    def release(self):
        ttnn.release_trace(self.device, self.tid)

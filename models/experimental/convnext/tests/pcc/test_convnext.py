# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.experimental.convnext.common import CONVNEXT_VARIANTS
from models.experimental.convnext.reference.convnext import convnext


@pytest.mark.parametrize("variant", list(CONVNEXT_VARIANTS.keys()), ids=list(CONVNEXT_VARIANTS.keys()))
def test_convnext_reference(variant):
    """Sanity check for the vendored reference model (all variants)."""
    batch_size = 1
    torch_model = convnext(variant, num_classes=1000).eval()
    torch_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        output = torch_model(torch_input)
    assert tuple(output.shape) == (batch_size, 1000)


@pytest.mark.parametrize(
    "device_params, variant",
    [({"l1_small_size": cfg["l1_small_size"]}, variant) for variant, cfg in CONVNEXT_VARIANTS.items()],
    indirect=["device_params"],
    ids=list(CONVNEXT_VARIANTS.keys()),
)
@pytest.mark.parametrize("use_pretrained_weight", [True], ids=["pretrained_weight_true"])
@pytest.mark.parametrize("batch_size", [1])
def test_convnext(device, variant, use_pretrained_weight, batch_size, reset_seeds, model_location_generator):
    import ttnn

    from models.experimental.convnext.common import load_torch_model
    from models.experimental.convnext.tt import ttnn_convnext
    from models.experimental.convnext.tt.model_preprocessing import create_convnext_input_tensors
    from tests.ttnn.utils_for_testing import assert_with_pcc

    torch_model = convnext(variant, num_classes=1000)
    if use_pretrained_weight:
        torch_model = load_torch_model(torch_model, variant=variant, model_location_generator=model_location_generator)
    torch_model.eval()

    torch_input_tensor, ttnn_input_tensor = create_convnext_input_tensors(
        batch=batch_size, input_height=224, input_width=224
    )
    torch_output_tensor = torch_model(torch_input_tensor)

    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device)
    ttnn_model = ttnn_convnext.TtConvNeXt(torch_model, device, batch_size=batch_size, resolution=224, variant=variant)
    output_tensor = ttnn_model(ttnn_input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.98)

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch

CONVNEXT_VARIANTS = {
    "tiny": {"weights": "convnext_tiny-983f1562.pth", "l1_small_size": 16 * 1024},
    "small": {"weights": "convnext_small-0c510722.pth", "l1_small_size": 24 * 1024},
    "base": {"weights": "convnext_base-6075fbad.pth", "l1_small_size": 32 * 1024},
    "large": {"weights": "convnext_large-ea097f82.pth", "l1_small_size": 48 * 1024},
}

_MODEL_DIR = "models/experimental/convnext"


def load_torch_model(torch_model, variant="small", model_location_generator=None):
    weights_filename = CONVNEXT_VARIANTS[variant]["weights"]

    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = os.path.join(_MODEL_DIR, "weight", weights_filename)
        if not os.path.exists(weights_path):
            os.system(f"bash {_MODEL_DIR}/weights_download.sh {variant}")
    else:
        weights_path = (
            model_location_generator("vision-models/convnext", model_subdir="", download_if_ci_v2=True)
            / weights_filename
        )

    state_dict = torch.load(weights_path, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model

# ConvNeXt

## Platforms:
    Blackhole (p150a)

### Introduction
ConvNeXt is a convolutional architecture introduced in
["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545). It modernizes a
standard ResNet with a patchify stem, depthwise 7x7 convolutions, LayerNorm,
GELU activations, an inverted bottleneck and layer scale, reaching
transformer-level accuracy while remaining a pure ConvNet. This package supports
all four variants pretrained on ImageNet-1k at 224x224; they differ only in
`depths`/`dims` and the loaded checkpoint:

| variant | depths | dims | checkpoint | top-1 |
|---|---|---|---|---|
| tiny  | [3, 3, 9, 3]  | [96, 192, 384, 768]    | convnext_tiny-983f1562.pth  | 82.52% |
| small | [3, 3, 27, 3] | [96, 192, 384, 768]    | convnext_small-0c510722.pth | 83.62% |
| base  | [3, 3, 27, 3] | [128, 256, 512, 1024]  | convnext_base-6075fbad.pth  | 84.06% |
| large | [3, 3, 27, 3] | [192, 384, 768, 1536]  | convnext_large-ea097f82.pth | 84.41% |

Variant geometry lives in `reference/convnext.py::VARIANT_CONFIG`; deployment
config (checkpoint, `l1_small_size`, pcc) lives in `common.py::CONVNEXT_VARIANTS`.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

Both tests are parametrized across all four variants (`tiny`/`small`/`base`/`large`).

### Reference model sanity check (CPU, no TT device required)
```
pytest models/experimental/convnext/tests/pcc/test_convnext.py::test_convnext_reference
```

### PCC test (TT device)
```
pytest models/experimental/convnext/tests/pcc/test_convnext.py::test_convnext
```

### Performant model (trace)
```
pytest models/experimental/convnext/tests/perf/test_e2e_performant.py
```

Pretrained weights are downloaded on demand via
`models/experimental/convnext/weights_download.sh <variant>`
(torchvision checkpoints, e.g. `convnext_small-0c510722.pth`).

Measured on Blackhole p150a (pretrained weights, batch=1, 224x224). PCC is against the
torch reference. The model is dispatch-bound at batch=1, so capturing the forward pass
once and replaying it with `ttnn.execute_trace` removes the op-to-op dispatch overhead:

| model | PCC   | eager latency | trace latency | trace FPS |
| ----- | ----- | ------------- | ------------- | --------- |
| tiny  | 0.987 | 9.5 ms        | 4.4 ms        | 227 fps   |
| small | 0.989 | 15.7 ms       | 7.7 ms        | 130 fps   |
| base  | 0.993 | 16.2 ms       | 10.3 ms       | 97 fps    |
| large | 0.993 | 18.3 ms       | 16.9 ms       | 59 fps    |

The trace runner lives in `models/experimental/convnext/runner/`.

## Details
- The entry point to the ConvNeXt model is `TtConvNeXt` in
  `models/experimental/convnext/tt/ttnn_convnext.py` (takes a `variant` argument).
- Reference model: `models/experimental/convnext/reference/convnext.py`
  (self-contained; module/parameter naming matches `torchvision.models.convnext_*`).
- Supported Input Resolution - (224, 224) (Height, Width)

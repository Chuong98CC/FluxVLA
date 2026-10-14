---
name: fluxvla-add-component
description: Add or extend a FluxVLA registry component (vision backbone, head, VLA, LLM/VLM backbone, projector, dataset, transform, collator, tokenizer, runner, operator, metric, processor) so it builds from a config and honors that registry's interface contract. Use when writing a new module under fluxvla/, swapping one module type for another in a recipe, or wiring a component into configs and inference models. Not for changing an existing component's internals or for training/eval recipes themselves.
---

# Adding a FluxVLA component

FluxVLA resolves modules through registries, and a config selects one by the
`type` string. The canonical procedure is four steps plus the interface
contract. The upstream tutorial is
https://fluxvla.limxdynamics.com/md_source/tutorials/private_module.html.

## Checklist

1. Add the component in its registry's directory.
2. Decorate the class with the matching `@REGISTRY.register_module()`.
3. Re-export it from that directory's `__init__.py`.
4. Reference it from a config with a `type` that exactly matches the class
   name and parameter names that exactly match `__init__`.
5. Implement the registry's interface methods, not a generic guess at them.

Registry, location and registry object:

| Registry | Location | Import |
| --- | --- | --- |
| `VISION_BACKBONES` | `fluxvla/models/backbones/visions/` | `from fluxvla.engines import VISION_BACKBONES` |
| `LLM_BACKBONES` | `fluxvla/models/backbones/llms/` | `from fluxvla.engines import LLM_BACKBONES` |
| `VLM_BACKBONES` | `fluxvla/models/backbones/vlms/` | `from fluxvla.engines import VLM_BACKBONES` |
| `HEADS` | `fluxvla/models/heads/` | `from fluxvla.engines import HEADS` |
| `VLAS` | `fluxvla/models/vlas/` | `from fluxvla.engines import VLAS` |
| `PROJECTORS` | `fluxvla/models/projectors/` | `from fluxvla.engines import PROJECTORS` |
| `DATASETS` | `fluxvla/datasets/` | `from fluxvla.engines import DATASETS` |
| `TRANSFORMS` | `fluxvla/transforms/` | `from fluxvla.engines import TRANSFORMS` |
| `COLLATORS` | `fluxvla/collators/` | `from fluxvla.engines import COLLATORS` |
| `TOKENIZERS` | `fluxvla/tokenizers/` | `from fluxvla.engines import TOKENIZERS` |
| `RUNNERS` | `fluxvla/engines/runners/` | `from fluxvla.engines import RUNNERS` |
| `OPERATORS` | `fluxvla/engines/operators/` | `from fluxvla.engines import OPERATORS` |
| `METRICS` | `fluxvla/engines/metrics/` | `from fluxvla.engines import METRICS` |
| `PROCESSORS` | `fluxvla/engines/processors/` | `from fluxvla.engines import PROCESSORS` |

## Interface contracts

Implement the real contract of the registry you are extending. Read the base
class or a neighboring implementation first; the head example's
`forward -> {'loss': ...}` shape is specific to `HEADS` and does not apply to
other registries.

For `VISION_BACKBONES`, the base is `fluxvla/models/backbones/visions/
base_vision.py` (`VisionBackbone`). Subclasses must provide `forward`,
`embed_dim`, `num_patches`, `half_precision_dtype` and
`get_fsdp_wrapping_policy`. Frozen feature encoders used by feature-space
consumers instead extend `FrozenVisionFeatures`, which adds `feature_dim`,
`feature_grid(height, width)`, `encode_features -> SpatialFeatureOutput` and a
`freeze()`/`train()` pair that keeps the encoder in evaluation mode. Keep the
input range in the component's own contract and document it: `[-1, 1]`
endpoint frames converted once to `[0, 1]` for C-RADIO, native `[-1, 1]` for
the VAE latent wrapper.

For `HEADS`, `forward(...)` returns a dict containing `'loss'` and
`predict_action(...)` returns an action tensor; implement
`get_fsdp_wrapping_policy()` when the head is large.

## Optional dependencies stay lazy

Import an optional dependency inside the function or method that needs it, not
at module import. Importing the package must not require it, and a component
that wraps a compiled artifact should be constructible from a config without
it. Name the exact requirement and version in the failure message, and reject
an unsupported major version explicitly instead of assuming compatible
behaviour. `RadioViTBackboneTensorRT` in
`fluxvla/models/backbones/visions/radio_trt.py` is a worked example: it imports
`tensorrt` only on the first encode and rejects anything but 10.x.

## Wiring a config

The `type` must be the registered class name, and every other key must be an
`__init__` parameter name. A key that is not a parameter raises `TypeError` at
build time. Extras that a consumer passes positionally are not config keys.

```python
import torch

from fluxvla.engines import build_vision_backbone_from_cfg

vision_backbone = dict(
    type='RadioViTBackbone',
    vision_backbone_id='c-radio_v4-h',
    model_path='/path/to/C-RADIOv4-H',
    patch_size=16,
    dtype=torch.bfloat16,
    default_image_resolution=(3, 256, 256),
)

encoder = build_vision_backbone_from_cfg(vision_backbone)
```

A component dictionary becomes a config slot only where a consumer declares
one: a VLA that takes `vision_backbone` accepts the dictionary above, a
different consumer may take the same class under another parameter name, and
one that hardcodes a fused tower takes none. Check the consumer's own
`__init__` before assuming a slot exists; the registry does not guarantee that
every consumer can host every component of a kind. When the component wraps a
compiled artifact, interchangeability is also bounded by what was compiled -
`RadioViTBackboneTensorRT` only serves its own fixed geometry.

Inference-only swaps belong in `inference_model`, which mirrors `model` when
it is absent. Copyable examples live under `configs/`, for instance
`configs/visions/radio_vit_tensorrt_example.py`.

## Verify before claiming done

- The class resolves: `REGISTRY.get('MyClass') is MyClass`.
- The config builds: `build_<kind>_from_cfg(dict(type='MyClass', ...))`
  (`build_vision_backbone_from_cfg` for vision backbones).
- Every config key is an `__init__` parameter, checked for the example config.
- The interface contract holds with a small stub: shapes, dtypes, freezing,
  and the failure paths a caller would otherwise hit.
- `python -m pytest <focused test files> -q` passes, and
  `git diff --check` is clean.
- New and edited files carry no copyright or license banner.

## Common pitfalls

- Adding the `# Copyright 2026 Limx Dynamics` / Apache License 2.0 banner to a
  new file. New files start with their module docstring or imports, and
  `docs/CONTRIBUTING.md` states that the repository-level `LICENSE` covers all
  contents and that new files must not include license headers.
- Forgetting the `__init__.py` re-export: the class exists but `type=` fails
  to resolve because the decorator never ran.
- A `type` string that does not match the class name exactly.
- Duplicating a loader or a range conversion instead of reusing the existing
  helper, which silently drifts from the sibling component.
- Importing an optional backend at module scope, which breaks every config
  that merely imports the package.
- Claiming a performance win without measuring it on the target device.

## References

- Tutorial:
  https://fluxvla.limxdynamics.com/md_source/tutorials/private_module.html
- Frozen visual backbones:
  `fluxvla/models/backbones/visions/frozen_vision.py` and
  `docs/vision-backbones.md`
- Reusable TensorRT execution:
  `fluxvla/models/backbones/visions/base_trt/{metadata,engine}.py` with the
  C-RADIO sidecar adapter beside them in `base_trt/radio_metadata.py`
- C-RADIO on TensorRT:
  `fluxvla/models/backbones/visions/radio_trt.py`, the offline helpers in
  `tools/trt_export.py` and `tools/c_radio_export.py`, and
  `docs/c-radio-v4-tensorrt.md`

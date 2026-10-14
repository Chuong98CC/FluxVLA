# Model and data contracts

## Locate the real extension point

The [custom-model tutorial](https://fluxvla.limxdynamics.com/md_source/tutorials/private_model.html)
provides architecture orientation. Inspect these repository-relative paths in
the target checkout; signatures and supported combinations can evolve.

| Component | Registry | Implementation directory | Recipe slot / builder |
| --- | --- | --- | --- |
| Top-level VLA | `VLAS` | `fluxvla/models/vlas/` | `model`; `build_vla_from_cfg` |
| VLM | `VLM_BACKBONES` | `fluxvla/models/backbones/vlms/` | `vlm_backbone`; `build_vlm_backbone_from_cfg` |
| Vision | `VISION_BACKBONES` | `fluxvla/models/backbones/visions/` | `vision_backbone`; `build_vision_backbone_from_cfg` |
| Language | `LLM_BACKBONES` | `fluxvla/models/backbones/llms/` | `llm_backbone`; `build_llm_backbone_from_cfg` |
| Projection | `PROJECTORS` | `fluxvla/models/projectors/` | `projector`; `build_projector_from_cfg` |
| Action head | `HEADS` | `fluxvla/models/heads/` | `vla_head`; `build_head_from_cfg` |

Registry definitions and builders live in `fluxvla/engines/utils/root.py` and
`builder.py`. Import registries from an established public export or `root.py`;
use `@REGISTRY.register_module()` and the relevant package `__init__.py` import.
A registry entry does not guarantee that a particular VLA accepts that component
slot or its input/output contract. Trace the consumer.

Read `base_vla.py`, the chosen VLA's constructor, `forward`, and `predict_action`.
Use the existing composition when it fits. Combined VLM and separate
vision/LLM/projector paths are alternative base compositions; architectures with
an additional action expert must follow their own constructors. In this checkout
`LlavaVLA` inherits directly from `BaseVLA`, so tutorial inheritance diagrams are
not sufficient implementation specifications.

## Action-head integration

Read `llava_vla.py`, `continuous_action_utils.py`, and the closest head. The
[custom-module tutorial](https://fluxvla.limxdynamics.com/md_source/tutorials/private_module.html)
illustrates an action-head replacement. Current callers additionally matter:

- `LlavaVLA.forward` sends features, states, attention/action masks, actions,
  embodiment IDs, and sample weights to its head. A new head must accept the
  caller's optional keys and implement their supported semantics explicitly.
- Training must reach the runner as a mapping with a scalar differentiable
  `loss`. The current wrapper can normalize some head outputs when composing
  auxiliary losses; returning a loss mapping is the straightforward contract.
- Action inference uses `predict_action`. Inspect its extra arguments, including
  previous actions, prefix length, RTC settings, and seed handling. Declare
  supported modes rather than silently claiming a capability through `**kwargs`.
- Preserve batch/horizon layout and define where padded action dimensions become
  physical controls. Check `ori_action_dim`, head output slicing, dataset
  normalization, and denormalization together; names alone do not define units
  or coordinate mappings.

For diffusion/flow methods, match the source's prediction target, time/noise
distribution, scheduler, integration/sampling rule, conditioning, and number of
inference steps. Keep these configurable and checkpoint-compatible. A new action
objective alone usually does not require a new distributed runner.

Mask before padded targets can contaminate conditioning or computations. Define
per-step/per-dimension validity, loss denominator and sample weighting. An empty
valid set needs an explicit policy, such as a differentiable zero action loss or
a clear rejection; a mean over an empty selection is not valid. Test arbitrary
padding and all-invalid examples. Any skip/update behavior affecting distributed
ranks belongs in the execution contract. Pool features using actual mask
positions; `mask.sum() - 1` assumes a specific padding layout.

## Backbones, additional objectives, and temporal inputs

Read `backbones/vlms/outputs.py` and the selected VLA. Legacy VLMs return
`(features, attention_mask, extra)`. Current `LlavaVLA` also consumes
`VLMBackboneOutput` with `last_hidden_state`, `attention_mask`, `auxiliary_losses`,
and `auxiliary_outputs`. Use the representation the caller handles. Match mask
length to the fused feature sequence, and accept forwarded image metadata such
as `image_grid_thw` when relevant.

Use `continuous_action_utils.py` for compatible loss normalization/composition.
In the checked checkout `LlavaVLA` adds backbone auxiliary losses with default
unit weights; configurable non-unit weights need an explicit, tested owner.
Log named terms and ensure the total counts each term once. Arbitrary new
training inputs are not automatically forwarded by this wrapper.

For future frames, history, teacher features, or additional modalities, map each
new field through sampling, episode boundaries, transforms, collator, batch
device/dtype conversion, and VLA forwarding. `**kwargs` accepting a field does
not prove that the field reaches the objective. Keep privileged future targets
out of deployment inputs; define the inference path with available observations.

Vision backbones may extend `base_vision.py` or, for frozen spatial features,
`frozen_vision.py`. Check the chosen base's forward/features interface, dimensions,
patch/grid metadata, precision, and freeze/train behavior. Do not apply an
action-head loss contract to a visual encoder or projector.

## Loading and distributed compatibility

Inspect `BaseVLA` initialization/freezing, `from_pretrained`, mapping resolution,
and `engines/utils/fsdp_wrapping.py`. Reuse the inherited policy if it covers the
new modules; expose appropriate policies for large custom blocks. Check trainable
parameters, gradient checkpointing and optimizer inclusion, including frozen
teacher branches.

Inspect the actual `name_mapping` direction and validate it with small known
checkpoint keys. Report missing/unexpected keys and shape mismatches; intentional
new heads need an explicit initialization/loading policy. Keep a separate
`inference_model` only when needed, then verify state-dict compatibility and
normalization. Inspect the config loader for the fallback to `model` rather than
assuming every inference runner handles a missing `inference_model` itself.

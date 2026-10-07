# PatchWAM integration design

Status: implementation authorized and starting. The user explicitly selected
`gpt-6.1-sol` as root and overrode the skill's Astra-only root requirement.

## Objective and user constraints

Integrate the implementation at `/home/chuong/workspace/PatchWAM/src/patchwam`
into FluxVLA as configurable native components. Inherit FluxVLA classes and
reuse its registries, dataset/statistics infrastructure, collators, training,
checkpointing, inference, and evaluation wherever their contracts match. Copy
PatchWAM's specialized VLM implementation locally. Preserve joint visual/action
flow matching, data preparation, and augmentation semantics without a broad
framework refactor or an installed `patchwam` dependency.

The user's reference to FluxFLA is interpreted as this FluxVLA repository.
The existing PatchWAM implementation is the source of behavior; reproducing a
paper from scratch is outside this task.

## Repository evidence and baseline

- FluxVLA: `main`, HEAD `be321105b5aa2331f88f3f9449fd4f87aa436fc2`.
- PatchWAM: HEAD `dea19d21182a26a8ee16efc2c4125ee7aecf44fb`.
- Both workspaces had no tracked, staged, or untracked changes at discovery.
- `fluxvla/models/vlas/base_vla.py`: component builders, freeze policy,
  checkpoint loading, and FSDP integration.
- `fluxvla/models/vlas/fastwam_vla.py`: existing world/action adapter with
  `vlm_backbone`, `vla_head`, native loss mapping, and `predict_action`.
- `fluxvla/engines/utils/root.py` and `builder.py`: existing component registries
  and `type`-based configuration construction.
- `fluxvla/datasets/parquet_dataset.py`: action/future-frame windows bounded by
  episode and data root, with valid action/frame masks.
- `fluxvla/datasets/dataset_wrapper.py`: `DistributedRepeatingDataset` expects
  child `.stats`, `__len__`, and `__getitem__(index, dataset_statistics)`.
- `fluxvla/transforms/transform_inputs.py`, `transform_images.py`, and
  `normalize.py`: decoding, state/action processing, image preparation, and
  normalization; `DictCollator` stacks tensors and preserves string metadata.
- `scripts/train.py`, `scripts/inference.py`, and existing runners are the
  entrypoints to retain. Training reads `output['loss']`; inference calls
  `predict_action(**inputs)` and then the configured action decoder.
- PatchWAM `models/flux.py` supplies the official FLUX.2 masked adapter and
  image/text encoding. `models/vision_language.py` supplies specialized Qwen3-VL
  readouts, visual history pooling, LoRA, and optional causal subtask supervision.
- PatchWAM `models/{codec,geometry,flow,policy,tiny}.py` supplies the parameter-free
  action patches and objective; `data/{augmentation,appearance,history}.py`
  supplies augmentation and causal history.

## Approach and alternatives

Use native registered wrappers around a small, vendored PatchWAM model core.
This preserves numerical behavior while exposing FluxVLA's Lego-style config
boundaries. Copying PatchWAM's entire dataset and engine would duplicate native
pipeline behavior. Reimplementing the FLUX.2 denoiser in existing Wan/Cosmos heads
would change the model and require an unnecessary refactor. Neither is selected.

FluxVLA's current Qwen3VL wrapper does not expose PatchWAM's instruction-layer
readouts, last-prompt summary, or pooled historical DeepStack path. Retain the
copied specialized encoder. Existing SigLIP and Wan VAE components do not produce
the required packed 128-channel FLUX.2 latents and must not be substituted merely
because they are image encoders.

## Component boundaries

1. Vendor the model core under `fluxvla/models/third_party_models/patchwam/`,
   retaining origin and license notices. Keep `vision_language.py` local and
   preserve its behavior. No imports from the sibling repository at runtime.
2. Register `PatchWAMHead` in `HEADS`, inheriting the vendored `PatchFlowPolicy`.
   It owns the denoiser, proprio projection, codec, and flow objective. Support
   a config-built tiny denoiser for offline tests and an official FLUX.2 denoiser
   for local assets. Preserve source masks and sampling semantics internally.
3. Register `PatchWAMBackbone` in `VLM_BACKBONES`. It owns only the autoencoder
   and instruction encoder and implements `encode(batch, target: bool) -> dict`.
   Reuse/extract the source `FluxAssetPolicy._encode` behavior without registering
   the head twice. A registered nn.Module frontend is acceptable, as in the
   existing Wan frontend; do not force a generation-oriented VLM base onto it.
4. Register `PatchWAMVLA(BaseVLA)` in `VLAS`. Build components from their config,
   adapt native batch names/masks once, encode, and call the head. Set
   `all_module_keys = ['vlm_backbone', 'vla_head']`. Keep parameter ownership
   disjoint for checkpointing and FSDP. Override freeze/train behavior only as
   needed to preserve frozen encoders and enabled language LoRA.
5. Add `PatchWAMParquetDataset(ParquetDataset)` only for temporal selection not
   available in the existing dataset. Use the inherited storage/statistics and
   wrapper interface. If required, introduce one protected hook in the base
   dataset before transform execution; its default must leave existing behavior
   unchanged. Avoid copying the entire `EpisodeDataset` reader or duplicating
   `ParquetDataset.__getitem__` wholesale.
6. Add registered PatchWAM-specific image/packing/augmentation transforms in
   `fluxvla/transforms/transform_patchwam.py`. Reuse existing action/state
   normalization and action decoding. Copy `ClipAugment` and optional
   `AppearanceRandomizer` when native transforms cannot preserve exact semantics.
7. Add Python configs under `configs/patchwam/` and usage guidance in
   `docs/patchwam.md`. Use existing DDP/FSDP runners and scripts. Prefer config
   integration and narrow capability handling to a new PatchWAM training engine.

## Public interfaces and data contracts

`PatchWAMVLA.forward(images=None, states=None, actions=None, action_masks=None,
frame_masks=None, img_masks=None, task_description=None, **kwargs) -> dict`
returns scalar `loss`, `loss_video`, and `loss_action`, plus `loss_subtask` when
enabled. Preserve weights exactly once. `predict_action(images=None, states=None,
task_description=None, action_horizon=None, num_inference_steps=None, seed=None,
**kwargs) -> Tensor` returns normalized `[B,H,A]` actions. The existing runner or
decoder applies denormalization once. Default inference uses ten solver steps,
with per-call override. Support explicit generators/noise in tests for parity.

| Native boundary field | Shape / meaning | Source adapter field |
| --- | --- | --- |
| `images` | `[B,V,3,T,H,W]`, separate cameras, or `[B,3,T,H,W]`, mosaic; RGB `[-1,1]` | `camera_video` or `video` |
| `states` | `[B,P]` or `[B,H,P]`, normalized | `proprio` |
| `actions` | `[B,H,A]`, normalized | `action` |
| `action_masks` | `[B,H]` or `[B,H,A]`, True means valid | split into invalid time and dimension masks |
| `action_dim_masks` | optional `[B,A]`/`[B,H,A]`, True means valid | `action_dim_is_pad`, inverted |
| `frame_masks` | `[B,T]`, True means valid | future raster validity / `image_is_pad` |
| `task_description` | one nonempty string per sample | `prompt` |
| `history_video`, `history_valid` | `[B,K,3,H,W]`, `[B,K]`, True means valid | same names |
| `vl_current` | unaugmented current head image `[B,3,Hv,Wv]` | same name |
| `vl_history`, `vl_history_valid` | unaugmented past images `[B,K,3,Hv,Wv]`, valid slots | same names |
| `subtask` | optional/required training sentence per sample | training only |
| `text_tokens`, `text_valid` | optional compatible cached `[B,L,C]`, valid tokens | same names |

Mask adaptation must preserve dimension and time validity separately. All-invalid
examples must produce finite losses and cannot supervise padded dimensions or
frames. Zero invalid inputs before attention so masked NaNs cannot contaminate
SDPA. Generated action padding must be handled as in the source implementation.

## Dataset and augmentation behavior

- A window with 17 sampled rows produces 16 action targets and the current/final
  visual endpoints. With stride one, the future image is at row `start+16`,
  while actions span `start` through `start+15`. Do not accidentally use the
  second row as the future target after reducing to two endpoint images.
- Preserve camera order and benchmark geometry: LIBERO's two-camera horizontal
  mosaic and RobotWin's configured compact mosaic; the RoboDojo history path
  encodes separate views with positional groups 10/11/12 and 0/1/2.
- Use episode/root-local padding and strict causal history on the original
  recording clock. Filtering/sample stride must not redefine time or leak across
  episodes. Validation and inference disable training augmentation.
- Reproduce source clip augmentation: one probability draw per sample, independent
  camera draws, temporal sharing within a camera, and the exact operation order
  and parameter names. History augmentation follows the source behavior.
- Clone clean VLM current/history images before world augmentation. Future images
  and target subtask sentences cannot affect conditioning representations.
- World history and VL history use separate jitter/dropout contracts; invalid
  slots remain explicitly masked and zeroed. Historical prefix length never
  changes the generated future raster length.
- Preserve native normalization/statistics ownership, source padded-delta zeroing,
  and inverse decoding of full action horizons. Test fixed inputs against the
  sibling source rather than assuming similarly named modes are identical.
- Online history must consume every observation at recording cadence and reset
  on episode boundaries. Reuse native eval dataset buffering when sufficient;
  history-enabled configs must arrange preprocessing every step. If native
  timestamps or reset signals are insufficient, fail that config explicitly.

## Assets, checkpoints, and failure behavior

Use local FLUX.2 source/assets and local Qwen directories. Preserve official Klein
base 4B/9B parameter names, strict loading, source identity checks, and packed
latent validation. Do not replace prefix-masked attention with the standard
image-generation forward. Do not download weights or execute remote model code.

Use native FluxVLA save/load and resume paths. FSDP wrappers must surround
modules whose forward is actually called; the copied FLUX adapter manually
executes inner blocks, so wrapping those blocks directly is unsafe. Prefer a
called head/denoiser boundary and keep distributed hardware coverage explicit.
Document any mapping needed to
initialize from a PatchWAM checkpoint; reject incompatible or incomplete mappings
instead of accepting random trainable weights. Missing files, bad dimensions,
invalid cache format, unsupported Qwen readout layers, and required missing subtask
labels must produce useful errors. Trainable VLM adapters cannot be bypassed by
text caches. Repeated module registration and blanket freezing of LoRA are errors.

EMA and Self-Flow require successful optimizer-update hooks and checkpointed
teacher state absent from the ordinary adapter contract. They are deferred from
the first native integration; expose no recipe claiming support and explicitly
reject nonzero Self-Flow/EMA training options. CFG, action isolation, VLM adapters,
causal history, subtask supervision, and source appearance transforms remain
within the integration scope. Any proposed expansion to shared runner state
requires Astra to revise this contract before implementation.

## Acceptance and practical limits

1. Native config construction works without an installed `patchwam` package;
   optional model dependencies are imported only when required.
2. With identical tiny-model weights, sigma/noise, and masks, wrapper losses,
   gradients, and sampled actions match the source numerically.
3. Synthetic LeRobot fixtures prove endpoint/action alignment, camera order,
   episode/root boundaries, normalization/inverse decoding, and mask polarity.
4. Fixed-seed augmentation matches source pixels; clean VLM images remain
   unchanged. History tests prove causality, reset, invalid-slot handling, and
   unchanged future sampling size. Optional subtask conditioning does not leak.
5. A synthetic native training update and checkpoint round trip work; real
   encoder parameters stay frozen while the head and configured LoRA get gradients.
6. LIBERO text-conditioning and RobotWin VLM configs use native runners, collators,
   statistics, denormalizers, and inference signatures. A separate-view/history
   example demonstrates the same configurable interfaces.
7. Existing affected dataset/config/model tests pass. Source licenses/provenance
   are retained and the PatchWAM checkout remains unmodified.

Full-size GPU training, distributed FSDP, real checkpoints, robot control, and
benchmark parity require assets/hardware not established during planning. Record
each as verified or unverified using actual evidence; CPU tests cannot establish
paper reproduction or hardware safety.

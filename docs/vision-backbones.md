# Swappable frozen visual backbones

`fluxvla/models/backbones/visions` exposes the frozen encoders used by
feature-space consumers (world-action models, auxiliary feature targets)
through one contract, so a recipe can switch between VAE latents and
C-RADIO foundation features without changing the consumer.

| Module | Class | Feature tokens |
| --- | --- | --- |
| `radio_vit.py` | `RadioViTBackbone` | C-RADIOv4 spatial features, `D = 1280` (H) or `1152` (SO400M) |
| `radio_trt.py` | `RadioViTBackboneTensorRT` | the same C-RADIOv4 features served by a TensorRT 10.x engine |
| `base_trt/` | `TensorRTEngine` (model-independent) | named-tensor execution for any serialized engine |
| `vae_latent.py` | `VAELatentVisionBackbone` | frozen autoencoder latents, `D =` latent channels |

## Contract

`FrozenVisionFeatures` extends the existing `VisionBackbone` interface with:

- `feature_dim`: width `D` of one feature token.
- `feature_grid(height, width) -> (h, w)`: the raster an `H x W` input
  produces, before any encoding.
- `encode_features(images) -> SpatialFeatureOutput`: finite `[B, 3, H, W]`
  RGB in `[-1, 1]` in, `[B, N, D]` tokens with `N == h * w` and the raster
  out. Batch entries are independent images; views are never mixed, so
  multi-view callers flatten views on the batch dimension and keep their
  own per-view identifiers.
- `freeze()`: clears `requires_grad` and sets evaluation mode. Encoders are
  frozen at construction and encode under `torch.no_grad()`; trainable
  projections belong to the consumer.
- `train()`: a frozen encoder never leaves evaluation mode, even when a parent
  module calls `train()` on the whole tree.

The input range is part of each encoder's own contract, not of the shared
interface: C-RADIO consumes `[0, 1]` and converts the dataset endpoint range,
while the VAE latent wrapper consumes `[-1, 1]` unchanged. Geometry and grid
validation live in the wrapper, not in the consumer: the prepared patch-aligned
input is reported back as `grid`, so consumers build positional geometry without
re-deriving it.

## C-RADIO

```python
from fluxvla.models.backbones.visions import RadioViTBackbone

backbone = RadioViTBackbone.from_local_export(
    '/path/to/C-RADIOv4-H',      # complete local official export
    vision_backbone_id='c-radio_v4-h',
)
output = backbone.encode_features(video)   # [B,3,H,W] in [-1,1]
```

Loading is local-only and strict. `from_local_export` reuses the
repository's C-RADIOv4-H export loader, which resolves the directory with
`local_files_only=True` and rejects a missing asset, a different version,
or a different feature width instead of substituting another model. No
asset is downloaded and no remote code is fetched.

The wrapper requests the `NCHW` spatial stream from the official call
contract (`model(x, feature_fmt='NCHW') -> summary, spatial_features`) and
accepts exports that instead return channels-last `[B, h, w, D]`, channels-first
`[B, D, h, w]`, or a `[B, N, D]` mapping. Every layout is normalized to
`[B, N, D]` and validated against the configured width and the patch-aligned
grid.

Input geometry follows the official helper when the export provides one, even
when it resolves *smaller* than the requested size: shrinking resamples the
image down (padding cannot remove pixels) and growing pads it up, per axis, to
the supported patch-aligned size. Without the helper the size rounds down to the
resolution step, and a resolution that is not a patch multiple fails instead of
producing a mismatched token count.

The version-to-width map (`c-radio_v4-h`: 1280, `c-radio_v4-so400m`: 1152)
is a declaration, not trust: the loaded model is validated against the
configured width, and any mismatch fails. Full-size C-RADIO exports were
not available in this environment, so only the contract and the stub-model
behaviour are verified by tests.

### C-RADIO on TensorRT

`RadioViTBackboneTensorRT` is the accelerated sibling of `RadioViTBackbone`:
the same frozen tokens and raster, served by a serialized TensorRT 10.x engine
built offline from the same local export. It satisfies the same contract, so
the two are interchangeable for any consumer of the generic vision interface.

The engine is compiled for one fixed patch-aligned resolution with a
dynamic-batch profile; an input of any other size is rejected rather than
silently resized, and `default_image_resolution` must equal that resolution so
`num_patches` cannot disagree with the engine. The two encoders are
interchangeable for a consumer whose expected token width and raster match the
compiled geometry; the consumer still owns the tensor device and its own
projection dtype. Construction reads only the metadata sidecar, and TensorRT
is imported on the first encode, so a config can be built where the optional
stack is absent. See [C-RADIO v4 TensorRT](c-radio-v4-tensorrt.md) for the
export/build/validate commands, the metadata contract, and the portability
and precision limits.

## VAE

```python
from fluxvla.models.backbones.visions import VAELatentVisionBackbone

backbone = VAELatentVisionBackbone(
    'vae-flux2',
    autoencoder=autoencoder,     # encode(images) -> [B,C,h,w]
    feature_dim=128,
)
```

The autoencoder is injected rather than re-implemented, which keeps a
single loading policy per encoder family. Latent inputs keep their native
`[-1, 1]` endpoint range and the wrapper does not rescale them.

## Registry

Both classes register with `VISION_BACKBONES`, so `build_vision_backbone_from_cfg`
selects them by `type`. Consumers outside this repository (for example
PatchWAM) import the wrapper instead of copying its encoder or its range
and geometry handling.

## Native PatchWAM integration

The native FluxVLA PatchWAM stack consumes the wrappers directly:

| Layer | Integration |
| --- | --- |
| `PatchWAMBackbone` | `visual_backbone={kind, model_path}` (or an injected `visual_encoder`) builds the shared wrapper; `encode_observations` then produces the reference/future/history tokens and their raster ids from it, with the frozen autoencoder path unchanged. |
| Vendored numerical core | `visual_dim` is independent of the action codec: `PatchFlowPolicy` carries both widths through denoising, the loss and all samplers, and the tiny/official denoisers add direct `D <-> hidden` visual maps when the widths differ (`128 -> hidden` action maps are untouched). |
| `PatchWAMHead` | `visual_dim`, `stage` (`stage1`/`stage2`) and `peft` config; `apply_stage_policy()` freezes everything and opens `visual_projection` (stage 1) or `visual_projection` + `peft` (stage 2), while `peft` with `stage=None` runs `apply_peft_finetune_policy()`: only the adapted `double_blocks`/`single_blocks` bases stay frozen and everything else in the head, the PEFT adapters included, trains. `peft` is a PEFT configuration mapping (`peft_type`, `r`, `lora_alpha`, ...) injected through `peft.inject_adapter_in_model`, so any module-level tuner can be selected; targets are confined to the DiT block containers. The native optimizer selects on `requires_grad`, so it sees exactly those groups. |
| `PatchWAMVLA` | `visual_stage`, `stage1_checkpoint` and `initialize_visual_stage()`: stage 1 inherits released/native pretrained PatchWAM weights, stage 2 inherits a stage 1 checkpoint; only the stage's own new keys may be absent and every inherited head tensor (including the frozen action maps, state projection, DiT blocks and, in stage 2, the stage 1 projections) must be present, with all validation completing before any write. An unstaged `peft` recipe uses the same allow-list for its adapter tensors through `finetune_new_model_keys()`, and resolves released keys onto the PEFT `base_layer` the adapter wraps. |

The staged source is wired into the runner lifecycle rather than being a helper
call: `PatchWAMVLA.from_pretrained()` loads the stage's own source through the
staged allow-list, and `PatchWAMVLA.freeze_backbones()` applies the stage policy.
The DDP runner freezes before it loads and the FSDP runner loads before it
freezes, and both orders end with the same frozen/trainable split: only the
visual projections (stage 1) or the projections plus the DiT PEFT adapters
(stage 2) train, while the backbone is frozen outright so no VL language adapter
is reopened (a staged recipe with trainable VL adapters is rejected). Stage 2
always uses `stage1_checkpoint`; a leftover `pretrained_name_or_path` is
ignored.

The generic legacy checkpoint mapping plus the default strict load are unchanged
(the allow-list only exists on the staged and unstaged-adapter paths).

An unstaged `peft` recipe re-applies its policy from `freeze_backbones()` after
the generic backbone freeze, so the frontend is frozen outright - including the
VL language LoRA residuals `PatchWAMBackbone.apply_trainable_policy()` would
otherwise reopen - and both runner call orders end with the same split.

### Launch

Launch instructions are in each recipe header. `scripts/train.sh` runs torchrun
with `NPROC_PER_NODE` processes per node. Checkpoints are written as
`<work_dir>/checkpoints/step-<step>-epoch-<epoch>[-loss=...].pt` with
`latest-checkpoint.pt`/`.safetensors` symlinks, and `runner.max_epochs` and
`runner.max_steps` are mutually exclusive.

### Native task subset

`PatchWAMParquetDataset(..., task_selection={...})` resolves the subset from the
per-root `tasks.jsonl`/`episodes.jsonl` metadata before sample indexes are built
and before statistics use the episodes. Task indices are root-local, so the task
name is the identity: an integer that means different tasks in different roots
fails, the per-task episode cap spans every root, and `selection_metadata()`
records the resolved task names plus the concrete root/episode ids. The stage 1
and stage 2 ablation recipes declare the same `task_selection`, so both stages
train on identical episodes. Statistics come from the selected episodes
(`selection_metadata()` matches the filtered `stats` list), the resolved
selection is written to `<work_dir>/dataset_selection.json` by `scripts/train.py`,
and a recipe that also configures pre-dataset statistics
(`auto_compute_statistics`/`dataset_statistics`/`dataset_statistics_path`) with a
selection is rejected because those cannot see the subset.

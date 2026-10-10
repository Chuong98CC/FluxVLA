# PatchWAM with C-RADIO feature supervision

`PatchWAMRadioAuxVLA` trains the original PatchWAM model with an extra
training-only objective: a weighted clean C-RADIOv4-H feature MSE read from the
**transformer hidden states of a configurable block** (the last block by
default). The original world/action flow, its shared
FLUX.2 blocks, conditioning, masks and sampling schedule are unchanged, and the
existing transformer initializes exactly as before (fine-tuning is allowed).

The auxiliary adapter is a training dependency. Inference uses the original
PatchWAM model, which loads the exported weights strictly and needs neither the
adapter nor any C-RADIO files.

## Objective

The future RGB endpoint is encoded twice while training: by the frozen VAE into
the original 128-wide `future_tokens`, and by the frozen C-RADIOv4-H teacher
into detached clean spatial targets `[B,S,D_RADIO]` (`D_RADIO = 1280`).

```
loss = video_weight * loss_video          # original VAE flow loss, default 0.5
     + radio_weight * loss_radio          # clean feature MSE,     default 0.5
     + action_weight * loss_action        # original action flow,  default 1.0
     + language auxiliary loss            # folded in once by the VLA
```

`loss_radio` is a masked per-sample feature MSE in float32:

```
hidden     = selected_block_hidden_future_tokens      # [B,S,3072] for Klein 4B
pred       = radio_projection(hidden)                 # [B,S,D_RADIO]
target     = stopgrad(RADIO(future_RGB))              # clean, never noised
loss_radio = mean_over_valid(pred, target)
```

The tap is a hidden future slice, never the 128-wide output projection, and it
reuses the same single transformer pass that produces the flow velocities (there
is no second backbone forward, and no additional adapter or loss). The target is
a clean detached feature map with no timestep weighting; the VAE and action
losses keep their original flow weighting.

### Selecting the adapter block (`adapter_layer_idex`)

`adapter_layer_idex` chooses which block's hidden future tokens feed
`radio_projection`. It is a **zero-based** block index and default `-1` selects
the last block, which is the original behaviour.

Blocks are ordered as all FLUX.2 double blocks followed by all single blocks. For
a model with `D` double blocks and `S` single blocks, `N = D + S` and:

| `adapter_layer_idex` | tapped block |
| --- | --- |
| `-1` (default) | last single block (`N - 1`) |
| `0` | first double block |
| `D - 1` | last double block |
| `D` | first single block (double/single boundary) |
| `N - 1` | last single block, identical to `-1` |

```python
# Supervise the last block (default, unchanged behaviour).
dict(type='PatchWAMRadioAuxHead', ..., adapter_layer_idex=-1)
# Supervise the first double block instead.
dict(type='PatchWAMRadioAuxHead', ..., adapter_layer_idex=0)
```

Values must be integers in `[-1, N - 1]`; booleans, non-integers, `-2` and any
index `>= N` are rejected at construction with a `ValueError` naming the option
and the valid range. Every block still runs, so the original 128-wide
future/action velocities and their losses are unchanged by the choice, and the
auxiliary gradient reaches the selected block and its predecessors while the
velocity objectives still train the later blocks. The selector is plain training
configuration: it adds no parameter or buffer and is removed from the original
PatchWAM `inference_model` config.

`loss_video` (also exposed as `loss_vae`), `loss_radio` and `loss_action` are
reported separately and all three weights are validated as finite and
non-negative. Setting `radio_weight=0` reproduces the original objective exactly
— identical draws, loss values and gradients — with no target requirement, no
teacher load and no hidden-state capture. Setting `video_weight=0` removes direct
world-velocity supervision while the sampler stays the original 128-wide one.

## Inference and the deployable artifact

Sampling is the untouched original PatchWAM sampler: 128-wide VAE future state
and action state, original schedule and CFG semantics, `predict_action` returning
normalized `[B,H,A]`. No teacher, no hidden-state capture and no
`radio_projection` runs at inference.

`PatchWAMRadioAuxVLA.export_inference_state_dict()` returns the original native
PatchWAM state with only the training adapter removed
(`vla_head.radio_projection.weight/bias`). The exported key set and parameter
count match the original model exactly, and a plain `PatchWAMVLA` /
`PatchWAMBackbone` / `PatchWAMHead` loads it with `strict=True` and reproduces
identical seeded actions. The frozen teacher is held outside the module tree, so
it is absent from every checkpoint and from the export.

## C-RADIO teacher

The teacher loads lazily from an explicitly configured **local** Hugging Face
C-RADIOv4-H export:

```python
dict(type='PatchWAMRadioAuxBackbone', ..., radio_model_path='/models/C-RADIOv4-H')
```

Loading uses `AutoConfig`/`AutoModel` with `local_files_only=True` and
`trust_remote_code=True` so the export's own local model code runs; nothing is
downloaded, no package is installed, and there is no fallback to an older RADIO
implementation. The loader validates the export's `version` (`c-radio_v4-h`),
`patch_size` (16) and output feature width, and fails with the offending path
when an asset is missing or mismatched.

The teacher contract:

* Always frozen, `eval()` and under `torch.no_grad()`; parameters are excluded
  from `state_dict`, the optimizer and FSDP units.
* Built and moved to the encoding device/dtype at the first training use.
* Returns spatial features `[N,S,D]` through the official
  `(summary, spatial_features)` tuple, a `RadioOutput` dataclass, a
  `ModelOutput` or a mapping with a spatial key. Summary/adaptor streams are
  never supervised.
* Consumes `[0,1]` RGB: the world model's explicit `[-1,1]` endpoint is
  converted once, and RADIO's own input conditioner normalizes it (no CLIP
  processor).
* Honours the teacher's patch grid and `min_resolution_step` (window x patch),
  pads only to satisfy them, processes each camera view independently, and
  spatially aligns the teacher grid to the existing per-view future raster.
* Rejects valid non-finite pixels and features; invalid frames/views are
  sanitized before the teacher runs and produce exactly zero target energy.
* Cached `radio_future_tokens` follow the same device/dtype convention as cached
  reference/future latents, and supervised coordinates are finiteness-checked
  while masked padding may hold anything.

Inference construction, prediction and checkpoint reload never need RADIO files.

## Config

`configs/patchwam/patchwam_radio_aux_robotwin.py` is the complete native RobotWin
training recipe: three cameras in the compact 288x256 mosaic, mean/std
normalization and the shared runner, with `radio_dim = 1280` and
`radio_patch_size = 16`. Its `inference_model` is the original
`PatchWAMVLA`/`PatchWAMBackbone`/`PatchWAMHead` model intended for
`export_inference_state_dict()` output, with no teacher and no adapter settings.

## Checkpoints

Training saves every trained weight, including `radio_projection`, and never
includes teacher weights. **Native resume is strict**: the generic
`load_state_dict` path (including the `strict=False` calls the DDP runners use)
and `load_patchwam_state_dict` / `from_pretrained` require every trainable
tensor. Missing, shape-mismatched or non-finite trainable tensors are rejected
before any parameter is mutated, so a training run can never silently continue
with a random adapter.

Starting from a pre-RADIO (VAE-lineage) PatchWAM checkpoint is a separate,
explicitly named operation:

```python
model.initialize_from_legacy_state_dict(torch.load(path)['model'])
```

That path leaves only `vla_head.radio_projection.*` at its fresh initialization
(reported as `left_fresh`) while every other trainable tensor is still validated
for presence, shape and finiteness. There is no constructor or config flag that
relaxes native resume.

## Scope and limitations

Full-size C-RADIO/FLUX.2 training and robot benchmarks are not covered here; the
CPU suites exercise reduced official FLUX.2 blocks, deterministic tiny models and
injected teachers. Semantic feature supervision is experimental: these tests do
not claim cheaper training or better control.

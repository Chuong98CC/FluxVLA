# PatchWAM 2D patch motion

`PatchWAM2DMotionVLA` replaces the robot action branch of PatchWAM with signed
2D patch motion. From one current/front RGB view and a language instruction it
jointly generates the future endpoint latent (the original visual flow) and the
per-step displacement of every 16-pixel image patch, then reconstructs absolute
traces. There is no robot state, no robot action chunk and no geometry, depth,
spline or segmentation model.

Public registered names:

| Registry | Name |
| --- | --- |
| `VLAS` | `PatchWAM2DMotionVLA` |
| `HEADS` | `PatchWAM2DMotionHead` |
| `DATASETS` | `PatchWAM2DTraceDataset` |
| (module) | `PatchMotion2DCodec` in `fluxvla.models.third_party_models.patchwam.motion2d` |

## Representation

| Quantity | Value |
| --- | --- |
| Patch spacing | 16 pixels; images are fixed `H x W`, both divisible by 16 |
| Patch grid | `h = H / 16`, `w = W / 16` |
| Motion block | one non-overlapping `8 x 8` region of patches (64 patches, 128 x 128 pixels) |
| Motion token | 128 wide = 64 patches x 2 coordinates |
| Coordinate order | `row, column` (motion is in pixels, not optical flow's column,row) |
| Target | `motion[t] = trace[t + 1] - trace[t]` (consecutive-step displacement) |
| Horizon | `T` consecutive direct future steps; default/example `T = 3` |
| Blocks per step | `K = ceil(h / 8) * ceil(w / 8)` |
| Tokens per sample | `T * K` (e.g. `256 x 256` gives `h = w = 16`, `K = 4`, `T = 3`, 12 tokens) |

Token order is `timestep`, `block row`, `block column`, `local row`,
`local column`, `coordinate`, so the coordinate axis varies fastest. Partial
edge blocks are padded with zero fill and their padded coordinates are masked
out of the loss.

Normalization happens exactly once inside the codec: public motion is signed
pixels, the codec divides by `(H, W)` per-coordinate axis before packing and
multiplies by `(H, W)` once when decoding. There is no sigmoid, clamping or
robot normalization on these values.

The public validity tensor is `[B, T, h, w]` and is shared by both coordinates
of a patch; the codec's mask helpers also accept the equivalent flattened
`[B, T, h*w]` form.

Traces are source-anchored: `trace[0]` is the patch-center grid
`(16 * r + 7.5, 16 * c + 7.5)` and `trace[t]` adds the cumulative sum of the
first `t` predicted displacements. Out-of-frame and negative future positions
are allowed as long as the point is valid.

Attention and positions: current visual tokens keep `(10, row, column, 0)`,
future visual tokens keep `(0, row, column, 0)`, motion tokens use
`(20, block_center_row, block_center_column, timestep)` in the patch-grid
coordinate system with a zero-based timestep. Partial blocks use the center of
their real patch extent. No state token exists on this variant.

## Model API

```python
from fluxvla.engines import build_vla_from_cfg

model = build_vla_from_cfg(dict(type='PatchWAM2DMotionVLA', motion_horizon=3,
                                num_inference_steps=10, vla_head=dict(...)))

# Training: images [B, 1, 3, 2, H, W] in [-1, 1] (current and final future RGB),
# motion in signed pixels, optional shared validity.
out = model(images=images, motion_2d=motion_2d, motion_valid=motion_valid,
            task_description=['pick the cube'])
out['loss'], out['loss_video'], out['loss_motion']

# Inference.
predicted = model.predict_motion(images=images,
                                 task_description=['pick the cube'], seed=0)
predicted['motion_2d']    # [B, T, h, w, 2]  signed pixels
predicted['traces_2d']    # [B, T+1, h, w, 2] absolute source-anchored pixels
predicted['future_tokens']  # [B, S, 128] jointly sampled future raster
```

`predict_motion` derives the motion grid from the raw image size. A caller that
already holds encoded tokens (no raw images) must pass
`motion_grid_shape=(h, w)`. Seeded inference is reproducible.

`predict_action` is a documented compatibility entry point: it returns the same
`[B, T, h, w, 2]` pixel motion tensor and never falls through to a robot
`[B, H, A]` action codec or shape. `loss_action` remains an alias of
`loss_motion` for runner compatibility.

`frame_masks [B, T]` (current first, future endpoint last) and `img_masks [B, V]`
(camera validity) are forwarded into the existing encoder path, so an invalid
future endpoint or padded view masks the reference/future rasters and the video
loss instead of supervising them. Cached-token callers pass `reference_valid` /
`future_valid` directly; supplying `frame_masks`/`img_masks` without a backbone
is rejected rather than silently ignored.

This variant is state-free: `states`/`actions`/`action_masks` inputs and a
non-`None` state projection are rejected at construction or on the first call
instead of being silently ignored. A robot PatchWAM checkpoint can still
initialize the shared transformer: its recognized `state_projection` /
`proprio_encoder` tensors are dropped deliberately and reported through
`load_patchwam_state_dict(...)['dropped_state_projection']`, while any other
missing or unexpected tensor still fails a strict load.

## Offline dataset format

```text
dataset/
  dataset.json
  train.jsonl
  val.jsonl
  rgb/<episode_id>/<frame_id>.png
  labels/<sample_id>.npz
```

`dataset.json`:

```json
{
  "schema_version": "patchwam_2d_trace_v1",
  "image_height": 256,
  "image_width": 256,
  "patch_size": 16,
  "motion_block_size": 8,
  "horizon": 3,
  "coordinate_order": "row_column",
  "timestep_offsets_s": [0.25, 0.5, 1.0]
}
```

`timestep_offsets_s` holds `T` strictly increasing positive sample times
relative to the current frame and must be a single fixed schedule per dataset.
`train.jsonl` and `val.jsonl` must use different episodes, and every
`sample_id` must be unique across both splits as well as within a split (ids are
the label-file key, so a repeat would make two rows share one
`labels/<sample_id>.npz`). Each row needs `sample_id`, `episode_id`,
`instruction`, `current_rgb`, `future_rgb` (the final selected target step) and
`labels`; every path is relative to the dataset root and may not escape it. The
current and future cameras must be static or mapped into the same view.

Each `labels/<sample_id>.npz` archive contains:

| Key | dtype / shape | Meaning |
| --- | --- | --- |
| `traces_2d` | float32 `[T+1, h, w, 2]` | absolute tracked `row, column` pixel positions |
| `trace_valid` | bool `[T+1, h, w]` | reliable point observations |
| `trace_t_seconds` | float64 `[T+1]` | `0` followed by the declared `timestep_offsets_s` |

At valid initial positions `traces_2d[0]` must match the patch centers within
`1e-3` pixels. Invalid positions may use nonfinite placeholders, but a nonfinite
value at a valid position is rejected, as is a valid displacement that overflows
after subtraction of finite float32 endpoints. `traces_2d` must be float32 and
`trace_t_seconds` must be float64 with every entry finite. Displacements and
their masks are derived by the dataset, never stored:
`motion[t] = trace[t+1] - trace[t]` and
`motion_valid[t] = trace_valid[0] & trace_valid[t] & trace_valid[t+1]`, so a
lost intermediate point invalidates both adjacent displacement targets.

`PatchWAM2DTraceDataset` validates the manifest, paths, NPZ shapes/dtypes,
timing, coordinate order, duplicate ids, split disjointness, RGB mode/size and
finite valid positions, and raises instead of silently converting a malformed
file into a valid sample. It returns native model inputs plus pixel motion;
normalization stays owned by the codec. There is no random spatial augmentation
in this version (it would need a coordinated trajectory transform) and no
robot joint statistic.

### Exporting a dataset

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
PATCH, BLOCK = 16, 8
OFFSETS_S = [0.25, 0.5, 1.0]
HORIZON = len(OFFSETS_S)
H, W = IMAGE_HEIGHT // PATCH, IMAGE_WIDTH // PATCH

root = Path('datasets/patchwam_2d_trace')
(root / 'rgb' / 'ep0').mkdir(parents=True, exist_ok=True)
(root / 'labels').mkdir(parents=True, exist_ok=True)

root.joinpath('dataset.json').write_text(json.dumps(dict(
    schema_version='patchwam_2d_trace_v1',
    image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH,
    patch_size=PATCH, motion_block_size=BLOCK,
    horizon=HORIZON, coordinate_order='row_column',
    timestep_offsets_s=OFFSETS_S)))

# The anchor every valid trace zero must reproduce.
rows = (np.arange(H) * PATCH + 7.5)[:, None]
columns = (np.arange(W) * PATCH + 7.5)[None, :]
centers = np.stack(np.broadcast_arrays(rows, columns), axis=-1).astype(np.float32)

def write_sample(sample_id, episode_id, instruction, frames, tracked, valid,
                 current_frame_id, future_frame_id, root=root):
    """Frames: RGB uint8 [T+1, IMAGE_HEIGHT, IMAGE_WIDTH, 3].

    Tracks: float32 [T+1, H, W, 2], where H,W are patch-grid dimensions.

    `current_frame_id`/`future_frame_id` are the tracker's own frame names, so
    two samples from one episode never overwrite each other's images.
    """
    label_path = root / 'labels' / f'{sample_id}.npz'
    if label_path.exists():
        raise ValueError(f'Duplicate sample_id: {sample_id}')
    tracked = np.asarray(tracked, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    # Trace zero is the patch centers wherever the point is observed.
    tracked[0] = np.where(valid[0][..., None], centers, tracked[0])
    episode_root = root / 'rgb' / episode_id
    episode_root.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for name, index in ((current_frame_id, 0), (future_frame_id, HORIZON)):
        path = episode_root / f'{name}.png'
        frame = np.asarray(frames[index], dtype=np.uint8)
        if path.exists():
            with Image.open(path) as saved:
                if not np.array_equal(np.asarray(saved), frame):
                    raise ValueError(f'Conflicting frame_id: {path}')
        else:
            Image.fromarray(frame).save(path)
        image_paths.append(str(path.relative_to(root)))
    np.savez(label_path,
             traces_2d=tracked,
             trace_valid=valid,
             trace_t_seconds=np.asarray([0.0, *OFFSETS_S], dtype=np.float64))
    return dict(sample_id=sample_id, episode_id=episode_id,
                instruction=instruction, current_rgb=image_paths[0],
                future_rgb=image_paths[1],
                labels=f'labels/{sample_id}.npz')

# `tracked`/`valid` come from your tracker, sampled at 0 and OFFSETS_S seconds.
# train_rows and val_rows must reference disjoint episodes and disjoint ids:
#   train_rows = [write_sample('ep0_s0', 'ep0', 'move', frames_a, tracked_a,
#                              valid_a, 'f000', 'f003'),
#                 write_sample('ep0_s1', 'ep0', 'move', frames_b, tracked_b,
#                              valid_b, 'f010', 'f013')]
#   val_rows   = [write_sample('ep1_s0', 'ep1', 'push', frames_c, tracked_c,
#                              valid_c, 'f000', 'f003')]
root.joinpath('train.jsonl').write_text(
    '\n'.join(json.dumps(row) for row in train_rows) + '\n')
root.joinpath('val.jsonl').write_text(
    '\n'.join(json.dumps(row) for row in val_rows) + '\n')
```

The loader decodes the PNGs itself (8-bit RGB) and normalizes them to `[-1, 1]`,
so no extra preprocessing is required.

## Training

`configs/patchwam/patchwam_2d_motion.py` is a complete native example:

```text
loss = video_weight * loss_video + motion_weight * loss_motion   # 0.5 / 1.0
```

It reuses the frozen `PatchWAMBackbone` frontends and the trainable joint flow
DiT, so only the new head/codec wrappers plus the dataset are variant specific.
Set `PATCHWAM_CKPT_ROOT`, `PATCHWAM_FLUX2_SRC` and `PATCHWAM_2D_TRACE_ROOT` to
your local asset and dataset locations, then launch with the repository's
standard runner entry point for a `DDPTrainRunner` config.

## Scope and limits

LoRA is not required for this integration; the existing trainable DiT is the
supported training path. Deferred/out of scope: VGGT, depth, cameras, geometry
adapters, changed visual RoPE, 3D coordinates, splines, segmentation or
probability channels, pseudo-label generation, robot execution/evaluation and
random spatial augmentation. Full-size FLUX.2 asset training, GPU verification
and any real training run are also out of scope here.

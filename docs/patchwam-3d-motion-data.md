# PatchWAM with future depth and 3D patch trajectories: offline data contract

Status: architecture agreed, including the ReChannel-style future-depth readout;
implementation has not started. This contract incorporates the user's confirmed
joint sampling, frozen VGGT conditioning, and cubic spline decisions. A loader
for this format has not been implemented. See the
[architecture specification](agent-work/patchwam-3d-motion/SPEC.md) and
[implementation plan](agent-work/patchwam-3d-motion/PLAN.md).

## Agreed model behavior

- Inputs are the front-view current RGB image and language, with no robot state.
- VGGT is frozen. Its observed-image patch features enter a trainable two-layer
  MLP adapter with a learnable gate initialized to zero. The base PatchWAM DiT
  is frozen and adapted through LoRA; the RGB autoencoder remains frozen.
- Current and future visual tokens share the same `(row, column, depth_bin)`
  coordinates, computed from **current-image depth**. Their existing group tags
  remain distinct: `10` for the current view and `0` for the future view.
- The camera is static, or future RGB and geometry are mapped into the current
  camera view before export. Traces still follow the same material points.
- Each non-overlapping 4x4 group of image patches produces a width-128 motion
  token per predicted spline control: 16 local patches times 8 coordinates.
- Three control planes are predicted. The current 3D point is supplied by the
  observed geometry and is not predicted.
- Moving/static classification applies to patches with a known movement label.
  Object/manipulator classification applies only to ground-truth moving patches
  with a known class. Static patches receive no object/manipulator loss.
- Sigmoid applies to classification logits. Signed 3D offsets and flow
  velocities remain unconstrained.
- The future branch predicts endpoint depth directly, replacing future RGB
  latent prediction. Following ReChannel's direct readout principle, depth
  targets do not enter the RGB VAE and use no target-side VAE decoder.
- Each 16x16 future depth patch contains 256 continuous log-depth values.
  Separate trainable depth input/output projections map `256 -> D -> 256`
  around the shared DiT hidden width `D`. Current RGB and motion retain their
  width-128 interfaces. Depth and motion use one joint flow schedule, with
  independent noise draws and attention between the generated branches.

## Export layout

Use one manifest row per current-image anchor and one numerical label archive per
row. Export numeric arrays with NumPy `savez` or `savez_compressed`; all members
must be ordinary numeric arrays, loadable with `allow_pickle=False`.

```text
dataset/
  dataset.json
  train.jsonl
  val.jsonl
  rgb/<episode_id>/<frame_id>.png
  labels/<sample_id>.npz
```

JPEG is also accepted. Paths in manifests are relative to the dataset directory.
Split whole episodes between training and validation, so overlapping trajectory
windows cannot cross splits. Existing robot-action Parquet exports can remain
separate; they are not required to generate this new export.

Example `dataset.json` (the numerical geometry and timing values are examples):

```json
{
  "schema_version": "patchwam_3d_depth_trace_v1",
  "image_height": 256,
  "image_width": 256,
  "patch_size": 16,
  "motion_block_size": 4,
  "trajectory_duration_s": 1.0,
  "trace_query_times_s": [0.25, 0.5, 1.0],
  "future_target": "endpoint_camera_z_depth",
  "depth_representation": "log_depth_over_current_median",
  "coordinate_frame": "current_camera_opencv",
  "length_unit": "vggt_scene",
  "meters_per_unit": null,
  "moving_label_rule": "maximum displacement from the current point exceeds 0.01 times median valid current patch depth",
  "anchor_tolerance_scene_fraction": 0.001,
  "geometry_provenance": {
    "model": "VGGT-Omega",
    "checkpoint_sha256": "REPLACE_WITH_ACTUAL_CHECKPOINT_HASH",
    "source_revision": "REPLACE_WITH_ACTUAL_SOURCE_COMMIT",
    "conditioning_context": "current_image_only",
    "preprocessing": "Describe the exact crop, resize, and padding used"
  },
  "trace_provenance": "Describe the trace pipeline and its version",
  "future_depth_provenance": "Describe endpoint depth generation and alignment to current geometry",
  "spline": {
    "degree": 3,
    "knots": [0, 0, 0, 0, 1, 1, 1, 1],
    "predicted_controls": 3,
    "control_representation": "offset_from_current_point"
  }
}
```

Replace example thresholds, duration, and provenance with actual generation
settings. `trajectory_duration_s` is fixed within a training dataset/config;
three control planes do not specify its duration. `trace_query_times_s` is a
fixed, finite, strictly increasing schedule in `(0, trajectory_duration_s]`,
including the endpoint. It controls decoded trajectory samples, not the control
count or number of raw tracking timestamps. Different duration policies must
be explicitly represented in a later model/data revision.

`patchwam_3d_depth_trace_v1` is distinct from the previously proposed
`patchwam_3d_trace_v1`, which required future RGB instead of future depth.
Do not silently accept an old export as a depth-supervised dataset.

Set `length_unit` to `meter` and `meters_per_unit` to `1.0` only when metric scale
is established. Otherwise use `vggt_scene` and `null`; all geometry in a sample
must share one scale, including current depth and the complete trace. If a
VGGT-scene reconstruction has a known conversion to meters, record that positive
conversion instead. Mixed or changing scale within one sample is invalid.

Example manifest row:

```json
{"sample_id":"ep0001_anchor000120","episode_id":"ep0001","current_frame_index":120,"future_frame_index":150,"current_timestamp_s":4.0,"future_timestamp_s":5.0,"instruction":"Move the cup to the left","current_rgb":"rgb/ep0001/000120.png","future_rgb":"rgb/ep0001/000150.png","labels":"labels/ep0001_anchor000120.npz"}
```

Only `current_rgb` and `labels` paths are required. `future_rgb` is optional
provenance/debugging data; it is not encoded or consumed by the model. Future
depth in the label archive is the visual target. Only the current image and
instruction are used at inference; frozen VGGT derives current geometry from
that image. Frame indices identify recorded frames; timestamps determine the
physical prediction interval. RGB is stored in RGB channel order as uint8
images, without application of model-specific normalization.

## Required arrays in each label archive

Let `H,W` be the exported image dimensions, `h=H/16`, `w=W/16`, and `F` the number
of trace timestamps, including current and endpoint times. All samples use the
same image dimensions. `F` may vary between samples.

| Array | dtype and shape | Meaning |
| --- | --- | --- |
| `depth_current` | float32 `[H,W]` | Observed-image camera-Z depth in the declared length unit; not min/max-normalized |
| `depth_valid` | bool `[H,W]` | Reliable, finite, positive current depth |
| `depth_future` | float32 `[H,W]` | Future endpoint camera-Z depth in the current view and the same length unit/scale as current depth and traces |
| `depth_future_valid` | bool `[H,W]` | Reliable, finite, positive endpoint depth |
| `intrinsics_current` | float32 `[3,3]` | Intrinsics for the exact exported image grid |
| `trace_t_seconds` | float64 `[F]` | Strictly increasing elapsed seconds; first `0`, last `trajectory_duration_s` |
| `traces_ref` | float32 `[F,h,w,3]` | Positions of the same tracked material points in the current camera coordinates |
| `trace_valid` | bool `[F,h,w]` | Reliability of each traced position; occlusion or tracking failure is invalidity, not static motion |
| `moving_label` | bool `[h,w]` | Whether the tracked patch-center point moves anywhere in the full prediction interval |
| `moving_valid` | bool `[h,w]` | Whether moving/static status is known |
| `object_label` | bool `[h,w]` | `True` = manipulated object; `False` = manipulator, where the class is valid |
| `object_valid` | bool `[h,w]` | Whether object/manipulator status is known; ignored for static patches |

Use finite zero fillers for invalid positions and depth values. Masks, rather
than NaNs or invented labels, specify invalidity. Valid depth must be positive.
Valid traces may have signed X/Y coordinates but must contain finite XYZ values.
Do not require a future point to remain inside the original image: a reliable
3D track can remain a valid motion target after leaving the image.

`intrinsics_current` must describe the final exported crop/resize/padding,
including principal-point shifts. It must use the same image-coordinate
convention as the geometry pipeline. Do not export original-camera intrinsics
beside resized images without transforming them.

Optional endpoint consistency supervision uses `trace_visible` and
`trace_visibility_valid`, both bool `[F,h,w]`. Visibility follows each tracked
material point and means visible on the depth surface, not merely in frame.
Unknown visibility is not positive visibility. A nonzero consistency-loss
weight requires these arrays. Dense future depth and sparse tracked XYZ labels
must be aligned to the same current-camera frame and scale before export;
independent per-frame min/max normalization is invalid.

No robot state, joint angles, prepacked action tokens, sigmoid values, depth
bins, or normalized motion values are required on disk. Binary classification
labels remain binary; the training loader owns the representation mapping.

## Point indexing and coordinates

The reference grid is anchored to the **current** image. In a zero-based pixel
index system where integer coordinates denote pixel centers, patch `(r,c)` has
center `(u,v) = (16*c + 7.5, 16*r + 7.5)`. Bilinearly sample its depth from the
four neighboring pixels; require their depth validity for the v1 sample.

Back-project with `intrinsics_current`, using the declared matching pixel
coordinate convention. Camera coordinates use OpenCV directions: X right,
Y down, Z forward. Every future trace position is expressed in this same frame.

`traces_ref[j,r,c]` follows the point initialized at patch `(r,c)` at time zero.
It is **not** the point currently occupying pixel `(r,c)` at timestamp `j`.
The user's correspondence pipeline supplies this association.

At valid anchors, `traces_ref[0,r,c]` must agree with the point reconstructed from
`depth_current` and intrinsics within the declared tolerance, scaled by median
valid patch depth. Align the trace reconstruction's frame and scale before
export. Reject or mask anchors that cannot be aligned reliably; do not silently
mix reconstruction gauges.

Future depth uses the same view grid as current RGB, as specified by the user.
It is a raster field describing the surface visible at each future pixel;
`traces_ref` follows material points initialized on the current grid. Shared
visual RoPE depth remains the depth of the current grid anchor even when scene
content has moved. This is an intentional positional convention.

## Future depth representation and joint sampling

Compute `s` as the median of reliable positive current patch-center depths;
require at least one reliable current patch and preserve `s` for decoding.
Pack `q = log(depth_future / s)` into `[B,h*w,256]`, one row-major 16x16 patch
per token. Depth validity has the same packed shape. Invalid fillers become
zero in log space before noising. These 256 values are continuous pixel depths,
independent of the 256 quantization bins used for visual positional IDs.

The future depth input projection is `Linear(256,D)`, and its output readout
is `Linear(D,256)` with the existing style of time-conditioned final
normalization. There is no width-128 bottleneck in this depth flow and no RGB
VAE target encoding/decoding. Current RGB and motion retain native width-128
projections; all branches share the transformer hidden width `D`.

With the existing native convention, `q_sigma = (1-sigma)*q + sigma*noise`
and the velocity target is `noise - q`: sigma zero is data, sigma one is noise.
Use the same sigma/schedule for depth and motion, independent Gaussian draws,
prefix-masked conditioning, and mutual depth/motion attention. Inference
integrates both states from sigma one to zero; changing the seed samples
another paired future. Recover clean log-depth as
`q_hat = q_sigma - sigma * predicted_velocity` for auxiliary training losses,
and decode `depth_hat = s * exp(q_hat)`. Do not exponentiate velocities.

Train masked depth flow and clean log-depth regression without independent
per-frame scale alignment. Optional geometric consistency projects the predicted
spline endpoint into predicted future depth and compares camera-Z only for
known visible, reliable, positive-Z, in-frame points with valid bilinear depth
support. Observed geometry is conditioning; future geometry is supervision.

## Confirmed spline target

The user confirmed this convention: use a clamped cubic B-spline with
four total controls `C0,C1,C2,C3`, normalized time `u=t/duration`, and knot vector
`[0,0,0,0,1,1,1,1]`. Fix `C0` at the current point. Predict three signed offsets:

```text
delta_C1 = C1 - C0
delta_C2 = C2 - C0
delta_C3 = C3 - C0
```

This knot convention gives a single cubic segment, equivalently a cubic Bezier
curve:

```text
P(u) = (1-u)^3*C0 + 3*(1-u)^2*u*C1 + 3*(1-u)*u^2*C2 + u^3*C3
```

The first and last controls are curve endpoints. Interior controls are not
future positions at one-third and two-thirds of the interval. Offsets are from
`C0`, not differences between consecutive controls or velocities per second.

Fit controls from the raw trace using valid timestamps and least squares with
`C0` fixed. For this proposal, require a reliable endpoint, fix `C3` to that
endpoint, and estimate `C1,C2` from the interior trace. Require at least two
distinct reliable interior timestamps plus the current and endpoint positions,
and a full-rank fit. More than four trace timestamps is preferable. Mark an
unreliable or ill-conditioned fit invalid rather than fabricating controls.

The canonical export keeps raw traces, so changing the spline convention does
not require regenerating correspondences. If controls are cached later, use
`control_offsets_ref` float32 `[3,h,w,3]` and `control_valid` bool `[h,w]`, and
record degree, knots, and fitting policy. Cached controls are optional and must
match the selected spline convention before they can replace fitting.

## Classification and masking

Movement is a trajectory-level label. For example, derive it from maximum
distance to the initial point across the trace, using the declared threshold;
using endpoint displacement alone would miss points that move and return.
Incomplete traces cannot establish static status without another reliable
label source. Export `moving_valid=False` when status is unknown.

The object label is the identity of the tracked material point, not the pixel
contents at a later timestamp. An unknown or unrelated moving entity can have a
known movement label with `object_valid=False`.

The object/manipulator supervision mask is:

```text
object_loss_valid = anchor_valid & moving_valid & moving_label & object_valid
moving_loss_valid = anchor_valid & moving_valid
motion_loss_valid = anchor_valid & spline_fit_valid
```

`anchor_valid` comes from current depth and initial-trace alignment. All masks
are extended with the spatial padding mask during token packing. Known static
patches have valid zero motion only when the trace/label source justifies that
target. Unknown patches must not become supervised zero-motion background.

Train object classification using ground-truth movement in the loss mask, not
the predicted movement probability. Thus early classifier mistakes do not
remove object supervision. A training dataset needs some valid moving-object
and moving-manipulator labels to learn both categories; 3D traces alone do not
provide their identities.

## Packing, normalization, and decoder behavior

Let `K=ceil(h/4)*ceil(w/4)`. Pad incomplete boundary blocks and exclude padded
local patches from every applicable loss. Enumerate controls first, then blocks
in row-major order, and local patches within each block in row-major order.

Each local patch contributes:

```text
[delta_Ck.x, delta_Ck.y, delta_Ck.z,
 delta_Ck.x, delta_Ck.y, delta_Ck.z,
 moving_code, object_code]
```

Each block is flattened `[16,8] -> [128]`. The model receives `[B,3*K,128]`,
and the decoder restores `[B,3,h,w,...]`. At 256x256, `K=16`: three predicted
controls require **48 motion tokens**. Keep `num_predicted_controls=3` separate
from the future-image frame offset and the number of raw trace timestamps.

A recommended VGGT-scale normalization divides all three offset axes by the
median valid current patch depth. Preserve that positive scale to restore
signed offsets after decoding; use current geometry only to compute it.
Do not min/max-normalize signed offsets to the depth-bin range. Alternative
motion statistics are a training configuration choice and must be recorded.

The proposed classification representation keeps the existing joint flow:
encode binary labels as finite signed logit codes, e.g. `-1,+1`, representing
logits `-L,+L` with `L=log((1-epsilon)/epsilon)` and `epsilon=0.01`. Repeat each
trajectory label across its three control tokens. A static or unknown object
coordinate has a finite zero filler and no object-coordinate flow or
classification supervision.

Apply the sigmoid to the recovered clean classification logits, not to flow
velocities. For additional classification supervision, the existing flow
convention gives `clean_hat = noisy - sigma * predicted_velocity`; apply
`BCEWithLogitsLoss` to the reconstructed logits. Aggregate the three recovered
control logits for each source patch by their mean before sigmoid/BCE to produce
one moving probability and one conditional object probability per trajectory.
Keep motion, segmentation flow, and classification terms separately weighted;
repetition must not multiply the effective segmentation weight.

Decode each signed control offset by averaging its two XYZ copies. Reconstruct
absolute controls with `Ck=C0+delta_Ck`, evaluate the spline at any desired times,
and return moving and conditional object probabilities in `[0,1]`. Object
classification has meaning only for moving patches. Inference may threshold
movement for a hard mask; training uses the ground-truth mask.

## Frozen geometry and preprocessing

`depth_current` and any cached adapter features must be computed using the
current observation only. The offline trace pipeline may use future frames to
produce **labels**, but its future-conditioned features or depth must not be
reused as model conditioning. Align its label frame and scale to observed
geometry before export.

Optional cached feature array: `geometry_features_current` float16 or float32
`[h,w,2048]` for the default VGGT-Omega final aggregator patch features. Strip
camera/register tokens. Record the actual feature tap, channel width, model
revision, checkpoint hash, and preprocessing in provenance. A different
geometry model requires an explicit feature-width/tap configuration.

Use matching spatial preprocessing for the autoencoder, VGGT, depth, and trace
grid. V1 should consume the already exported image geometry without additional
random crops, flips, or resizes. Color augmentation is possible separately.
Spatial augmentation needs an explicit coordinated transform of image, depth,
intrinsics, patch identities, and motion labels; the existing RGB-only spatial
transforms cannot be reused unchanged.

## Validation and integration acceptance

Before accepting an export, check:

1. Unique sample IDs, existing required paths, matching current RGB/future depth
   dimensions, optional future RGB dimensions, and episode-level split separation.
2. Positive fixed prediction duration, consistent manifest/archive clocks,
   increasing trace timestamps, and image dimensions divisible by 16.
3. Exact array shapes/dtypes, finite fillers and valid values, positive valid
   depths, valid intrinsics, declared scale, and initial-point alignment.
4. Reliable correspondence masks, known movement labels, conditional class
   masks, and nonempty object/manipulator supervision across the training set.
5. Spline rank/endpoint rules, constant traces decoding to zero offsets, signed
   motion preserved, and masks retained through partial 4x4 boundary blocks.
6. Equal current/future `(row,column,depth_bin)` IDs with distinct group tags;
   depth bins are computed from valid current patch depths with clamping and a
   constant-depth fallback. Feature extraction never consumes target frames.
7. Static object channels contribute no loss; changing a static object's filler
   label cannot change the supervised loss. Sigmoid outputs remain bounded.
8. Depth patch pack/unpack is exact, future values are continuous rather than
   positional bins, shared scale restores physical depth, and no future RGB
   target encoding or independently normalized depth enters training.

Implementation will need a dedicated native dataset/transform path, depth and
motion codecs/masks, branch-specific DiT input/output projections and LoRA,
custom spatial/control RoPE IDs, classification-aware losses, and joint
sampling/decoding. It must preserve existing robot-action, RADIO, and 2D motion
models and pre-existing workspace changes. These components are not implemented.

## References

- [Current native PatchWAM architecture](../papers/PatchWAM_Architecture_Simplified.md)
- [Native dataset conventions](../fluxvla/datasets/patchwam_dataset.py)
- [Native token geometry](../fluxvla/models/third_party_models/patchwam/geometry.py)
- [Flow timing and clean reconstruction convention](../fluxvla/models/third_party_models/patchwam/flow.py)
- [VGGT-Omega feature interface](https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/models/vggt_omega.py)
- [SciPy B-spline degree, control, and knot definitions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)
- [ReChannel direct field readout](https://arxiv.org/html/2607.06553v2)
- [Local ReChannel depth readout](/home/chuong/workspace/ReChannel/train.py)

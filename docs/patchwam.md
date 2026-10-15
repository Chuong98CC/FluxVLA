# PatchWAM in FluxVLA

PatchWAM is a joint world/action rectified-flow model. FluxVLA integrates it as
native components: a vendored numerical core, two registered model components,
native dataset/transform extensions, and configs that reuse the existing
runners, statistics, collators, normalizers, and inference scripts.

## Components

| Registered type | Registry | Owns |
| --- | --- | --- |
| `PatchWAMVLA` | `VLAS` | Wires backbone + head; native forward/prediction |
| `PatchWAMBackbone` | `VLM_BACKBONES` | Frozen packed-latent autoencoder + instruction encoder |
| `PatchWAMHead` | `HEADS` | Denoiser, proprio projection, action codec, flow objective |
| `PatchWAMTinyDenoiser` | `HEADS` | Offline tiny joint transformer (CPU tests only) |
| `PatchWAMFluxDenoiser` | `HEADS` | Official FLUX.2 prefix-masked denoiser loader |
| `PatchWAMParquetDataset`, `PatchWAMLiberoEvalDataset` | `DATASETS` | Training dataset + endpoint/history metadata; eval dataset retaining raw prompts |
| `PatchWAMRawInputs`, `PatchWAMDecodeFrames`, `PatchWAMResizeCameras`, `PatchWAMEvalPack`, `PatchWAMSelectWindow`, `PatchWAMSelectHistory`, `PatchWAMEndpoints`, `PatchWAMCleanVL`, `PatchWAMWorldAugment`, `PatchWAMComposeWorld`, `PatchWAMWorldHistory`, `PatchWAMKeep`, `PatchWAMHistoryBuffer` | `TRANSFORMS` | Frame decode, geometry, augmentation, packing, online history |

The head and backbone hold disjoint parameters; `PatchWAMVLA.all_module_keys`
is `['vlm_backbone', 'vla_head']` for checkpointing and FSDP.

## Data flow

1. `PatchWAMParquetDataset(training=...)` exposes the native action window
   (16 targets with
   `window_start_idx=0`, `action_window_size=16`) and adds, before the transform
   list, episode/root-bounded endpoint metadata (`patchwam_future_row` =
   `start + 16`) and original-clock history rows (`patchwam_history_rows`) via
   `PastFrameSelector`.
2. `PatchWAMDecodeFrames` decodes only the requested episode-local world and VL
   history timestamps (torchcodec/pyav) and zeroes invalid slots;
   `PatchWAMSelectWindow` / `PatchWAMSelectHistory` remain for a decoded-tensor
   path and stay episode-bounded.
3. `PatchWAMCleanVL` snapshots the clean head-camera current/history images
   before world-camera resizing and augmentation for the Qwen3-VL encoder.
4. `PatchWAMResizeCameras` resizes each world/history camera to its source
   geometry (LIBERO 224x224, RobotWin 240x320, RoboDojo 256x256) before
   augmentation.
5. `PatchWAMEndpoints` selects the current/future endpoint pair and emits
   `frame_masks` `[2]` and `img_masks` `[V]` (`True` means valid).
6. `PatchWAMWorldAugment` applies the copied `ClipAugment` /
   `AppearanceRandomizer` to the world and history clips only (one probability
   draw per sample, independent cameras, shared temporal parameters).
7. `PatchWAMComposeWorld` builds the mosaic and/or separate views;
   `PatchWAMWorldHistory` stacks world history; `PatchWAMKeep` prunes to the
   collator contract.

`PatchWAMBackbone.encode(batch, target)` then produces `reference_tokens`,
`future_tokens`, `reference_ids` (`group=10+view`), `future_ids` (`group=view`),
`reference_valid` / `future_valid`, `text_tokens`, and `sampling_future_length`.
The generated future raster is always the current-view raster; an appended
historical prefix never changes it.

## Configs

`configs/patchwam/`:

* `patchwam_libero_full_finetune.py` - LIBERO, two-camera horizontal mosaic,
  online Qwen3 text encoder.
* `patchwam_robotwin_vlm_full_finetune.py` - RobotWin compact 288x256 mosaic
  with three cameras and the **frozen, no-LoRA, current-image-only** Qwen3-VL
  encoder (source `robotwin_vlm.yaml`; no history).
* `patchwam_robodojo_vlm_history_example.py` - separate views with causal
  world-model + VL history.
* `patchwam_robotwin_vlm_5tasks_full_finetune.py` - the RobotWin VLM recipe
  restricted to the five tasks `turn_switch`, `rotate_qrcode`,
  `blocks_ranking_size`, `move_stapler_pad`, and `hanging_mug`, each listed as
  its own RoboTwin-Clean root, with isolated `robotwin_5tasks` statistics.

All of them point at local assets under `_ckpt_root` and the local black-forest-labs
`flux2` checkout via `flux_source`. Nothing is downloaded.

Each config defines a complete native `runner` (`DDPTrainRunner`) with the
collator, optimizer, LR schedule, and metric in the native locations, a child
dataset whose `statistic_name` matches the `DistributedRepeatingDataset`
wrapper, `action_key='action'`, explicit name mappings, native normalization
(`NormalizeStatesAndActions`) with the source padded-delta recipe, and (for
LIBERO) a `LiberoEvalRunner` eval block with `DenormalizeLiberoAction`. Train and
validation configs are independent objects: validation disables world
augmentation and history jitter/dropout.

### Running

```
python scripts/train.py --config configs/patchwam/patchwam_libero_full_finetune.py
python scripts/inference.py --config configs/patchwam/patchwam_libero_full_finetune.py \
    --ckpt-path <native_checkpoint>
```

The RobotWin recipe matches the source non-idle training subset, so it reads
that filter from the environment:

```
PATCHWAM_NONIDLE_FILTER=/path/to/nonidle_rows.json \
    python scripts/train.py --config configs/patchwam/patchwam_robotwin_vlm_full_finetune.py
```

The JSON maps each recording's episode index to a list of half-open `[start,
end)` physical row intervals, either at the top level or under an `"episodes"`
key: `{"episodes": {"0": [[12, 200], [260, 900]]}}`. Retained rows are the
sorted union of the clipped intervals, so overlaps collapse and indices past
the recording are ignored. An episode without a key keeps every row; an
explicit `[]` keeps none. The action window and the future endpoint advance
over the *retained* rows, while history keeps resolving on the original
recording clock, including rows the filter removed. The environment variable
holds one JSON filename; for a separate file per root, pass a mapping to the
dataset config in Python instead: `nonidle_filter_path={root: path}`, in which
case every root must be listed. The recipe sets `require_nonidle_filter=True`,
so a missing value fails at dataset construction instead of silently training
on idle frames. The other recipes leave the dataset unfiltered.

The five-task ablation narrows both splits to five RoboTwin-Clean tasks. In this
release every task is its own LeRobot v2.1 root under one parent directory, so
the recipe lists five explicit roots and no `datasets.task_selection` is used —
listing the roots *is* the filter, and its `tasks.jsonl` holds ordinary
instruction strings. It inherits the RobotWin VLM recipe unchanged and only
overrides the roots, the (absent) non-idle filter, the asset paths, and the
statistics namespace:

```
python scripts/train.py \
    --config configs/patchwam/patchwam_robotwin_vlm_5tasks_full_finetune.py \
    --work-dir work_dirs/patchwam_robotwin_vlm_5tasks_full_finetune
```

`PATCHWAM_DATA_ROOT` names the *parent* directory (default
`/data/RoboTwin/RoboTwin-Clean`) and each root is `$PATCHWAM_DATA_ROOT/<task>`.
The five requested roots stay listed even when the release has not published one
yet: a missing folder fails at dataset construction instead of silently training
on a smaller subset. This release ships no non-idle row filter, so
`nonidle_filter_path` defaults to `None` and `require_nonidle_filter` to `False`;
an explicit `PATCHWAM_NONIDLE_FILTER` still applies it and switches the required
contract back on. Its task roots report 15 fps, so with the unchanged 16-step
action horizon a window spans about 1.07 s of recording rather than 0.32 s at
50 fps; no resampling is applied and the dataset reads FPS from metadata.

The subset records its statistics under the `robotwin_5tasks` namespace so its
keys do not collide with the full-data `robotwin` recipe. `scripts/train.py`
writes that file into the run's `--work-dir`, so keep the ablation in its own
directory (as above); the separate namespace is what keeps the two recipes'
statistics distinct. The normalization method is unchanged.

`tools/patchwam_robotwin_5tasks_smoke.py` preflights the five configured task
roots and reproduces the dataset and forward/loss checks against the local
release without writing into it. `selection` reports per-folder episodes, FPS,
cameras and instruction provenance, and exits nonzero naming exactly which
requested folders the release does not contain; `dataset`/`train` refuse to run
while any configured root is missing. The helper sets
`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`/`HF_DATASETS_OFFLINE=1` itself, and with
`--backward` also asserts a finite scalar objective, a nonzero trainable
gradient and finite gradients:

```
cd /home/chuong/workspace/FluxVLA
PY=/home/chuong/miniconda3/envs/fluxvla/bin/python
$PY tools/patchwam_robotwin_5tasks_smoke.py selection --report work_dirs/patchwam_5tasks_smoke/robotwin_clean/selection.json
$PY tools/patchwam_robotwin_5tasks_smoke.py dataset   --report work_dirs/patchwam_5tasks_smoke/robotwin_clean/dataset.json
$PY tools/patchwam_robotwin_5tasks_smoke.py train --backward --report work_dirs/patchwam_5tasks_smoke/robotwin_clean/train_backward.json
```

`train` reports the objective from `outputs['loss']` alone, matching
`BaseTrainRunner._training_step`; the other head outputs are diagnostics and are
reported separately, never summed into the objective.

On the supplied machine only `blocks_ranking_size`, `move_stapler_pad` and
`hanging_mug` exist (50 episodes, 15 fps each), so the five-task smoke stops at
the preflight with a nonzero exit. The earlier `selection.json`, `dataset.json`,
`train_backward.json` and `summary.json` directly under
`work_dirs/patchwam_5tasks_smoke/` are historical records of the previous merged
`/data/robotwin2.0` release and are intentionally left unchanged.

Samples carry both axes explicitly. `patchwam_row` / `patchwam_window_rows` /
`patchwam_future_row` are dataset rows of the filtered sequence, while
`patchwam_raw_row` / `patchwam_raw_window_rows` / `patchwam_raw_future_row` and
`patchwam_raw_history_rows` are exact original-recording rows (per-root offsets
preserved), with `patchwam_raw_total_rows` giving the recording length and
`patchwam_nonidle_filtered` marking the path. `PatchWAMSelectWindow` and
`PatchWAMSelectHistory` prefer the raw keys, so their `decoded_frames` must be
the *unfiltered original-row* tensor; any other length fails loudly instead of
gathering substituted frames. Without a non-idle filter both axes coincide and
the behavior is unchanged. Non-idle datasets must keep LeRobot's contiguous
0-based `frame_index` recording axis, otherwise construction fails.

The RobotWin and RoboDojo runners also declare `expected_global_batch_size=256`
(the source effective batch). The runner compares it with
`per_device_batch_size x world_size x grad_accumulation_steps` and warns once on
rank zero when they differ; it never changes the batch, the optimizer, or the
budget, and training continues with the effective size.

The three recipes share the shipped `schedule_style='openpi'` warmup/cosine
policy with `warmup_ratio=0.05` and `min_lr_rate=0.01`; no PatchWAM-specific
schedule branch exists. OpenPI's `(step + 1) / (W + 1)` warmup is intentionally
used in place of the original PatchWAM `1/W + (1 - 1/W) * step/W` ramp: for
an illustrative 20000-update run (W = 1000), the largest absolute rate
difference is
`1 / (W * (W + 1))` ~ 9.99e-7 at the first warmup step (about 0.1 % of the
source start value), shrinks monotonically, and vanishes at the warmup
boundary; the cosine phase is identical. Library reuse is preferred over
carrying a near-duplicate policy. PatchWAM has no fixed default warmup count:
its actual W is computed as 5% of the total optimizer-update budget, which
depends on the dataset and training settings.

`--ckpt-path` is consumed by the native eval runner via
`BaseEvalRunner.load_eval_state_dict` → `model.load_state_dict`, so it accepts
**native** `vla_head.*` / `vlm_backbone.*` checkpoints only. A source PatchWAM
checkpoint (bare head-only policy weights, or compact `policy.*` + trainable
`text_encoder.*` adapters) is initialized through
`PatchWAMVLA.pretrained_name_or_path` + `from_pretrained()`, which validates
every required trainable tensor for presence, shape, and finiteness before any
parameter is mutated; frozen autoencoder/Qwen assets load separately from config
paths.

## Checkpoints

`PatchWAMVLA` ships a default FluxVLA-to-source key mapping
(`DEFAULT_PATCHWAM_NAME_MAPPING`) that maps a source `FluxAssetPolicy`
checkpoint onto the native split model:

```
vlm_backbone.autoencoder.* -> autoencoder.*
vlm_backbone.text_encoder.* -> text_encoder.*
vla_head.*                  -> policy.*
```

`PatchWAMVLA.checkpoint_mapping_report(checkpoint_keys)` resolves every model key
and reports missing/unexpected keys; with `strict_mapping=True` any mismatch
raises instead of leaving random trainable weights. Native
`torch.save`/`load_state_dict` resume works as-is.

## Normalization and decoding

PatchWAM normalization/statistics stay owned by the native transforms
(`NormalizeStatesAndActions`, `RelativeActions`, `ProcessLiberoActions`, ...).
`PatchWAMVLA.predict_action` returns normalized `[B, H, A]` actions; the
native LIBERO runner applies the configured decoder once to each action before
stepping the environment. The model does not denormalize a second time.

## Freeze and FSDP policy

`PatchWAMBackbone.apply_trainable_policy` freezes the autoencoder and the
encoder base/vision tower while keeping the language LoRA residuals trainable.
`PatchWAMVLA.get_fsdp_wrapping_policy` wraps the head only, because
`OfficialFluxDenoiser` executes the inner FLUX blocks via
`_paired_block`/`_unified_block` rather than their `forward`; auto-wrapping those
blocks would bypass their FSDP all-gather hooks.

## Inference

Only the LIBERO config ships a runnable native inference block
(`inference`/`eval` = `LiberoEvalRunner` with `norm_stats_key` and
`PatchWAMLiberoEvalDataset`). Run it with
`python scripts/inference.py --config configs/patchwam/patchwam_libero_full_finetune.py --ckpt-path <ckpt>`.
RobotWin and RoboDojo have no native benchmark inference runner in FluxVLA;
their configs are training recipes and their benchmark evaluation is unverified.

## History (training vs online inference)

Training history is generated per sample by `PatchWAMParquetDataset` +
`CausalObservationBuffer`-compatible selectors and is bounded by episode and
the original recording clock. Online history requires the eval loop to call
`PatchWAMHistoryBuffer` once per recorded observation with
`patchwam_episode_index`, `patchwam_timestamp`, and `patchwam_reset`. Without
those signals the buffer raises rather than returning stale history; no config
claims history inference support on its own.

The buffer converts inputs to the training `[-1, 1]` range and resizes to the
world geometry you configure. Because nonnegative frames in `[-1, 1]` are
ambiguous with `[0, 1]`, pass `input_range='minus_one_one'` explicitly for that
case; `auto` raises on ambiguous 4-D `[V, 3, H, W]` / `[T, 3, H, W]` layouts and
requires `frame_layout`.

## Deferred

EMA and Self-Flow training (optimizer/teacher hooks) are intentionally rejected
rather than silently ignored. `frame_masks`/`img_masks` are enforced, but
full-size GPU training, official checkpoint parity, and robot benchmarks require
assets and hardware that are not established here.
Native statistics are aggregated by `DistributedRepeatingDataset` under the
native `proprio`/`action` names; the configs' normalizer reads `state_key='proprio'`.

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

Datasets, runners, and configs hand the model a raw `task_description`. Both
training (`PatchWAMVLA.forward`) and inference
(`PatchWAMVLA.predict_action`) build their batch through the same
`_prepare_batch`, which applies the source template
`DEFAULT_PATCHWAM_PROMPT_TEMPLATE`
(`A video recorded from a robot's point of view executing the following
instruction: {task}`) once and passes the formatted `prompt` list to the
instruction encoder. The raw `task_description` stays untemplated metadata, a
single instruction string becomes a one-element prompt list, an already
formatted instruction is not wrapped twice, and a caller-supplied explicit
`prompt` is treated as prepared text and passed through unchanged. Tuning the
template is not exposed as a knob; the fallback exists only to match the source
checkpoint's conditioning.

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
* `patchwam_robotwin_vlm_lora_finetune.py` - the RobotWin VLM recipe with
  ordinary (unstaged) DiT PEFT finetuning: the released RoboTwin C2R policy is
  warm-started from `pretrained_name_or_path`, the FLUX.2
  `double_blocks`/`single_blocks` keep their base weights frozen behind
  `peft_type='LORA'`, `r=32`, `lora_alpha=64`, `lora_dropout=0` adapters, and
  every other policy layer stays trainable. Its paired
  `patchwam_robotwin_vlm_lora_eval.py` carries the same `vla_head.peft` block,
  so a checkpoint evaluates with the architecture it was trained under.

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
overrides the roots, the initial weights, the (absent) non-idle filter, the
asset paths, and the statistics namespace:

```
python scripts/train.py \
    --config configs/patchwam/patchwam_robotwin_vlm_5tasks_full_finetune.py \
    --work-dir work_dirs/patchwam_robotwin_vlm_5tasks_full_finetune
```

`PATCHWAM_DATA_ROOT` names the *parent* directory (default
`/data/RoboTwin/RoboTwin-Clean`) and each root is `$PATCHWAM_DATA_ROOT/<task>`.
The run also starts from the released RoboTwin 2.0 C2R policy instead of the raw
FLUX.2 base: `model.pretrained_name_or_path` defaults to
`checkpoints/Pretrained_models/patchwam-robotwin/step_140000.pt`, overridable
with `PATCHWAM_PRETRAINED_CHECKPOINT`. That payload is the ImageWAM-lineage
Action-as-Patch release, which maps onto exactly the trainable set — the
denoiser transformer and the proprio/state projection — so the strict load fills
all 151 trainable tensors and leaves the frozen Qwen3-VL encoder from config.
The released policy was trained against a Qwen3 text-only encoder, so the
conditioning features this recipe feeds its head come from the configured
Qwen3-VL encoder, not from the checkpoint.
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
release without writing into it. Its `train` mode loads the recipe's initial
weights in the DDP runner's order, so the reported model is the one the recipe
trains. `selection` reports per-folder episodes, FPS,
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

All five requested roots are published in the RoboTwin-Clean release as of
2026-10-15: each holds 50 episodes at 15 fps with the head and both wrist
cameras, so the preflight passes and the smoke runs to completion. The
`selection`, `dataset` and `train --backward` reports recorded on that release
live under `work_dirs/patchwam_5tasks_smoke/robotwin_clean/`; the `dataset` pass
indexes 60395 samples over the five roots, and `train --backward` reaches a
finite scalar objective with nonzero finite gradients on all 151 trainable
tensors at 23.3 GB peak VRAM on one RTX 5090. The objective value tracks the
sampled window, so read the checks rather than the number. The earlier
`selection.json`, `dataset.json`, `train_backward.json` and `summary.json`
directly under `work_dirs/patchwam_5tasks_smoke/` are historical records of the
previous merged `/data/robotwin2.0` release and are intentionally left unchanged.

That smoke covers one forward/backward; a real run also carries AdamW state and
does not fit one 32 GB card. `freeze_backbones` leaves the FLUX denoiser
trainable (151 tensors, 3.88e9 parameters, 7.22 GiB) and freezes the Qwen3-VL
backbone, so the bf16 weights are 15.6 GiB, `DDPTrainRunner` allocates 7.8 GiB of
gradient buckets at setup for those trainable parameters, and the two optimizer
moment buffers land on top of the activation memory. Training the ablation needs
a GPU larger than 32 GB, or several of them.

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

### Ordinary DiT adapter finetuning (PEFT)

`patchwam_robotwin_vlm_lora_finetune.py` changes only the adaptation policy of
the RobotWin VLM recipe. Leaving `vla_head.stage` unset and setting
`vla_head.peft` selects ordinary finetuning: the FLUX.2
`transformer.double_blocks`/`transformer.single_blocks` subtrees keep their base
weights frozen (modulation and normalization parameters included) and carry the
adapters built by `attach_dit_adapters`, which goes through the HuggingFace
`peft` library (`peft.get_peft_config` + `peft.inject_adapter_in_model`). The
recipe selects LoRA with `dict(peft_type='LORA', r=32, lora_alpha=64,
lora_dropout=0.0)` over the official attention/feed-forward target list; the
PEFT option names match the `LoraConfig(...)` keys the native DDP runner already
uses for whole-model LoRA. Meanwhile the denoiser's own input/output/time/text
projections, the shared stream modulations, and the head's proprio/state
projection and action maps stay fully trainable; the fixed action codec carries
no trainable parameters. The visual frontend and the Qwen3-VL instruction
encoder stay frozen, including any VL language LoRA residuals, so
`vl_trainable_adapters=True` cannot leak trainable encoder parameters into this
mode.

`vla_head.peft` is a PEFT configuration mapping, so another module-level tuner
can be selected by configuration alone:

```python
model = dict(
    vla_head=dict(peft=dict(peft_type='LOHA', r=8, alpha=16,
                            target_modules=['img_attn.qkv', 'txt_attn.qkv'])),
)
```

In a recipe derived from
`patchwam_robotwin_vlm_lora_finetune.py`, replacing the tuner must replace the
mapping wholesale (`peft=dict(_delete_=True, peft_type='LOHA', r=8, alpha=16,
...)`): config inheritance deep-merges dictionaries, so a plain `peft=...`
override keeps the inherited LoRA options (`lora_alpha`, `lora_dropout`, ...)
and PEFT rejects the mixed mapping with an invalid-options error.

PEFT keeps the wrapped weights in `<module>.base_layer` and adds its own
adapter tensors beside them, so the base payload the recipe warm-starts from
still loads strictly and only the adapter tensors count as new keys. Adapters
are confined to the block containers (`double_blocks`/`single_blocks` by
default, overridable with the reserved `blocks` key of `vla_head.peft`): a
target that resolves outside them, an unsupported tuner type (prompt-learning
and adaption-prompt tuners do not adapt modules and are rejected), an unmatched
target, or a second injection into the same denoiser fails with a diagnostic
instead of training the wrong tensors.

The pre-PEFT `vla_head.lora` alias (`rank`/`alpha`/`dropout`/`targets`) is still
accepted with a `DeprecationWarning`: it keeps the historical LoRA defaults
(`r=32`, `lora_alpha=64`, `lora_dropout=0`, official target list) for whatever
the mapping leaves unset, because PEFT's own LoRA defaults differ. The legacy
option names are not accepted inside `vla_head.peft`, which is PEFT's own
configuration surface. Checkpoints written by the removed private
implementation (`denoiser.dit_lora.*`) are rejected with an actionable message
because their adapter tensors cannot be mapped onto a PEFT tuner.

```
PATCHWAM_PRETRAINED_CHECKPOINT=checkpoints/Pretrained_models/patchwam-robotwin/step_140000.pt \
    python scripts/train.py \
    --config configs/patchwam/patchwam_robotwin_vlm_lora_finetune.py \
    --work-dir work_dirs/patchwam_robotwin_vlm_lora_finetune
```

The base payload is loaded through an adapter allow-list: every inherited
`vla_head.*` tensor, the frozen DiT blocks included, must be present with a
matching shape, and only the new PEFT adapter tensors may be missing. A base
payload names those tensors without the `base_layer` segment PEFT adds, so the
loader resolves each released key onto the wrapped layer it now belongs to.
This mode does not use `stage1_checkpoint`; combining it with `visual_stage` is
rejected.

`PatchWAMVLA.trainable_parameter_report()` returns the exact split for a built
model - model-level names, the frozen DiT block patterns, and the head's
collapsed module patterns - for example:

```
>>> model.trainable_parameter_report()['dit_block_patterns']
['vla_head.denoiser.transformer.double_blocks.*', 'vla_head.denoiser.transformer.single_blocks.*']
>>> model.vla_head.trainable_parameter_report()['adapter_module_patterns']
['denoiser.transformer.double_blocks.*.img_attn.proj', 'denoiser.transformer.double_blocks.*.img_attn.qkv',
 ... , 'denoiser.transformer.single_blocks.*.linear2']
```

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

Staged recipes and the unstaged adapter finetune use the same strict loader with
an allow-list: a stage may omit its own new modules, and an adapter finetune may
omit exactly its PEFT adapter tensors, while every other inherited `vla_head.*`
tensor stays mandatory and any unexpected key fails before a parameter is
written. A native checkpoint written by the adapter mode carries the adapters,
so it loads with an empty missing set, and the evaluator's strict
`load_state_dict` - which builds the same `vla_head.peft` config - restores them
exactly.

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

The unstaged adapter mode owns its own policy and re-applies it from
`PatchWAMVLA.freeze_backbones()` after `BaseVLA.freeze_backbones()`, so the
runners' opposite call orders (DDP freezes before `from_pretrained`, FSDP after
it) end with the same split. The re-application matters because the generic
`freeze_vlm_backbone` pass runs `PatchWAMBackbone.apply_trainable_policy()`
afterwards, which would otherwise reopen the VL language LoRA residuals for a
backbone configured with `vl_trainable_adapters=True`. An adapter finetune recipe
that sets `freeze_vlm_backbone=False` is rejected instead of silently training
the encoders.

`PatchWAMHead.parameter_groups()` reports the optimizer groups the active policy
opens: `visual_projection` (stage 1), `visual_projection` + `peft` (stage 2),
`policy` + `peft` (unstaged adapter finetune), and nothing when no policy is
configured. The native optimizer selects on `requires_grad` either way.

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

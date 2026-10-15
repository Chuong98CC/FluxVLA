# PatchWAM RoboTwin 2.0 evaluation

PatchWAM evaluates on RoboTwin 2.0 through the shared FluxVLA RoboTwin
evaluator, `RobotwinEvalRunner`. The runner embeds RoboTwin's own evaluation
harness in the FluxVLA process and drives it through the harness
`eval_policy.py` contract (`get_model` / `eval` / `reset_model` /
`close_model`), so every policy in the repository shares one evaluation
implementation. RoboTwin is installed into the FluxVLA environment by
`scripts/install_env.sh --with-robotwin`; there is no separate simulator
environment and no FluxVLA-owned rollout loop.

## Configs

| Config | Purpose |
| --- | --- |
| `configs/patchwam/patchwam_robotwin_eval.py` | Trained PatchWAM checkpoint, single-task smoke |
| `configs/patchwam/patchwam_robotwin_official_benchmark.py` | Released 50-task / clean+random / 100-trial protocol |
| `configs/patchwam/patchwam_lit_eval_robotwin.py` | Trained PatchWAM + LIT Stage-2 checkpoint, single-task smoke |
| `configs/patchwam/patchwam_lit_official_benchmark.py` | Same released protocol for the LIT head |

All four set `eval.type='RobotwinEvalRunner'`. The checkpoint comes from the
CLI (`--ckpt-path`), exactly like every other RoboTwin config, because
`scripts/eval.py` assigns `eval.ckpt_path` from its arguments:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
  --config configs/patchwam/patchwam_robotwin_eval.py \
  --ckpt-path work_dirs/patchwam_robotwin_vlm_full_finetune/checkpoints/latest-checkpoint.pt
```

The official protocol goes through the shared manager, which builds one
single-task worker per task and aggregates the per-worker `summary.json`
files:

```bash
CONFIG=configs/patchwam/patchwam_robotwin_official_benchmark.py \
CKPT=work_dirs/patchwam_robotwin_vlm_full_finetune/checkpoints/latest-checkpoint.pt \
  bash scripts/eval_robotwin_manager.sh
```

`my_scripts/eval_patchwam.sh` is a thin convenience wrapper over that manager
(`--ckpt`, `--config`, `--task`, `--condition`, `--episodes`, `--smoke`).

## Observation preprocessing

The runner's `_build_observation` hands the policy three `uint8 [H, W, 3]`
cameras (`cam_high`, `cam_left_wrist`, `cam_right_wrist`), the 14-D `qpos` and
the raw instruction. `PatchWAMRoboTwinEvalDataset` adapts that dict into the
native PatchWAM prediction batch, and the config names the transform chain, so
the preprocessing lives at the dataset/transform boundary rather than in the
evaluation runner:

1. `ProcessEvalInputs(img_keys=[...])` - load the three frames as `uint8`.
2. `PatchWAMEvalFrames` - dtype-based `uint8` to `[0, 1]` scaling plus the
   `world_frames` view stack. Dtype decides the scaling, so a dark frame whose
   maximum sample is 1 is not mistaken for white.
3. `PatchWAMCleanVL(image_size=448, head_index=0)` - the unaugmented head view
   an image-conditioned encoder requires (`vl_current`, with empty history).
4. `PatchWAMEvalPack(layout='robotwin', video_size=[288, 256], ...)` - resize
   each camera and compose the compact 288x256 top/wrist mosaic in `[-1, 1]`,
   the same layout `PatchWAMComposeWorld(layout='robotwin')` builds in training.
5. `NormalizeStatesAndActions(state_key='proprio', action_key=None, ...,
   normalization_epsilon=1e-8)` - the trainer's own normalizer and epsilon.

Actions are absolute joint targets normalized with mean/std in training, so
`DenormalizePrivateAction` reverses that transform with the shared
`x * std + mean` convention (no epsilon). Training-only augmentation
(`PatchWAMWorldAugment`) is deliberately absent from the evaluation chain.

`test/test_datasets/test_patchwam_robotwin_eval_dataset.py` drives the real
canonical `LocalFluxVLAPolicy.predict` with synthetic cameras and asserts the
emitted mosaic geometry, the normalized state and the denormalized action
chunk.

## Statistics and checkpoints

`norm_stats_path` points at the run's `dataset_statistics.json` (written by
`scripts/train.py`), i.e. the statistics the checkpoint was trained with. The
canonical policy loads the checkpoint with `strict=True`; there is no silent
partial load. Evaluation reuses the trainer's own state transform
(`NormalizeStatesAndActions(..., normalization_epsilon=1e-8)`), while actions
are denormalized by `DenormalizePrivateAction` with the shared `x * std + mean`
convention (no epsilon), so the two directions invert each other only where
the statistics are well conditioned.

The released PatchWAM asset
(`QZWang/patchwam-robotwin-c2r-step140000`) is **unsupported** on this path
until a conversion is established and verified:

* its `dataset_stats.json` uses the legacy
  `{state: {default: {global_mean, global_std}}, action: ...}` schema, while
  the shared transforms expect
  `{statistic_name: {proprio: {mean, std, ...}, action: {...}}}`; and
* the released state-dict layout has not been verified against the
  `PatchWAMVLA` architecture built from the training config, so a strict load
  may need an explicit state-dict conversion.

Neither conversion is provided or claimed here. In particular,
`fluxvla.datasets.utils.statistics.convert_robotwin_statistics_file` exports the
*opposite* direction (native training statistics to the released schema) for
publishing a trained run; it is not a way to consume the released asset.

Newly trained PatchWAM / PatchWAM-LIT RoboTwin checkpoints evaluate with their
own statistics and need no conversion.

## Environment

RoboTwin lives in the FluxVLA environment:

```bash
bash scripts/install_env.sh sim-only --profile cu128 --with-robotwin
```

`--with-robotwin` clones the validated RoboTwin ref, downloads the simulator
assets, regenerates the embodiment config paths and rebuilds cuRobo against the
selected PyTorch profile. `ROBOTWIN_ROOT` (or `PATCHWAM_ROBOTWIN_ROOT`) selects
the checkout at evaluation time.

## What was retired

The previous PatchWAM-only evaluation path was a FluxVLA-owned
`RoboTwinEvalRunner` that drove the simulator in a separate conda environment
behind `tools/robotwin_sim_worker.py`, together with its dedicated environment
installer (`scripts/install_robotwin_env.sh`, `scripts/robotwin_env_checks.py`,
`scripts/patch_robotwin_runtime.py`, `scripts/requirements-robotwin-sim.txt`),
the standalone official-benchmark launcher
(`tools/patchwam_official_benchmark.py`) and the released-C2R asset configs.
All of it was removed in favour of the shared runner described above; the
repository history holds the previous design.

# PatchWAM + Latent Interface Training (LIT)

This document covers the FluxVLA integration of ImageWAM's two-stage Latent
Interface Training on top of the native PatchWAM stack. It assumes the reader
has read [`docs/patchwam.md`](patchwam.md) first.

## What was added

| Component | Registry | Role |
| --- | --- | --- |
| `PatchWAMLITVLA` | `VLAS` | Wires `vlm_backbone` + `vla_head` + `lit_head`; joint/`lit`/`patchwam` training modes, head selection, checkpoint routing |
| `PatchWAMLITFlowMatchingHead` | `HEADS` | LIT goal encoder (Stage 1), 100-latent semantic/visual aggregator, pose decoder, layer-wise action expert |
| `PatchWAMLITActionDiT` | `HEADS` | Registered native FastWAM `ActionDiT` used as the LIT action expert |
| `PatchWAMLITParquetDataset` | `DATASETS` | `PatchWAMParquetDataset` subclass adding the episode/root-bounded terminal goal state |
| `PatchWAMLITGoal` | `TRANSFORMS` | Normalizes the terminal goal with the **state/proprio** statistics |
| `OfficialFluxDenoiser.forward_prefix`, `SmallJointTransformer.forward_prefix` | - | Streaming prefix feature API over the shared FLUX blocks |

LIT modules are copied from ImageWAM (`imagewam/models/backbones/goal_pose_prior.py`,
itself adapted from the MolmoAct2 V3 goal-pose prior) under Apache-2.0, and
narrowed to the **strict** variant: the action expert never attends the raw
reference image, and the relaxed Plan-B / grid-latent ablations are removed
rather than left inert. Unsupported knobs (`latent_layout != 'free'`,
`action_sees_ref=True`) raise at construction time.

## Architecture

The released PatchWAM parameter names, ownership and checkpoint keys are
retained, but joint Stage 2 *trains* the shared FLUX.2 transformer and proprio
projection rather than leaving them at their released values. `vla_head`
(`PatchWAMHead`) owns the FLUX.2 transformer and the proprio projection;
`lit_head` owns only the LIT parameters and calls `forward_prefix` on the same
transformer instance. There is exactly one transformer and one proprio
projection in `state_dict()`, and `all_module_keys` is
`['vlm_backbone', 'vla_head', 'lit_head']`.

Stage 1 (`lit_stage='stage1'`, image-free):

1. The shared encoder produces `[language | state]` tokens from the instruction
   and the current normalized proprioception. No camera frame is decoded,
   encoded, or attended.
2. The goal encoder maps the oracle normalized terminal state
   (`observation.state` at `t + 16`) to 8 tokens appended to that prefix.
3. The frozen FLUX prefix runs once (weights frozen, activations live), then the
   native `ActionDiT` flow-matches the normalized `[B, 16, 14]` action window
   against the goal-conditioned prefix. Only the goal encoder and action expert
   are trained.

Stage 2 (`lit_stage='stage2'`, joint):

1. The prefix pass reads the language, the current proprioception and the
   current (plus history) visual tokens only. Futures and goals never enter it.
2. Every FLUX layer is a coupling layer (default `1..25` for Klein base 4B).
   The pass streams the text/image hidden states into the aggregator
   immediately; only the 768-wide latent contexts are retained, never
   full-resolution layer features.
3. The aggregator runs `num_action_layers` depth steps (default 25, five shared
   parameter groups of five updates each), each doing latent self-attention,
   then semantic cross-attention, then visual cross-attention, each layer
   seeing its own updated latents.
4. The pose decoder reconstructs the normalized goal from the first 8 latents
   after `semantic_visual_pose_norm`.
5. The layer-wise action expert runs the native `ActionDiT` blocks where layer
   `i` consumes **only** the projected latents of depth `i` and their validity
   mask. There is no text/image prefix on this path: the strict LIT bottleneck
   keeps all backbone memory inside the aggregator, so the action head cannot
   learn an image-to-action bypass through the semantic stream.

The default expert is the native `ActionDiT` at its native width:
`hidden_dim=1024`, `ffn_dim=4096`, `num_heads=16`, `attn_head_dim=64`,
`text_dim=3072` (the latents arrive through `context_projection`) and
`num_layers=25`. Configs may override these, but `num_layers` must stay
divisible by `num_layer_groups` so depth sharing stays exact.

The released PatchWAM parameter names, module layout and checkpoint keys are
retained. Joint Stage 2 *updates* the shared FLUX transformer and proprio
projection rather than freezing them: the released `step_140000.pt` is the
initialization, not the final state. In Stage-2 `lit` mode the original head's
output modules (`final_layer`, or the tiny `output_norm`/`token_output`) are
frozen because the LIT path never executes them.

Adapter limit: this integration ports the source LIT core
(goal encoder, 100-latent aggregator, pose decoder) and feeds the projected
latents through the native `ActionDiT` cross-attention. It is not a bit-exact
reproduction of ImageWAM's MoT synthetic-K/V injection, and whole-head image
co-training remains off by design.

Total Stage-2 loss:

```
patchwam_loss_weight * (0.5 * video + 1.0 * action)     # original head
+ lit_loss_weight     * (1.0 * FM action + 0.3 * goal)  # LIT interface
```

Both branch weights default to 1. The auxiliary language loss is folded in
exactly once by the shared `PatchWAMVLA._run_head` path.

## Goal contract

The goal is the normalized `observation.state` at the exact same
episode/root-bounded row as the PatchWAM future endpoint (`index + 16`). It is
not the last action, not `t + 15`, and never a row from another episode or data
root. Normalization reuses the native `NormalizeStatesAndActions` arithmetic
with the state/proprio statistics (`stats['proprio']`), never the action
statistics, and with the same `normalization_epsilon=1e-8` the real state
transform uses. A valid goal with missing or width-mismatched statistics raises
instead of silently disabling supervision; an in-episode row that lacks the
state column is a schema error. Invalid *boundary* goals (past the end of an
episode/root) are emitted as zeros with `goal_valid=False` and are excluded from
the reconstruction loss, and non-finite values inside a supervised coordinate
raise rather than poisoning gradients. This integration is not an 8-D SE(3)
paper reproduction: RobotWin uses the 14-D joint/gripper state.

## Commands

```bash
# Stage 1 (image-free goal-conditioned prior)
python scripts/train.py \
  --config configs/patchwam/patchwam_lit_stage1_robotwin.py \
  --work-dir work_dirs/patchwam_lit_stage1_robotwin

# Stage 2 (joint original + LIT). `PATCHWAM_CHECKPOINT` initializes the released
# PatchWAM weights; `PATCHWAM_LIT_STAGE1_CHECKPOINT` bridges the trained Stage-1
# run (the native runner writes `latest-checkpoint.pt`). Training refuses to
# start without it unless the random-init ablation is explicit.
PATCHWAM_LIT_STAGE1_CHECKPOINT=work_dirs/patchwam_lit_stage1_robotwin/checkpoints/latest-checkpoint.pt \
python scripts/train.py \
  --config configs/patchwam/patchwam_lit_stage2_robotwin.py \
  --work-dir work_dirs/patchwam_lit_stage2_robotwin

# RobotWin smoke evaluation of a Stage-2 checkpoint with the LIT head. The
# native runner writes `latest-checkpoint.pt` inside
# `work_dirs/<run>/checkpoints/`, and `--ckpt-path` is required because the
# shared evaluator takes it from the CLI. Statistics default to the Stage-2
# training file
# `work_dirs/patchwam_lit_stage2_robotwin/dataset_statistics.json`.
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
  --config configs/patchwam/patchwam_lit_eval_robotwin.py \
  --ckpt-path work_dirs/patchwam_lit_stage2_robotwin/checkpoints/latest-checkpoint.pt

# Evaluate the original PatchWAM head from the same dual-head model. The
# evaluator builds `inference_model` when the config defines it
# (base_eval_runner.prepare_eval_model_cfg), so both keys must be overridden;
# RobotwinEvalRunner itself takes no `head` argument.
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
  --config configs/patchwam/patchwam_lit_eval_robotwin.py \
  --ckpt-path work_dirs/patchwam_lit_stage2_robotwin/checkpoints/latest-checkpoint.pt \
  --cfg-options model.active_head=patchwam inference_model.active_head=patchwam

# Official 50-task / clean+random / 100-trial released protocol. The shared
# manager builds one single-task worker per task and aggregates the summaries.
CONFIG=configs/patchwam/patchwam_lit_official_benchmark.py \
CKPT=work_dirs/patchwam_lit_stage2_robotwin/checkpoints/latest-checkpoint.pt \
  bash scripts/eval_robotwin_manager.sh
```

`scripts/train.sh CONFIG WORK_DIR` is the distributed wrapper around the same
commands. Every asset and data path is environment overridable
(`PATCHWAM_ASSETS_ROOT`, `PATCHWAM_FLUX2_ROOT`, `PATCHWAM_FLUX2_SRC`,
`PATCHWAM_TRANSFORMER`, `PATCHWAM_AE`, `PATCHWAM_TEXT_ENCODER`,
`PATCHWAM_TEXT_TOKENIZER`, `PATCHWAM_CHECKPOINT`, `PATCHWAM_LIT_STAGE1_CHECKPOINT`,
`PATCHWAM_LIT_CHECKPOINT`, `PATCHWAM_DATA_ROOT`, `PATCHWAM_DATASET_STATS`).
Training normalization statistics are aggregated from the dataset metadata by
`DistributedRepeatingDataset`; `PATCHWAM_DATASET_STATS` is the evaluation-only
statistics file.

## Evaluation statistics

`scripts/train.py` writes the native training statistics to
`<work_dir>/dataset_statistics.json` in the
`{statistic_name: {field: {min, max, mean, std, ...}}}` schema. Trained LIT
weights must normalize with those Stage-2 statistics, so
`patchwam_lit_eval_robotwin.py` sets `norm_stats_path` to
`work_dirs/patchwam_lit_stage2_robotwin/dataset_statistics.json`
(`PATCHWAM_DATASET_STATS` overrides it). Evaluation reuses the trainer's own
state transform: the config runs `NormalizeStatesAndActions(state_key='proprio',
..., normalization_epsilon=1e-8)`, the exact class and epsilon the Stage-1 and
Stage-2 recipes use, so proprio is z-scored identically at evaluation. Actions
use `DenormalizePrivateAction`, which applies the shared `x * std + mean`
convention *without* the training epsilon, so the two directions invert each
other only where the statistics are well conditioned (`std` large compared
with `1e-8`); it is the shared convention, not a bit-exact inverse.
`test/test_datasets/test_patchwam_lit_statistics.py` checks the release-schema
conversion and the training transform parity, and
`test/test_datasets/test_patchwam_robotwin_eval_dataset.py` checks that the
evaluation chain matches the trainer on ordinary, zero and near-zero state
`std`.

If a released-schema file is needed elsewhere, convert explicitly instead of
hand-editing:

```bash
python -c "from fluxvla.datasets.utils.statistics import \
  convert_robotwin_statistics_file as c; \
  c('work_dirs/patchwam_lit_stage2_robotwin/dataset_statistics.json', \
    'work_dirs/patchwam_lit_stage2_robotwin/dataset_stats_robotwin.json')"
```

## Checkpoints

`PatchWAMLITVLA` keeps the native conventions and never uses a silent
`strict=False`:

* `from_pretrained()` (the native DDP/FSDP initialization hook) loads the
  released PatchWAM weights from `pretrained_name_or_path` and then bridges
  `lit_stage1_checkpoint` with `bridge_from_stage1=True`. Stage-1 itself also
  initializes the shared FLUX transformer and proprio projection from the same
  released checkpoint, so the frozen Stage-1 prior never starts from a random
  proprio projection.
* A Stage-2 recipe that has no LIT tensors in its primary checkpoint and no
  Stage-1 file fails loudly (`lit_stage1_required`, the default) instead of
  training a random action prior. `lit_stage1_required=False` is the explicit
  random-initialization ablation. Evaluation loads a Stage-2 checkpoint that
  already contains both halves and therefore never needs the Stage-1 file.
  Configuring a Stage-1 bridge while the primary checkpoint already carries LIT
  weights raises, rather than silently overwriting a trained interface.
* `required_original_keys()` is computed independently of `requires_grad`, so a
  Stage-1 model (where the whole original branch is frozen) still rejects
  partial original payloads. Only the frozen AE/Qwen base assets — loaded from
  local files — may be omitted, plus the LoRA adapters when adapters are off.
* `plan_state_dict_payload()` / `load_state_dict_payload()` validate **both**
  halves (keys, shapes, finiteness) before any tensor is written; a malformed
  original or LIT entry leaves the model completely untouched. No multi-GB
  clone is taken for rollback. `load_lit_state_dict(state,
  bridge_from_stage1=...)` uses the same plan/apply split: with the bridge the
  Stage-2-only modules may be new and the Stage-1-only `goal_pose_encoder.*`
  tensors may be dropped, while every shared/action-expert tensor must match.
  Reports list the exact `missing` / `unexpected` categories.
* A payload that carries `lit_head.*` tensors must also carry the complete
  original/shared half; a LIT-only payload is rejected by the full-restore API
  (use `load_lit_state_dict` for a deliberate head-only load). `from_pretrained`
  reads and validates the primary and the Stage-1 bridge before mutating
  anything, requires the bridge to contain a real `goal_pose_encoder.*`, and
  verifies that the Stage-1 original/shared tensors match the released weights
  the Stage-2 model is about to load (compared after normal bfloat16 storage
  rounding, so a different or edited backbone fails atomically).

## Scale and FSDP

The shipped training configs use the native **DDP** runner. FSDP for this
integration wraps the **outer VLA unit only**
(`get_fsdp_wrapping_policy()` returns a policy for `PatchWAMLITVLA`): the LIT
prefix/conditioning helpers call borrowed `vla_head`/`lit_head` methods outside
any per-head `forward`, and the official FLUX blocks execute through adapter
helpers rather than `block.forward`, so wrapping those modules would bypass
FSDP all-gather/unshard hooks. Outer-unit-only sharding is memory-hungry
(the FLUX transformer plus the 25-layer expert stay in one unit) and has not
been validated on multiple GPUs; treat multi-GPU FSDP as unverified.

## Inference

`predict_action(..., head='lit')` (the `active_head` default for the LIT model)
reads the current observation only, streams the aggregator, and denoises
normalized `[B, 16, 14]` actions with the configured scheduler steps. Sampling is
reproducible via `seed=`. Oracle goal and future-token inputs are rejected for
Stage 2. Stage 1 inference is an explicit oracle-goal diagnostic and requires
`oracle_goal=True`. `head='patchwam'` keeps evaluating the original head from
the same model.

## Reading order for review

1. `fluxvla/models/third_party_models/patchwam/lit.py` - goal encoder/decoder,
   aggregator groups, layer-wise action expert, checkpoint key rules.
2. `fluxvla/models/third_party_models/patchwam/flux.py` - `forward_prefix`, the
   streaming capture API over the official FLUX blocks.
3. `fluxvla/models/heads/patchwam_lit_head.py` - Stage-1/Stage-2 losses,
   masking, flow scheduler use, inference.
4. `fluxvla/models/vlas/patchwam_lit_vla.py` - shared-prefix orchestration,
   stage policy, joint loss, head selection, checkpoint routing.
5. `fluxvla/datasets/patchwam_lit_dataset.py` and
   `fluxvla/transforms/transform_patchwam_lit.py` - terminal goal metadata and
   normalization.
6. `configs/patchwam/patchwam_lit_*.py` - shipped recipes.

## Tests

```
OMP_NUM_THREADS=1 python -m pytest -q \
  test/test_models/patchwam/test_patchwam_lit.py \
  test/test_models/patchwam/test_patchwam_lit_prefix.py \
  test/test_models/test_patchwam_lit_vla.py \
  test/test_datasets/test_patchwam_lit_dataset.py \
  test/test_datasets/test_patchwam_lit_statistics.py \
  test/test_configs/test_patchwam_lit_config.py
```

CPU-only; the official FLUX prefix test skips itself when the local `flux2`
source checkout is unavailable. Full-size training, benchmark rollouts, and
checkpoint downloads are out of scope for these checks.

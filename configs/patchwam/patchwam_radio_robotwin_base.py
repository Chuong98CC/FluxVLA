# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# PatchWAM with a frozen C-RADIOv4 visual backbone, base recipe.
#
# The frozen shared wrapper in `fluxvla/models/backbones/visions` replaces the
# VAE latent frontend: C-RADIO spatial features (`D = 1280`) become the
# reference and future tokens, the pretrained 128-wide action maps keep their
# contract, and the new visual input/output projections map directly between
# `D` and the transformer hidden width (3072 for Klein base 4B).
#
# This base trains every task of the dataset and declares no stage. The staged
# recipes inherit it with `_base_`:
#   patchwam_radio_stage1_robotwin.py               full task, projections only
#   patchwam_radio_stage2_robotwin.py          full task, projections + LoRA
#   patchwam_radio_ablation_default5_stage1_robotwin.py   five-task subset
#   patchwam_radio_ablation_default5_stage2_robotwin.py   five-task subset
#
# Assets are strict and local (no downloads). Every path is env-overridable:
#   PATCHWAM_ASSETS_ROOT, PATCHWAM_FLUX2_ROOT, PATCHWAM_FLUX2_SRC,
#   PATCHWAM_TRANSFORMER, PATCHWAM_AE, PATCHWAM_TEXT_ENCODER,
#   PATCHWAM_TEXT_TOKENIZER, PATCHWAM_RADIO_ROOT, PATCHWAM_DATA_ROOT,
#   PATCHWAM_CHECKPOINT, PATCHWAM_STAGE1_CHECKPOINT.
#
# Launch (single node, eight processes, from the FluxVLA root).
# `scripts/train.sh` runs torchrun with `NPROC_PER_NODE` processes per node and
# resolves the work directory itself:
#   NPROC_PER_NODE=8 bash scripts/train.sh \
#     configs/patchwam/patchwam_radio_stage1_robotwin.py \
#     work_dirs/patchwam_radio_stage1_robotwin
#
# The runner writes `<work_dir>/checkpoints/step-<6-digit step>-epoch-<3-digit
# epoch>[-loss=...].pt` plus `.safetensors`, and keeps `latest-checkpoint.pt` /
# `latest-checkpoint.safetensors` symlinks on the newest one.
# `runner.max_epochs` and `runner.max_steps` are mutually exclusive: use
# `max_steps` for a bounded step budget, and `save_iter_interval` to control
# how often step checkpoints appear. The resolved task/episode selection is
# written to `<work_dir>/dataset_selection.json`.

import os

seed = 42

_assets_root = os.environ.get('PATCHWAM_ASSETS_ROOT',
                              '/home/chuong/workspace/wam/ImageWAM')
_flux2_root = os.environ.get('PATCHWAM_FLUX2_ROOT',
                             os.path.join(_assets_root, 'checkpoints',
                                          'flux2'))
_flux_source = os.environ.get(
    'PATCHWAM_FLUX2_SRC', os.path.join(_assets_root, 'third_party', 'flux2'))
_transformer = os.environ.get(
    'PATCHWAM_TRANSFORMER',
    os.path.join(_flux2_root, 'FLUX.2-klein-base-4B',
                 'flux-2-klein-base-4b.safetensors'))
_text_encoder = os.environ.get(
    'PATCHWAM_TEXT_ENCODER',
    os.path.join(_flux2_root, 'FLUX.2-klein-base-4B', 'text_encoder'))
_tokenizer = os.environ.get(
    'PATCHWAM_TEXT_TOKENIZER',
    os.path.join(_flux2_root, 'FLUX.2-klein-base-4B', 'tokenizer'))
_radio_root = os.environ.get('PATCHWAM_RADIO_ROOT',
                             os.path.join(_assets_root, 'checkpoints',
                                          'C-RADIOv4-H'))
_data_root_paths = [
    os.environ.get('PATCHWAM_DATA_ROOT', 'datasets/robotwin_lerobotv2.1')
]
# Released or native pretrained PatchWAM weights for stage 1, and the stage 1
# checkpoint for stage 2.
_checkpoint = os.environ.get(
    'PATCHWAM_CHECKPOINT',
    '/data/Pretrained_models/patchwam-robotwin/step_140000.pt')
_stage1_checkpoint = os.environ.get('PATCHWAM_STAGE1_CHECKPOINT', '')

_text_dim = 7680           # Qwen3 FLUX.2 readout width (3 x 2560)
_hidden_dim = 3072         # Klein base 4B transformer width
_action_dim = 14           # RobotWin normalized joint/gripper action and state
_state_dim = 14
_token_dim = 128           # pretrained action/latent codec width
_visual_dim = 1280         # C-RADIOv4-H spatial feature width
_radio_version = 'c-radio_v4-h'
_statistic_name = 'robotwin'
_history_pool_size = 4

_name_mappings = {
    'observation.state': ['proprio'],
    'action': ['action'],
}
_statistic_keys = ['observation.state', 'timestamp', 'action']
_video_keys = ['observation.images.cam_high',
               'observation.images.cam_left_wrist',
               'observation.images.cam_right_wrist']
_augmentation = dict(
    p=0.5,
    augment_types=('corrupt_only', 'color_only', 'both'),
    color_jitter=dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    gamma=dict(range=[0.9, 1.1]),
    gaussian_noise=dict(std=0.01),
    random_resized_crop=dict(scale=[0.95, 1.0], ratio='preserve'),
    rotate=dict(degrees=5.0, fill='mean'),
)
_keep_keys = [
    'video', 'camera_video', 'frame_masks', 'img_masks', 'history_video',
    'history_valid', 'vl_current', 'vl_history', 'vl_history_valid',
    'actions', 'action_masks', 'states', 'task_description', 'subtask',
]
_collator = dict(
    type='DictCollator',
    keys=[
        'video', 'camera_video', 'frame_masks', 'img_masks', 'history_video',
        'history_valid', 'vl_current', 'vl_history', 'vl_history_valid',
        'actions', 'action_masks', 'states'
    ],
    meta_keys=['task_description', 'subtask'],
)


def _transforms(training: bool):
    return [
        dict(type='PatchWAMRawInputs'),
        dict(type='PatchWAMDecodeFrames', video_keys=_video_keys,
             backend='torchcodec', head_index=0),
        dict(type='PatchWAMCleanVL', image_size=448, head_index=0),
        dict(type='PatchWAMResizeCameras', camera_size=[240, 320]),
        dict(type='NormalizeStatesAndActions', state_key='proprio',
             action_key='action', action_dim=_action_dim, state_dim=_state_dim,
             norm_type='mean_std', statistics_key='stats',
             normalization_epsilon=1e-8),
        dict(type='PatchWAMEndpoints', horizon=16),
        dict(type='PatchWAMWorldAugment', config=_augmentation,
             enabled=training),
        dict(type='PatchWAMComposeWorld', layout='robotwin',
             video_size=[288, 256], robotwin_layout='compact_288x256',
             separate_views=False),
        dict(type='PatchWAMWorldHistory', head_index=0),
        dict(type='PatchWAMKeep', keys=_keep_keys, require=False),
    ]


def _dataset(training: bool, task_selection=None):
    """Native RoboTwin dataset; ``task_selection`` narrows it.

    The filter resolves from the per-root metadata before sample indexes and
    statistics are built, so both training stages of a recipe train on exactly
    the subset the recipe declares.
    """
    options = dict(
        type='PatchWAMParquetDataset',
        data_root_path=_data_root_paths,
        action_window_size=16,
        window_start_idx=0,
        frame_window_size=17,
        frame_sample_stride=1,
        future_offset=16,
        horizon=16,
        action_key='action',
        statistic_name=_statistic_name,
        training=training,
        history=dict(enabled=False),
        transforms=_transforms(training),
    )
    if task_selection is not None:
        options['task_selection'] = dict(task_selection)
    return dict(
        type='DistributedRepeatingDataset',
        reshuffle_each_epoch=True,
        seed=seed,
        name_mappings=_name_mappings,
        statistic_keys=_statistic_keys,
        statistic_name=_statistic_name,
        datasets=options,
    )


model = dict(
    type='PatchWAMVLA',
    action_horizon=16,
    action_norm_type='mean_std',
    num_inference_steps=10,
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='PatchWAMBackbone',
        token_dim=_token_dim,
        text_dim=_text_dim,
        history_pool_size=_history_pool_size,
        visual_backbone=dict(
            kind=_radio_version,
            model_path=_radio_root,
        ),
        text_encoder_path=_text_encoder,
        tokenizer_path=_tokenizer,
        flux_source=_flux_source,
        text_encoder_kind='qwen3',
        dtype='bfloat16',
        device='cuda',
        vl_trainable_adapters=False,
    ),
    vla_head=dict(
        type='PatchWAMHead',
        action_dim=_action_dim,
        text_dim=_text_dim,
        proprio_dim=_state_dim,
        token_dim=_token_dim,
        visual_dim=_visual_dim,
        video_weight=0.5,
        action_weight=1.0,
        isolate_actions=False,
        pack_proprio_after_text=True,
        denoiser=dict(
            type='PatchWAMFluxDenoiser',
            transformer_path=_transformer,
            flux_source=_flux_source,
            variant='klein-base-4b',
            dtype='bfloat16',
            device='cuda',
            gradient_checkpointing=True,
            visual_dim=_visual_dim,
        ),
    ),
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=16,
    dataset=_dataset(training=True),
)
val_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=_dataset(training=False),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=10,
    optimizer=dict(type='AdamW', lr=1.0e-4, weight_decay=0.01,
                   betas=(0.9, 0.95)),
    max_grad_norm=1.0,
    grad_accumulation_steps=1,
    collator=_collator,
    sampler='distributed',
    metric=dict(type='VLAMetric', active_trackers=('jsonl', 'wandb'),
                run_dir='work_dirs', window_size=1),
    lr_scheduler=dict(type='linear-warmup+cosine-decay', warmup_ratio=0.05),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    seed=42,
)

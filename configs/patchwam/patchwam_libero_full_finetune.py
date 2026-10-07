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
# PatchWAM joint world/action flow model finetuned on every LIBERO suite.
#
# LIBERO geometry: two cameras concatenated horizontally into one 224x448
# mosaic. A 17-row sampled window yields 16 action targets; the current frame is
# at the window start and the future endpoint 16 rows later.
#
# Asset loading is strict and local (no downloads). Point `_ckpt_root` at the
# official FLUX.2 Klein transformer/autoencoder, the local Qwen3 text encoder,
# and the black-forest-labs `flux2` source checkout. Assets/GPU were not
# available during the CPU integration, so full-size training is unverified.

import copy

seed = 42

# Local assets only. Klein base 4B packs 128-channel latents and consumes a
# 3x hidden-state Qwen3 text readout (Klein4BParams.context_in_dim == 7680).
_ckpt_root = './checkpoints'
_flux_source = './third_party/flux2'
_transformer = (_ckpt_root +
                '/flux2_klein_base_4b/transformer/diffusion_pytorch_model.safetensors')
_autoencoder = (_ckpt_root +
                '/flux2_klein_base_4b/vae/diffusion_pytorch_model.safetensors')
_text_encoder = _ckpt_root + '/flux2_klein_base_4b/text_encoder'
_norm_stats = _ckpt_root + '/patchwam_libero_norm_stats.json'
_text_dim = 7680
_action_dim = 7
_state_dim = 8

_data_root_paths = [
    'datasets/libero_spatial_no_noops_lerobotv2.1',
    'datasets/libero_object_no_noops_lerobotv2.1',
    'datasets/libero_goal_no_noops_lerobotv2.1',
    'datasets/libero_10_no_noops_lerobotv2.1',
]
_statistic_name = 'libero_all_no_noops'
_video_keys = ['observation.images.image', 'observation.images.wrist_image']

# LIBERO source recipe: two 224x224 cameras concatenated horizontally into one
# 224x448 mosaic, min/max normalization, and the source ClipAugment recipe.
_augmentation = dict(
    p=0.65,
    augment_types=('corrupt_only', 'color_only', 'both', 'both'),
    color_jitter=dict(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.04),
    gamma=dict(range=[0.8, 1.25]),
    gaussian_noise=dict(std=0.015),
    random_resized_crop=dict(scale=[0.92, 1.0], ratio='preserve'),
    rotate=dict(degrees=8.0, fill='mean'),
    exposure=dict(ev_range=[-0.25, 0.25]),
)
_keep_keys = [
    'video', 'camera_video', 'frame_masks', 'img_masks', 'history_video',
    'history_valid', 'vl_current', 'vl_history', 'vl_history_valid',
    'actions', 'action_masks', 'states', 'task_description', 'subtask',
]
_collator = dict(
    type='DictCollator',
    keys=['video', 'camera_video', 'frame_masks', 'img_masks', 'history_video',
          'history_valid', 'vl_current', 'vl_history', 'vl_history_valid',
          'actions', 'action_masks', 'states'],
    meta_keys=['task_description', 'subtask'],
)


def _transforms(training: bool):
    chain = [
        dict(type='PatchWAMRawInputs'),
        dict(type='PatchWAMDecodeFrames', video_keys=_video_keys,
             backend='torchcodec', head_index=0),
        dict(type='PatchWAMCleanVL', image_size=448, head_index=0),
        dict(type='PatchWAMResizeCameras', camera_size=[224, 224]),
        dict(type='NormalizeStatesAndActions', state_key='proprio',
             action_key='action', action_dim=_action_dim, state_dim=_state_dim,
             norm_type='min_max', pad_invalid_action_delta_dims=True,
             delta_action_dim_mask=[True, True, True, True, True, True, False],
             statistics_key='stats'),
        dict(type='PatchWAMEndpoints', horizon=16),
    ]
    if training:
        chain.append(dict(type='PatchWAMWorldAugment', config=_augmentation))
    else:
        chain.append(dict(type='PatchWAMWorldAugment', enabled=False))
    chain.extend([
        dict(type='PatchWAMComposeWorld', layout='horizontal',
             video_size=[224, 448], separate_views=False),
        dict(type='PatchWAMWorldHistory', head_index=0),
        dict(type='PatchWAMKeep', keys=_keep_keys, require=False),
    ])
    return chain


def _dataset(training: bool):
    return dict(
        type='DistributedRepeatingDataset',
        reshuffle_each_epoch=True,
        seed=42,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        statistic_keys=['observation.state', 'timestamp', 'action'],
        statistic_name=_statistic_name,
        datasets=dict(
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
        ),
    )


model = dict(
    type='PatchWAMVLA',
    action_horizon=16,
    action_norm_type='min_max',
    num_inference_steps=10,
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='PatchWAMBackbone',
        token_dim=128,
        text_dim=_text_dim,
        autoencoder_path=_autoencoder,
        text_encoder_path=_text_encoder,
        flux_source=_flux_source,
        text_encoder_kind='qwen3',
        dtype='bfloat16',
        device='cuda',
    ),
    vla_head=dict(
        type='PatchWAMHead',
        action_dim=_action_dim,
        text_dim=_text_dim,
        proprio_dim=_state_dim,
        token_dim=128,
        video_weight=0.5,
        action_weight=1.0,
        shift=5.0,
        isolate_actions=False,
        denoiser=dict(
            type='PatchWAMFluxDenoiser',
            transformer_path=_transformer,
            flux_source=_flux_source,
            variant='klein-base-4b',
            dtype='bfloat16',
            device='cuda',
            gradient_checkpointing=True,
        ),
    ),
)

inference_model = model.copy()

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=16,
    dataset=_dataset(training=True),
)

# Independent validation config (no shared nested dicts): no augmentation and
# deterministic history.
val_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=_dataset(training=False),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=50,
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

_eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_10',
    model_family='pi0',
    eval_chunk_size=10,
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    norm_stats_key=_statistic_name,
    dataset=dict(
        type='PatchWAMLiberoEvalDataset',
        norm_stats=_norm_stats,
        norm_stats_key=_statistic_name,
        task_suite_name='libero_10',
        device='cuda',
        transforms=[
            dict(type='ProcessLiberoEvalInputs',
                 img_keys=['agentview_image', 'robot0_eye_in_hand_image']),
            dict(type='PatchWAMEvalPack', layout='horizontal',
                 video_size=[224, 448], camera_size=[224, 224],
                 head_index=0),
            dict(type='LiberoProprioFromInputs', norm_type='min_max',
                 pos_key='robot0_eef_pos', quat_key='robot0_eef_quat',
                 gripper_key='robot0_gripper_qpos', state_dim=8,
                 stat_key='proprio', out_key='states'),
        ]),
    denormalize_action=dict(type='DenormalizeLiberoAction',
                            norm_type='min_max', action_dim=7),
)

# `scripts/inference.py` builds `cfg.inference` (not `eval`).
eval = copy.deepcopy(_eval)
inference = copy.deepcopy(_eval)

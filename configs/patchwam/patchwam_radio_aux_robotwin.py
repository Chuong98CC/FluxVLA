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
# PatchWAM on RobotWin with C-RADIO auxiliary supervision: the original
# 128-wide VAE future flow and the original action objective stay exactly as
# they are, and a weighted clean C-RADIOv4-H feature MSE is added from the selected
# FLUX.2 block's hidden future tokens through one `radio_projection` adapter.
#
#   loss = 0.5 * loss_video + 0.5 * loss_radio + 1.0 * loss_action
#
# C-RADIO is a training dependency: `inference_model` below is the deployable
# original PatchWAM model for the weights produced by
# `PatchWAMRadioAuxVLA.export_inference_state_dict()`, with no teacher and no
# adapter. `radio_model_path` points at a complete local C-RADIOv4-H export.
#
# Training resume is strict: a checkpoint missing the trained `radio_projection`
# is rejected even under the runner's `strict=False` load. Starting from a
# VAE-lineage PatchWAM checkpoint is an explicit operation:
#   model.initialize_from_legacy_state_dict(torch.load(path)['model'])

import os
from copy import deepcopy

seed = 42

_ckpt_root = os.environ.get('PATCHWAM_CKPT_ROOT', './checkpoints')
_flux_source = os.environ.get('PATCHWAM_FLUX2_SRC', './third_party/flux2')
_transformer = (_ckpt_root +
                '/flux2_klein_base_4b/transformer/diffusion_pytorch_model.safetensors')
_autoencoder = (_ckpt_root +
                '/flux2_klein_base_4b/vae/diffusion_pytorch_model.safetensors')
_text_encoder = _ckpt_root + '/flux2_klein_base_4b/text_encoder'
_radio_model = os.environ.get('PATCHWAM_RADIO_ROOT',
                              _ckpt_root + '/C-RADIOv4-H')
_text_dim = 7680
_action_dim = 14
_state_dim = 14
_radio_dim = 1280

_data_root_paths = ['datasets/robotwin_lerobotv2.1']
_statistic_name = 'robotwin'
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
    keys=['video', 'camera_video', 'frame_masks', 'img_masks', 'history_video',
          'history_valid', 'vl_current', 'vl_history', 'vl_history_valid',
          'actions', 'action_masks', 'states'],
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


def _dataset(training: bool):
    return dict(
        type='DistributedRepeatingDataset',
        reshuffle_each_epoch=True,
        seed=42,
        name_mappings={'observation.state': ['proprio'],
                       'action': ['action']},
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
    type='PatchWAMRadioAuxVLA',
    action_horizon=16,
    action_norm_type='mean_std',
    num_inference_steps=10,
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='PatchWAMRadioAuxBackbone',
        token_dim=128,
        text_dim=_text_dim,
        history_pool_size=4,
        autoencoder_path=_autoencoder,
        text_encoder_path=_text_encoder,
        flux_source=_flux_source,
        text_encoder_kind='qwen3_vl',
        dtype='bfloat16',
        device='cuda',
        vl_trainable_adapters=False,
        vl_instruction_layers=[9, 18, 27],
        vl_summary_layers=[18, 27, 36],
        vl_image_size=448,
        vl_require_subtasks=False,
        radio_model_path=_radio_model,
        radio_dim=_radio_dim,
        radio_patch_size=16,
        radio_min_resolution=16,
    ),
    vla_head=dict(
        type='PatchWAMRadioAuxHead',
        action_dim=_action_dim,
        text_dim=_text_dim,
        proprio_dim=_state_dim,
        token_dim=128,
        radio_dim=_radio_dim,
        radio_weight=0.5,
        video_weight=0.5,
        action_weight=1.0,
        # Zero-based transformer block whose hidden future tokens feed the
        # adapter; -1 keeps the original last-block behavior. Blocks are ordered
        # as all FLUX.2 double blocks followed by all single blocks.
        adapter_layer_idex=-1,
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

# Deployable artifact for `export_inference_state_dict()`: the original native
# PatchWAM model with no teacher and no training adapter.
inference_model = deepcopy(model)
inference_model['type'] = 'PatchWAMVLA'
inference_model['vlm_backbone'] = {
    key: value for key, value in inference_model['vlm_backbone'].items()
    if key not in ('radio_model_path', 'radio_dim', 'radio_patch_size',
                   'radio_min_resolution')
}
inference_model['vlm_backbone']['type'] = 'PatchWAMBackbone'
inference_model['vla_head']['type'] = 'PatchWAMHead'
inference_model['vla_head']['denoiser']['type'] = 'PatchWAMFluxDenoiser'
for key in ('radio_dim', 'radio_weight', 'adapter_layer_idex'):
    inference_model['vla_head'].pop(key, None)

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=16,
    dataset=_dataset(training=True),
)
val_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=8,
    dataset=_dataset(training=False),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=5,
    optimizer=dict(type='AdamW', lr=1.0e-4, weight_decay=0.01,
                   betas=(0.9, 0.95)),
    max_grad_norm=1.0,
    grad_accumulation_steps=2,
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

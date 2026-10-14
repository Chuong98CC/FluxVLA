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
# PatchWAM separate-view history example (RoboDojo-style).
#
# This demonstrates the same configurable interfaces with causal world-model
# history and the specialized Qwen3-VL visual history path:
#   * world history is pooled and encoded positionally (group -1, -2, ...);
#   * VL current/history come from the clean head-camera frames before world
#     augmentation and are encoded with their own jitter/dropout contract;
#   * the generated future raster length is the current view raster, never the
#     appended historical prefix (``sampling_future_length``).
#
# ONLINE HISTORY (inference/eval): causal history is only valid when the eval
# loop calls the buffer once per recorded observation with an episode id, a
# monotonically increasing timestamp, and a reset on episode change. Wire
# `PatchWAMHistoryBuffer` into the eval transform list and supply
# `patchwam_episode_index` / `patchwam_timestamp` / `patchwam_reset` every step.
# Without those signals the buffer raises instead of returning stale history,
# so this file does not claim history inference support on its own.

# RoboDojo source recipe: separate camera views, 20-slot causal world + VL
# history, trainable VL LoRA, and required subtask sentence labels. Klein base
# 4B is the source variant (not 9B).
seed = 42

_ckpt_root = './checkpoints'
_flux_source = './third_party/flux2'
_transformer = (_ckpt_root +
                '/flux2_klein_base_4b/transformer/diffusion_pytorch_model.safetensors')
_autoencoder = (_ckpt_root +
                '/flux2_klein_base_4b/vae/diffusion_pytorch_model.safetensors')
_text_encoder = _ckpt_root + '/flux2_klein_base_4b/text_encoder'
_text_dim = 7680
_action_dim = 14
_state_dim = 14

_data_root_paths = ['datasets/robodojo_lerobotv2.1']
_statistic_name = 'robodojo'
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
        dict(type='PatchWAMCleanVL', image_size=448, head_index=0,
             history_key='vl_history_frames',
             valid_key='patchwam_vl_history_valid'),
        dict(type='PatchWAMResizeCameras', camera_size=[256, 256]),
        dict(type='NormalizeStatesAndActions', state_key='proprio',
             action_key='action', action_dim=_action_dim, state_dim=_state_dim,
             norm_type='mean_std', statistics_key='stats',
             normalization_epsilon=1e-8),
        dict(type='PatchWAMEndpoints', horizon=16),
        dict(type='PatchWAMWorldAugment', config=_augmentation,
             enabled=training),
        dict(type='PatchWAMComposeWorld', layout='robotwin',
             video_size=[288, 256], robotwin_layout='compact_288x256',
             separate_views=True),
        dict(type='PatchWAMWorldHistory', head_index=0),
        dict(type='PatchWAMKeep', keys=_keep_keys, require=True),
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
            history=dict(enabled=True, slots=20, interval_s=1.0,
                         jitter_s=0.4, whole_dropout=0.2, slot_dropout=0.2,
                         truncate_recent=True, vl_jitter_frames=1),
            transforms=_transforms(training),
        ),
    )


model = dict(
    type='PatchWAMVLA',
    action_horizon=16,
    action_norm_type='mean_std',
    num_inference_steps=10,
    freeze_vlm_backbone=True,
    vlm_backbone=dict(
        type='PatchWAMBackbone',
        token_dim=128,
        text_dim=_text_dim,
        history_pool_size=4,
        autoencoder_path=_autoencoder,
        text_encoder_path=_text_encoder,
        flux_source=_flux_source,
        text_encoder_kind='qwen3_vl',
        dtype='bfloat16',
        device='cuda',
        vl_trainable_adapters=True,
        vl_lora_rank=64,
        vl_lora_alpha=128,
        vl_lora_dropout=0.05,
        vl_instruction_layers=[9, 18, 27],
        vl_summary_layers=[18, 27, 36],
        vl_history_pool_size=4,
        vl_image_size=448,
        vl_auxiliary_weight=0.1,
        vl_label_smoothing=0.1,
        vl_require_subtasks=True,
    ),
    vla_head=dict(
        type='PatchWAMHead',
        action_dim=_action_dim,
        text_dim=_text_dim,
        proprio_dim=_state_dim,
        token_dim=128,
        video_weight=0.5,
        action_weight=1.0,
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
val_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=8,
    dataset=_dataset(training=False),
)

runner = dict(
    type='DDPTrainRunner',
    max_epochs=100,
    optimizer=dict(type='AdamW', lr=1.0e-4, weight_decay=0.01,
                   betas=(0.9, 0.95)),
    max_grad_norm=1.0,
    grad_accumulation_steps=1,
    # Source RoboDojo effective batch: 8 x 32 x 1 = 256. Advisory only; the
    # runner warns once on rank zero and keeps running if the world size
    # changes.
    expected_global_batch_size=256,
    collator=_collator,
    sampler='distributed',
    metric=dict(type='VLAMetric', active_trackers=('jsonl', 'wandb'),
                run_dir='work_dirs', window_size=1),
    lr_scheduler=dict(type='linear-warmup+cosine-decay',
                      schedule_style='openpi', warmup_ratio=0.05,
                      min_lr_rate=0.01),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    seed=42,
)

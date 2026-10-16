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
# PatchWAM + LIT, Stage 2: joint training of the original PatchWAM branch and
# the LIT interface.
#
# Total loss =
#     patchwam_loss_weight * (0.5 * video + 1.0 * action)      # original head
#   + lit_loss_weight      * (1.0 * FM action + 0.3 * goal)    # LIT interface
#
# The shared FLUX.2 transformer and proprio projection are trained once and
# receive gradients from both branches; AE/Qwen stay frozen exactly as in
# PatchWAM. The LIT path reads the current observation prefix only and never the
# decoded future target or the goal.
#
# `pretrained_name_or_path` initializes the released PatchWAM weights
# (`step_140000.pt`) and `lit_stage1_checkpoint` bridges the trained Stage-1
# LIT weights. `from_pretrained()` (called by the native DDP/FSDP runner) loads
# the released weights, then the Stage-1 LIT payload with
# `bridge_from_stage1=True`, verifying every shared/action-expert tensor and
# reporting the new Stage-2 modules. Training fails loudly when the Stage-1
# checkpoint is missing; `lit_stage1_required=False` is the explicit
# random-init ablation.
#
# Env overrides:
#   PATCHWAM_ASSETS_ROOT, PATCHWAM_FLUX2_ROOT, PATCHWAM_FLUX2_SRC,
#   PATCHWAM_TRANSFORMER, PATCHWAM_AE, PATCHWAM_TEXT_ENCODER,
#   PATCHWAM_TEXT_TOKENIZER, PATCHWAM_CHECKPOINT, PATCHWAM_DATA_ROOT,
#   PATCHWAM_LIT_STAGE1_CHECKPOINT.
#
# Training normalization statistics are aggregated from the dataset metadata;
# `PATCHWAM_DATASET_STATS` belongs to the evaluation config.
#
# Smoke (single node):
#   python scripts/train.py \
#     --config configs/patchwam/patchwam_lit_stage2_robotwin.py \
#     --work-dir work_dirs/patchwam_lit_stage2_robotwin

import os

seed = 42

_patchwam_root = os.environ.get('PATCHWAM_ASSETS_ROOT',
                                '/home/chuong/workspace/wam/ImageWAM')
_flux2_root = os.environ.get('PATCHWAM_FLUX2_ROOT',
                             os.path.join(_patchwam_root, 'checkpoints',
                                          'flux2'))
_flux_source = os.environ.get(
    'PATCHWAM_FLUX2_SRC', os.path.join(_patchwam_root, 'third_party', 'flux2'))
_transformer = os.environ.get(
    'PATCHWAM_TRANSFORMER',
    os.path.join(_flux2_root, 'FLUX.2-klein-base-4B',
                 'flux-2-klein-base-4b.safetensors'))
_autoencoder = os.environ.get(
    'PATCHWAM_AE', os.path.join(_flux2_root, 'FLUX.2-dev', 'ae.safetensors'))
_text_encoder = os.environ.get(
    'PATCHWAM_TEXT_ENCODER',
    os.path.join(_flux2_root, 'FLUX.2-klein-base-4B', 'text_encoder'))
_tokenizer = os.environ.get(
    'PATCHWAM_TEXT_TOKENIZER',
    os.path.join(_flux2_root, 'FLUX.2-klein-base-4B', 'tokenizer'))
_checkpoint = os.environ.get(
    'PATCHWAM_CHECKPOINT',
    '/data/Pretrained_models/patchwam-robotwin/step_140000.pt')
_stage1_checkpoint = os.environ.get('PATCHWAM_LIT_STAGE1_CHECKPOINT', '')
_data_root_paths = [
    os.environ.get('PATCHWAM_DATA_ROOT', 'datasets/robotwin_lerobotv2.1')
]
_text_dim = 7680
_hidden_dim = 3072
_action_dim = 14
_state_dim = 14
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
    'actions', 'action_masks', 'states', 'goal_state', 'goal_valid',
    'task_description', 'subtask',
]
_collator = dict(
    type='DictCollator',
    keys=['video', 'camera_video', 'frame_masks', 'img_masks', 'history_video',
          'history_valid', 'vl_current', 'vl_history', 'vl_history_valid',
          'actions', 'action_masks', 'states', 'goal_state', 'goal_valid'],
    meta_keys=['task_description', 'subtask'],
)


def _transforms(training: bool):
    return [
        dict(type='PatchWAMRawInputs'),
        dict(type='PatchWAMDecodeFrames', video_keys=_video_keys,
             backend='torchcodec', head_index=0),
        # CleanVL must run before the world augmentation.
        dict(type='PatchWAMCleanVL', image_size=448, head_index=0),
        dict(type='PatchWAMResizeCameras', camera_size=[240, 320]),
        dict(type='NormalizeStatesAndActions', state_key='proprio',
             action_key='action', action_dim=_action_dim, state_dim=_state_dim,
             norm_type='mean_std', statistics_key='stats',
             normalization_epsilon=1e-8),
        # Terminal goal is normalized with the same state statistics and no
        # action statistics, before any world augmentation runs.
        dict(type='PatchWAMLITGoal', state_key='proprio',
             state_dim=_state_dim, norm_type='mean_std',
             normalization_epsilon=1e-8,
             statistics_key='stats'),
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
            type='PatchWAMLITParquetDataset',
            data_root_path=_data_root_paths,
            action_window_size=16,
            window_start_idx=0,
            frame_window_size=17,
            frame_sample_stride=1,
            future_offset=16,
            horizon=16,
            goal_offset=16,
            goal_state_key='observation.state',
            action_key='action',
            statistic_name=_statistic_name,
            training=training,
            history=dict(enabled=False),
            transforms=_transforms(training),
        ),
    )


model = dict(
    type='PatchWAMLITVLA',
    lit_stage='stage2',
    training_mode='joint',
    active_head='lit',
    patchwam_loss_weight=1.0,
    lit_loss_weight=1.0,
    lit_stage1_checkpoint=(_stage1_checkpoint or None),
    lit_stage1_required=True,
    action_horizon=16,
    action_norm_type='mean_std',
    num_inference_steps=10,
    freeze_vlm_backbone=True,
    pretrained_name_or_path=_checkpoint,
    vlm_backbone=dict(
        type='PatchWAMBackbone',
        token_dim=128,
        text_dim=_text_dim,
        history_pool_size=4,
        autoencoder_path=_autoencoder,
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
        token_dim=128,
        video_weight=0.5,
        action_weight=1.0,
        isolate_actions=False,
        # The released RobotWin C2R recipe packs the state token directly after
        # the valid instruction prefix.
        pack_proprio_after_text=True,
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
    lit_head=dict(
        type='PatchWAMLITFlowMatchingHead',
        stage='stage2',
        action_dim=_action_dim,
        context_dim=_hidden_dim,
        prefix_input_dim=_text_dim,
        proprio_dim=_state_dim,
        num_latents=100,
        num_pose_tokens=8,
        num_layer_groups=5,
        latent_dim=768,
        inner_dim=512,
        num_heads=8,
        ffn_ratio=4.0,
        dropout=0.0,
        pose_loss_weight=0.3,
        fm_action_weight=1.0,
        action_horizon=16,
        prefix_timestep=0.0,
        # Coupling layers are derived from the loaded FLUX depth (25 layers for
        # Klein base 4B -> 5, 10, 15, 20, 25).
        action_expert=dict(
            type='PatchWAMLITActionDiT',
            hidden_dim=1024,
            action_dim=_action_dim,
            ffn_dim=4096,
            text_dim=_hidden_dim,
            freq_dim=256,
            eps=1e-6,
            num_heads=16,
            attn_head_dim=64,
            num_layers=25,
            use_gradient_checkpointing=True,
        ),
    ),
)

inference_model = model.copy()

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
    auto_scale_lr=True,
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

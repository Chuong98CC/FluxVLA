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
# PatchWAM + LIT, Stage 1: the image-free, goal-conditioned action prior.
#
# Stage 1 observes only the language instruction, the current normalized
# proprioception and the *oracle* terminal goal state (observation.state at
# t + 16, normalized with the state/proprio statistics). No camera frame enters
# the autoencoder, the frozen FLUX prefix, or the action expert, so the trained
# goal encoder and flow-matching action expert are the only moving parts.
#
# Assets are strict and local (no downloads). Every path is env-overridable:
#   PATCHWAM_ASSETS_ROOT, PATCHWAM_FLUX2_ROOT, PATCHWAM_FLUX2_SRC,
#   PATCHWAM_TRANSFORMER, PATCHWAM_AE, PATCHWAM_TEXT_ENCODER,
#   PATCHWAM_TEXT_TOKENIZER, PATCHWAM_CHECKPOINT, PATCHWAM_DATA_ROOT.
#
# Training normalization statistics are aggregated from the dataset metadata by
# `DistributedRepeatingDataset`; `PATCHWAM_DATASET_STATS` is used by the
# evaluation config only. `PATCHWAM_CHECKPOINT` initializes the shared FLUX
# transformer and proprio projection from the same released PatchWAM weights the
# Stage-2 recipe uses.
#
# Smoke (single node):
#   python scripts/train.py \
#     --config configs/patchwam/patchwam_lit_stage1_robotwin.py \
#     --work-dir work_dirs/patchwam_lit_stage1_robotwin

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
_data_root_paths = [
    os.environ.get('PATCHWAM_DATA_ROOT', 'datasets/robotwin_lerobotv2.1')
]
_text_dim = 7680          # Qwen3 FLUX.2 readout width (3 x 2560)
_hidden_dim = 3072        # Klein base 4B transformer width
_action_dim = 14          # RobotWin normalized joint/gripper action and state
_state_dim = 14
_statistic_name = 'robotwin'

# Stage 1 never decodes frames: no world augmentation, no mosaic, no VL images,
# no camera or frame masks. Only the language instruction, the current state,
# the action window and the terminal goal survive `PatchWAMKeep`.
_keep_keys = [
    'actions', 'action_masks', 'states', 'goal_state', 'goal_valid',
    'task_description',
]
_collator = dict(
    type='DictCollator',
    keys=['actions', 'action_masks', 'states', 'goal_state', 'goal_valid'],
    meta_keys=['task_description'],
)


def _transforms(training: bool):
    del training
    return [
        dict(type='PatchWAMRawInputs'),
        dict(type='NormalizeStatesAndActions', state_key='proprio',
             action_key='action', action_dim=_action_dim, state_dim=_state_dim,
             norm_type='mean_std', statistics_key='stats',
             normalization_epsilon=1e-8),
        dict(type='PatchWAMLITGoal', state_key='proprio',
             state_dim=_state_dim, norm_type='mean_std',
             normalization_epsilon=1e-8,
             statistics_key='stats'),
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
    lit_stage='stage1',
    training_mode='lit',
    active_head='lit',
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
        stage='stage1',
        action_dim=_action_dim,
        context_dim=_hidden_dim,
        prefix_input_dim=_text_dim,
        proprio_dim=_state_dim,
        num_goal_tokens=8,
        inner_dim=512,
        pose_loss_weight=0.3,
        fm_action_weight=1.0,
        action_horizon=16,
        prefix_timestep=0.0,
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

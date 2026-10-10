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
# PatchWAM 2D patch-motion: one current/front RGB view plus language jointly
# predict the future endpoint latent and the signed per-step displacement of
# every 16-pixel image patch. Each 8x8 region of patches packs its 64
# (delta_row, delta_column) pairs into one width-128 motion token, so three
# steps at 256x256 (16x16 patch grid, K=4) give 12 motion tokens.
#
#   loss = 0.5 * loss_video + 1.0 * loss_motion
#
# There is no robot state, no robot action chunk and no action normalization:
# `motion_2d` stays in signed pixels and the codec owns the single internal
# (H, W) normalization. The dataset manifest owns the fixed sample schedule;
# `horizon`/`timestep_offsets_s` below only restate it and are validated
# against `dataset.json`. See `docs/patchwam-2d-motion.md` for the exact
# offline export layout.

import os

seed = 42

_ckpt_root = os.environ.get('PATCHWAM_CKPT_ROOT', './checkpoints')
_flux_source = os.environ.get('PATCHWAM_FLUX2_SRC', './third_party/flux2')
_dataset_root = os.environ.get('PATCHWAM_2D_TRACE_ROOT',
                               './datasets/patchwam_2d_trace')
_transformer = (_ckpt_root +
                '/flux2_klein_base_4b/transformer/diffusion_pytorch_model.safetensors')
_autoencoder = (_ckpt_root +
                '/flux2_klein_base_4b/vae/diffusion_pytorch_model.safetensors')
_text_encoder = _ckpt_root + '/flux2_klein_base_4b/text_encoder'

_text_dim = 7680
_image_height = 256
_image_width = 256
_motion_horizon = 3
_timestep_offsets_s = [0.25, 0.5, 1.0]

_collator = dict(
    type='DictCollator',
    keys=['images', 'motion_2d', 'motion_valid'],
    meta_keys=['task_description', 'sample_id', 'episode_id'],
)


def _dataset(split):
    return dict(
        type='PatchWAM2DTraceDataset',
        dataset_root=_dataset_root,
        split=split,
        horizon=_motion_horizon,
        patch_size=16,
        motion_block_size=8,
        coordinate_order='row_column',
        timestep_offsets_s=_timestep_offsets_s,
        check_split_disjoint=True,
    )


model = dict(
    type='PatchWAM2DMotionVLA',
    motion_horizon=_motion_horizon,
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
        text_encoder_kind='qwen3',
        dtype='bfloat16',
        device='cuda',
        context_length=128,
    ),
    vla_head=dict(
        type='PatchWAM2DMotionHead',
        text_dim=_text_dim,
        token_dim=128,
        image_height=_image_height,
        image_width=_image_width,
        patch_size=16,
        block_size=8,
        motion_horizon=_motion_horizon,
        video_weight=0.5,
        motion_weight=1.0,
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

train_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=8,
    dataset=_dataset('train'),
)
val_dataloader = dict(
    per_device_batch_size=4,
    per_device_num_workers=4,
    dataset=_dataset('val'),
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

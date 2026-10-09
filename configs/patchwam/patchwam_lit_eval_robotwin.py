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
# RobotWin 2.0 evaluation of a trained PatchWAM + LIT Stage-2 checkpoint.
#
# The LIT inference path is current-observation only: it reads the language,
# the current proprioception and the current camera frames, runs the shared
# FLUX prefix once, streams the aggregator latents and denoises normalized
# [B, 16, 14] actions. Oracle goals are rejected.
#
# To evaluate the original PatchWAM head from the same dual-head model, switch
# the active head in the model config. `RoboTwinEvalRunner` takes no `head`
# argument, and the evaluator builds `inference_model` when a config defines it
# (`base_eval_runner.prepare_eval_model_cfg`), so override both keys:
#   --cfg-options model.active_head=patchwam inference_model.active_head=patchwam
#
# This config is the single-task integration smoke (episodes_per_task defaults
# to 1 and can be raised with PATCHWAM_EPISODES). The official 50-task / 100
# episode protocol lives in `patchwam_lit_official_benchmark.py`, mirroring the
# released PatchWAM benchmark config.
#
# Normalization: `dataset_stats_path` defaults to the Stage-2 run's
# `dataset_statistics.json` (written by `scripts/train.py`), so trained LIT
# weights evaluate with their own Stage-2 training statistics instead of the
# released PatchWAM asset. The native payload is accepted directly; see
# docs/patchwam-lit.md for the explicit conversion command if the released
# schema is needed elsewhere.
#
# Smoke (one task, one episode):
#   CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
#     --config configs/patchwam/patchwam_lit_eval_robotwin.py
#
# Env overrides: PATCHWAM_LIT_CHECKPOINT (defaults to the Stage-2 run dir
# checkpoint), plus the asset/data variables used by the training configs.

import os

seed = 42

_patchwam_root = os.environ.get('PATCHWAM_ASSETS_ROOT',
                                '/home/chuong/workspace/wam/ImageWAM')
_robotwin_root = os.environ.get(
    'PATCHWAM_ROBOTWIN_ROOT',
    os.path.join(_patchwam_root, 'third_party', 'RoboTwin'))
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
    'PATCHWAM_LIT_CHECKPOINT',
    'work_dirs/patchwam_lit_stage2_robotwin/checkpoints/latest-checkpoint.pt')
_stats = os.environ.get(
    'PATCHWAM_DATASET_STATS',
    'work_dirs/patchwam_lit_stage2_robotwin/dataset_statistics.json')
_sim_python = os.environ.get(
    'PATCHWAM_SIM_PYTHON',
    os.path.expanduser('~/miniconda3/envs/Robotwin/bin/python'))

_text_dim = 7680
_hidden_dim = 3072
_action_dim = 14
_state_dim = 14

model = dict(
    type='PatchWAMLITVLA',
    lit_stage='stage2',
    training_mode='joint',
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
            gradient_checkpointing=False,
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
        pose_loss_weight=0.3,
        fm_action_weight=1.0,
        action_horizon=16,
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
            use_gradient_checkpointing=False,
        ),
    ),
)

inference_model = model.copy()

eval = dict(
    type='RoboTwinEvalRunner',
    seed=seed,
    task_name='adjust_bottle',
    task_config='demo_clean',
    robotwin_root=_robotwin_root,
    sim_python=_sim_python,
    dataset_stats_path=_stats,
    instruction_type='unseen',
    expert_check=True,
    setup_overrides=dict(clear_cache_freq=1),
    episodes_per_task=int(os.environ.get('PATCHWAM_EPISODES', '1')),
    action_horizon=16,
    replan_steps=16,
    num_inference_steps=10,
    output_dir='runs/evaluation/patchwam_lit_robotwin',
)

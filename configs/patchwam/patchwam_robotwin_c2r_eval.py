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
# Native FluxVLA evaluation of the released PatchWAM RoboTwin 2.0 C2R
# checkpoint (``QZWang/patchwam-robotwin-c2r-step140000``, step 140000).
#
# The checkpoint stores the fine-tuned FLUX.2 Klein base 4B transformer plus
# the 14-D proprio projection; the FLUX.2 autoencoder, Qwen3-4B instruction
# encoder and RoboTwin 2.0 simulator come from local assets. Paths are
# environment-overridable so the same config runs on another checkout.
#
# Smoke (one task, one episode):
#   CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
#     --config configs/patchwam/patchwam_robotwin_c2r_eval.py
#
# This is an integration smoke, not the 50-task / 100-episode benchmark.

import os

seed = 42

_patchwam_root = os.environ.get('PATCHWAM_ASSETS_ROOT',
                                '/home/chuong/workspace/wam/ImageWAM')
_robotwin_root = os.environ.get(
    'PATCHWAM_ROBOTWIN_ROOT',
    os.path.join(_patchwam_root, 'third_party', 'RoboTwin'))

_ckpt = os.environ.get(
    'PATCHWAM_CHECKPOINT',
    '/data/Pretrained_models/patchwam-robotwin/step_140000.pt')
_flux2_root = os.environ.get(
    'PATCHWAM_FLUX2_ROOT',
    os.path.join(_patchwam_root, 'checkpoints', 'flux2'))
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
_stats = os.environ.get(
    'PATCHWAM_DATASET_STATS',
    '/home/chuong/workspace/RoboDojo/XPolicyLab/policy/PatchWAM/assets/'
    'dataset_stats.json')
_sim_python = os.environ.get(
    'PATCHWAM_SIM_PYTHON',
    os.path.join(_patchwam_root, '.venv_rb2', 'bin', 'python'))

_action_dim = 14
_state_dim = 14
_text_dim = 7680

model = dict(
    type='PatchWAMVLA',
    action_horizon=16,
    action_norm_type='mean_std',
    num_inference_steps=10,
    freeze_vlm_backbone=True,
    pretrained_name_or_path=_ckpt,
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
        # The released ImageWAM-lineage recipe packs the proprio token directly
        # after the valid instruction prefix (``pack_proprio_after_text``).
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
)

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
    episodes_per_task=int(os.environ.get('PATCHWAM_EPISODES', '1')),
    action_horizon=16,
    replan_steps=16,
    num_inference_steps=10,
    output_dir='runs/evaluation/patchwam_robotwin',
)

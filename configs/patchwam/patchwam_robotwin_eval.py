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
# RobotWin 2.0 evaluation of a trained PatchWAM checkpoint.
#
# Evaluation uses the shared FluxVLA RoboTwin evaluator (`RobotwinEvalRunner`),
# which embeds RoboTwin's own evaluation harness in this process and drives it
# through the harness `eval_policy.py` contract. There is no separate simulator
# environment and no FluxVLA-owned rollout loop.
#
# The model block mirrors `patchwam_robotwin_vlm_full_finetune.py`, so a
# checkpoint from that recipe evaluates with the architecture it was trained
# under. Observation preprocessing is PatchWAM-specific and lives at the
# dataset boundary (`PatchWAMRoboTwinEvalDataset`): the three RoboTwin cameras
# compose into the same compact 288x256 mosaic the trainer builds and proprio
# is z-scored with the run's statistics. Training-only augmentation is
# deliberately absent. Actions are absolute joint targets normalized with
# mean/std, matching `NormalizeStatesAndActions` in the training recipe, so
# `DenormalizePrivateAction` reverses exactly that transform.
#
# `norm_stats_path` defaults to the run's `dataset_statistics.json`, i.e. the
# statistics the checkpoint was trained with. The released PatchWAM
# `dataset_stats.json` asset uses a different schema and checkpoint layout; see
# docs/patchwam-robotwin-evaluation.md for that migration note.
#
# Smoke (one task, one trial). `--ckpt-path` is required, exactly like the other
# RoboTwin configs, because `scripts/eval.py` takes the checkpoint from the CLI:
#   CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
#     --config configs/patchwam/patchwam_robotwin_eval.py \
#     --ckpt-path work_dirs/patchwam_robotwin_vlm_full_finetune/checkpoints/latest-checkpoint.pt
#
# The official 50-task / clean+random protocol is
# `patchwam_robotwin_official_benchmark.py`, launched through
# `scripts/eval_robotwin_manager.sh`.
#
# Env overrides: PATCHWAM_CHECKPOINT, PATCHWAM_DATASET_STATS,
# PATCHWAM_ROBOTWIN_ROOT, PATCHWAM_ROBOTWIN_TASKS, PATCHWAM_TASK_SUITE,
# PATCHWAM_EPISODES.

import os

seed = 42

_assets_root = os.environ.get('PATCHWAM_ASSETS_ROOT', '.')
_flux_source = os.environ.get('PATCHWAM_FLUX2_SRC',
                              './third_party/flux2')
_ckpt_root = os.environ.get('PATCHWAM_FLUX2_ROOT',
                            os.path.join(_assets_root, 'checkpoints'))
_transformer = os.environ.get(
    'PATCHWAM_TRANSFORMER',
    os.path.join(_ckpt_root, 'flux2_klein_base_4b', 'transformer',
                 'diffusion_pytorch_model.safetensors'))
_autoencoder = os.environ.get(
    'PATCHWAM_AE',
    os.path.join(_ckpt_root, 'flux2_klein_base_4b', 'vae',
                 'diffusion_pytorch_model.safetensors'))
_text_encoder = os.environ.get(
    'PATCHWAM_TEXT_ENCODER',
    os.path.join(_ckpt_root, 'flux2_klein_base_4b', 'text_encoder'))
_tokenizer = os.environ.get(
    'PATCHWAM_TEXT_TOKENIZER',
    os.path.join(_ckpt_root, 'flux2_klein_base_4b', 'tokenizer'))
_checkpoint = os.environ.get(
    'PATCHWAM_CHECKPOINT',
    'work_dirs/patchwam_robotwin_vlm_full_finetune/checkpoints/'
    'latest-checkpoint.pt')
_stats = os.environ.get(
    'PATCHWAM_DATASET_STATS',
    'work_dirs/patchwam_robotwin_vlm_full_finetune/dataset_statistics.json')
_robotwin_root = os.environ.get(
    'PATCHWAM_ROBOTWIN_ROOT',
    os.environ.get('ROBOTWIN_ROOT', 'src/RoboTwin'))

_text_dim = 7680
_action_dim = 14
_state_dim = 14
_statistic_name = 'robotwin'

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
        tokenizer_path=_tokenizer,
        flux_source=_flux_source,
        text_encoder_kind='qwen3_vl',
        dtype='bfloat16',
        device='cuda',
        vl_trainable_adapters=False,
        vl_instruction_layers=[9, 18, 27],
        vl_summary_layers=[18, 27, 36],
        vl_image_size=448,
        vl_require_subtasks=False,
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
            gradient_checkpointing=False,
        ),
    ),
)

inference_model = model.copy()

eval = dict(
    type='RobotwinEvalRunner',
    seed=seed,
    model_family='patchwam',
    robotwin_root=_robotwin_root,
    task_suite_name=os.environ.get('PATCHWAM_TASK_SUITE', 'clean'),
    task_list=os.environ.get('PATCHWAM_ROBOTWIN_TASKS', 'adjust_bottle'),
    # The training recipe predicts a 16-step horizon; execute all of it.
    eval_chunk_size=16,
    instruction_type='unseen',
    num_trials_per_task=int(os.environ.get('PATCHWAM_EPISODES', '1')),
    unnorm_key=_statistic_name,
    norm_stats_path=_stats,
    result_output_dir='runs/evaluation/patchwam_robotwin',
    dataset=dict(
        type='PatchWAMRoboTwinEvalDataset',
        statistic_name=_statistic_name,
        device='cuda',
        transforms=[
            dict(type='ProcessEvalInputs',
                 img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist']),
            dict(type='PatchWAMEvalPack',
                 layout='robotwin',
                 video_size=[288, 256],
                 camera_size=[240, 320],
                 robotwin_layout='compact_288x256',
                 head_index=0),
            dict(type='StateFromInputs',
                 state_key='qpos',
                 stat_key='proprio',
                 norm_type='mean_std',
                 state_dim=_state_dim,
                 out_key='states'),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=_action_dim,
    ),
)

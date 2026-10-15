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
# Evaluation uses the shared FluxVLA RoboTwin evaluator (`RobotwinEvalRunner`),
# which embeds RoboTwin's own evaluation harness in this process and drives it
# through the harness `eval_policy.py` contract. There is no separate simulator
# environment and no FluxVLA-owned rollout loop.
#
# The LIT inference path is current-observation only: it reads the language,
# the current proprioception and the current camera frames, runs the shared
# FLUX prefix once, streams the aggregator latents and denoises normalized
# [B, 16, 14] actions. Oracle goals are rejected.
#
# To evaluate the original PatchWAM head from the same dual-head model, switch
# the active head in the model config. `RobotwinEvalRunner` takes no `head`
# argument, and the evaluator builds `inference_model` when a config defines it
# (`base_eval_runner.prepare_eval_model_cfg`), so override both keys:
#   --cfg-options model.active_head=patchwam inference_model.active_head=patchwam
#
# Observation preprocessing is PatchWAM-specific and lives at the dataset
# boundary (`PatchWAMRoboTwinEvalDataset`): the three RoboTwin cameras are
# composed into the same compact 288x256 mosaic the trainer builds, and proprio
# is z-scored with the run's statistics. Training-only augmentation is
# deliberately absent. Actions are absolute joint targets normalized with
# mean/std, matching `NormalizeStatesAndActions` in the Stage-1/Stage-2 recipes,
# so `DenormalizePrivateAction` reverses exactly that transform.
#
# Normalization: `norm_stats_path` points at the Stage-2 run's
# `dataset_statistics.json` (written by `scripts/train.py`), so trained LIT
# weights evaluate with their own Stage-2 training statistics. The released
# PatchWAM `dataset_stats.json` asset uses a different schema and checkpoint
# layout; see docs/patchwam-robotwin-evaluation.md for the migration note.
#
# Smoke (one task, one trial). `--ckpt-path` is required, exactly like the
# other RoboTwin configs, because `scripts/eval.py` takes the checkpoint from
# the CLI:
#   CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
#     --config configs/patchwam/patchwam_lit_eval_robotwin.py \
#     --ckpt-path work_dirs/patchwam_lit_stage2_robotwin/checkpoints/latest-checkpoint.pt
#
# Env overrides: PATCHWAM_ROBOTWIN_TASKS, PATCHWAM_TASK_SUITE,
# PATCHWAM_EPISODES, plus the asset/data variables used by the training configs.

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

_text_dim = 7680
_hidden_dim = 3072
_action_dim = 14
_state_dim = 14
_statistic_name = 'robotwin'

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
    type='RobotwinEvalRunner',
    seed=seed,
    model_family='patchwam_lit',
    robotwin_root=_robotwin_root,
    task_suite_name=os.environ.get('PATCHWAM_TASK_SUITE', 'clean'),
    task_list=os.environ.get('PATCHWAM_ROBOTWIN_TASKS', 'adjust_bottle'),
    # The released protocol executes the full trained horizon per prediction.
    eval_chunk_size=16,
    instruction_type='unseen',
    num_trials_per_task=int(os.environ.get('PATCHWAM_EPISODES', '1')),
    unnorm_key=_statistic_name,
    norm_stats_path=_stats,
    result_output_dir='runs/evaluation/patchwam_lit_robotwin',
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

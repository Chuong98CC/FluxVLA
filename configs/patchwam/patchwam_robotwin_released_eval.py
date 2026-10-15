"""RobotWin 2.0 evaluation of the released PatchWAM C2R checkpoint.

This is the ImageWAM-lineage released policy (`QZWang/patchwam-robotwin-c2r-step140000`,
step 140000), evaluated through the shared FluxVLA RoboTwin evaluator
(`RobotwinEvalRunner`). It is *not* the newly trained Qwen3-VL recipe: the
released checkpoint was trained with `text_encoder_kind='qwen3'` (text-only
instruction encoder) and `pack_proprio_after_text=True`, so the model block
below matches that architecture.

The released checkpoint and statistics are consumed through explicit
conversions that never modify the originals:

    python tools/convert_patchwam_robotwin_released.py

which writes a native safetensors (`PATCHWAM_CONVERTED_CHECKPOINT`) whose keys
match the built model exactly, and a native `dataset_statistics.json`
(`PATCHWAM_CONVERTED_STATS`) built from the released `global_mean`/`global_std`
z-score fields. The shared evaluator loads the converted checkpoint with a
strict state-dict load.

Run it with the wrapper (defaults point here) or directly:

    CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
      --config configs/patchwam/patchwam_robotwin_released_eval.py \
      --ckpt-path work_dirs/patchwam_robotwin_released_conversion/step_140000_native.safetensors

Env overrides: PATCHWAM_ASSETS_ROOT, PATCHWAM_FLUX2_ROOT, PATCHWAM_FLUX2_SRC,
PATCHWAM_TRANSFORMER, PATCHWAM_AE, PATCHWAM_TEXT_ENCODER,
PATCHWAM_TEXT_TOKENIZER, PATCHWAM_RELEASED_CHECKPOINT,
PATCHWAM_RELEASED_STATS, PATCHWAM_CONVERTED_ROOT,
PATCHWAM_CONVERTED_CHECKPOINT, PATCHWAM_CONVERTED_STATS, PATCHWAM_ROBOTWIN_ROOT,
PATCHWAM_ROBOTWIN_TASKS, PATCHWAM_TASK_SUITE, PATCHWAM_EPISODES.
"""

import os

seed = 42

_assets_root = os.environ.get(
    'PATCHWAM_ASSETS_ROOT', '/home/chuong/workspace/wam/ImageWAM')
_flux2_root = os.environ.get('PATCHWAM_FLUX2_ROOT',
                             os.path.join(_assets_root, 'checkpoints', 'flux2'))
_flux_source = os.environ.get(
    'PATCHWAM_FLUX2_SRC', os.path.join(_assets_root, 'third_party', 'flux2'))
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

_robotwin_root = os.environ.get(
    'PATCHWAM_ROBOTWIN_ROOT',
    os.environ.get('ROBOTWIN_ROOT', 'src/RoboTwin'))

# Released assets (read-only inputs to the conversion).
PATCHWAM_RELEASED_CHECKPOINT = os.environ.get(
    'PATCHWAM_RELEASED_CHECKPOINT',
    'checkpoints/Pretrained_models/patchwam-robotwin/step_140000.pt')
PATCHWAM_RELEASED_STATS = os.environ.get(
    'PATCHWAM_RELEASED_STATS',
    '/home/chuong/workspace/RoboDojo/XPolicyLab/policy/PatchWAM/assets/'
    'dataset_stats.json')

# Converted native artifacts written by tools/convert_patchwam_robotwin_released.py.
_converted_root = os.environ.get(
    'PATCHWAM_CONVERTED_ROOT',
    'work_dirs/patchwam_robotwin_released_conversion')
PATCHWAM_CONVERTED_CHECKPOINT = os.environ.get(
    'PATCHWAM_CONVERTED_CHECKPOINT',
    os.path.join(_converted_root, 'step_140000_native.safetensors'))
PATCHWAM_CONVERTED_STATS = os.environ.get(
    'PATCHWAM_CONVERTED_STATS',
    os.path.join(_converted_root, 'dataset_statistics.json'))

# ``RobotwinEvalRunner`` accepts a list of tasks or a single task name; a bare
# string is one task, so the env override is split into a real list.
_task_list = os.environ.get(
    'PATCHWAM_ROBOTWIN_TASKS', 'adjust_bottle rotate_qrcode').split()

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
    pretrained_name_or_path=PATCHWAM_RELEASED_CHECKPOINT,
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
        # Released ImageWAM-lineage recipe: the proprio token follows the valid
        # instruction prefix.
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

inference_model = model.copy()

eval = dict(
    type='RobotwinEvalRunner',
    seed=seed,
    model_family='patchwam_released',
    robotwin_root=_robotwin_root,
    task_suite_name=os.environ.get('PATCHWAM_TASK_SUITE', 'clean'),
    task_list=_task_list,
    # The released protocol executes the full trained horizon per prediction.
    eval_chunk_size=16,
    instruction_type='unseen',
    num_trials_per_task=int(os.environ.get('PATCHWAM_EPISODES', '1')),
    unnorm_key=_statistic_name,
    norm_stats_path=PATCHWAM_CONVERTED_STATS,
    result_output_dir='runs/evaluation/patchwam_released_robotwin',
    dataset=dict(
        type='PatchWAMRoboTwinEvalDataset',
        statistic_name=_statistic_name,
        device='cuda',
        transforms=[
            dict(type='ProcessEvalInputs',
                 img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist']),
            dict(type='PatchWAMEvalFrames'),
            dict(type='PatchWAMEvalPack',
                 layout='robotwin',
                 video_size=[288, 256],
                 camera_size=[240, 320],
                 robotwin_layout='compact_288x256',
                 head_index=0),
            # The released z-score uses the same mean/std normalizer the
            # trainer uses, so the shared transform and epsilon are reused.
            dict(type='NormalizeStatesAndActions',
                 state_key='proprio',
                 action_key=None,
                 norm_type='mean_std',
                 state_dim=_state_dim,
                 statistics_key='norm_stats',
                 normalization_epsilon=1e-8,
                 output_dtype='float32'),
        ]),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=_action_dim,
    ),
)

"""Fine-tune and evaluate the released text-only Qwen3 PatchWAM policy."""

import copy
import os


_ROBOTWIN_TASK_LIST = [
    'adjust_bottle',
    'beat_block_hammer',
    'blocks_ranking_rgb',
    'blocks_ranking_size',
    'click_alarmclock',
    'click_bell',
    'dump_bin_bigbin',
    'grab_roller',
    'handover_block',
    'handover_mic',
    'hanging_mug',
    'lift_pot',
    'move_can_pot',
    'move_pillbottle_pad',
    'move_playingcard_away',
    'move_stapler_pad',
    'open_laptop',
    'open_microwave',
    'pick_diverse_bottles',
    'pick_dual_bottles',
    'place_a2b_left',
    'place_a2b_right',
    'place_bread_basket',
    'place_bread_skillet',
    'place_burger_fries',
    'place_can_basket',
    'place_cans_plasticbox',
    'place_container_plate',
    'place_dual_shoes',
    'place_empty_cup',
    'place_fan',
    'place_mouse_pad',
    'place_object_basket',
    'place_object_scale',
    'place_object_stand',
    'place_phone_stand',
    'place_shoe',
    'press_stapler',
    'put_bottles_dustbin',
    'put_object_cabinet',
    'rotate_qrcode',
    'scan_object',
    'shake_bottle',
    'shake_bottle_horizontally',
    'stack_blocks_three',
    'stack_blocks_two',
    'stack_bowls_three',
    'stack_bowls_two',
    'stamp_seal',
    'turn_switch',
]

_train_seed = 42
_flux2_pretrained = (os.environ.get('FLUX2_PRETRAINED_ROOT')
                or 'checkpoints/Pretrained_models/flux2')
_autoencoder    = os.path.join(_flux2_pretrained, 'FLUX.2-dev/ae.safetensors')
_transformer    = os.path.join(_flux2_pretrained, 'FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors')
_text_encoder   = os.path.join(_flux2_pretrained, 'FLUX.2-klein-base-4B/text_encoder')
_tokenizer      = os.path.join(_flux2_pretrained, 'FLUX.2-klein-base-4B/tokenizer')
_parent_data_root = (os.environ.get('PATCHWAM_DATA_ROOT')
                    or '/data/RoboTwin/RoboTwin-Clean')
_data_root_paths = [os.path.join(_parent_data_root, task) for task in _ROBOTWIN_TASK_LIST]

_checkpoint = (os.environ.get('PATCHWAM_PRETRAINED_CHECKPOINT')
               or 'work_dirs/patchwam_robotwin_released_conversion/'
                  'step_140000_native.safetensors')

_text_dim = 7680
_action_dim = 14
_state_dim = 14

_statistic_name = 'robotwin'
_video_keys = ['observation.images.cam_high',
               'observation.images.cam_left_wrist',
               'observation.images.cam_right_wrist']

# RobotWin source recipe: three cameras in the compact 288x256 mosaic, z-score
# normalization, a frozen text-only Qwen3 encoder, and the source
# ClipAugment recipe.
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
    chain = [
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
    return chain

def _dataset(training: bool):
    return dict(
        type='DistributedRepeatingDataset',
        reshuffle_each_epoch=True,
        seed=_train_seed,
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
            nonidle_filter_path=os.environ.get('PATCHWAM_NONIDLE_FILTER'),
            require_nonidle_filter=True,
            transforms=_transforms(training),
        ),
    )

model = dict(
    type='PatchWAMVLA',
    pretrained_name_or_path=_checkpoint,
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
            variant='klein-base-4b',
            dtype='bfloat16',
            device='cuda',
            gradient_checkpointing=True,
        ),
    ),
)

# Training and evaluation share the architecture and initial checkpoint.
inference_model = copy.deepcopy(model)
inference_model['vla_head']['denoiser']['gradient_checkpointing'] = False

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
    max_epochs=100,
    optimizer=dict(type='AdamW', lr=1.0e-4, weight_decay=0.01,
                   betas=(0.9, 0.95)),
    max_grad_norm=1.0,
    grad_accumulation_steps=2,
    # Source RobotWin effective batch: 4 x 32 x 2 = 256. The runner warns once
    # on rank zero and keeps running if the world size changes, then scales the
    # configured LRs by effective / expected.
    expected_global_batch_size=256,
    auto_scale_lr=True,
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
    seed=_train_seed,
)

# ---------------------------------------------------------------------------
# RoboTwin 2.0 evaluation of this recipe.
#
# Evaluation defaults to the official RoboTwin protocol: seed
# 0, all 50 tasks, `clean` + `random`, 100 trials per task. Training keeps its
# own `seed = 42` above.
#
# One task, one trial, single condition:
#
#   python scripts/eval.py \
#     --config configs/patchwam/robotwin/patchwam_robotwin_vlm_full_finetune.py \
#     --ckpt-path work_dirs/<run>/checkpoints/latest-checkpoint.pt \
#     --cfg-options eval.task_list=[adjust_bottle] \
#       eval.task_suite_name=clean eval.num_trials_per_task=1
#
# The full suite goes through the shared manager, which reads `eval.task_list`
# and `eval.task_suite_name`:
#
#   CONFIG=configs/patchwam/robotwin/patchwam_robotwin_vlm_full_finetune.py \
#   CKPT=work_dirs/<run>/checkpoints/latest-checkpoint.pt \
#     bash scripts/eval_robotwin_manager.sh
#
# Env overrides: PATCHWAM_ROBOTWIN_ROOT / ROBOTWIN_ROOT, PATCHWAM_ROBOTWIN_TASKS
# (whitespace-separated), PATCHWAM_TASK_SUITE (whitespace-separated conditions),
# PATCHWAM_EPISODES, PATCHWAM_DATASET_STATS. Manager names TASK_NAMES,
# TASK_SUITE_NAME and NUM_TRIALS_PER_TASK take priority over PATCHWAM aliases.
#
# `norm_stats_path` stays None, so the runner takes the statistics from the
# checkpoint itself (`<ckpt>/../../dataset_statistics.json`).
# ---------------------------------------------------------------------------

_eval_seed = 0
_eval_robotwin_root = 'src/RoboTwin'
_eval_task_list = (os.environ.get('TASK_NAMES')
                   or '').split() or list(_ROBOTWIN_TASK_LIST)
_eval_task_suite_name = (os.environ.get('TASK_SUITE_NAME')
                         or '').split() or ['clean', 'random']
_num_trials_per_task = int(os.environ.get('NUM_TRIALS_PER_TASK')
                           or os.environ.get('PATCHWAM_EPISODES') or '100')

_eval_transform = [
    dict(type='ProcessEvalInputs',
            img_keys=['cam_high', 'cam_left_wrist',
                    'cam_right_wrist']),
    # Dtype-based uint8 -> [0, 1] scaling before the mosaic pack.
    dict(type='PatchWAMEvalFrames'),
    # Unaugmented head snapshot required by image-conditioned encoders.
    dict(type='PatchWAMCleanVL', image_size=448, head_index=0),
    dict(type='PatchWAMEvalPack',
            layout='robotwin',
            video_size=[288, 256],
            camera_size=[240, 320],
            robotwin_layout='compact_288x256',
            head_index=0),
    # The exact training normalizer and epsilon, not an equivalent.
    dict(type='NormalizeStatesAndActions',
            state_key='proprio',
            action_key=None,
            norm_type='mean_std',
            state_dim=_state_dim,
            statistics_key='norm_stats',
            normalization_epsilon=1e-8,
            output_dtype='float32'),
]

eval = dict(
    type='RobotwinEvalRunner',
    seed=_eval_seed,
    model_family='patchwam_released',
    robotwin_root=_eval_robotwin_root,
    task_suite_name=_eval_task_suite_name ,
    task_list=_eval_task_list ,
    # The recipe predicts a 16-step horizon; execute all of it.
    eval_chunk_size=16,
    instruction_type='unseen',
    num_trials_per_task=_num_trials_per_task,
    unnorm_key=_statistic_name,
    norm_stats_path=os.environ.get('PATCHWAM_DATASET_STATS') or None,
    result_output_dir='runs/evaluation/patchwam_robotwin',
    dataset=dict(
        type='PatchWAMRoboTwinEvalDataset',
        statistic_name=_statistic_name,
        device='cuda',
        transforms=_eval_transform),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        action_dim=_action_dim,
    ),
)

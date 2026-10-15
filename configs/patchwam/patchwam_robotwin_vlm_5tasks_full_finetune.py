# PatchWAM RobotWin five-task ablation.
#
# Inherits the frozen, no-LoRA, current-image Qwen3-VL source recipe
# (`patchwam_robotwin_vlm_full_finetune.py`) and narrows both splits to five
# exact RoboTwin tasks. Model, conditioning, augmentation, non-idle row filter,
# horizon, dimensions, normalization method, optimizer, five epochs, LR and
# batch settings are unchanged. Task names are the identity, so the recipe is
# safe for a merged root whose local task indices differ per root.

import os

_base_ = './patchwam_robotwin_vlm_full_finetune.py'

# Exact RoboTwin tasks, in the order requested for the ablation.
_tasks = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
          'move_stapler_pad', 'hanging_mug']

# The subset aggregates its own statistics under a distinct namespace so the
# full-data `robotwin` statistics file is never overwritten.
_statistic_name = 'robotwin_5tasks'

# Honour PATCHWAM_DATA_ROOT when it is set; otherwise keep the recipe default.
_data_root_paths = [os.environ.get('PATCHWAM_DATA_ROOT')
                    or 'datasets/robotwin_lerobotv2.1']

# Independent train/validation selectors: both name the same five tasks, but
# they are separate objects so either split can be narrowed on its own.
_train_task_selection = dict(tasks=list(_tasks))
_val_task_selection = dict(tasks=list(_tasks))

train_dataloader = dict(
    dataset=dict(
        statistic_name=_statistic_name,
        datasets=dict(
            data_root_path=_data_root_paths,
            statistic_name=_statistic_name,
            task_selection=_train_task_selection,
        ),
    ),
)
val_dataloader = dict(
    dataset=dict(
        statistic_name=_statistic_name,
        datasets=dict(
            data_root_path=_data_root_paths,
            statistic_name=_statistic_name,
            task_selection=_val_task_selection,
        ),
    ),
)

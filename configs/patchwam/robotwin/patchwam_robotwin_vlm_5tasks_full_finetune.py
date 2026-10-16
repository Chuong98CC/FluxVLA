"""Five-task RoboTwin ablation; inherit all other settings from the full recipe.

Normalization statistics use the selected episodes, as in the native dataset.
"""
import os

_base_ = './patchwam_robotwin_vlm_full_finetune.py'

# The five tasks of the ablation, in the order requested. Each name is a task
# declared by the inherited root's `tasks.jsonl`.
_tasks = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
          'move_stapler_pad', 'hanging_mug']
_parent_data_root = (os.environ.get('PATCHWAM_DATA_ROOT')
                    or '/data/RoboTwin/RoboTwin-Clean')
_data_root_paths = [os.path.join(_parent_data_root, task) for task in _tasks]

def _task_selection():
    """A fresh five-task selection, so the two splits stay independent."""
    return dict(tasks=list(_tasks))


# Only the child dataset's task selection is overridden. The wrapper, the
# statistics keys, the transforms, the batch sizes and the runner come from the
# source recipe.
train_dataloader = dict(
    dataset=dict(datasets=dict(task_selection=_task_selection())),
)
# Independent selection objects, so either split can be scoped on its own.
val_dataloader = dict(
    dataset=dict(datasets=dict(task_selection=_task_selection())),
)

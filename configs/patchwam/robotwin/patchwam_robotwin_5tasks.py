"""Five-task RoboTwin ablation; inherit all other settings from the full recipe.

Normalization statistics use the selected episodes, as in the native dataset.
"""
import os

_base_ = './patchwam_robotwin.py'

# The five tasks of the ablation, in the order requested. Each name is a task
# declared by the inherited root's `tasks.jsonl`.
_tasks = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
          'move_stapler_pad', 'hanging_mug']
# _parent_data_root = (os.environ.get('PATCHWAM_DATA_ROOT')
#                     or '/data/RoboTwin/RoboTwin-Clean')
# _data_root_paths = [os.path.join(_parent_data_root, task) for task in _tasks]

def _task_selection():
    """A fresh five-task selection, so the two splits stay independent."""
    return dict(tasks=list(_tasks))

# The PEFT adapter this recipe trains with, stated explicitly so the recipe
# records the adaptation policy it was trained with.
_peft = dict(peft_type='LORA', r=32, lora_alpha=64, lora_dropout=0.0)

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

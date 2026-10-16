"""RobotWin 2.0 evaluation of an ordinary PatchWAM DiT adapter finetune.

Derivative of `patchwam_robotwin_eval.py` that adds the same `vla_head.peft`
block to `model` and `inference_model`, so the evaluator builds the PEFT
adapter tensors a checkpoint from
`patchwam_robotwin_vlm_lora_finetune.py` carries before the strict
`load_state_dict`. Every other observation, normalization, denormalization and
harness setting is inherited unchanged; only the default `norm_stats_path`
follows the adapter run's work directory.

`--ckpt-path` is still the CLI checkpoint for `scripts/eval.py`:

    CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
      --config configs/patchwam/patchwam_robotwin_vlm_lora_eval.py \
      --ckpt-path work_dirs/patchwam_robotwin_vlm_lora_finetune/checkpoints/latest-checkpoint.pt

Env overrides: PATCHWAM_DATASET_STATS plus every variable the base eval config
reads.
"""

import copy
import os

_base_ = './patchwam_robotwin_eval.py'

_peft = dict(peft_type='LORA', r=32, lora_alpha=64, lora_dropout=0.0)
_stats = os.environ.get(
    'PATCHWAM_DATASET_STATS',
    'work_dirs/patchwam_robotwin_vlm_lora_finetune/dataset_statistics.json')

model = dict(vla_head=dict(peft=dict(_peft)))
inference_model = copy.deepcopy(model)
eval = dict(norm_stats_path=_stats)

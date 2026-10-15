"""PatchWAM RobotWin: ordinary (unstaged) DiT LoRA finetuning.

Derivative of `patchwam_robotwin_vlm_full_finetune.py`. The model dimensions,
data pipeline, augmentation, normalization, three-camera mosaic, optimizer,
LR schedule, batch settings and the frozen Qwen3-VL instruction encoder are
inherited unchanged; only the adaptation policy changes, through
`vla_head.lora` with `stage` left unset:

  * the FLUX.2 DiT transformer blocks (`double_blocks` / `single_blocks`,
    their modulation and normalization parameters included) keep their base
    weights frozen and carry the canonical zero-initialized LoRA adapters
    (`rank=32`, `alpha=64`, `dropout=0`),
  * the denoiser's input/output/time/text projections, the shared stream
    modulations, the parameter-free action codec and the proprio/state and
    action maps stay fully trainable,
  * the autoencoder, the visual frontend and the instruction encoder stay
    frozen, including any VL language LoRA residuals
    (`vl_trainable_adapters=False` here).

`pretrained_name_or_path` warm-starts the head from the released RoboTwin 2.0
C2R PatchWAM policy (`step_140000.pt`). Only the new `dit_lora` adapter keys
may be absent from that payload; every inherited tensor - the frozen DiT
blocks included - must be present, or the load fails atomically.
`PATCHWAM_PRETRAINED_CHECKPOINT` overrides the path, and the inherited
`PATCHWAM_NONIDLE_FILTER` still gates the non-idle row filter.

`PatchWAMVLA.trainable_parameter_report()` prints the exact trainable names
and the frozen DiT block patterns for a built model.

Smoke (single node):
  python scripts/train.py \
    --config configs/patchwam/patchwam_robotwin_vlm_lora_finetune.py \
    --work-dir work_dirs/patchwam_robotwin_vlm_lora_finetune
"""

import copy
import os

_base_ = './patchwam_robotwin_vlm_full_finetune.py'

# Initial weights: the published RoboTwin 2.0 C2R policy, relative to the
# repository root like the five-task ablation. The frozen FLUX.2/Qwen assets
# and the instruction encoder still come from the inherited config.
_checkpoint = (os.environ.get('PATCHWAM_PRETRAINED_CHECKPOINT')
               or 'checkpoints/Pretrained_models/patchwam-robotwin/'
                  'step_140000.pt')

# Canonical `attach_dit_lora` defaults, stated explicitly so the recipe records
# the adaptation policy it was trained with.
_lora = dict(rank=32, alpha=64, dropout=0.0)

model = dict(
    pretrained_name_or_path=_checkpoint,
    vla_head=dict(lora=dict(_lora)),
)
inference_model = copy.deepcopy(model)

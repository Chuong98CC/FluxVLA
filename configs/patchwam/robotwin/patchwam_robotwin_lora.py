"""PatchWAM RobotWin: ordinary (unstaged) DiT adapter finetuning.

Derivative of `patchwam_robotwin_vlm_full_finetune.py`. The model dimensions,
data pipeline, augmentation, normalization, three-camera mosaic, optimizer,
LR schedule, batch settings and the frozen Qwen3-VL instruction encoder are
inherited unchanged; only the adaptation policy changes, through
`vla_head.peft` with `stage` left unset:

  * the FLUX.2 DiT transformer blocks (`double_blocks` / `single_blocks`,
    their modulation and normalization parameters included) keep their base
    weights frozen and carry zero-initialized PEFT LoRA adapters
    (`peft_type='LORA'`, `r=32`, `lora_alpha=64`, `lora_dropout=0`; PEFT keeps
    the wrapped weights in `base_layer` and zero-initializes the adapter),
  * the denoiser's own input/output/time/text projections (the FLUX.2
    `img_in`/`txt_in`/`time_in`/`final_layer` maps) and the shared stream
    modulations, plus the head's proprio/state projection and action maps,
    stay fully trainable; the fixed action codec has no trainable parameters
    at all,
  * the autoencoder, the visual frontend and the instruction encoder stay
    frozen, including any VL language LoRA residuals
    (`vl_trainable_adapters=False` here).

`vla_head.peft` is a PEFT configuration mapping, so another module-level tuner
can be selected without code changes - for example
`dict(peft_type='LOHA', r=8, alpha=16, target_modules=[...])`. Swapping the
tuner from a derived recipe must replace that mapping wholesale
(`peft=dict(_delete_=True, peft_type='LOHA', ...)`): config inheritance
deep-merges dictionaries, so a plain override keeps the inherited LoRA options
and PEFT rejects the mixed mapping. The default targets are the DiT block
attention and feed-forward projections, and any configured target is validated
to stay inside the block containers.

`pretrained_name_or_path` warm-starts the head from the released RoboTwin 2.0
C2R PatchWAM policy (`step_140000.pt`). Only the new PEFT adapter tensors may
be absent from that payload; every inherited tensor - the frozen DiT blocks
included - must be present, or the load fails atomically.
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

_base_ = './patchwam_robotwin.py'

# Initial weights: the published RoboTwin 2.0 C2R policy, relative to the
# repository root like the five-task ablation. The frozen FLUX.2/Qwen assets
# and the instruction encoder still come from the inherited config.
_checkpoint = (os.environ.get('PATCHWAM_PRETRAINED_CHECKPOINT')
               or 'checkpoints/Pretrained_models/patchwam-robotwin/'
                  'step_140000.pt')

# The PEFT adapter this recipe trains with, stated explicitly so the recipe
# records the adaptation policy it was trained with.
_peft = dict(peft_type='LORA', r=32, lora_alpha=64, lora_dropout=0.0)

model = dict(
    pretrained_name_or_path=_checkpoint,
    vla_head=dict(peft=dict(_peft)),
)
inference_model = copy.deepcopy(model)

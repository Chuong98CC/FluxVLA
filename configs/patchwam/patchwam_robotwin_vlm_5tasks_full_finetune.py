# PatchWAM RoboTwin-Clean five-task ablation.
#
# Inherits the frozen, no-LoRA, current-image Qwen3-VL source recipe
# (`patchwam_robotwin_vlm_full_finetune.py`) and narrows both splits to five
# released RoboTwin tasks. Model, conditioning, augmentation, horizon,
# dimensions, normalization method, optimizer, five epochs, LR and batch
# settings are unchanged; only the task roots, the asset paths, the initial
# weights, the statistics namespace and the (absent) non-idle row filter differ.
#
# RoboTwin-Clean publishes one LeRobot v2.1 root per task, named after the task,
# under a single parent directory. The explicit root list *is* the task filter,
# so no `task_selection` is used. A requested folder the release does not
# contain fails the run instead of silently training on a smaller subset.
#
# The run starts from the released RoboTwin 2.0 C2R policy (step 140000) rather
# than the raw FLUX.2 base: the released payload maps onto exactly the trainable
# set (the denoiser transformer and the proprio/state projection), so the
# ablation warm-starts from the published Action-as-Patch weights. The frozen
# Qwen3-VL instruction encoder is not in that payload and still comes from
# config, so its conditioning features are the recipe's, not the released
# checkpoint's.

import copy
import os

_base_ = './patchwam_robotwin_vlm_full_finetune.py'

# The five tasks of the ablation, in the order requested. Each name is also the
# on-disk folder name under `$PATCHWAM_DATA_ROOT`.
_tasks = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
          'move_stapler_pad', 'hanging_mug']
# PATCHWAM_DATA_ROOT names the parent directory holding one root per task.
_parent = (os.environ.get('PATCHWAM_DATA_ROOT')
           or '/data/RoboTwin/RoboTwin-Clean')
_data_root_paths = [os.path.join(_parent, task) for task in _tasks]

# The subset aggregates its own statistics under a distinct namespace so its
# keys do not collide with the full-data `robotwin` recipe. The file itself
# still lands in the run's --work-dir (scripts/train.py), so point the ablation
# at its own directory.
_statistic_name = 'robotwin_5tasks'

# This release ships no non-idle row filter, so rows stay unfiltered by default.
# An explicit PATCHWAM_NONIDLE_FILTER still enables the required-filter contract,
# which then fails rather than training on idle rows.
_nonidle_filter = os.environ.get('PATCHWAM_NONIDLE_FILTER') or None
_require_nonidle_filter = _nonidle_filter is not None

# Local asset defaults: the Fast-WAM/ImageWAM FLUX.2 checkout plus the Qwen3-VL
# instruction encoder this recipe uses. The same environment variables as the
# source recipe override each path. The native loader consumes the original
# black-forest-labs single-file/flux2-format weights, so the transformer points
# at the combined release file and the VAE at the flux2 `ae.safetensors` rather
# than the diffusers-format copies under `transformer/` and `vae/`.
_flux_assets = (os.environ.get('PATCHWAM_FLUX_ASSETS')
                or '/home/chuong/workspace/wam/ImageWAM/checkpoints/flux2/'
                   'FLUX.2-klein-base-4B')
_transformer = (os.environ.get('PATCHWAM_TRANSFORMER')
                or _flux_assets + '/flux-2-klein-base-4b.safetensors')
_autoencoder = (os.environ.get('PATCHWAM_AUTOENCODER')
                or '/home/chuong/workspace/wam/ImageWAM/checkpoints/flux2/'
                   'FLUX.2-dev/ae.safetensors')
_text_encoder = (os.environ.get('PATCHWAM_VLM')
                 or '/data/Pretrained_models/Qwen3-VL-4B-Instruct')
_flux_source = (os.environ.get('FLUX2_SRC')
                or '/home/chuong/workspace/wam/ImageWAM/third_party/flux2')

# Initial weights: the published RoboTwin 2.0 C2R policy. Relative to the
# repository root, matching the released-evaluation recipe; `PATCHWAM_PRETRAINED_CHECKPOINT`
# overrides it, and the frozen assets above still come from config.
_checkpoint = (os.environ.get('PATCHWAM_PRETRAINED_CHECKPOINT')
               or 'checkpoints/Pretrained_models/patchwam-robotwin/step_140000.pt')

# Repoint only the asset locations; architecture, dimensions, dtype and
# conditioning are inherited unchanged.
_assets = dict(
    pretrained_name_or_path=_checkpoint,
    vlm_backbone=dict(
        autoencoder_path=_autoencoder,
        text_encoder_path=_text_encoder,
        flux_source=_flux_source,
    ),
    vla_head=dict(
        denoiser=dict(transformer_path=_transformer, flux_source=_flux_source),
    ),
)
model = copy.deepcopy(_assets)
inference_model = copy.deepcopy(_assets)

train_dataloader = dict(
    dataset=dict(
        statistic_name=_statistic_name,
        datasets=dict(
            data_root_path=list(_data_root_paths),
            statistic_name=_statistic_name,
            nonidle_filter_path=_nonidle_filter,
            require_nonidle_filter=_require_nonidle_filter,
        ),
    ),
)
# Independent train/validation root lists: separate objects, so either split can
# be scoped on its own.
val_dataloader = dict(
    dataset=dict(
        statistic_name=_statistic_name,
        datasets=dict(
            data_root_path=list(_data_root_paths),
            statistic_name=_statistic_name,
            nonidle_filter_path=_nonidle_filter,
            require_nonidle_filter=_require_nonidle_filter,
        ),
    ),
)

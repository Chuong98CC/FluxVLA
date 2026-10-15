# PatchWAM RobotWin 2.0 five-task ablation.
#
# Inherits the frozen, no-LoRA, current-image Qwen3-VL source recipe
# (`patchwam_robotwin_vlm_full_finetune.py`) and narrows both splits to five
# released RoboTwin tasks. Model, conditioning, augmentation, non-idle row
# filter, horizon, dimensions, normalization method, optimizer, five epochs, LR
# and batch settings are unchanged; only the task subset, the default dataset
# root, the asset paths and the statistics namespace differ.
#
# The RoboTwin 2.0 release stores instruction paraphrases rather than benchmark
# names, so one episode declares dozens of instruction variants and exact-name
# identity cannot name a benchmark task. `episode_ranges` defines benchmark
# identity by root-local half-open episode ranges instead (50 released tasks x
# 550 episodes = 27500), and instruction text is never interpreted. The
# exact-name `tasks` mode stays available for other datasets.

import copy
import os

_base_ = './patchwam_robotwin_vlm_full_finetune.py'

# The five released tasks, in the order requested for the ablation, with the
# half-open episode ranges that identify them in the merged release root. Every
# range covers all 550 episodes of that task (both rendered conditions); no
# episode cap or sampling is applied.
_tasks = ['turn_switch', 'rotate_qrcode', 'blocks_ranking_size',
          'move_stapler_pad', 'hanging_mug']
_episode_ranges = {
    'turn_switch': [[26950, 27500]],
    'rotate_qrcode': [[22000, 22550]],
    'blocks_ranking_size': [[1650, 2200]],
    'move_stapler_pad': [[8250, 8800]],
    'hanging_mug': [[5500, 6050]],
}
_expected_episodes = 27500

# The subset aggregates its own statistics under a distinct namespace so its
# keys do not collide with the full-data `robotwin` recipe. The file itself
# still lands in the run's --work-dir (scripts/train.py), so point the ablation
# at its own directory.
_statistic_name = 'robotwin_5tasks'

# Honour PATCHWAM_DATA_ROOT when it is set; otherwise use the supplied release.
_data_root_paths = [os.environ.get('PATCHWAM_DATA_ROOT')
                    or '/data/robotwin2.0']

# The release ships its own non-idle row filter. An explicit
# PATCHWAM_NONIDLE_FILTER still wins, and `require_nonidle_filter=True` is
# inherited so a missing filter fails instead of training on idle rows.
_nonidle_filter = (os.environ.get('PATCHWAM_NONIDLE_FILTER')
                   or os.path.join(_data_root_paths[0], 'nonidle_ranges.json'))

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

# Independent train/validation selectors: both name the same five tasks, but
# they are separate objects so either split can be narrowed on its own.
_train_task_selection = dict(tasks=list(_tasks),
                             episode_ranges=copy.deepcopy(_episode_ranges),
                             expected_episodes=_expected_episodes)
_val_task_selection = dict(tasks=list(_tasks),
                           episode_ranges=copy.deepcopy(_episode_ranges),
                           expected_episodes=_expected_episodes)

# Repoint only the asset locations; architecture, dimensions, dtype and
# conditioning are inherited unchanged.
_assets = dict(
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
            data_root_path=_data_root_paths,
            statistic_name=_statistic_name,
            nonidle_filter_path=_nonidle_filter,
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
            nonidle_filter_path=_nonidle_filter,
            task_selection=_val_task_selection,
        ),
    ),
)

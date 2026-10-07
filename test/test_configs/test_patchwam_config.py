# Copyright 2026 Limx Dynamics
"""CPU units validating the native PatchWAM config placement and recipes."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

import fluxvla.datasets  # noqa: F401
import fluxvla.engines.runners  # noqa: F401  (runner registration)
import fluxvla.transforms  # noqa: F401
from fluxvla.collators.dict_collator import DictCollator
from fluxvla.engines import (DATASETS, HEADS, RUNNERS, TRANSFORMS, VLAS,
                             VLM_BACKBONES, build_collator_from_cfg,
                             build_vla_from_cfg)
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.vlas import patchwam_vla  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'

CONFIGS = {
    'patchwam_libero_full_finetune.py': dict(
        action_dim=7, state_dim=8, norm='min_max', layout='horizontal',
        video_size=[224, 448], camera_size=[224, 224], separate=False,
        variant='klein-base-4b',
        slots=0, subtasks=False, adapters=False),
    'patchwam_robotwin_vlm_full_finetune.py': dict(
        action_dim=14, state_dim=14, norm='mean_std', layout='robotwin',
        video_size=[288, 256], camera_size=[240, 320], separate=False,
        variant='klein-base-4b', slots=0, subtasks=False, adapters=False),
    'patchwam_robodojo_vlm_history_example.py': dict(
        action_dim=14, state_dim=14, norm='mean_std', layout='robotwin',
        video_size=[288, 256], camera_size=[256, 256], separate=True,
        variant='klein-base-4b',
        slots=20, subtasks=True, adapters=True),
}


def _load(name):
    return Config.fromfile(str(CONFIG_DIR / name))


@pytest.mark.parametrize('name,recipe', CONFIGS.items())
def test_runner_collator_statistics_and_recipe(name, recipe):
    cfg = _load(name)
    model = cfg.model
    assert VLAS.get(model.type) is not None
    assert VLM_BACKBONES.get(model.vlm_backbone.type) is not None
    assert HEADS.get(model.vla_head.type) is not None
    assert HEADS.get(model.vla_head.denoiser.type) is not None
    # Klein base 4B consumes a 3x Qwen3 hidden-state readout.
    assert model.vlm_backbone.text_dim == 7680
    assert model.vla_head.text_dim == model.vlm_backbone.text_dim
    assert model.vla_head.action_dim == recipe['action_dim']
    assert model.vla_head.proprio_dim == recipe['state_dim']
    assert model.vla_head.isolate_actions is False
    assert model.vla_head.denoiser.variant == recipe['variant']
    assert model.vlm_backbone.get('vl_trainable_adapters',
                                  False) is recipe['adapters']
    if recipe['subtasks']:
        assert model.vlm_backbone.get('vl_require_subtasks', False) is True

    # The runner owns the collator/optimizer/scheduler/metric.
    runner = cfg.runner
    assert RUNNERS.get(runner.type) is not None
    assert 'collator' in runner and 'optimizer' in runner
    assert 'metric' in runner and 'lr_scheduler' in runner
    assert 'collator' not in cfg.train_dataloader
    collator = build_collator_from_cfg(dict(runner.collator))
    assert isinstance(collator, DictCollator)

    dataset = cfg.train_dataloader.dataset
    child = dataset.datasets
    assert DATASETS.get(child.type) is not None
    assert dataset.statistic_name == child.statistic_name
    assert child.action_key == 'action'
    assert child.future_offset == 16 and child.horizon == 16
    assert child.action_window_size == 16 and child.window_start_idx == 0
    if recipe['slots']:
        assert child.history['enabled'] is True
        assert child.history['slots'] == recipe['slots']
    else:
        assert child.history.get('enabled') is False

    types = [t['type'] for t in child.transforms]
    assert types[0] == 'PatchWAMRawInputs'
    assert 'PatchWAMDecodeFrames' in types
    assert 'NormalizeStatesAndActions' in types
    norm = next(t for t in child.transforms
                if t['type'] == 'NormalizeStatesAndActions')
    assert norm['norm_type'] == recipe['norm']
    # Stats are aggregated under the native ``proprio`` name.
    assert norm['state_key'] == 'proprio'
    assert norm['action_dim'] == recipe['action_dim']
    assert norm['state_dim'] == recipe['state_dim']
    compose = next(t for t in child.transforms
                   if t['type'] == 'PatchWAMComposeWorld')
    assert compose['layout'] == recipe['layout']
    assert list(compose['video_size']) == recipe['video_size']
    assert compose['separate_views'] is recipe['separate']
    resize = next(t for t in child.transforms
                  if t['type'] == 'PatchWAMResizeCameras')
    assert list(resize['camera_size']) == recipe['camera_size']
    for transform in child.transforms:
        assert TRANSFORMS.get(transform['type']) is not None


def test_train_and_validation_configs_are_independent():
    cfg = _load('patchwam_robotwin_vlm_full_finetune.py')
    train_transforms = cfg.train_dataloader.dataset.datasets.transforms
    val_transforms = cfg.val_dataloader.dataset.datasets.transforms
    assert cfg.train_dataloader is not cfg.val_dataloader
    train_augment = next(t for t in train_transforms
                         if t['type'] == 'PatchWAMWorldAugment')
    val_augment = next(t for t in val_transforms
                       if t['type'] == 'PatchWAMWorldAugment')
    assert train_augment['enabled'] is True
    assert val_augment['enabled'] is False
    # Mutating the training config must not mutate validation.
    original = val_augment['enabled']
    train_augment['enabled'] = False
    assert val_augment['enabled'] is original


def test_libero_eval_config_is_present():
    cfg = _load('patchwam_libero_full_finetune.py')
    assert 'eval' in cfg
    assert cfg.eval['type'] == 'LiberoEvalRunner'
    assert 'inference' in cfg
    assert cfg.inference['type'] == 'LiberoEvalRunner'
    # The runner carries the norm stats key (it overwrites the dataset field).
    assert cfg.inference['norm_stats_key'] == 'libero_all_no_noops'
    assert cfg.inference['dataset']['type'] == 'PatchWAMLiberoEvalDataset'
    assert cfg.eval['denormalize_action']['type'] == 'DenormalizeLiberoAction'
    transforms = [t['type'] for t in cfg.eval['dataset']['transforms']]
    assert 'PatchWAMEvalPack' in transforms


def test_shipped_datasets_pass_their_training_flag():
    for name, recipe in CONFIGS.items():
        cfg = _load(name)
        assert cfg.train_dataloader.dataset.datasets['training'] is True
        assert cfg.val_dataloader.dataset.datasets['training'] is False


class TinyAutoencoder(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(3, channels, 1)

    def encode(self, video):
        return self.proj(video)


class TinyTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self, dim: int = 32, length: int = 3):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(length, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


def test_config_model_builds_with_injected_modules():
    cfg = _load('patchwam_libero_full_finetune.py')
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone',
        autoencoder=TinyAutoencoder(channels=128),
        text_encoder=TinyTextEncoder(dim=32),
        token_dim=128, text_dim=32, history_pool_size=2)
    model_cfg['vla_head'] = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        action_dim=7, text_dim=32, proprio_dim=8, token_dim=128)
    model_cfg['num_inference_steps'] = 2
    vla = build_vla_from_cfg(model_cfg)
    out = vla(images=torch.rand(2, 3, 2, 32, 64), states=torch.randn(2, 8),
              actions=torch.randn(2, 16, 7),
              task_description=['a', 'b'])
    assert torch.isfinite(out['loss'])

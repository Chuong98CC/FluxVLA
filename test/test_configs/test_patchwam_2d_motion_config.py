# Copyright 2026 Limx Dynamics
"""CPU units validating the shipped PatchWAM 2D patch-motion recipe."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

import fluxvla.datasets  # noqa: F401
import fluxvla.engines.runners  # noqa: F401  (runner registration)
from fluxvla.collators.dict_collator import DictCollator
from fluxvla.engines import (DATASETS, HEADS, RUNNERS, VLAS, VLM_BACKBONES,
                             build_collator_from_cfg, build_vla_from_cfg)
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_2d_motion_head  # noqa: F401
from fluxvla.models.vlas import patchwam_2d_motion_vla  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'
CONFIG_NAME = 'patchwam_2d_motion.py'


def _load():
    return Config.fromfile(str(CONFIG_DIR / CONFIG_NAME))


class TinyAutoencoder(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.pool = nn.AvgPool2d(16)
        self.proj = nn.Conv2d(3, channels, 1)

    def encode(self, video):
        return self.proj(self.pool(video))


class TinyTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self, dim: int = 32, length: int = 3):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(length, dim))
        self.feature_dim = dim

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


def test_recipe_registers_native_2d_motion_components():
    cfg = _load()
    model = cfg.model
    assert model.type == 'PatchWAM2DMotionVLA'
    assert VLAS.get(model.type) is not None
    assert model.vlm_backbone.type == 'PatchWAMBackbone'
    assert VLM_BACKBONES.get(model.vlm_backbone.type) is not None
    assert model.vla_head.type == 'PatchWAM2DMotionHead'
    assert HEADS.get(model.vla_head.type) is not None
    assert model.vla_head.denoiser.type == 'PatchWAMFluxDenoiser'
    assert HEADS.get(model.vla_head.denoiser.type) is not None
    assert model.motion_horizon == 3
    assert model.vla_head.motion_horizon == 3
    assert model.vlm_backbone.text_dim == model.vla_head.text_dim
    assert model.vla_head.token_dim == 128
    assert model.vla_head.patch_size == 16
    assert model.vla_head.block_size == 8
    assert model.vla_head.image_height == model.vla_head.image_width == 256
    assert model.vla_head.video_weight == 0.5
    assert model.vla_head.motion_weight == 1.0
    # This variant is state-free and action-free by construction.
    assert 'proprio_dim' not in model.vla_head
    assert 'action_dim' not in model.vla_head
    assert 'action_horizon' not in model
    assert 'action_norm_type' not in model
    runner = cfg.runner
    assert RUNNERS.get(runner.type) is not None
    collator = build_collator_from_cfg(dict(runner.collator))
    assert isinstance(collator, DictCollator)
    assert set(collator.keys) == {'images', 'motion_2d', 'motion_valid'}
    assert set(collator.meta_keys) == {'task_description', 'sample_id',
                                       'episode_id'}
    for loader in (cfg.train_dataloader, cfg.val_dataloader):
        dataset = loader.dataset
        assert dataset.type == 'PatchWAM2DTraceDataset'
        assert DATASETS.get(dataset.type) is not None
        assert dataset.coordinate_order == 'row_column'
        assert dataset.patch_size == 16
        assert dataset.motion_block_size == 8
        assert dataset.horizon == 3
        assert list(dataset.timestep_offsets_s) == [0.25, 0.5, 1.0]
        # No robot state/action normalization transform anywhere.
        assert 'transforms' not in dataset
    assert cfg.train_dataloader.dataset.split == 'train'
    assert cfg.val_dataloader.dataset.split == 'val'


def test_config_builds_and_trains_with_injected_modules():
    cfg = _load()
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone', autoencoder=TinyAutoencoder(),
        text_encoder=TinyTextEncoder(), token_dim=128, text_dim=32,
        history_pool_size=2)
    model_cfg['vla_head'] = dict(
        type='PatchWAM2DMotionHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        text_dim=32, token_dim=128, image_height=64, image_width=64,
        motion_horizon=3)
    model_cfg['num_inference_steps'] = 2
    vla = build_vla_from_cfg(model_cfg)
    motion = torch.randn(2, 3, 4, 4, 2) * 20
    out = vla(images=torch.rand(2, 1, 3, 2, 64, 64) * 2 - 1,
              motion_2d=motion,
              motion_valid=torch.ones(2, 3, 4, 4, dtype=torch.bool),
              task_description=['a', 'b'])
    assert torch.isfinite(out['loss'])
    assert set(('loss_video', 'loss_motion', 'loss_action')).issubset(out)
    predicted = vla.predict_motion(images=torch.rand(2, 1, 3, 2, 64, 64) * 2 - 1,
                                   task_description=['a', 'b'], seed=3,
                                   num_inference_steps=2)
    assert predicted['motion_2d'].shape == (2, 3, 4, 4, 2)
    assert predicted['traces_2d'].shape == (2, 4, 4, 4, 2)
    assert predicted['future_tokens'].shape == (2, 16, 128)


def test_robot_action_chunks_and_state_are_not_configured():
    cfg = _load()
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMBackbone', autoencoder=TinyAutoencoder(),
        text_encoder=TinyTextEncoder(), token_dim=128, text_dim=32,
        history_pool_size=2)
    model_cfg['vla_head'] = dict(
        type='PatchWAM2DMotionHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128, text_dim=32,
                      width=32, heads=4, depth=2),
        text_dim=32, token_dim=128, image_height=64, image_width=64)
    sample = dict(images=torch.rand(2, 1, 3, 2, 64, 64) * 2 - 1,
                  motion_2d=torch.randn(2, 3, 4, 4, 2),
                  task_description=['a', 'b'])
    with pytest.raises(ValueError, match='action chunk|robot action'):
        build_vla_from_cfg(dict(model_cfg, action_horizon=16))
    vla = build_vla_from_cfg(model_cfg)
    with pytest.raises(ValueError, match='state|proprio'):
        vla(**sample, states=torch.randn(2, 14))
    with pytest.raises(ValueError, match='state|proprio'):
        build_vla_from_cfg(dict(
            model_cfg, vla_head=dict(model_cfg['vla_head'], proprio_dim=14)))
    with pytest.raises(ValueError, match='action chunk|robot action'):
        vla(**sample, actions=torch.randn(2, 4, 7))

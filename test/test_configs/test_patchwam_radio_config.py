# Copyright 2026 Limx Dynamics
"""CPU units validating the shipped PatchWAM C-RADIO auxiliary recipe."""

import copy
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config

import fluxvla.datasets  # noqa: F401
import fluxvla.engines.runners  # noqa: F401  (runner registration)
import fluxvla.transforms  # noqa: F401
from fluxvla.collators.dict_collator import DictCollator
from fluxvla.engines import (DATASETS, HEADS, RUNNERS, TRANSFORMS, VLAS,
                             VLM_BACKBONES, build_collator_from_cfg,
                             build_vla_from_cfg)
from fluxvla.models.backbones.vlms import patchwam_radio_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_radio_head  # noqa: F401
from fluxvla.models.vlas import patchwam_radio_vla  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / 'configs' / 'patchwam'
CONFIG_NAME = 'patchwam_radio_aux_robotwin.py'
RADIO_DIM = 6


def _load():
    return Config.fromfile(str(CONFIG_DIR / CONFIG_NAME))


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


class TinyRadioTeacher(nn.Module):
    def __init__(self, feature_dim: int = RADIO_DIM, patch_size: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.kernel = nn.Parameter(
            torch.randn(feature_dim, 3, patch_size, patch_size) * 0.1)

    def encode(self, images):
        spatial = F.conv2d(images, self.kernel,
                           stride=self.patch_size).flatten(2).transpose(1, 2)
        return spatial.mean(1), spatial


def test_radio_recipe_registers_and_keeps_the_native_dataset():
    cfg = _load()
    model = cfg.model
    assert model.type == 'PatchWAMRadioAuxVLA'
    assert VLAS.get(model.type) is not None
    assert model.vlm_backbone.type == 'PatchWAMRadioAuxBackbone'
    assert VLM_BACKBONES.get(model.vlm_backbone.type) is not None
    assert model.vla_head.type == 'PatchWAMRadioAuxHead'
    assert HEADS.get(model.vla_head.type) is not None
    assert model.vla_head.denoiser.type == 'PatchWAMFluxDenoiser'
    assert HEADS.get(model.vla_head.denoiser.type) is not None
    # RobotWin recipe: three cameras, 14-D actions, mean_std normalization.
    assert model.vla_head.action_dim == 14
    assert model.vla_head.proprio_dim == 14
    assert model.vlm_backbone.text_dim == model.vla_head.text_dim == 7680
    assert model.vla_head.isolate_actions is False
    assert model.vla_head.radio_weight == 0.5
    assert model.vla_head.video_weight == 0.5
    assert model.vla_head.action_weight == 1.0
    assert model.vla_head.radio_dim == model.vlm_backbone.radio_dim == 1280
    # Explicitly configurable zero-based adapter block (-1 keeps the last one).
    assert model.vla_head.adapter_layer_idex == -1
    # Native resume is strict: no missing-projection opt-out exists anywhere.
    assert 'allow_missing_radio_projections' not in model
    assert 'allow_missing_radio_projections' not in inspect.signature(
        VLAS.get(model.type).__init__).parameters
    assert model.vlm_backbone.radio_model_path.endswith('C-RADIOv4-H')
    runner = cfg.runner
    assert RUNNERS.get(runner.type) is not None
    collator = build_collator_from_cfg(dict(runner.collator))
    assert isinstance(collator, DictCollator)
    dataset = cfg.train_dataloader.dataset
    child = dataset.datasets
    assert DATASETS.get(child.type) is not None
    assert child.action_window_size == 16 and child.future_offset == 16
    for transform in child.transforms:
        assert TRANSFORMS.get(transform['type']) is not None
    norm = next(t for t in child.transforms
                if t['type'] == 'NormalizeStatesAndActions')
    assert norm['norm_type'] == 'mean_std' and norm['action_dim'] == 14


def test_inference_model_is_the_teacher_free_original_model():
    cfg = _load()
    inference = cfg.inference_model
    assert inference['type'] == 'PatchWAMVLA'
    assert inference['vlm_backbone']['type'] == 'PatchWAMBackbone'
    assert inference['vla_head']['type'] == 'PatchWAMHead'
    assert inference['vlm_backbone'].get('radio_model_path') is None
    assert 'radio_teacher' not in inference['vlm_backbone']
    assert 'radio_dim' not in inference['vla_head']
    assert 'radio_weight' not in inference['vla_head']
    assert 'adapter_layer_idex' not in inference['vla_head']
    assert 'allow_missing_radio_projections' not in inference
    assert inference['vla_head']['denoiser'].get('future_dim') is None
    # Mutating the training copy must not change the inference copy.
    inference_path = inference['vlm_backbone'].get('radio_model_path')
    cfg.model['vlm_backbone']['radio_model_path'] = '/tmp/mutated'
    assert inference['vlm_backbone'].get('radio_model_path') == inference_path


def test_removed_version_one_recipe_is_gone():
    assert not (CONFIG_DIR / 'patchwam_radio_robotwin.py').exists()


def test_radio_config_builds_with_injected_modules():
    cfg = _load()
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMRadioAuxBackbone',
        autoencoder=TinyAutoencoder(),
        text_encoder=TinyTextEncoder(),
        token_dim=128,
        text_dim=32,
        history_pool_size=2,
        radio_teacher=TinyRadioTeacher(),
        radio_dim=RADIO_DIM,
        radio_patch_size=2,
        radio_min_resolution=2,
    )
    model_cfg['vla_head'] = dict(
        type='PatchWAMRadioAuxHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                      text_dim=32, width=32, heads=4, depth=2),
        action_dim=7, text_dim=32, proprio_dim=8, token_dim=128,
        radio_dim=RADIO_DIM, radio_weight=0.5)
    model_cfg['action_horizon'] = 4
    model_cfg['num_inference_steps'] = 2
    vla = build_vla_from_cfg(model_cfg)
    video = torch.rand(2, 2, 3, 2, 4, 6) * 2 - 1
    out = vla(images=video, states=torch.randn(2, 8),
              actions=torch.randn(2, 4, 7),
              action_masks=torch.ones(2, 4, dtype=torch.bool),
              task_description=['a', 'b'])
    assert torch.isfinite(out['loss'])
    assert set(('loss_video', 'loss_radio', 'loss_action')).issubset(out)
    prediction = vla.predict_action(
        images=video, states=torch.randn(2, 8),
        task_description=['a', 'b'], seed=3)
    assert prediction.shape == (2, 4, 7)


def test_adapter_layer_idex_is_plumbed_and_validated_from_config():
    cfg = _load()
    model_cfg = copy.deepcopy(Config(cfg.model).to_dict())
    model_cfg['vlm_backbone'] = dict(
        type='PatchWAMRadioAuxBackbone',
        autoencoder=TinyAutoencoder(),
        text_encoder=TinyTextEncoder(),
        token_dim=128, text_dim=32, history_pool_size=2,
        radio_teacher=TinyRadioTeacher(), radio_dim=RADIO_DIM,
        radio_patch_size=2, radio_min_resolution=2)
    head_cfg = dict(
        type='PatchWAMRadioAuxHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                      text_dim=32, width=32, heads=4, depth=2),
        action_dim=7, text_dim=32, proprio_dim=8, token_dim=128,
        radio_dim=RADIO_DIM, radio_weight=0.5)
    model_cfg['action_horizon'] = 4
    model_cfg['num_inference_steps'] = 2
    sample = dict(images=torch.rand(2, 2, 3, 2, 4, 6) * 2 - 1,
                  states=torch.randn(2, 8), actions=torch.randn(2, 4, 7),
                  action_masks=torch.ones(2, 4, dtype=torch.bool),
                  task_description=['a', 'b'])
    early = build_vla_from_cfg(
        dict(model_cfg, vla_head=dict(head_cfg, adapter_layer_idex=0)))
    assert early.vla_head.adapter_layer_idex == 0
    assert early.vla_head.capture_layer == 1
    default = build_vla_from_cfg(dict(model_cfg, vla_head=head_cfg))
    assert default.vla_head.adapter_layer_idex == -1
    assert default.vla_head.capture_layer == 2
    default.vla_head.load_state_dict(early.vla_head.state_dict())
    generator = torch.Generator().manual_seed(41)
    flow = dict(sigma=torch.tensor([0.3, 0.7]),
                future_noise=torch.randn(2, 48, 128, generator=generator),
                action_noise=torch.randn(2, 4, 128, generator=generator))
    early_out = early(**copy.deepcopy(sample), **copy.deepcopy(flow))
    default_out = default(**copy.deepcopy(sample), **copy.deepcopy(flow))
    # Only the auxiliary tap changes; the original flow objectives are equal.
    for key in ('loss_video', 'loss_action'):
        torch.testing.assert_close(early_out[key], default_out[key], rtol=0,
                                   atol=0)
    assert torch.isfinite(early_out['loss_radio'])
    with pytest.raises(ValueError, match='adapter_layer_idex'):
        build_vla_from_cfg(
            dict(model_cfg, vla_head=dict(head_cfg, adapter_layer_idex=2)))
    with pytest.raises(ValueError, match='adapter_layer_idex'):
        build_vla_from_cfg(
            dict(model_cfg, vla_head=dict(head_cfg, adapter_layer_idex=True)))

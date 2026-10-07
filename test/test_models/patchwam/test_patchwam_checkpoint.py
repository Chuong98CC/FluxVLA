# Copyright 2026 Limx Dynamics
"""PatchWAM checkpoint initialization regression tests (bare/compact/native)."""

import torch
import torch.nn as nn
import pytest
from safetensors.torch import save_file

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.vlas import patchwam_vla  # noqa: F401


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


class TinyLoRAEncoder(nn.Module):
    requires_images = False
    trainable_adapters = True

    def __init__(self, dim: int = 32):
        super().__init__()
        self.base = nn.Linear(dim, dim)
        self.residual_in = nn.Linear(dim, 4, bias=False)
        self.residual_out = nn.Linear(4, dim, bias=False)
        self.feature_dim = dim

    def forward(self, prompts):
        return self.base.weight[:1].expand(len(prompts), 3, -1)


def _model(text_encoder=None, **overrides):
    cfg = dict(
        type='PatchWAMVLA',
        action_horizon=16,
        num_inference_steps=2,
        vlm_backbone=dict(
            type='PatchWAMBackbone',
            autoencoder=TinyAutoencoder(),
            text_encoder=text_encoder or TinyTextEncoder(),
            token_dim=128, text_dim=32, history_pool_size=2),
        vla_head=dict(
            type='PatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=7, text_dim=32, proprio_dim=8, token_dim=128))
    cfg.update(overrides)
    return build_vla_from_cfg(cfg)


def test_native_full_bare_and_compact_loads(tmp_path):
    source = _model(text_encoder=TinyLoRAEncoder())
    source.vla_head.state_projection.weight.data.fill_(0.25)
    state = source.state_dict()

    native_path = tmp_path / 'native.pth'
    torch.save({'model': state}, native_path)
    native = _model(text_encoder=TinyLoRAEncoder(), strict_mapping=True,
                    pretrained_name_or_path=str(native_path))
    native.from_pretrained()
    torch.testing.assert_close(native.vla_head.state_projection.weight,
                               source.vla_head.state_projection.weight)

    head = {key: value for key, value in state.items()
            if key.startswith('vla_head.')}
    adapters = {key: value for key, value in state.items()
                if key.startswith('vlm_backbone.text_encoder.')
                and ('residual_in' in key or 'residual_out' in key)}
    bare_path = tmp_path / 'bare.safetensors'
    save_file({key[len('vla_head.'):]: value for key, value in head.items()},
              str(bare_path))
    bare = _model()
    bare.pretrained_name_or_path = str(bare_path)
    bare.from_pretrained()
    torch.testing.assert_close(bare.vla_head.state_projection.weight,
                               source.vla_head.state_projection.weight)

    compact_path = tmp_path / 'compact.safetensors'
    compact_state = {'policy.' + key[len('vla_head.'):]: value
                     for key, value in head.items()}
    compact_state.update(
        {'text_encoder.' + key[len('vlm_backbone.text_encoder.'):]: value
         for key, value in adapters.items()})
    save_file(compact_state, str(compact_path))
    compact = _model(text_encoder=TinyLoRAEncoder())
    compact.pretrained_name_or_path = str(compact_path)
    compact.from_pretrained()
    torch.testing.assert_close(compact.vla_head.state_projection.weight,
                               source.vla_head.state_projection.weight)


def test_checkpoint_validation_rejects_before_mutation(tmp_path):
    source = _model(text_encoder=TinyLoRAEncoder())
    state = source.state_dict()
    head = {key[len('vla_head.'):]: value for key, value in state.items()
            if key.startswith('vla_head.')}

    missing = dict(head)
    missing.pop('denoiser.token_output.weight')
    missing_path = tmp_path / 'missing.safetensors'
    save_file(missing, str(missing_path))
    model = _model()
    before = model.vla_head.state_projection.weight.detach().clone()
    model.pretrained_name_or_path = str(missing_path)
    with pytest.raises(ValueError, match='missing='):
        model.from_pretrained()
    torch.testing.assert_close(model.vla_head.state_projection.weight, before)

    shaped = dict(head)
    shaped['state_projection.weight'] = torch.zeros(3, 3)
    shape_path = tmp_path / 'shape.safetensors'
    save_file(shaped, str(shape_path))
    model = _model()
    model.pretrained_name_or_path = str(shape_path)
    with pytest.raises(ValueError, match='shape differs'):
        model.from_pretrained()

    unexpected = dict(head)
    unexpected['not_a_real_key'] = torch.zeros(1)
    unexpected_path = tmp_path / 'unexpected.safetensors'
    save_file(unexpected, str(unexpected_path))
    model = _model()
    model.pretrained_name_or_path = str(unexpected_path)
    with pytest.raises(ValueError, match='unexpected='):
        model.from_pretrained()

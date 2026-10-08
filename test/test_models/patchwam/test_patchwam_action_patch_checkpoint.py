# Copyright 2026 Limx Dynamics
"""Released Action-as-Patch checkpoint mapping and loading tests."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401,F403
from fluxvla.models.heads import patchwam_head  # noqa: F401,F403
from fluxvla.models.vlas import patchwam_vla  # noqa: F401,F403
from fluxvla.models.vlas.patchwam_vla import released_action_patch_state

RELEASED_CHECKPOINT = Path(
    '/data/Pretrained_models/patchwam-robotwin/step_140000.pt')


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


class TinyTransformer(nn.Module):
    """Minimal ``denoiser.transformer`` stand-in with FLUX.2 input names."""

    def __init__(self, token_dim: int = 128, text_dim: int = 32):
        super().__init__()
        self.img_in = nn.Linear(token_dim, 8, bias=False)
        self.txt_in = nn.Linear(text_dim, 8, bias=False)


class TinyDenoiser(nn.Module):
    def __init__(self, token_dim: int = 128, text_dim: int = 32):
        super().__init__()
        self.transformer = TinyTransformer(token_dim, text_dim)


def _model(proprio_dim: int = 14):
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        action_horizon=16,
        num_inference_steps=2,
        vlm_backbone=dict(
            type='PatchWAMBackbone',
            autoencoder=TinyAutoencoder(),
            text_encoder=TinyTextEncoder(),
            token_dim=128,
            text_dim=32,
            history_pool_size=2),
        vla_head=dict(
            type='PatchWAMHead',
            denoiser=TinyDenoiser(),
            action_dim=14,
            text_dim=32,
            proprio_dim=proprio_dim,
            token_dim=128)))


def _released_payload(model):
    """Build a released-style payload from a built tiny model."""
    state = model.state_dict()
    mot = {}
    for key, value in state.items():
        prefix = 'vla_head.denoiser.transformer.'
        if not key.startswith(prefix):
            continue
        body = key[len(prefix):]
        mot['mixtures.video.transformer.' + body] = value
        if body.startswith(('double_blocks.', 'single_blocks.')):
            mot['mixtures.video.' + body] = value
    proprio = {
        key[len('vla_head.state_projection.'):]: value
        for key, value in state.items()
        if key.startswith('vla_head.state_projection.')
    }
    return {'mot': mot, 'proprio_encoder': proprio, 'step': 140000,
            'torch_dtype': 'torch.bfloat16'}


def test_released_payload_maps_to_native_keys():
    transformer = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    payload = {
        'mot': {
            'mixtures.video.transformer.img_in.weight': transformer,
            'mixtures.video.transformer.double_blocks.0.img_mlp.0.weight':
                transformer,
            'mixtures.video.double_blocks.0.img_mlp.0.weight': transformer,
        },
        'proprio_encoder': {'weight': transformer, 'bias': torch.ones(3)},
        'step': 140000,
        'torch_dtype': 'torch.bfloat16',
    }
    resolved = released_action_patch_state(payload)
    assert set(resolved) == {
        'vla_head.denoiser.transformer.img_in.weight',
        'vla_head.denoiser.transformer.double_blocks.0.img_mlp.0.weight',
        'vla_head.state_projection.weight',
        'vla_head.state_projection.bias',
    }
    assert resolved[
        'vla_head.denoiser.transformer.img_in.weight'] is transformer


def test_released_payload_rejects_conflicting_block_alias():
    value = torch.zeros(2, 3)
    payload = {
        'mot': {
            'mixtures.video.transformer.img_in.weight': value,
            'mixtures.video.transformer.double_blocks.0.img_mlp.0.weight':
                value,
            'mixtures.video.double_blocks.0.img_mlp.0.weight':
                torch.ones(2, 3),
        },
        'proprio_encoder': {'weight': value, 'bias': torch.zeros(3)},
    }
    with pytest.raises(ValueError, match='differs from its transformer copy'):
        released_action_patch_state(payload)


def test_released_payload_is_ignored_for_other_layouts():
    assert released_action_patch_state({'policy.x': torch.zeros(1)}) is None


def test_released_payload_rejects_unexpected_tensors():
    transformer = torch.zeros(2, 3)
    nested = {
        'mot': {'mixtures.video.transformer.img_in.weight': transformer},
        'proprio_encoder': {'weight': transformer, 'bias': torch.zeros(3)},
        'unexpected': torch.ones(1),
    }
    with pytest.raises(ValueError, match='Unexpected tensor entry'):
        released_action_patch_state(nested)
    flat = {
        'mot.mixtures.video.transformer.img_in.weight': transformer,
        'mot.other_tensor': torch.ones(1),
        'proprio_encoder.weight': transformer,
        'proprio_encoder.bias': torch.zeros(3),
    }
    with pytest.raises(ValueError, match='Unexpected tensor entry'):
        released_action_patch_state(flat)
    metadata = {
        'mot': {'mixtures.video.transformer.img_in.weight': transformer},
        'proprio_encoder': {'weight': transformer, 'bias': torch.zeros(3)},
        'step': 140000,
        'torch_dtype': 'torch.bfloat16',
    }
    assert released_action_patch_state(metadata) is not None


def test_released_payload_requires_proprio_section():
    payload = {'mot': {'mixtures.video.transformer.img_in.weight':
                       torch.zeros(2, 3)}}
    with pytest.raises(ValueError, match='proprio_encoder'):
        released_action_patch_state(payload)


def test_load_released_payload_into_native_model(tmp_path):
    source = _model()
    payload = _released_payload(source)
    target = _model()
    report = target.load_patchwam_state_dict(payload)
    assert report['format'] == 'action_patch'
    for name, tensor in target.state_dict().items():
        if name.startswith(('vla_head.denoiser.transformer.',
                            'vla_head.state_projection.')):
            assert torch.equal(tensor, source.state_dict()[name]), name


def test_pt_payload_loads_through_from_pretrained(tmp_path):
    source = _model()
    payload = _released_payload(source)
    path = tmp_path / 'released.pt'
    torch.save(payload, path)
    target = _model()
    target.pretrained_name_or_path = str(path)
    target.from_pretrained()
    assert torch.equal(
        target.vla_head.state_projection.weight,
        source.vla_head.state_projection.weight)


@pytest.mark.checkpoint
@pytest.mark.skipif(not RELEASED_CHECKPOINT.is_file(),
                    reason='released RoboTwin checkpoint is unavailable')
def test_real_released_checkpoint_contract():
    payload = torch.load(RELEASED_CHECKPOINT, map_location='cpu',
                         weights_only=True, mmap=True)
    assert payload['step'] == 140000
    assert set(payload) == {'mot', 'step', 'torch_dtype', 'proprio_encoder'}
    resolved = released_action_patch_state(payload)
    assert len(resolved) == 151
    assert resolved['vla_head.state_projection.weight'].shape == (7680, 14)
    assert resolved[
        'vla_head.denoiser.transformer.img_in.weight'].shape == (3072, 128)
    assert resolved[
        'vla_head.denoiser.transformer.txt_in.weight'].shape == (3072, 7680)

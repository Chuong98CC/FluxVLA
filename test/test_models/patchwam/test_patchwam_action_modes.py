# Copyright 2026 Limx Dynamics
"""CPU units for the PatchWAM action-representation modes."""

import copy

import pytest
import torch
import torch.nn as nn

from fluxvla.engines import build_head_from_cfg, build_vla_from_cfg
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.vlas import patchwam_vla  # noqa: F401
from fluxvla.models.third_party_models.patchwam.codec import (
    ACTION_MODES, RepeatedActionCodec, normalize_action_mode)

TINY = dict(token_dim=128, text_dim=32, width=32, heads=4, depth=2)


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


def _head_cfg(action_dim, mode, joint_dim=None, ee_dim=None, **overrides):
    cfg = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
        action_dim=action_dim, text_dim=32, proprio_dim=8, token_dim=128,
        action_mode=mode, joint_action_dim=joint_dim,
        end_effector_action_dim=ee_dim)
    cfg.update(overrides)
    return cfg


def _build_vla(action_dim, mode, joint_dim=None, ee_dim=None, horizon=4):
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        vlm_backbone=dict(type='PatchWAMBackbone',
                          autoencoder=TinyAutoencoder(),
                          text_encoder=TinyTextEncoder(), token_dim=128,
                          text_dim=32, history_pool_size=2),
        vla_head=_head_cfg(action_dim, mode, joint_dim, ee_dim),
        action_horizon=horizon,
        num_inference_steps=2,
    ))


def _video(batch=2, views=2, frames=2, height=4, width=6):
    generator = torch.Generator().manual_seed(11)
    return torch.rand(batch, views, 3, frames, height, width,
                      generator=generator) * 2 - 1


def _flow_kwargs(batch=2, future=48, horizon=4, seed=17):
    generator = torch.Generator().manual_seed(seed)
    return dict(sigma=torch.tensor([0.3, 0.7]),
                future_noise=torch.randn(batch, future, 128,
                                         generator=generator),
                action_noise=torch.randn(batch, horizon, 128,
                                         generator=generator))


# ----------------------------------------------------------------------
# Codec
# ----------------------------------------------------------------------
def test_codec_default_is_bit_identical_to_explicit_joint_only():
    legacy = RepeatedActionCodec(7, 128, 0.75)
    explicit = RepeatedActionCodec(7, 128, 0.75, action_mode='joint_only',
                                   joint_action_dim=7)
    assert legacy.action_mode == explicit.action_mode == 'joint_only'
    assert legacy.coordinate_layout == explicit.coordinate_layout
    assert legacy.copies == explicit.copies == 18
    actions = torch.randn(3, 5, 7)
    tokens = legacy.encode(actions)
    torch.testing.assert_close(explicit.encode(actions), tokens, rtol=0,
                               atol=0)
    torch.testing.assert_close(explicit.decode(tokens), legacy.decode(tokens),
                               rtol=0, atol=0)
    mask = torch.ones(3, 5, 7, dtype=torch.bool)
    mask[1, 2, 3] = False
    torch.testing.assert_close(explicit.coordinate_validity(mask),
                               legacy.coordinate_validity(mask), rtol=0,
                               atol=0)


def test_codec_dual_packs_joint_then_end_effector_and_splits():
    codec = RepeatedActionCodec(14, 128, action_mode='dual',
                                joint_action_dim=7,
                                end_effector_action_dim=7)
    assert codec.coordinate_layout == {'joint': slice(0, 7),
                                       'end_effector': slice(7, 14)}
    joint = torch.randn(2, 4, 7)
    end_effector = torch.randn(2, 4, 7) + 10.0
    packed = torch.cat((joint, end_effector), -1)
    tokens = codec.encode(packed)
    assert tokens.shape == (2, 4, 128)
    # Parameter-free coordinate-major repetition of the packed vector.
    torch.testing.assert_close(
        tokens[..., :codec.occupied],
        packed.repeat_interleave(codec.copies, dim=-1))
    assert torch.count_nonzero(tokens[..., codec.occupied:]) == 0
    decoded = codec.decode(tokens)
    torch.testing.assert_close(decoded, packed, rtol=1e-6, atol=1e-6)
    parts = codec.split(decoded)
    assert set(parts) == {'joint', 'end_effector'}
    torch.testing.assert_close(parts['joint'], joint, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(parts['end_effector'], end_effector, rtol=1e-6,
                               atol=1e-6)
    assert not torch.allclose(parts['joint'], parts['end_effector'])


def test_single_modes_expose_their_own_layout():
    ee = RepeatedActionCodec(14, 128, action_mode='end_effector_only',
                             end_effector_action_dim=14)
    assert ee.coordinate_layout == {'end_effector': slice(0, 14)}
    values = torch.randn(2, 3, 14)
    decoded = ee.decode(ee.encode(values))
    torch.testing.assert_close(ee.split(decoded)['end_effector'], values,
                               rtol=1e-6, atol=1e-6)
    single = RepeatedActionCodec(14, 128)
    torch.testing.assert_close(single.encode(values), ee.encode(values),
                               rtol=0, atol=0)


def test_codec_rejects_invalid_modes_and_dimensions():
    for mode in ACTION_MODES:
        assert normalize_action_mode(mode) == mode
    with pytest.raises(ValueError, match='action_mode'):
        normalize_action_mode('dual_arm')
    with pytest.raises(ValueError, match='action_mode'):
        RepeatedActionCodec(7, 128, action_mode='end_effector')
    with pytest.raises(ValueError, match='positive integer'):
        RepeatedActionCodec(7, 128, action_mode='dual', joint_action_dim=True,
                            end_effector_action_dim=7)
    with pytest.raises(ValueError, match='positive integer'):
        RepeatedActionCodec(7, 128, action_mode='dual', joint_action_dim=0,
                            end_effector_action_dim=7)
    with pytest.raises(ValueError, match='joint_action_dim'):
        RepeatedActionCodec(7, 128, joint_action_dim=6)
    with pytest.raises(ValueError, match='end_effector_action_dim'):
        RepeatedActionCodec(7, 128, action_mode='end_effector_only',
                            end_effector_action_dim=6)
    with pytest.raises(ValueError, match='dual action mode requires both'):
        RepeatedActionCodec(14, 128, action_mode='dual', joint_action_dim=7)
    with pytest.raises(ValueError, match='joint_action_dim'):
        RepeatedActionCodec(14, 128, action_mode='dual', joint_action_dim=6,
                            end_effector_action_dim=7)
    with pytest.raises(ValueError, match='token_dim'):
        RepeatedActionCodec(16, 8, action_mode='dual', joint_action_dim=8,
                            end_effector_action_dim=8)


# ----------------------------------------------------------------------
# Heads and native end-to-end
# ----------------------------------------------------------------------
@pytest.mark.parametrize('mode,action_dim,joint,ee', [
    ('joint_only', 7, None, None),
    ('end_effector_only', 7, None, None),
    ('dual', 14, 7, 7),
])
def test_head_plumbing_and_end_to_end(mode, action_dim, joint, ee):
    head = build_head_from_cfg(_head_cfg(action_dim, mode, joint, ee))
    assert head.action_mode == mode
    assert head.codec.action_mode == mode
    if mode == 'dual':
        assert head.joint_action_dim == 7
        assert head.end_effector_action_dim == 7
    if mode == 'joint_only':
        assert head.joint_action_dim == action_dim

    vla = _build_vla(action_dim, mode, joint, ee)
    video = _video()
    out = vla(images=video, states=torch.randn(2, 8),
              actions=torch.randn(2, 4, action_dim),
              action_masks=torch.ones(2, 4, dtype=torch.bool),
              task_description=['a', 'b'], **_flow_kwargs())
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert vla.vla_head.denoiser.token_output.weight.grad.abs().sum() > 0
    prediction = vla.predict_action(images=video, states=torch.randn(2, 8),
                                    task_description=['a', 'b'], seed=3)
    assert prediction.shape == (2, 4, action_dim)


def test_legacy_joint_mode_is_numerically_identical_to_the_default():
    explicit = _build_vla(7, 'joint_only', 7)
    default = _build_vla(7, 'joint_only', None)
    default.load_state_dict(explicit.state_dict())
    video = _video()
    kwargs = _flow_kwargs(seed=23)
    actions = torch.randn(2, 4, 7)
    states = torch.randn(2, 8)
    first = explicit(images=video, states=states, actions=actions,
                     action_masks=torch.ones(2, 4, dtype=torch.bool),
                     task_description=['a', 'b'], **copy.deepcopy(kwargs))
    second = default(images=video, states=states, actions=actions,
                     action_masks=torch.ones(2, 4, dtype=torch.bool),
                     task_description=['a', 'b'], **copy.deepcopy(kwargs))
    for key in ('loss', 'loss_video', 'loss_action', 'sigma_mean'):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert set(explicit.vla_head.state_dict()) == set(
        default.vla_head.state_dict())


def test_action_modes_add_no_learned_weights():
    joint = build_head_from_cfg(_head_cfg(7, 'joint_only', None, None))
    ee = build_head_from_cfg(_head_cfg(7, 'end_effector_only', None, None))
    dual = build_head_from_cfg(_head_cfg(14, 'dual', 7, 7))
    joint_params = sum(p.numel() for p in joint.parameters())
    ee_params = sum(p.numel() for p in ee.parameters())
    dual_params = sum(p.numel() for p in dual.parameters())
    assert joint_params == ee_params == dual_params
    for head in (joint, ee, dual):
        assert not any(name.startswith('codec')
                       for name in head.state_dict())
        assert not hasattr(head.codec, 'parameters')


def test_dual_decode_returns_both_coordinate_sets_distinctly():
    vla = _build_vla(14, 'dual', 7, 7)
    sample = dict(reference_tokens=torch.randn(2, 4, 128),
                  text_tokens=torch.randn(2, 3, 32))
    with torch.no_grad():
        prediction = vla.predict_action(states=torch.randn(2, 8), seed=7,
                                        **sample)
    assert prediction.shape == (2, 4, 14)
    parts = vla.vla_head.codec.split(prediction)
    assert torch.isfinite(prediction).all()
    assert not torch.allclose(parts['joint'], parts['end_effector'])

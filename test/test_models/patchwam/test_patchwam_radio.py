# Copyright 2026 Limx Dynamics
"""CPU units for the PatchWAM C-RADIO auxiliary supervision variant.

The model keeps the original 128-wide VAE future flow and adds a training-only
clean C-RADIO feature MSE from the final transformer hidden future tokens.
Teachers here are deterministic local stand-ins, and the official FLUX.2
fixtures are the real upstream module built with reduced dimensions.
"""

import copy
import dataclasses
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxvla.engines import (HEADS, VLAS, VLM_BACKBONES, build_head_from_cfg,
                             build_vla_from_cfg)
from fluxvla.models.backbones.vlms import patchwam_radio_backbone  # noqa: F401
from fluxvla.models.heads import patchwam_radio_head  # noqa: F401
from fluxvla.models.third_party_models.patchwam.flow import masked_sample_mse
from fluxvla.models.third_party_models.patchwam.flux import \
    OfficialFluxDenoiser
from fluxvla.models.third_party_models.patchwam.geometry import (
    joint_visibility, sequence_coordinates)
from fluxvla.models.third_party_models.patchwam.radio import (
    RadioAuxFlowPolicy, load_local_radio_teacher, radio_feature_targets,
    radio_spatial_features)
from fluxvla.models.vlas import patchwam_radio_vla  # noqa: F401

RADIO_DIM = 6
TINY = dict(token_dim=128, text_dim=32, width=32, heads=4, depth=2)
FLUX_SOURCE = Path('/home/chuong/workspace/wam/ImageWAM/third_party/flux2/src')


@dataclasses.dataclass
class RadioOutput:
    """Mirror of the official ``(summary, spatial_features)`` return type."""

    summary: torch.Tensor
    spatial_features: torch.Tensor


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
    """Deterministic patch-embedding stand-in for C-RADIOv4-H features.

    ``api`` reproduces the genuine upstream returns: a ``(summary, spatial)``
    pair, a ``RadioOutput`` dataclass, a mapping, or a bare tensor.
    """

    def __init__(self, feature_dim: int = RADIO_DIM, patch_size: int = 2,
                 *, api: str = 'tuple', min_resolution_step: int = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.min_resolution_step = min_resolution_step
        self.api = api
        self.calls = 0
        self.kernel = nn.Parameter(
            torch.randn(feature_dim, 3, patch_size, patch_size) * 0.1)

    def encode(self, images):
        self.calls += 1
        spatial = F.conv2d(images, self.kernel,
                           stride=self.patch_size).flatten(2).transpose(1, 2)
        summary = spatial.mean(1)
        if self.api == 'tuple':
            return summary, spatial
        if self.api == 'dataclass':
            return RadioOutput(summary=summary, spatial_features=spatial)
        if self.api == 'dict':
            return {'summary': summary, 'spatial_features': spatial}
        return spatial


class ExplodingTeacher(TinyRadioTeacher):
    """Any evaluation of this teacher is a hard failure."""

    def encode(self, images):
        raise AssertionError('the RADIO teacher must not run at inference')


class RecordingDenoiser(nn.Module):
    """Tiny denoiser that forwards and records the captured hidden states."""

    def __init__(self, *, zero_velocities: bool = True, **kwargs):
        super().__init__()
        from fluxvla.models.third_party_models.patchwam.tiny import \
            SmallJointTransformer
        self.inner = SmallJointTransformer(**kwargs)
        self.captured = None
        self.zero_velocities = zero_velocities
        self.representation_dim = self.inner.representation_dim
        self.representation_layers = self.inner.representation_layers

    def forward(self, reference, noisy, sigma, context, **kwargs):
        result = self.inner(reference, noisy, sigma, context, **kwargs)
        if isinstance(result, tuple):
            self.captured = result[1].detach().clone()
            if self.zero_velocities:
                return tuple(torch.zeros_like(value) for value in result)
            return result
        return torch.zeros_like(result) if self.zero_velocities else result


def aux_head_cfg(**overrides):
    cfg = dict(
        type='PatchWAMRadioAuxHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
        action_dim=7, text_dim=32, proprio_dim=8, token_dim=128,
        radio_dim=RADIO_DIM, radio_weight=0.5)
    cfg.update(overrides)
    return cfg


def radio_backbone_cfg(teacher, **overrides):
    cfg = dict(
        type='PatchWAMRadioAuxBackbone',
        autoencoder=TinyAutoencoder(),
        text_encoder=TinyTextEncoder(),
        token_dim=128, text_dim=32, history_pool_size=2,
        radio_teacher=teacher, radio_dim=RADIO_DIM,
        radio_patch_size=2, radio_min_resolution=2)
    cfg.update(overrides)
    return cfg


def build_radio_backbone(teacher=None, **overrides):
    from fluxvla.engines import build_vlm_backbone_from_cfg
    return build_vlm_backbone_from_cfg(radio_backbone_cfg(teacher, **overrides))


def build_vla(teacher=None, **overrides):
    cfg = dict(
        type='PatchWAMRadioAuxVLA',
        vlm_backbone=radio_backbone_cfg(teacher),
        vla_head=aux_head_cfg(),
        action_horizon=5,
        num_inference_steps=2,
    )
    cfg.update(overrides)
    return build_vla_from_cfg(cfg)


def preencoded(batch=2, reference=4, future=6, text=3, horizon=5,
               action_dim=7, proprio_dim=8):
    generator = torch.Generator().manual_seed(7)
    return {
        'reference_tokens':
        torch.randn(batch, reference, 128, generator=generator),
        'future_tokens':
        torch.randn(batch, future, 128, generator=generator),
        'radio_future_tokens':
        torch.randn(batch, future, RADIO_DIM, generator=generator),
        'text_tokens':
        torch.randn(batch, text, 32, generator=generator),
        'action':
        torch.randn(batch, horizon, action_dim, generator=generator),
        'proprio':
        torch.randn(batch, horizon, proprio_dim, generator=generator),
    }


def build_video(batch=2, views=2, frames=2, height=4, width=6):
    generator = torch.Generator().manual_seed(11)
    return torch.rand(batch, views, 3, frames, height, width,
                      generator=generator) * 2 - 1


# ----------------------------------------------------------------------
# Registration and defaults
# ----------------------------------------------------------------------
def test_registries_expose_only_the_auxiliary_components():
    assert HEADS.get('PatchWAMRadioAuxHead') is not None
    assert VLM_BACKBONES.get('PatchWAMRadioAuxBackbone') is not None
    assert VLAS.get('PatchWAMRadioAuxVLA') is not None
    # The feature-space flow variant (and its split-width denoisers) is gone.
    for removed in ('PatchWAMRadioHead', 'PatchWAMRadioTinyDenoiser',
                    'PatchWAMRadioFluxDenoiser'):
        assert HEADS.get(removed) is None
    assert VLM_BACKBONES.get('PatchWAMRadioBackbone') is None
    assert VLAS.get('PatchWAMRadioVLA') is None
    head = build_head_from_cfg(aux_head_cfg())
    assert isinstance(head, RadioAuxFlowPolicy)
    assert head.radio_dim == RADIO_DIM and head.radio_weight == 0.5
    assert head.codec.token_dim == 128
    assert not hasattr(head, 'split_future')
    assert not hasattr(head.denoiser, 'split_modalities')
    assert not hasattr(head.denoiser, 'feature_input')
    assert tuple(head.radio_projection.weight.shape) == (RADIO_DIM, TINY['width'])
    with pytest.raises(ValueError, match='radio_weight'):
        build_head_from_cfg(aux_head_cfg(radio_weight=-1))
    with pytest.raises(ValueError, match='video_weight'):
        build_head_from_cfg(aux_head_cfg(video_weight=float('nan')))


# ----------------------------------------------------------------------
# Teacher contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize('api', ['tuple', 'dataclass', 'dict', 'tensor'])
def test_teacher_spatial_unpacking_never_uses_the_summary(api):
    teacher = TinyRadioTeacher(api=api)
    targets = radio_feature_targets(torch.zeros(1, 3, 4, 6), teacher=teacher,
                                    raster=(4, 6), patch_size=2,
                                    minimum_resolution=2,
                                    expected_dim=RADIO_DIM)
    assert targets.shape == (1, 24, RADIO_DIM)
    assert teacher.calls == 1
    # A summary-only return is rejected rather than silently supervised.
    with pytest.raises(TypeError, match='spatial'):
        radio_spatial_features({'summary': torch.zeros(2, RADIO_DIM)})
    with pytest.raises(TypeError, match='spatial'):
        radio_spatial_features((torch.zeros(2, RADIO_DIM), 'summary'))


def test_teacher_conversion_padding_and_group_grid():
    seen = {}

    class Recorder(nn.Module):
        feature_dim = RADIO_DIM
        patch_size = 2
        min_resolution_step = 6

        def encode(self, images):
            seen['shape'] = tuple(images.shape)
            height, width = images.shape[-2:]
            return torch.zeros(images.shape[0], (height // 2) * (width // 2),
                               RADIO_DIM)

    video = torch.full((1, 3, 5, 7), -1.0)
    targets = radio_feature_targets(video, teacher=Recorder(), raster=(5, 7),
                                    patch_size=2, minimum_resolution=2,
                                    expected_dim=RADIO_DIM)
    # min_resolution_step=6 rounds the input up to 6x12 in patches of two.
    assert seen['shape'] == (1, 3, 6, 12)
    assert targets.shape == (1, 35, RADIO_DIM)

    class UnitRange(nn.Module):
        feature_dim = RADIO_DIM
        patch_size = 2

        def encode(self, images):
            seen['range'] = (float(images.min()), float(images.max()))
            return torch.zeros(images.shape[0], 6, RADIO_DIM)

    video = torch.ones(1, 3, 4, 6)
    radio_feature_targets(video, teacher=UnitRange(), raster=(4, 6),
                          patch_size=2, minimum_resolution=2,
                          expected_dim=RADIO_DIM)
    assert seen['range'] == (1.0, 1.0)
    with pytest.raises(ValueError, match='finite'):
        bad = video.clone()
        bad[0, 0, 0, 0] = float('nan')
        radio_feature_targets(bad, teacher=UnitRange(), raster=(4, 6),
                              patch_size=2, minimum_resolution=2,
                              expected_dim=RADIO_DIM)
    with pytest.raises(ValueError, match='radio_dim'):
        radio_feature_targets(video, teacher=UnitRange(), raster=(4, 6),
                              patch_size=2, minimum_resolution=2,
                              expected_dim=RADIO_DIM + 1)
    fixed_grid = TinyRadioTeacher(patch_size=2, api='tensor')
    fixed_grid.encode = lambda images: torch.zeros(images.shape[0], 5,
                                                   RADIO_DIM)
    with pytest.raises(ValueError, match='grid'):
        radio_feature_targets(video, teacher=fixed_grid, raster=(4, 6),
                              patch_size=2, minimum_resolution=2,
                              expected_dim=RADIO_DIM)


def test_local_teacher_loading_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match='local Hugging Face export'):
        load_local_radio_teacher(str(tmp_path / 'missing'))
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match='config.json'):
        load_local_radio_teacher(str(empty))
    wrong = tmp_path / 'wrong'
    wrong.mkdir()
    (wrong / 'config.json').write_text(json.dumps({'version': 'radio_v2.5-h'}))
    with pytest.raises(ValueError, match='version'):
        load_local_radio_teacher(str(wrong))
    mismatched = tmp_path / 'patch'
    mismatched.mkdir()
    (mismatched / 'config.json').write_text(
        json.dumps({'version': 'c-radio_v4-h', 'patch_size': 14}))
    with pytest.raises(ValueError, match='patch_size'):
        load_local_radio_teacher(str(mismatched))


# ----------------------------------------------------------------------
# Auxiliary targets, loss mix and the final-hidden tap
# ----------------------------------------------------------------------
def test_backbone_emits_both_vae_and_radio_targets():
    from fluxvla.models.backbones.vlms.patchwam_backbone import \
        PatchWAMBackbone
    video = build_video()
    request = dict(video=video, prompt=['a', 'b'])
    autoencoder, text_encoder = TinyAutoencoder(), TinyTextEncoder()
    legacy = PatchWAMBackbone(autoencoder=autoencoder,
                              text_encoder=text_encoder, token_dim=128,
                              text_dim=32, history_pool_size=2)
    legacy_out = legacy.encode(request, target=True)
    teacher = TinyRadioTeacher()
    backbone = build_radio_backbone(teacher, autoencoder=autoencoder,
                                    text_encoder=text_encoder)
    out = backbone.encode(dict(request, need_radio=True), target=True)
    assert out['future_tokens'].shape == (2, 48, 128)
    assert out['radio_future_tokens'].shape == (2, 48, RADIO_DIM)
    assert out['radio_future_tokens'].requires_grad is False
    torch.testing.assert_close(out['reference_tokens'],
                               legacy_out['reference_tokens'])
    for key in ('future_tokens', 'future_ids', 'future_valid'):
        torch.testing.assert_close(out[key], legacy_out[key])
    assert out['sampling_future_length'] == legacy_out['sampling_future_length']
    assert teacher.calls == 1
    # No teacher work when the auxiliary weight is zero.
    teacher.calls = 0
    skipped = backbone.encode(dict(request, need_radio=False), target=True)
    assert 'radio_future_tokens' not in skipped
    assert teacher.calls == 0
    assert torch.equal(skipped['future_tokens'], out['future_tokens'])


def test_targets_are_per_view_and_invalid_padding_is_sanitized():
    teacher = TinyRadioTeacher()
    views = torch.zeros(1, 2, 3, 2, 4, 6)
    views[0, 0] = -1.0
    views[0, 1] = 1.0
    backbone = build_radio_backbone(teacher)
    tokens = backbone.encode(dict(video=views, prompt=['move'], need_radio=True),
                             target=True)['radio_future_tokens'][0]
    first, second = tokens[:24], tokens[24:]
    one = build_radio_backbone(teacher).encode(
        dict(video=views[:, :1], prompt=['move'], need_radio=True),
        target=True)['radio_future_tokens'][0]
    two = build_radio_backbone(teacher).encode(
        dict(video=views[:, 1:], prompt=['move'], need_radio=True),
        target=True)['radio_future_tokens'][0]
    torch.testing.assert_close(first, one)
    torch.testing.assert_close(second, two)
    assert not torch.allclose(first, second)

    video = build_video().clone()
    video[1, 1] = float('nan')
    frame_masks = torch.tensor([[True, True], [True, False]])
    img_masks = torch.tensor([[True, True], [True, False]])
    backbone = build_radio_backbone(TinyRadioTeacher())
    encoded = backbone.encode(
        dict(video=video, prompt=['a', 'b'], frame_masks=frame_masks,
             img_masks=img_masks, need_radio=True), target=True)
    valid = encoded['future_valid']
    assert not valid[1].any()
    assert torch.count_nonzero(encoded['radio_future_tokens'][1]) == 0
    assert torch.isfinite(encoded['radio_future_tokens']).all()


def test_loss_mix_and_final_hidden_tap():
    denoiser = RecordingDenoiser(zero_velocities=False, **TINY)
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=32,
                                proprio_dim=8, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=0.5,
                                video_weight=0.5, action_weight=1.0)
    sample = preencoded()
    result = policy(sample, sigma=torch.tensor([0.25, 0.75]))
    assert set(('loss', 'loss_video', 'loss_radio', 'loss_action',
                'loss_vae')).issubset(result)
    assert torch.equal(result['loss_vae'], result['loss_video'])
    expected = (0.5 * result['loss_video'] + 0.5 * result['loss_radio']
                + 1.0 * result['loss_action'])
    torch.testing.assert_close(result['loss'], expected)
    # The capture covers future + action hidden tokens in the same pass, and the
    # adapter tap is the final hidden width rather than the 128-wide output.
    assert denoiser.captured.shape == (2, 6 + 5, TINY['width'])
    assert denoiser.captured.shape[-1] != 128
    result['loss'].backward()
    assert policy.radio_projection.weight.grad.abs().sum() > 0
    assert denoiser.inner.blocks[-1].qkv.weight.grad.abs().sum() > 0


def test_radio_loss_uses_only_the_future_slice():
    denoiser = RecordingDenoiser(zero_velocities=False, **TINY)
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=32,
                                proprio_dim=8, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=0.5)
    sample = preencoded()
    result = policy(sample, sigma=torch.tensor([0.5, 0.5]))
    hidden = denoiser.captured[:, :6]
    expected = masked_sample_mse(policy.radio_projection(hidden),
                                 sample['radio_future_tokens']).mean()
    torch.testing.assert_close(result['loss_radio'], expected)
    # The adapter is fed only the future slice of the final hidden states.
    recorded = {}
    handle = policy.radio_projection.register_forward_hook(
        lambda module, args, output: recorded.update(
            value=args[0].detach().clone()))
    policy(sample, sigma=torch.tensor([0.5, 0.5]))
    handle.remove()
    assert recorded['value'].shape == (2, 6, TINY['width'])
    torch.testing.assert_close(recorded['value'], denoiser.captured[:, :6])


def test_zero_radio_weight_is_the_original_objective():
    torch.manual_seed(3)
    legacy = build_head_from_cfg(
        dict(type='PatchWAMHead',
             denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
             action_dim=7, text_dim=32, proprio_dim=8, token_dim=128,
             video_weight=0.5, action_weight=1.0))
    torch.manual_seed(3)
    aux = build_head_from_cfg(aux_head_cfg(radio_weight=0.0))
    legacy.load_state_dict(aux.state_dict(), strict=False)
    sample = preencoded()
    sample.pop('radio_future_tokens')  # no RADIO target is required
    sigma = torch.tensor([0.2, 0.8])
    generator = torch.Generator().manual_seed(12)
    kwargs = dict(sigma=sigma,
                  future_noise=torch.randn(2, 6, 128, generator=generator),
                  action_noise=torch.randn(2, 5, 128, generator=generator))
    legacy_out = legacy(copy.deepcopy(sample), **kwargs)
    aux_out = aux(copy.deepcopy(sample), **kwargs)
    for key in ('loss', 'loss_video', 'loss_action', 'sigma_mean'):
        torch.testing.assert_close(aux_out[key], legacy_out[key], rtol=0,
                                   atol=0)
    assert torch.equal(aux_out['loss_radio'], torch.zeros(()))
    legacy_out['loss'].backward()
    legacy_grads = {name: parameter.grad.detach().clone()
                    for name, parameter in legacy.named_parameters()
                    if parameter.grad is not None}
    aux_out['loss'].backward()
    for name, parameter in aux.named_parameters():
        if name in legacy_grads:
            torch.testing.assert_close(parameter.grad, legacy_grads[name],
                                       rtol=0, atol=0)
    assert aux.radio_projection.weight.grad is None
    # No teacher work either: an unusable path must not be touched at zero weight.
    unusable = build_vla(None, vlm_backbone=radio_backbone_cfg(
        None, radio_model_path='/nonexistent/radio'),
        vla_head=aux_head_cfg(radio_weight=0.0))
    loss = unusable(
        images=build_video(), states=torch.randn(2, 5, 8),
        actions=torch.randn(2, 5, 7),
        action_masks=torch.ones(2, 5, dtype=torch.bool),
        task_description=['a', 'b'])['loss']
    assert torch.isfinite(loss)


# ----------------------------------------------------------------------
# Export, teacher-free inference and checkpoints
# ----------------------------------------------------------------------
def test_export_loads_strictly_and_reproduces_actions():
    from fluxvla.models.vlas.patchwam_vla import PatchWAMVLA
    aux = build_vla(TinyRadioTeacher())
    sample = preencoded()
    aux.eval()
    states = torch.randn(2, 8)
    with torch.no_grad():
        predicted = aux.predict_action(
            reference_tokens=sample['reference_tokens'],
            text_tokens=sample['text_tokens'],
            future_ids=sequence_coordinates(2, 6, group=0),
            states=states, seed=21)
    exported = aux.export_inference_state_dict()
    report = aux.exported_inference_keys_report()
    assert set(report['removed']) == set(aux.RADIO_PROJECTION_KEYS)
    assert report['teacher_keys'] == []
    assert not report['teacher_in_state_dict']
    assert not any(key.startswith('vla_head.radio_projection')
                   for key in exported)

    plain_cfg = dict(
        type='PatchWAMVLA',
        vlm_backbone=dict(
            type='PatchWAMBackbone',
            autoencoder=TinyAutoencoder(),
            text_encoder=TinyTextEncoder(),
            token_dim=128, text_dim=32, history_pool_size=2),
        vla_head=dict(
            type='PatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
            action_dim=7, text_dim=32, proprio_dim=8, token_dim=128),
        action_horizon=5,
        num_inference_steps=2,
    )
    plain = build_vla_from_cfg(plain_cfg)
    assert isinstance(plain, PatchWAMVLA)
    plain.load_state_dict(exported, strict=True)
    # The exported artifact is exactly the original PatchWAM inference model.
    original_params = sum(tensor.numel() for tensor in plain.state_dict().values())
    exported_params = sum(tensor.numel() for tensor in exported.values())
    adapter_params = sum(aux.state_dict()[key].numel()
                         for key in aux.RADIO_PROJECTION_KEYS)
    trained_params = sum(tensor.numel() for tensor in aux.state_dict().values())
    assert adapter_params > 0
    assert exported_params == original_params
    assert trained_params == original_params + adapter_params
    plain.eval()
    with torch.no_grad():
        restored = plain.predict_action(
            reference_tokens=sample['reference_tokens'],
            text_tokens=sample['text_tokens'],
            future_ids=sequence_coordinates(2, 6, group=0),
            states=states, seed=21)
    torch.testing.assert_close(predicted, restored, rtol=0, atol=0)


def test_teacher_is_never_needed_for_inference_or_export():
    teacher = ExplodingTeacher()
    aux = build_vla(teacher)
    aux.eval()
    with torch.no_grad():
        actions = aux.predict_action(
            images=build_video(), states=torch.randn(2, 8),
            task_description=['a', 'b'], seed=4)
    assert actions.shape == (2, 5, 7) and torch.isfinite(actions).all()
    assert teacher.calls == 0
    aux.export_inference_state_dict()
    assert teacher.calls == 0


def test_checkpoint_strictness_and_legacy_initialization():
    source = build_vla(TinyRadioTeacher())
    state = copy.deepcopy(source.state_dict())
    assert 'vla_head.radio_projection.weight' in state
    assert not any(key.startswith('vlm_backbone.radio_teacher')
                   for key in state)
    clone = build_vla(None)
    clone.load_state_dict(copy.deepcopy(state))
    torch.testing.assert_close(clone.vla_head.radio_projection.weight,
                               source.vla_head.radio_projection.weight)
    report = clone.load_patchwam_state_dict(copy.deepcopy(state), strict=True)
    assert report['format'] == 'native'

    legacy = copy.deepcopy(state)
    for key in ('vla_head.radio_projection.weight',
                'vla_head.radio_projection.bias'):
        legacy.pop(key)
    strict = build_vla(None)
    fresh = strict.vla_head.radio_projection.weight.detach().clone()
    with pytest.raises(ValueError, match='missing='):
        strict.load_patchwam_state_dict(copy.deepcopy(legacy), strict=True)
    # Runners resume with strict=False; the trained adapter is still required.
    with pytest.raises(ValueError, match='missing'):
        strict.load_state_dict(copy.deepcopy(legacy), strict=False)
    torch.testing.assert_close(strict.vla_head.radio_projection.weight, fresh)
    poisoned = copy.deepcopy(state)
    poisoned['vla_head.radio_projection.weight'] = torch.full_like(
        poisoned['vla_head.radio_projection.weight'], float('nan'))
    with pytest.raises(ValueError, match='invalid'):
        strict.load_state_dict(copy.deepcopy(poisoned), strict=False)

    # The explicit legacy initializer is the only path that may leave exactly
    # the declared new layers fresh.
    relaxed = build_vla(None)
    init = relaxed.initialize_from_legacy_state_dict(copy.deepcopy(legacy),
                                                     strict=True)
    assert init['left_fresh'] == sorted(relaxed.RADIO_PROJECTION_KEYS)
    torch.testing.assert_close(relaxed.vla_head.state_projection.weight,
                               source.vla_head.state_projection.weight)
    corrupt = copy.deepcopy(legacy)
    corrupt['vla_head.denoiser.blocks.0.qkv.weight'] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match='shape differs'):
        relaxed.initialize_from_legacy_state_dict(copy.deepcopy(corrupt),
                                                  strict=True)


def test_vla_forwards_cached_radio_targets():
    vla = build_vla(None)
    sample = preencoded()
    cached = dict(
        reference_tokens=sample['reference_tokens'],
        future_tokens=sample['future_tokens'],
        radio_future_tokens=sample['radio_future_tokens'],
        text_tokens=sample['text_tokens'],
        actions=sample['action'],
        states=sample['proprio'][:, 0],
        action_masks=torch.ones(2, 5, dtype=torch.bool))
    out = vla(**cached)
    assert torch.isfinite(out['loss']) and 'loss_radio' in out
    without = {key: value for key, value in cached.items()
               if key != 'radio_future_tokens'}
    with pytest.raises(ValueError, match='radio_future_tokens'):
        vla(**without)


# ----------------------------------------------------------------------
# Target device/dtype alignment and resume strictness
# ----------------------------------------------------------------------
def test_align_supervised_features_moves_and_validates():
    from fluxvla.models.third_party_models.patchwam.radio import \
        align_supervised_features
    tokens = torch.randn(2, 3, RADIO_DIM, dtype=torch.float64,
                         requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    aligned = align_supervised_features(
        tokens, valid, name='radio_future_tokens',
        device=torch.device('cpu'), dtype=torch.bfloat16)
    assert aligned.dtype == torch.bfloat16
    assert aligned.device == torch.device('cpu')
    assert not aligned.requires_grad
    # Masked padding may hold anything; supervised coordinates must be finite.
    padded = torch.randn(2, 3, RADIO_DIM)
    padded[0, 2] = float('nan')
    align_supervised_features(padded, valid, name='radio_future_tokens')
    poisoned = padded.clone()
    poisoned[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        align_supervised_features(poisoned, valid,
                                  name='radio_future_tokens')
    with pytest.raises(ValueError, match='finite'):
        align_supervised_features(poisoned, None, name='radio_future_tokens')


def test_cached_radio_targets_follow_the_reference_dtype_and_device():
    from fluxvla.models.third_party_models.patchwam.encoding import \
        encode_observations
    batch = {
        'reference_tokens': torch.randn(2, 4, 128),
        'future_tokens': torch.randn(2, 6, 128),
        'radio_future_tokens': torch.randn(2, 6, RADIO_DIM).to(torch.float64),
        'text_tokens': torch.randn(2, 3, 32),
        'prompt': ['a', 'b'],
    }
    encoded = encode_observations(
        batch, autoencoder=TinyAutoencoder(), text_encoder=TinyTextEncoder(),
        token_dim=128, text_dim=32, history_pool_size=2, target=True)
    reference = encoded['reference_tokens']
    radio = encoded['radio_future_tokens']
    assert radio.dtype == reference.dtype
    assert radio.device == reference.device


def test_cached_radio_targets_reject_nonfinite_supervised_values():
    vla = build_vla(None)
    sample = preencoded()
    sample['future_valid'] = torch.ones(2, 6, dtype=torch.bool)
    sample['future_valid'][0, -1] = False
    sample['radio_future_tokens'][0, -1] = float('nan')
    cached = dict(
        reference_tokens=sample['reference_tokens'],
        future_tokens=sample['future_tokens'],
        radio_future_tokens=sample['radio_future_tokens'],
        future_valid=sample['future_valid'],
        text_tokens=sample['text_tokens'],
        actions=sample['action'],
        states=sample['proprio'][:, 0],
        action_masks=torch.ones(2, 5, dtype=torch.bool))
    # NaN in masked padding is ignored and the loss stays finite.
    out = vla(**cached)
    assert torch.isfinite(out['loss']) and torch.isfinite(out['loss_radio'])
    poisoned = dict(cached)
    poisoned['radio_future_tokens'] = sample['radio_future_tokens'].clone()
    poisoned['radio_future_tokens'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='radio_future_tokens'):
        vla(**poisoned)


def test_native_resume_rejects_missing_or_corrupt_radio_layers():
    source = build_vla(TinyRadioTeacher())
    state = copy.deepcopy(source.state_dict())
    target = build_vla(None)
    target.load_state_dict(copy.deepcopy(state))
    broken = copy.deepcopy(state)
    for key in target.RADIO_PROJECTION_KEYS:
        broken.pop(key)
    fresh = {key: dict(target.named_parameters())[key].detach().clone()
             for key in target.RADIO_PROJECTION_KEYS}
    with pytest.raises(ValueError, match='missing'):
        target.load_state_dict(copy.deepcopy(broken), strict=False)
    with pytest.raises(ValueError, match='missing'):
        target.load_patchwam_state_dict(copy.deepcopy(broken), strict=True)
    for key, value in fresh.items():
        torch.testing.assert_close(dict(target.named_parameters())[key], value)
    legacy = build_vla(None)
    report = legacy.initialize_from_legacy_state_dict(copy.deepcopy(broken),
                                                     strict=True)
    assert report['left_fresh'] == sorted(legacy.RADIO_PROJECTION_KEYS)


# ----------------------------------------------------------------------
# Genuine official FLUX.2 blocks
# ----------------------------------------------------------------------
def _small_transformer():
    if str(FLUX_SOURCE) not in sys.path:
        sys.path.insert(0, str(FLUX_SOURCE))
    model_api = importlib.import_module('flux2.model')
    params = model_api.Klein4BParams(
        in_channels=128,
        context_in_dim=48,
        hidden_size=64,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[4, 4, 4, 4],
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )
    return model_api.Flux2(params)


flux_required = pytest.mark.skipif(
    not (FLUX_SOURCE / 'flux2' / 'model.py').is_file(),
    reason='local official flux2 source is unavailable')


@flux_required
def test_official_flux_auxiliary_taps_the_final_hidden_state():
    class RecordingFlux(OfficialFluxDenoiser):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captured = None

        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            if isinstance(result, tuple):
                self.captured = result[1].detach().clone()
            return result

    denoiser = RecordingFlux(_small_transformer(), gradient_checkpointing=True)
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=48,
                                proprio_dim=None, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=0.5)
    sample = preencoded(reference=3, future=4, text=5, horizon=5, action_dim=7)
    sample['text_tokens'] = torch.randn(2, 5, 48)
    sample['proprio'] = torch.randn(2, 7)
    result = policy(sample, sigma=torch.tensor([0.4, 0.6]))
    features = denoiser.captured
    assert features.shape == (2, 9, denoiser.transformer.hidden_size)
    assert features.shape[-1] != 128  # the final hidden tap, not the output
    result['loss'].backward()
    assert policy.radio_projection.weight.grad.abs().sum() > 0
    assert denoiser.transformer.single_blocks[-1].linear1.weight.grad.abs().sum() > 0
    assert denoiser.transformer.final_layer.linear.weight.grad.abs().sum() > 0


@flux_required
def test_aux_accepts_fp32_cached_targets_on_a_bf16_transformer():
    denoiser = OfficialFluxDenoiser(_small_transformer().to(torch.bfloat16))
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=48,
                                proprio_dim=None, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=0.5)
    assert policy.radio_projection.weight.dtype == torch.bfloat16
    sample = {
        'reference_tokens': torch.randn(2, 3, 128).to(torch.bfloat16),
        'future_tokens': torch.randn(2, 4, 128).to(torch.bfloat16),
        # Raw float32 teacher features cached beside a bfloat16 model.
        'radio_future_tokens': torch.randn(2, 4, RADIO_DIM),
        'text_tokens': torch.randn(2, 5, 48).to(torch.bfloat16),
        'action': torch.randn(2, 2, 7),
    }
    out = policy(sample, sigma=torch.tensor([0.4, 0.6]))
    assert torch.isfinite(out['loss']) and torch.isfinite(out['loss_radio'])
    out['loss'].backward()
    assert policy.radio_projection.weight.grad.abs().sum() > 0


# ----------------------------------------------------------------------
# adapter_layer_idex (zero-based block selector, -1 = last block)
# ----------------------------------------------------------------------
def _policy(denoiser, **overrides):
    cfg = dict(action_dim=7, text_dim=32, proprio_dim=8, token_dim=128,
               radio_dim=RADIO_DIM, radio_weight=0.5)
    cfg.update(overrides)
    return RadioAuxFlowPolicy(denoiser, **cfg)


def _fixed_kwargs(seed=17):
    generator = torch.Generator().manual_seed(seed)
    return dict(sigma=torch.tensor([0.35, 0.65]),
                future_noise=torch.randn(2, 6, 128, generator=generator),
                action_noise=torch.randn(2, 5, 128, generator=generator))


def test_adapter_layer_idex_validation_and_translation():
    from fluxvla.models.third_party_models.patchwam.radio import \
        resolve_capture_layer
    # -1 and the last zero-based index both select the final block.
    assert resolve_capture_layer(-1, 4) == 4
    assert resolve_capture_layer(3, 4) == 4
    assert resolve_capture_layer(0, 4) == 1
    assert resolve_capture_layer(2, 4) == 3
    for bad in (True, False, 1.0, '1', None):
        with pytest.raises(ValueError, match='adapter_layer_idex'):
            resolve_capture_layer(bad, 4)
    for bad in (-2, 4, 99):
        with pytest.raises(ValueError, match='adapter_layer_idex'):
            resolve_capture_layer(bad, 4)
    with pytest.raises(ValueError, match=r'\[0, 3\]'):
        resolve_capture_layer(4, 4)

    # The head validates against the real denoiser depth at construction.
    head = build_head_from_cfg(aux_head_cfg())
    assert head.adapter_layer_idex == -1
    assert head.capture_layer == TINY['depth']
    assert build_head_from_cfg(
        aux_head_cfg(adapter_layer_idex=0)).capture_layer == 1
    for bad in (True, 1.5, TINY['depth'], -2):
        with pytest.raises(ValueError, match='adapter_layer_idex'):
            build_head_from_cfg(aux_head_cfg(adapter_layer_idex=bad))


def test_adapter_layer_uses_real_early_hidden_values():
    denoiser_last = RecordingDenoiser(zero_velocities=False, **TINY)
    last_policy = _policy(denoiser_last, adapter_layer_idex=-1)
    denoiser_first = RecordingDenoiser(zero_velocities=False, **TINY)
    first_policy = _policy(denoiser_first, adapter_layer_idex=0)
    first_policy.load_state_dict(last_policy.state_dict())

    sample = preencoded()
    block_outputs = []
    block_inputs = []

    def record_block(module, args, output):
        block_inputs.append(args[0].detach().clone())
        block_outputs.append(output.detach().clone())

    handle = denoiser_first.inner.blocks[0].register_forward_hook(record_block)
    projection_inputs = {}
    handle_projection = first_policy.radio_projection.register_forward_hook(
        lambda module, args, output: projection_inputs.update(
            value=args[0].detach().clone()))
    kwargs = _fixed_kwargs()
    first_out = first_policy(copy.deepcopy(sample), **kwargs)
    handle_projection.remove()
    handle.remove()

    # The first block ran once and its future slice is exactly what the
    # projection consumed: a real early hidden state, not a passed argument.
    assert len(block_outputs) == 1
    noisy_length = sample['future_tokens'].shape[1] + sample['action'].shape[1]
    offset = block_inputs[0].shape[1] - noisy_length
    assert offset > 0
    expected = block_outputs[0][:, offset:]
    assert expected.shape == (2, 6 + 5, TINY['width'])
    torch.testing.assert_close(projection_inputs['value'],
                               expected[:, :6], rtol=0, atol=0)

    last_out = last_policy(copy.deepcopy(sample), **kwargs)
    torch.testing.assert_close(projection_inputs['value'],
                               denoiser_first.captured[:, :6])
    last_hidden = denoiser_last.captured[:, :6]
    assert not torch.allclose(last_hidden, projection_inputs['value'])
    # All blocks still run and the original flow losses are untouched.
    for key in ('loss_video', 'loss_action'):
        torch.testing.assert_close(first_out[key], last_out[key], rtol=0,
                                   atol=0)
    assert not torch.allclose(first_out['loss_radio'], last_out['loss_radio'])
    assert torch.isfinite(first_out['loss']).all()


def test_adapter_layer_default_parity_and_state_keys():
    last = build_head_from_cfg(aux_head_cfg())
    explicit_last = build_head_from_cfg(
        aux_head_cfg(adapter_layer_idex=TINY['depth'] - 1))
    last.load_state_dict(explicit_last.state_dict())
    assert last.adapter_layer_idex == -1
    assert last.capture_layer == explicit_last.capture_layer == TINY['depth']
    sample = preencoded()
    kwargs = _fixed_kwargs(seed=5)
    default_out = last(copy.deepcopy(sample), **kwargs)
    explicit_out = explicit_last(copy.deepcopy(sample), **kwargs)
    for key in ('loss', 'loss_video', 'loss_radio', 'loss_action',
                'sigma_mean'):
        torch.testing.assert_close(default_out[key], explicit_out[key],
                                   rtol=0, atol=0)
    # The selector is configuration only: identical parameter/state keys.
    assert set(last.state_dict()) == set(explicit_last.state_dict())
    assert not any('adapter_layer_idex' in name for name in last.state_dict())


def test_adapter_layer_zero_weight_parity_with_non_default_index():
    torch.manual_seed(3)
    legacy = build_head_from_cfg(
        dict(type='PatchWAMHead',
             denoiser=dict(type='PatchWAMTinyDenoiser', **TINY),
             action_dim=7, text_dim=32, proprio_dim=8, token_dim=128,
             video_weight=0.5, action_weight=1.0))
    torch.manual_seed(3)
    aux = build_head_from_cfg(
        aux_head_cfg(radio_weight=0.0, adapter_layer_idex=0))
    legacy.load_state_dict(aux.state_dict(), strict=False)
    sample = preencoded()
    sample.pop('radio_future_tokens')
    kwargs = _fixed_kwargs(seed=12)
    legacy_out = legacy(copy.deepcopy(sample), **kwargs)
    aux_out = aux(copy.deepcopy(sample), **kwargs)
    for key in ('loss', 'loss_video', 'loss_action', 'sigma_mean'):
        torch.testing.assert_close(aux_out[key], legacy_out[key], rtol=0,
                                   atol=0)
    assert aux.capture_layer is None
    legacy_out['loss'].backward()
    legacy_grads = {name: parameter.grad.detach().clone()
                    for name, parameter in legacy.named_parameters()
                    if parameter.grad is not None}
    aux_out['loss'].backward()
    for name, parameter in aux.named_parameters():
        if name in legacy_grads:
            torch.testing.assert_close(parameter.grad, legacy_grads[name],
                                       rtol=0, atol=0)
    assert aux.radio_projection.weight.grad is None


# ----------------------------------------------------------------------
# adapter_layer_idex on genuine official reduced FLUX.2 blocks
# ----------------------------------------------------------------------
def _deeper_small_transformer():
    if str(FLUX_SOURCE) not in sys.path:
        sys.path.insert(0, str(FLUX_SOURCE))
    model_api = importlib.import_module('flux2.model')
    # Two double blocks followed by two single blocks make both sides of the
    # double/single boundary addressable.
    params = model_api.Klein4BParams(
        in_channels=128,
        context_in_dim=48,
        hidden_size=64,
        num_heads=4,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[4, 4, 4, 4],
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )
    return model_api.Flux2(params)


def _flux_sample(reference=3, future=4, text=5, horizon=5):
    generator = torch.Generator().manual_seed(29)
    sample = preencoded(reference=reference, future=future, text=text,
                        horizon=horizon, action_dim=7)
    sample['text_tokens'] = torch.randn(2, text, 48, generator=generator)
    sample['proprio'] = torch.randn(2, 7, generator=generator)
    return sample


def _flux_policy_state():
    """One shared parameter set so selector runs differ only by the tap."""
    denoiser = OfficialFluxDenoiser(_deeper_small_transformer())
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=48,
                                proprio_dim=None, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=0.5)
    return policy.state_dict()


def _seed_adapter(policy, width=64):
    """Deterministic, non-degenerate adapter shared by every selector run."""
    generator = torch.Generator().manual_seed(31)
    with torch.no_grad():
        policy.radio_projection.weight.copy_(
            torch.randn(RADIO_DIM, width, generator=generator) * 0.5)
        policy.radio_projection.bias.zero_()


def _flux_kwargs(seed=23, future=4, horizon=5):
    """Fixed noise so different selector runs see identical flow inputs."""
    generator = torch.Generator().manual_seed(seed)
    return dict(sigma=torch.tensor([0.4, 0.6]),
                future_noise=torch.randn(2, future, 128, generator=generator),
                action_noise=torch.randn(2, horizon, 128, generator=generator))


def _flux_capture_run(adapter_layer_idex, state, sample):
    """Run one forward, recording every block output and the adapter input."""
    denoiser = OfficialFluxDenoiser(_deeper_small_transformer())
    double_outputs, single_outputs = [], []
    original_double = denoiser._paired_block
    original_single = denoiser._unified_block

    def record_double(block, text, image, *args, **kwargs):
        result = original_double(block, text, image, *args, **kwargs)
        double_outputs.append(result[1].detach().clone())
        return result

    def record_single(block, states, *args, **kwargs):
        result = original_single(block, states, *args, **kwargs)
        single_outputs.append(result.detach().clone())
        return result

    denoiser._paired_block = record_double
    denoiser._unified_block = record_single
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=48,
                                proprio_dim=None, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=0.5,
                                adapter_layer_idex=adapter_layer_idex)
    policy.load_state_dict(state)
    _seed_adapter(policy)
    captured = {}
    handle = policy.radio_projection.register_forward_hook(
        lambda module, args, output: captured.update(
            value=args[0].detach().clone()))
    out = policy(copy.deepcopy(sample), **_flux_kwargs())
    handle.remove()
    return out, captured['value'], double_outputs, single_outputs


@flux_required
def test_official_flux_layer_selector_captures_the_selected_block():
    reference_length, text_length, future_length = 3, 5, 4
    state, sample = _flux_policy_state(), _flux_sample()
    baseline_out = None
    radio_losses = {}
    cases = {
        0: ('double', 0),
        1: ('double', 1),   # last double block
        2: ('single', 0),   # first single block: the boundary
        3: ('single', 1),
        -1: ('single', 1),
    }
    for index, (kind, position) in cases.items():
        out, adapter_input, double_outputs, single_outputs = \
            _flux_capture_run(index, state, sample)
        # One pass: every block still executes for the original velocities.
        assert len(double_outputs) == 2 and len(single_outputs) == 2
        if kind == 'double':
            expected = double_outputs[position][:, reference_length:]
        else:
            expected = single_outputs[position][
                :, text_length + reference_length:]
        assert adapter_input.shape == (2, future_length, 64)
        torch.testing.assert_close(adapter_input,
                                   expected[:, :future_length], rtol=0,
                                   atol=0)
        assert torch.isfinite(out['loss_video']).all()
        radio_losses[index] = out['loss_radio'].detach()
        if baseline_out is None:
            baseline_out = out
        else:
            # The selector never changes the original flow objectives.
            for key in ('loss_video', 'loss_action'):
                torch.testing.assert_close(out[key], baseline_out[key],
                                           rtol=0, atol=0)
    # Different taps really do produce different auxiliary losses.
    assert not torch.allclose(radio_losses[0], radio_losses[-1])
    assert not torch.allclose(radio_losses[0], radio_losses[2])


def _flux_gradient_sums(adapter_layer_idex, state, sample, *, video_weight,
                        action_weight):
    denoiser = OfficialFluxDenoiser(_deeper_small_transformer(),
                                    gradient_checkpointing=True)
    policy = RadioAuxFlowPolicy(denoiser, action_dim=7, text_dim=48,
                                proprio_dim=None, token_dim=128,
                                radio_dim=RADIO_DIM, radio_weight=0.5,
                                video_weight=video_weight,
                                action_weight=action_weight,
                                adapter_layer_idex=adapter_layer_idex)
    policy.load_state_dict(state)
    _seed_adapter(policy)
    out = policy(copy.deepcopy(sample), **_flux_kwargs())
    out['loss'].backward()
    transformer = denoiser.transformer
    sums = {
        'double0': next(transformer.double_blocks[0].parameters()).grad,
        'double1': next(transformer.double_blocks[1].parameters()).grad,
        'single0': next(transformer.single_blocks[0].parameters()).grad,
        'single1': next(transformer.single_blocks[1].parameters()).grad,
        'final': transformer.final_layer.linear.weight.grad,
        'adapter': policy.radio_projection.weight.grad,
    }
    return out, {name: None if grad is None else float(grad.abs().sum())
                 for name, grad in sums.items()}


@flux_required
def test_official_flux_layer_selector_gradients_with_checkpointing():
    # Radio-only objective: the auxiliary loss must reach the selected block and
    # its predecessors, and must not reach strictly later blocks.
    state, sample = _flux_policy_state(), _flux_sample()
    for index, later in ((1, ('single0', 'single1')), (2, ('single1',))):
        out, sums = _flux_gradient_sums(index, state, sample, video_weight=0.0,
                                        action_weight=0.0)
        assert torch.isfinite(out['loss']) and torch.isfinite(out['loss_radio'])
        assert sums['adapter'] > 0
        assert sums['double0'] > 0 and sums['double1'] > 0
        if index == 2:
            assert sums['single0'] > 0
        for name in later:
            assert sums[name] == 0
        assert sums['final'] == 0
    # With the original weights the later blocks are trained by the velocity
    # objectives, exactly as before this feature.
    out, sums = _flux_gradient_sums(1, state, sample, video_weight=0.5,
                                    action_weight=1.0)
    assert sums['single1'] > 0 and sums['final'] > 0
    assert torch.isfinite(out['loss_video']).all()

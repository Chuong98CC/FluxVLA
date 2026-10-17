"""Inference contracts: one current extraction, feedback carry, and reset."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.backbones.vlms import patchwam_backbone  # noqa: F401
from fluxvla.models.heads import teacher_forcing_patchwam_head  # noqa: F401
from fluxvla.models.vlas import teacher_forcing_patchwam_vla  # noqa: F401
from fluxvla.models.backbones.visions.frozen_vision import SpatialFeatureOutput
from fluxvla.models.third_party_models.patchwam.teacher_forcing import (
    TeacherForcingConfig,
    teacher_representation_fingerprint,
)
from fluxvla.models.third_party_models.patchwam.geometry import (
    raster_coordinates,
)


class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 128, 4, stride=4)

    def encode(self, video):
        return self.proj(video)


class TinyTextEncoder(nn.Module):
    requires_images = False
    trainable_adapters = False

    def __init__(self):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(3, 32))
        self.feature_dim = 32

    def forward(self, prompts):
        return self.embed.unsqueeze(0).expand(len(prompts), -1, -1)


class FakeTeacherEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 8
        self.calls = []
        self.means = []

    def feature_grid(self, height, width):
        if (height, width) != (8, 12):
            raise ValueError('fixed input 8x12')
        return 2, 3

    def encode_features(self, images):
        self.calls.append(images.shape[0])
        self.means.append(float(images.mean()))
        batch = images.shape[0]
        base = torch.arange(6, device=images.device, dtype=torch.float32)
        value = images.mean(dim=(1, 2, 3))[:, None]
        tokens = torch.cat((
            base[None].expand(batch, -1)[..., None],
            value.expand(-1, 6)[..., None],
            torch.zeros(batch, 6, 6, device=images.device),
        ), -1)
        return SpatialFeatureOutput(tokens=tokens, grid=(2, 3))


SPEC = dict(
    name='radio',
    feature_dim=8,
    representation_id='fake-radio-sampling',
    checkpoint_sha256='f' * 64,
    stream='spatial_features',
    preprocessing='unit_range_once',
    input_size=(8, 12),
    native_grid=(2, 3),
)


def config(**overrides):
    values = dict(
        encoders=[dict(SPEC)],
        current_conditioning=True,
        future_prediction=True,
        future_feedback=True,
        predictor=dict(type='linear'),
        self_cond_drop_prob=0.1,
        true_feature_prob=0.05,
    )
    values.update(overrides)
    return values


def build(encoder=None, teacher_config=None, num_steps=3):
    cfg = dict(
        type='TeacherForcingPatchWAMVLA',
        action_horizon=16,
        num_inference_steps=num_steps,
        vlm_backbone=dict(
            type='PatchWAMBackbone',
            autoencoder=TinyAutoencoder(),
            text_encoder=TinyTextEncoder(),
            token_dim=128, text_dim=32, history_pool_size=2),
        vla_head=dict(
            type='TeacherForcingPatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=14, text_dim=32, proprio_dim=14, token_dim=128,
            teacher_forcing=teacher_config or config()),
    )
    if encoder is not None:
        cfg['teacher_extractors'] = {'radio': encoder}
        cfg['teacher_provenance'] = {
            'radio': dict(
                representation_id=SPEC['representation_id'],
                test_contract='explicit-unit-fixture')}
    vla = build_vla_from_cfg(cfg)
    with torch.no_grad():
        for conditioner in vla.vla_head.teacher_conditioners.values():
            conditioner.weight.normal_()
    return vla


def video(batch=2, views=2, frames=5, value=None):
    result = torch.zeros(batch, views, 3, frames, 8, 12)
    if value is not None:
        result.fill_(value)
    return result


def preencoded_tokens(batch=2, length=12):
    spec = TeacherForcingConfig.from_mapping(config()).encoders[0]
    fp = torch.tensor(list(teacher_representation_fingerprint(spec)),
                      dtype=torch.uint8)[None].expand(batch, -1)
    return {
        'teacher_current_tokens': {
            'radio': torch.randn(batch, length, 8) + 0.2},
        'teacher_current_valid': {
            'radio': torch.ones(batch, length, dtype=torch.bool)},
        'teacher_current_fingerprints': {'radio': fp},
        'current_raster_layout': {
            'views': 2, 'video_size': [8, 12], 'grid': [2, 3],
            'view_order': 'view_major_row_major'},
        'reference_ids': torch.cat([
            raster_coordinates(batch, 2, 3, group=10 + view)
            for view in range(2)
        ], 1),
        'current_raster_length': length,
        'sampling_future_length': length,
    }


def test_current_extraction_once_and_no_future_extraction():
    encoder = FakeTeacherEncoder()
    vla = build(encoder)
    result = vla.predict_action(
        video=video(value=0.25), states=torch.zeros(2, 14),
        task_description=['move', 'lift'], seed=7)
    assert result.shape == (2, 16, 14)
    assert len(encoder.calls) == 1
    assert encoder.calls == [4]


def test_training_extracts_current_and_future_in_separate_calls():
    encoder = FakeTeacherEncoder()
    vla = build(encoder)
    clip = video()
    clip[:, :, :, 0] = -0.5
    clip[:, :, :, -1] = 0.5
    out = vla(
        video=clip,
        states=torch.zeros(2, 16, 14),
        actions=torch.randn(2, 16, 14),
        task_description=['move', 'lift'],
        sigma=torch.tensor([0.3, 0.7]))
    assert len(encoder.calls) == 2
    assert encoder.calls == [4, 4]
    assert encoder.means[0] == pytest.approx(-0.5)
    assert encoder.means[1] == pytest.approx(0.5)
    assert torch.isfinite(out['loss'])
    assert 'loss_teacher_radio' in out


def test_guided_sampling_uses_two_calls_per_step_and_resets_feedback():
    vla = build()
    calls = []
    original = vla.vla_head._velocity

    def recording(*args, **kwargs):
        calls.append(kwargs.get('future_hidden_residual'))
        return original(*args, **kwargs)

    tokens = preencoded_tokens()
    reference = torch.randn(2, 12, 128)
    text = torch.randn(2, 3, 32)
    projector_calls = []
    handle = vla.vla_head.teacher_current_projectors[
        'radio'].register_forward_hook(
            lambda module, inputs, output: projector_calls.append(1))
    with patch.object(vla.vla_head, '_velocity', recording):
        first = vla.predict_action(
            reference_tokens=reference,
            text_tokens=text,
            states=torch.zeros(2, 14),
            guidance_scale=1.5,
            action_guidance_scale=1.2,
            seed=11,
            **tokens)
    handle.remove()
    assert len(calls) == 6
    assert len(projector_calls) == 1
    assert torch.count_nonzero(calls[0]) == 0
    assert torch.count_nonzero(calls[1]) == 0
    with patch.object(vla.vla_head, '_velocity', recording):
        second = vla.predict_action(
            reference_tokens=reference,
            text_tokens=text,
            states=torch.zeros(2, 14),
            guidance_scale=1.5,
            action_guidance_scale=1.2,
            seed=11,
            **tokens)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_future_targets_cannot_change_sampling():
    vla = build()
    tokens = preencoded_tokens()
    reference = torch.randn(2, 12, 128)
    text = torch.randn(2, 3, 32)
    base = vla.predict_action(
        reference_tokens=reference, text_tokens=text,
        states=torch.zeros(2, 14), seed=13, **tokens)
    extra = dict(tokens)
    extra['teacher_future_tokens'] = {
        'radio': torch.randn(2, 12, 8)}
    extra['teacher_future_valid'] = {
        'radio': torch.ones(2, 12, dtype=torch.bool)}
    spec = TeacherForcingConfig.from_mapping(config()).encoders[0]
    extra['teacher_future_fingerprints'] = {
        'radio': torch.tensor(
            list(teacher_representation_fingerprint(spec)),
            dtype=torch.uint8)[None].expand(2, -1)}
    changed = vla.predict_action(
        reference_tokens=reference, text_tokens=text,
        states=torch.zeros(2, 14), seed=13, **extra)
    torch.testing.assert_close(base, changed, rtol=0, atol=0)
    partial = dict(tokens)
    partial['teacher_future_tokens'] = {
        'radio': torch.randn(2, 12, 8)}
    partial_changed = vla.predict_action(
        reference_tokens=reference, text_tokens=text,
        states=torch.zeros(2, 14), seed=13, **partial)
    torch.testing.assert_close(base, partial_changed, rtol=0, atol=0)


def test_current_only_and_supervision_only_switches():
    current_only = build(teacher_config=config(
        future_prediction=False, future_feedback=False))
    result = current_only.predict_action(
        reference_tokens=torch.randn(2, 12, 128),
        text_tokens=torch.randn(2, 3, 32),
        states=torch.zeros(2, 14), seed=3,
        **preencoded_tokens())
    assert result.shape == (2, 16, 14)
    assert len(current_only.vla_head.teacher_predictors) == 0
    assert len(current_only.vla_head.teacher_conditioners) == 0

    supervision = build(teacher_config=config(
        current_conditioning=False, future_feedback=False))
    result = supervision.predict_action(
        reference_tokens=torch.randn(2, 12, 128),
        text_tokens=torch.randn(2, 3, 32),
        states=torch.zeros(2, 14), seed=3)
    assert result.shape == (2, 16, 14)
    assert len(supervision.vla_head.teacher_current_projectors) == 0


def test_future_only_feedback_inference_needs_no_teacher_encoder():
    vla = build(teacher_config=config(current_conditioning=False))
    result = vla.predict_action(
        reference_tokens=torch.randn(2, 12, 128),
        text_tokens=torch.randn(2, 3, 32),
        states=torch.zeros(2, 14), seed=4,
        sampling_future_length=12)
    assert result.shape == (2, 16, 14)
    assert torch.isfinite(result).all()
    assert len(vla.vla_head.teacher_current_projectors) == 0


def test_combined_inference_requires_current_features_or_encoder():
    vla = build()
    with pytest.raises(ValueError, match='Missing teacher RGB'):
        vla.predict_action(
            reference_tokens=torch.randn(2, 12, 128),
            text_tokens=torch.randn(2, 3, 32),
            states=torch.zeros(2, 14),
            sampling_future_length=12)

"""Multiple encoders and rank-independent pass structure."""

from unittest.mock import patch

import torch

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.heads import teacher_forcing_patchwam_head  # noqa: F401
from fluxvla.models.heads.patchwam_head import PatchWAMHead
from fluxvla.models.vlas import teacher_forcing_patchwam_vla  # noqa: F401
from fluxvla.models.third_party_models.patchwam.teacher_forcing import (
    TeacherForcingConfig,
    teacher_representation_fingerprint,
)
from fluxvla.models.third_party_models.patchwam.geometry import (
    raster_coordinates,
)


SPECS = [
    dict(
        name='radio', feature_dim=8,
        representation_id='fake-radio-multi',
        checkpoint_sha256='3' * 64,
        stream='spatial_features', preprocessing='unit_range_once',
        input_size=(8, 12), native_grid=(2, 3), loss_weight=0.1),
    dict(
        name='radio_alt', feature_dim=6,
        representation_id='fake-radio-alt-multi',
        checkpoint_sha256='4' * 64,
        stream='spatial_features', preprocessing='unit_range_once',
        input_size=(8, 12), native_grid=(2, 3), loss_weight=0.2),
]


def config(**overrides):
    values = dict(
        encoders=[dict(spec) for spec in SPECS],
        predictor=dict(type='linear'),
        self_cond_drop_prob=0.0,
        true_feature_prob=0.0,
    )
    values.update(overrides)
    return values


def model(**overrides):
    return build_vla_from_cfg(dict(
        type='TeacherForcingPatchWAMVLA',
        action_horizon=16,
        num_inference_steps=2,
        vla_head=dict(
            type='TeacherForcingPatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=14, text_dim=32, proprio_dim=14, token_dim=128,
            teacher_forcing=config(**overrides)),
    ))


def fingerprints():
    specs = TeacherForcingConfig.from_mapping(config()).encoders
    return {
        spec.name: torch.tensor(
            list(teacher_representation_fingerprint(spec)),
            dtype=torch.uint8)[None].expand(2, -1)
        for spec in specs
    }


def batch(seed=5):
    generator = torch.Generator().manual_seed(seed)
    fp = fingerprints()
    return {
        'reference_tokens': torch.randn(2, 6, 128, generator=generator),
        'future_tokens': torch.randn(2, 6, 128, generator=generator),
        'text_tokens': torch.randn(2, 3, 32, generator=generator),
        'action': torch.randn(2, 16, 14, generator=generator),
        'proprio': torch.randn(2, 16, 14, generator=generator),
        'teacher_current_tokens': {
            'radio': torch.randn(2, 6, 8, generator=generator) + 0.1,
            'radio_alt': torch.randn(2, 6, 6, generator=generator) + 0.2,
        },
        'teacher_current_valid': {
            name: torch.ones(2, 6, dtype=torch.bool)
            for name in fp},
        'teacher_current_fingerprints': fp,
        'teacher_future_tokens': {
            'radio': torch.randn(2, 6, 8, generator=generator) + 0.3,
            'radio_alt': torch.randn(2, 6, 6, generator=generator) + 0.4,
        },
        'teacher_future_valid': {
            name: torch.ones(2, 6, dtype=torch.bool)
            for name in fp},
        'teacher_future_fingerprints': {
            name: value.clone() for name, value in fp.items()},
        'current_raster_layout': {
            'views': 1, 'video_size': [8, 12], 'grid': [2, 3],
            'view_order': 'view_major_row_major'},
        'future_raster_layout': {
            'views': 1, 'video_size': [8, 12], 'grid': [2, 3],
            'view_order': 'view_major_row_major'},
        'reference_ids': raster_coordinates(2, 2, 3, group=10),
        'future_ids': raster_coordinates(2, 2, 3, group=0),
        'current_raster_length': 6,
    }


def run(vla, data):
    return vla(
        reference_tokens=data['reference_tokens'],
        future_tokens=data['future_tokens'],
        text_tokens=data['text_tokens'],
        actions=data['action'],
        states=data['proprio'],
        sigma=torch.tensor([0.3, 0.7]),
        teacher_current_tokens=data['teacher_current_tokens'],
        teacher_current_valid=data['teacher_current_valid'],
        teacher_current_fingerprints=data[
            'teacher_current_fingerprints'],
        teacher_future_tokens=data['teacher_future_tokens'],
        teacher_future_valid=data['teacher_future_valid'],
        teacher_future_fingerprints=data[
            'teacher_future_fingerprints'],
        current_raster_layout=data['current_raster_layout'],
        future_raster_layout=data['future_raster_layout'],
        reference_ids=data['reference_ids'],
        future_ids=data['future_ids'],
        current_raster_length=6,
    )


def test_two_encoders_have_independent_branches_and_losses():
    vla = model()
    head = vla.vla_head
    assert isinstance(head, PatchWAMHead)
    assert set(head.teacher_current_projectors) == {'radio', 'radio_alt'}
    assert head.teacher_current_projectors['radio'].weight.shape == (32, 8)
    assert head.teacher_current_projectors['radio_alt'].weight.shape == (
        32, 6)
    with torch.no_grad():
        head.teacher_current_projectors['radio'].weight.fill_(0.1)
        head.teacher_current_projectors['radio_alt'].weight.fill_(0.2)
        head.teacher_conditioners['radio'].weight.fill_(0.3)
        head.teacher_conditioners['radio_alt'].weight.fill_(0.4)
    data = batch()
    out = run(vla, data)
    assert 'loss_teacher_radio' in out
    assert 'loss_teacher_radio_alt' in out
    expected = (0.1 * out['loss_teacher_radio']
                + 0.2 * out['loss_teacher_radio_alt'])
    torch.testing.assert_close(out['loss_teacher'], expected)
    out['loss'].backward()
    for name in ('radio', 'radio_alt'):
        assert head.teacher_current_projectors[name].weight.grad.abs().sum() > 0
        assert head.teacher_conditioners[name].weight.grad.abs().sum() > 0
        assert head.teacher_predictors[name].weight.grad.abs().sum() > 0


def test_per_sample_masks_do_not_skip_lookahead_on_any_rank():
    vla = model(true_feature_prob=1.0, self_cond_drop_prob=0.5)
    data = batch()
    calls = []
    original = vla.vla_head.denoiser.forward

    def recording(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return original(*args, **kwargs)

    generator = torch.Generator().manual_seed(2)
    with patch.object(vla.vla_head.denoiser, 'forward', recording):
        out = vla(
            reference_tokens=data['reference_tokens'],
            future_tokens=data['future_tokens'],
            text_tokens=data['text_tokens'],
            actions=data['action'],
            states=data['proprio'],
            sigma=torch.tensor([0.3, 0.7]),
            generator=generator,
            teacher_current_tokens=data['teacher_current_tokens'],
            teacher_current_valid=data['teacher_current_valid'],
            teacher_current_fingerprints=data[
                'teacher_current_fingerprints'],
            teacher_future_tokens=data['teacher_future_tokens'],
            teacher_future_valid=data['teacher_future_valid'],
            teacher_future_fingerprints=data[
                'teacher_future_fingerprints'],
            current_raster_layout=data['current_raster_layout'],
            future_raster_layout=data['future_raster_layout'],
            reference_ids=data['reference_ids'],
            future_ids=data['future_ids'],
            current_raster_length=6,
        )
    assert calls == [False, True]
    assert out['teacher_true_fraction'] == 1
    assert 0 <= out['teacher_feedback_drop_fraction'] <= 1


def test_partial_encoder_batches_are_rejected():
    vla = model()
    data = batch()
    data['teacher_current_tokens'].pop('radio_alt')
    try:
        run(vla, data)
    except ValueError as exc:
        assert 'radio_alt' in str(exc)
    else:  # pragma: no cover
        raise AssertionError('partial encoder batch was accepted')

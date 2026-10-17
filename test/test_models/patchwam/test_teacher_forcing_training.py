"""Training contracts for current conditioning and feature forcing."""

import pytest
import torch

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.heads import teacher_forcing_patchwam_head  # noqa: F401
from fluxvla.models.vlas import teacher_forcing_patchwam_vla  # noqa: F401
from fluxvla.models.third_party_models.patchwam.teacher_forcing import (
    TeacherForcingConfig,
    teacher_representation_fingerprint,
)
from fluxvla.models.third_party_models.patchwam.geometry import (
    raster_coordinates,
)


SPEC = dict(
    name='radio',
    feature_dim=8,
    representation_id='fake-radio-training',
    checkpoint_sha256='e' * 64,
    stream='spatial_features',
    preprocessing='unit_range_once',
    input_size=(8, 12),
    native_grid=(2, 3),
    loss_weight=0.1,
)


def config(**overrides):
    values = dict(
        encoders=[dict(SPEC)],
        predictor=dict(type='linear'),
        self_cond_drop_prob=0.1,
        true_feature_prob=0.05,
    )
    values.update(overrides)
    return values


def model(**overrides):
    cfg = dict(
        type='TeacherForcingPatchWAMVLA',
        action_horizon=16,
        num_inference_steps=3,
        vla_head=dict(
            type='TeacherForcingPatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=14, text_dim=32, proprio_dim=14, token_dim=128,
            teacher_forcing=config(**overrides),
        ),
    )
    result = build_vla_from_cfg(cfg)
    with torch.no_grad():
        result.vla_head.teacher_conditioners['radio'].weight.normal_()
    return result


def fingerprint():
    spec = TeacherForcingConfig.from_mapping(config()).encoders[0]
    return torch.tensor(list(teacher_representation_fingerprint(spec)),
                        dtype=torch.uint8)


def sample(batch_size=2, length=6, horizon=16, seed=5):
    generator = torch.Generator().manual_seed(seed)
    fp = fingerprint()[None].expand(batch_size, -1)
    return {
        'reference_tokens': torch.randn(
            batch_size, length, 128, generator=generator),
        'future_tokens': torch.randn(
            batch_size, length, 128, generator=generator),
        'text_tokens': torch.randn(
            batch_size, 3, 32, generator=generator),
        'action': torch.randn(
            batch_size, horizon, 14, generator=generator),
        'proprio': torch.randn(
            batch_size, horizon, 14, generator=generator),
        'teacher_current_tokens': {
            'radio': torch.randn(
                batch_size, length, 8, generator=generator) + 0.1},
        'teacher_current_valid': {
            'radio': torch.ones(batch_size, length, dtype=torch.bool)},
        'teacher_current_fingerprints': {'radio': fp},
        'teacher_future_tokens': {
            'radio': torch.randn(
                batch_size, length, 8, generator=generator) + 0.2},
        'teacher_future_valid': {
            'radio': torch.ones(batch_size, length, dtype=torch.bool)},
        'teacher_future_fingerprints': {'radio': fp.clone()},
        'current_raster_layout': {
            'views': 1, 'video_size': [8, 12], 'grid': [2, 3],
            'view_order': 'view_major_row_major'},
        'future_raster_layout': {
            'views': 1, 'video_size': [8, 12], 'grid': [2, 3],
            'view_order': 'view_major_row_major'},
        'reference_ids': raster_coordinates(
            batch_size, 2, 3, group=10),
        'future_ids': raster_coordinates(batch_size, 2, 3, group=0),
        'current_raster_length': length,
        'sampling_future_length': length,
    }


def run(vla, data, *, sigma=(0.2, 0.8), future_tokens=None):
    generator = torch.Generator().manual_seed(31)
    future_noise = torch.randn(
        data['future_tokens'].shape, generator=generator)
    action_noise = torch.randn(
        data['action'].shape[0], data['action'].shape[1], 128,
        generator=generator)
    teacher_future = data['teacher_future_tokens']
    if future_tokens is not None:
        teacher_future = {'radio': future_tokens}
    layout_args = {
        key: data[key] for key in (
            'current_raster_layout', 'future_raster_layout',
            'reference_ids', 'future_ids', 'current_raster_length',
        ) if key in data
    }
    return vla(
        reference_tokens=data['reference_tokens'],
        future_tokens=data['future_tokens'],
        text_tokens=data['text_tokens'],
        actions=data['action'],
        states=data['proprio'],
        sigma=torch.tensor(sigma),
        future_noise=future_noise,
        action_noise=action_noise,
        teacher_current_tokens=data['teacher_current_tokens'],
        teacher_current_valid=data['teacher_current_valid'],
        teacher_current_fingerprints=data[
            'teacher_current_fingerprints'],
        teacher_future_tokens=teacher_future,
        teacher_future_valid=data['teacher_future_valid'],
        teacher_future_fingerprints=data[
            'teacher_future_fingerprints'],
        **layout_args,
    )


def test_total_loss_composition_counts_feature_term_once():
    vla = model()
    out = run(vla, sample())
    flow = (vla.vla_head.video_weight * out['loss_video']
            + vla.vla_head.action_weight * out['loss_action'])
    torch.testing.assert_close(out['loss'], flow + out['loss_teacher'])
    assert 'loss_teacher_radio' in out


def test_all_teacher_descriptors_are_detached_from_autograd():
    vla = model(true_feature_prob=1.0, self_cond_drop_prob=0.0)
    data = sample()
    data['teacher_current_tokens']['radio'].requires_grad_()
    data['teacher_future_tokens']['radio'].requires_grad_()
    out = run(vla, data)
    out['loss'].backward()
    assert data['teacher_current_tokens']['radio'].grad is None
    assert data['teacher_future_tokens']['radio'].grad is None
    head = vla.vla_head
    for module in (
            head.teacher_current_projectors['radio'],
            head.teacher_conditioners['radio'],
            head.teacher_predictors['radio']):
        assert module.weight.grad is not None
        assert module.weight.grad.abs().sum() > 0


def test_layout_identity_and_raster_order_are_strict():
    vla = model()
    data = sample()
    data['current_raster_layout']['video_size'] = [12, 8]
    data['current_raster_layout']['grid'] = [3, 2]
    data['future_raster_layout']['video_size'] = [12, 8]
    data['future_raster_layout']['grid'] = [3, 2]
    with pytest.raises(ValueError, match='does not match encoder'):
        run(vla, data)

    data = sample()
    data.pop('current_raster_layout')
    with pytest.raises(ValueError, match='explicit current_raster_layout'):
        vla(
            reference_tokens=data['reference_tokens'],
            future_tokens=data['future_tokens'],
            text_tokens=data['text_tokens'],
            actions=data['action'],
            states=data['proprio'],
            teacher_current_tokens=data['teacher_current_tokens'],
            teacher_current_valid=data['teacher_current_valid'],
            teacher_current_fingerprints=data[
                'teacher_current_fingerprints'],
            teacher_future_tokens=data['teacher_future_tokens'],
            teacher_future_valid=data['teacher_future_valid'],
            teacher_future_fingerprints=data[
                'teacher_future_fingerprints'],
            current_raster_length=data['current_raster_length'],
        )

    data = sample()
    data.pop('future_raster_layout')
    with pytest.raises(ValueError, match='explicit future_raster_layout'):
        run(vla, data)

    data = sample()
    data['reference_ids'] = data['reference_ids'][:, [1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError, match='reference_ids'):
        run(vla, data)


def test_future_length_mismatch_is_rejected():
    vla = model()
    data = sample()
    with pytest.raises(ValueError, match='future_length'):
        vla.predict_action(
            reference_tokens=data['reference_tokens'],
            text_tokens=data['text_tokens'],
            states=data['proprio'][:, 0],
            future_length=5,
            teacher_current_tokens=data['teacher_current_tokens'],
            teacher_current_valid=data['teacher_current_valid'],
            teacher_current_fingerprints=data[
                'teacher_current_fingerprints'],
            current_raster_layout=data['current_raster_layout'],
            reference_ids=data['reference_ids'],
            current_raster_length=data['current_raster_length'],
        )


def test_feature_loss_still_trains_at_flow_weight_endpoints():
    vla = model()
    out = run(vla, sample(), sigma=(0.0, 1.0))
    assert torch.isfinite(out['loss_teacher_radio'])
    out['loss_teacher'].backward()
    assert vla.vla_head.teacher_predictors['radio'].weight.grad.abs().sum() > 0


def test_future_targets_do_not_affect_conditioning_without_substitution():
    vla = model(true_feature_prob=0.0, self_cond_drop_prob=0.0)
    data = sample()
    other = torch.randn_like(data['teacher_future_tokens']['radio'])
    first = run(vla, data)
    second = run(vla, data, future_tokens=other)
    torch.testing.assert_close(first['loss_video'], second['loss_video'])
    torch.testing.assert_close(first['loss_action'], second['loss_action'])
    assert not torch.allclose(
        first['loss_teacher_radio'], second['loss_teacher_radio'])


def test_true_substitution_dependency_and_dropout_order():
    substituted = model(true_feature_prob=1.0, self_cond_drop_prob=0.0)
    data = sample(seed=7)
    other = torch.randn_like(data['teacher_future_tokens']['radio'])
    first = run(substituted, data)
    second = run(substituted, data, future_tokens=other)
    assert not torch.allclose(first['loss_action'], second['loss_action'])
    assert first['teacher_true_fraction'] == 1

    dropped = model(true_feature_prob=1.0, self_cond_drop_prob=1.0)
    dropped_first = run(dropped, data)
    dropped_second = run(dropped, data, future_tokens=other)
    torch.testing.assert_close(
        dropped_first['loss_action'], dropped_second['loss_action'])
    torch.testing.assert_close(
        dropped_first['loss_video'], dropped_second['loss_video'])
    assert dropped_first['teacher_feedback_drop_fraction'] == 1


def test_feedback_dropout_keeps_g_execution_but_zero_weight_gradient():
    vla = model(self_cond_drop_prob=1.0)
    out = run(vla, sample())
    out['loss'].backward()
    conditioner = vla.vla_head.teacher_conditioners['radio'].weight
    assert conditioner.grad is not None
    assert torch.count_nonzero(conditioner.grad) == 0
    assert vla.vla_head.teacher_predictors['radio'].weight.grad is not None
    assert (vla.vla_head.teacher_predictors['radio'].weight.grad.abs().sum()
            > 0)
    assert (vla.vla_head.teacher_current_projectors[
        'radio'].weight.grad.abs().sum() > 0)


def test_validation_is_deterministic_without_substitution_or_dropout():
    vla = model(true_feature_prob=1.0, self_cond_drop_prob=1.0)
    vla.eval()
    data = sample()
    other = torch.randn_like(data['teacher_future_tokens']['radio'])
    first = run(vla, data)
    second = run(vla, data)
    changed_target = run(vla, data, future_tokens=other)
    for key in ('loss_video', 'loss_action'):
        torch.testing.assert_close(first[key], second[key])
        torch.testing.assert_close(first[key], changed_target[key])
    assert 'teacher_true_fraction' not in first
    assert 'teacher_feedback_drop_fraction' not in first


def test_partial_and_all_invalid_targets_stay_finite_and_masked():
    vla = model()
    data = sample()
    data['teacher_future_valid']['radio'][:, 2:] = False
    data['future_valid'] = data['teacher_future_valid']['radio'].clone()
    partial = run(vla, data)
    assert torch.isfinite(partial['loss'])
    assert torch.isfinite(partial['loss_teacher_radio'])
    partial['loss'].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in vla.parameters() if parameter.grad is not None)

    vla = model()
    data = sample()
    data['teacher_future_valid']['radio'][:] = False
    data['future_valid'] = torch.zeros_like(
        data['teacher_future_valid']['radio'])
    all_invalid = run(vla, data)
    torch.testing.assert_close(
        all_invalid['loss_teacher_radio'], torch.zeros(()))
    assert torch.isfinite(all_invalid['loss'])


def test_padded_current_nans_are_ignored_and_flow_masks_unchanged():
    vla = model()
    data = sample()
    data['teacher_current_valid']['radio'][0, 4:] = False
    data['teacher_current_tokens']['radio'][0, 4:] = float('nan')
    data['reference_valid'] = torch.ones(
        2, 6, dtype=torch.bool)
    data['reference_valid'][0, 4:] = False
    out = run(vla, data)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in vla.parameters() if parameter.grad is not None)


def test_missing_current_conditioning_fails_without_silent_zeros():
    vla = model()
    data = sample()
    with pytest.raises(ValueError, match='Missing teacher RGB'):
        vla(
            reference_tokens=data['reference_tokens'],
            future_tokens=data['future_tokens'],
            text_tokens=data['text_tokens'],
            actions=data['action'],
            states=data['proprio'],
            teacher_future_tokens=data['teacher_future_tokens'],
            teacher_future_valid=data['teacher_future_valid'],
            teacher_future_fingerprints=data[
                'teacher_future_fingerprints'],
        )

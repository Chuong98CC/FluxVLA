"""Fusion, prefix reuse, and gradient tests for teacher-feature PatchWAM."""

import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.third_party_models.patchwam.teacher_forcing import (
    TeacherForcingConfig,
    teacher_representation_fingerprint,
)
from fluxvla.models.third_party_models.patchwam.geometry import (
    raster_coordinates,
)
from fluxvla.models.heads import teacher_forcing_patchwam_head  # noqa: F401
from fluxvla.models.vlas.teacher_forcing_patchwam_vla import (
    TeacherForcingPatchWAMVLA,
)
from fluxvla.models.vlas.patchwam_vla import _TOKEN_KEYS

FLUX_SOURCE = Path(
    '/home/chuong/workspace/wam/ImageWAM/third_party/flux2/src')


SPEC = dict(
    name='radio',
    feature_dim=8,
    representation_id='fake-radio-v1',
    checkpoint_sha256='d' * 64,
    stream='spatial_features',
    preprocessing='unit_range_once',
    input_size=(8, 12),
    native_grid=(2, 3),
    loss_weight=0.1,
)


def teacher_config(**overrides):
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


def head_config(config=None, **overrides):
    values = dict(
        type='TeacherForcingPatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                      text_dim=32, width=32, heads=4, depth=2),
        action_dim=14,
        text_dim=32,
        proprio_dim=14,
        token_dim=128,
        teacher_forcing=config or teacher_config(),
    )
    values.update(overrides)
    return values


def vla(config=None, **overrides):
    values = dict(
        type='TeacherForcingPatchWAMVLA',
        vla_head=head_config(config),
        action_horizon=16,
        num_inference_steps=3,
    )
    values.update(overrides)
    return build_vla_from_cfg(values)


def baseline_vla():
    return build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        vla_head=dict(
            type='PatchWAMHead',
            denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                          text_dim=32, width=32, heads=4, depth=2),
            action_dim=14, text_dim=32, proprio_dim=14, token_dim=128),
        action_horizon=16,
        num_inference_steps=3,
    ))


def copy_common(source, target):
    state = source.state_dict()
    available = target.state_dict()
    common = {name: value for name, value in state.items()
              if name in available and not name.startswith('teacher_')}
    target.load_state_dict(common, strict=False)


def fingerprint():
    spec = TeacherForcingConfig.from_mapping(
        teacher_config()).encoders[0]
    return torch.tensor(list(teacher_representation_fingerprint(spec)),
                        dtype=torch.uint8)


def batch(batch_size=2, length=6, horizon=16):
    generator = torch.Generator().manual_seed(17)
    current = torch.randn(batch_size, length, 8, generator=generator) + 0.2
    future = torch.randn(batch_size, length, 8, generator=generator) + 0.4
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
        'teacher_current_tokens': {'radio': current},
        'teacher_current_valid': {
            'radio': torch.ones(batch_size, length, dtype=torch.bool)},
        'teacher_current_fingerprints': {'radio': fp},
        'teacher_future_tokens': {'radio': future},
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


def teacher_kwargs(data):
    return {
        key: data[key] for key in (
            'teacher_current_tokens', 'teacher_current_valid',
            'teacher_current_fingerprints', 'teacher_future_tokens',
            'teacher_future_valid', 'teacher_future_fingerprints',
            'current_raster_layout', 'future_raster_layout',
            'current_raster_length',
        )
    }


def fixed_noise(length=6, horizon=16, batch_size=2):
    generator = torch.Generator().manual_seed(29)
    return (torch.randn(batch_size, length, 128, generator=generator),
            torch.randn(batch_size, horizon, 128, generator=generator))


def test_registered_vla_builds_teacher_modules_with_zero_maps():
    model = vla()
    assert isinstance(model, TeacherForcingPatchWAMVLA)
    head = model.vla_head
    assert head.teacher_current_projectors['radio'].weight.shape == (32, 8)
    assert head.teacher_conditioners['radio'].weight.shape == (32, 8)
    assert torch.count_nonzero(
        head.teacher_current_projectors['radio'].weight) == 0
    assert torch.count_nonzero(
        head.teacher_conditioners['radio'].weight) == 0
    assert head.teacher_predictors['radio'].weight.shape == (8, 32)
    assert head.teacher_contract_digest.shape == (32,)


def test_base_token_keys_remain_teacher_free():
    teacher_keys = {
        'teacher_current_tokens', 'teacher_future_tokens',
        'teacher_current_valid', 'teacher_future_valid',
        'teacher_current_fingerprints', 'teacher_future_fingerprints',
    }
    assert teacher_keys.isdisjoint(_TOKEN_KEYS)
    assert teacher_keys <= set(TeacherForcingPatchWAMVLA.TOKEN_KEYS)
    assert 'current_raster_layout' in _TOKEN_KEYS
    assert 'future_raster_layout' in _TOKEN_KEYS


def test_lookahead_mode_restores_exact_descendant_flags_and_exceptions():
    model = vla()
    head = model.vla_head
    denoiser = head.denoiser
    denoiser.train()
    denoiser.blocks[0].eval()
    denoiser.blocks[1].train()
    predictor = head.teacher_predictors['radio']
    predictor.eval()
    with head._teacher_lookahead_mode():
        assert denoiser.training is False
        assert denoiser.blocks[0].training is False
        assert denoiser.blocks[1].training is False
        assert predictor.training is False
    assert denoiser.training is True
    assert denoiser.blocks[0].training is False
    assert denoiser.blocks[1].training is True
    assert predictor.training is False

    with pytest.raises(RuntimeError, match='lookahead failure'):
        with head._teacher_lookahead_mode():
            raise RuntimeError('lookahead failure')
    assert denoiser.training is True
    assert denoiser.blocks[0].training is False
    assert denoiser.blocks[1].training is True


def test_teacher_head_rejects_unsupported_ema_and_token_sigma_paths():
    model = vla()
    data = batch()
    with pytest.raises(ValueError, match='representation_only'):
        model.vla_head(
            data, representation_only=True, representation_layer=1)
    with pytest.raises(ValueError, match='teacher_sampling'):
        model.vla_head(data, teacher_sampling=True)
    with pytest.raises(ValueError, match='token_sigma'):
        model.vla_head(
            data, token_sigma=torch.zeros(2, 6 + 16))


def test_prepared_prefix_matches_ordinary_projection():
    model = vla()
    denoiser = model.vla_head.denoiser
    reference = torch.randn(2, 6, 128)
    noisy = torch.randn(2, 6 + 3, 128)
    context = torch.randn(2, 4, 32)
    sigma = torch.tensor([0.2, 0.7])
    reference_ids = torch.zeros(2, 6, 4)
    noisy_ids = torch.zeros(2, 9, 4)
    context_ids = torch.zeros(2, 4, 4)
    visibility = torch.ones(2, 1, 19, 19, dtype=torch.bool)
    ordinary = denoiser(
        reference, noisy, sigma, context, reference_ids=reference_ids,
        noisy_ids=noisy_ids, context_ids=context_ids,
        visibility=visibility)
    context_hidden = denoiser.text_input(context)
    reference_hidden = denoiser.token_input(reference)
    from fluxvla.models.third_party_models.patchwam.conditioning import (
        PreparedDenoiserPrefix,
    )
    prefix = PreparedDenoiserPrefix(
        reference_hidden=reference_hidden, context_hidden=context_hidden)
    prepared = denoiser(
        reference, noisy, sigma, context, reference_ids=reference_ids,
        noisy_ids=noisy_ids, context_ids=context_ids,
        visibility=visibility, prepared_prefix=prefix)
    torch.testing.assert_close(ordinary, prepared, rtol=1e-5, atol=1e-6)


def test_zero_maps_preserve_baseline_flow_losses():
    teacher = vla()
    baseline = baseline_vla()
    copy_common(teacher, baseline)
    sample = batch()
    sigma = torch.tensor([0.25, 0.75])
    future_noise, action_noise = fixed_noise()
    common = dict(
        reference_tokens=sample['reference_tokens'],
        future_tokens=sample['future_tokens'],
        text_tokens=sample['text_tokens'],
        actions=sample['action'],
        states=sample['proprio'],
        sigma=sigma,
        future_noise=future_noise,
        action_noise=action_noise,
        reference_ids=sample['reference_ids'],
        future_ids=sample['future_ids'],
    )
    base_out = baseline(**common)
    teacher_out = teacher(
        **common,
        **teacher_kwargs(sample),
    )
    torch.testing.assert_close(
        teacher_out['loss_video'], base_out['loss_video'],
        rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        teacher_out['loss_action'], base_out['loss_action'],
        rtol=1e-5, atol=1e-6)
    assert 'loss_teacher_radio' in teacher_out


def test_two_denoiser_passes_reuse_noisy_inputs_and_prefix():
    model = vla()
    with torch.no_grad():
        model.vla_head.teacher_conditioners['radio'].weight.normal_()
    sample = batch()
    sigma = torch.tensor([0.3, 0.6])
    future_noise, action_noise = fixed_noise()
    seen = []
    original = model.vla_head.denoiser.forward

    def recording(*args, **kwargs):
        seen.append({
            'reference': args[0].detach().clone(),
            'noisy': args[1].detach().clone(),
            'sigma': args[2].detach().clone(),
            'prefix': kwargs.get('prepared_prefix'),
            'residual': kwargs.get('future_hidden_residual'),
            'grad': torch.is_grad_enabled(),
        })
        return original(*args, **kwargs)

    with patch.object(model.vla_head.denoiser, 'forward', recording):
        out = model(
            reference_tokens=sample['reference_tokens'],
            future_tokens=sample['future_tokens'],
            text_tokens=sample['text_tokens'],
            actions=sample['action'],
            states=sample['proprio'],
            sigma=sigma, future_noise=future_noise,
            action_noise=action_noise,
            **teacher_kwargs(sample),
        )
    assert len(seen) == 2
    assert seen[0]['grad'] is False
    assert seen[1]['grad'] is True
    torch.testing.assert_close(seen[0]['noisy'], seen[1]['noisy'])
    torch.testing.assert_close(seen[0]['sigma'], seen[1]['sigma'])
    assert seen[0]['prefix'] is seen[1]['prefix']
    assert torch.count_nonzero(seen[0]['residual']) == 0
    assert torch.count_nonzero(seen[1]['residual']) > 0
    assert torch.isfinite(out['loss'])


def test_gradients_reach_teacher_and_shared_trainable_projections():
    model = vla()
    sample = batch()
    out = model(
        reference_tokens=sample['reference_tokens'],
        future_tokens=sample['future_tokens'],
        text_tokens=sample['text_tokens'],
        actions=sample['action'],
        states=sample['proprio'],
        sigma=torch.tensor([0.4, 0.6]),
        **teacher_kwargs(sample),
    )
    out['loss'].backward()
    head = model.vla_head
    assert head.teacher_current_projectors['radio'].weight.grad is not None
    assert head.teacher_current_projectors['radio'].weight.grad.abs().sum() > 0
    assert head.teacher_conditioners['radio'].weight.grad is not None
    assert head.teacher_conditioners['radio'].weight.grad.abs().sum() > 0
    assert head.teacher_predictors['radio'].weight.grad is not None
    assert head.teacher_predictors['radio'].weight.grad.abs().sum() > 0
    assert head.denoiser.token_input.weight.grad is not None
    assert head.state_projection.weight.grad is not None
    assert sample['teacher_current_tokens']['radio'].grad is None


def test_current_residual_covers_only_known_raster_then_history_zeros():
    model = vla()
    sample = batch()
    head = model.vla_head
    encoded = head._teacher_validate_encoded(sample, target=True)
    residual = head._teacher_current_residual(encoded, reference_length=8)
    assert residual.shape == (2, 8, 32)
    assert torch.count_nonzero(residual[:, 6:]) == 0


def test_optional_scaled_xavier_initialization_is_independent():
    cfg = teacher_config(
        current_projector_init='scaled_xavier_uniform',
        current_projector_init_scale=0.5,
        feedback_projector_init='scaled_xavier_uniform',
        feedback_projector_init_scale=0.25,
    )
    model = vla(cfg)
    head = model.vla_head
    assert torch.count_nonzero(
        head.teacher_current_projectors['radio'].weight) > 0
    assert torch.count_nonzero(
        head.teacher_conditioners['radio'].weight) > 0
    assert not torch.equal(
        head.teacher_current_projectors['radio'].weight,
        head.teacher_conditioners['radio'].weight)


@pytest.mark.skipif(
    not (FLUX_SOURCE / 'flux2' / 'model.py').is_file(),
    reason='local official flux2 source is unavailable')
def test_official_flux_prepared_prefix_and_future_residual():
    if str(FLUX_SOURCE) not in sys.path:
        sys.path.insert(0, str(FLUX_SOURCE))
    import flux2.model as model_api
    from fluxvla.models.third_party_models.patchwam.conditioning import (
        PreparedDenoiserPrefix,
    )
    from fluxvla.models.third_party_models.patchwam.flux import (
        OfficialFluxDenoiser,
    )

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
    denoiser = OfficialFluxDenoiser(model_api.Flux2(params))
    reference = torch.randn(2, 3, 128)
    noisy = torch.randn(2, 2 + 2, 128)
    context = torch.randn(2, 4, 48)
    sigma = torch.tensor([0.3, 0.8])
    reference_ids = torch.zeros(2, 3, 4)
    noisy_ids = torch.zeros(2, 4, 4)
    context_ids = torch.zeros(2, 4, 4)
    visibility = torch.ones(2, 1, 11, 11, dtype=torch.bool)
    ordinary = denoiser(
        reference, noisy, sigma, context, reference_ids=reference_ids,
        noisy_ids=noisy_ids, context_ids=context_ids,
        visibility=visibility)
    prefix = denoiser.prepare_prefix(reference, context)
    prepared = denoiser(
        reference, noisy, sigma, context, reference_ids=reference_ids,
        noisy_ids=noisy_ids, context_ids=context_ids,
        visibility=visibility, prepared_prefix=prefix)
    torch.testing.assert_close(ordinary, prepared, rtol=1e-5, atol=1e-6)

    residual = torch.randn(2, 2, 64)
    with_residual = denoiser(
        reference, noisy, sigma, context, reference_ids=reference_ids,
        noisy_ids=noisy_ids, context_ids=context_ids,
        visibility=visibility, prepared_prefix=prefix,
        future_hidden_residual=residual)
    assert not torch.allclose(with_residual, prepared)
    with_residual.square().mean().backward()
    assert denoiser.transformer.img_in.weight.grad is not None

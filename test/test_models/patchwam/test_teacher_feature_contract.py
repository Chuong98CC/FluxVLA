"""CPU contracts for teacher-feature configuration, predictors and losses."""

import math

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from fluxvla.models.third_party_models.patchwam.teacher_forcing import (
    TeacherEncoderSpec,
    TeacherForcingConfig,
    build_feature_predictor,
    masked_teacher_cosine_loss,
    prepare_teacher_fusion_input,
    teacher_policy_digest,
    teacher_representation_fingerprint,
)


def spec(**overrides):
    values = dict(
        name='radio',
        feature_dim=8,
        representation_id='c-radio-v4-h@local-review',
        checkpoint_sha256='a' * 64,
        stream='spatial_features',
        preprocessing='world_rgb_minus_one_to_one_to_unit_range',
        input_size=(8, 12),
        native_grid=(2, 3),
        view_context='independent_rasters',
        view_order='view_major_row_major',
        alignment='native_exact',
        loss_weight=0.1,
    )
    values.update(overrides)
    return TeacherEncoderSpec(**values)


def config(**overrides):
    values = dict(
        encoders=[spec()],
        current_conditioning=True,
        future_prediction=True,
        future_feedback=True,
        predictor=dict(type='repa_mlp'),
        self_cond_drop_prob=0.1,
        true_feature_prob=0.05,
    )
    values.update(overrides)
    return TeacherForcingConfig.from_mapping(values)


@pytest.mark.parametrize(
    'overrides,match',
    [
        (dict(name='1bad'), 'name'),
        (dict(feature_dim=0), 'feature_dim'),
        (dict(loss_weight=-0.1), 'loss_weight'),
        (dict(alignment='interpolate'), 'native_exact'),
        (dict(view_context='joint_views'), 'independent_rasters'),
        (dict(native_grid=(0, 3)), 'native_grid'),
    ],
)
def test_spec_rejects_invalid_contracts(overrides, match):
    with pytest.raises(ValueError, match=match):
        spec(**overrides)


@pytest.mark.parametrize(
    'overrides,match',
    [
        (dict(encoders=[]), 'encoder'),
        (dict(encoders=[spec(), spec()]), 'unique'),
        (dict(self_cond_drop_prob=math.nan), 'self_cond_drop_prob'),
        (dict(self_cond_drop_prob=1.1), 'self_cond_drop_prob'),
        (dict(true_feature_prob=-0.1), 'true_feature_prob'),
        (dict(future_prediction=False, future_feedback=True), 'feedback'),
        (dict(current_conditioning=False, future_prediction=False,
              future_feedback=False), 'least one'),
        (dict(predictor=dict(type='mlp2', hidden_dim=0)), 'hidden_dim'),
    ],
)
def test_config_rejects_invalid_contracts(overrides, match):
    with pytest.raises(ValueError, match=match):
        config(**overrides)


def test_repa_predictor_matches_source_architecture_and_initialization():
    predictor = build_feature_predictor(
        dict(type='repa_mlp'), input_dim=6, output_dim=4)
    linears = [m for m in predictor.modules() if isinstance(m, nn.Linear)]
    assert [m.in_features for m in linears] == [6, 2048, 2048]
    assert [m.out_features for m in linears] == [2048, 2048, 4]
    assert len([m for m in predictor.modules() if isinstance(m, nn.SiLU)]) == 2
    assert not any(isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.Dropout))
                   for m in predictor.modules())
    assert all(layer.bias is not None for layer in linears)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linears)
    # Xavier uniform: variance ~= 2 / (fan_in + fan_out).
    for layer in linears:
        expected = math.sqrt(6 / (layer.in_features + layer.out_features))
        assert float(layer.weight.detach().abs().max()) <= expected + 1e-6


def test_linear_and_mlp2_alternatives_and_parameter_counts():
    linear = build_feature_predictor(
        dict(type='linear'), input_dim=6, output_dim=4)
    assert isinstance(linear, nn.Linear)
    assert linear.weight.shape == (4, 6)

    mlp2 = build_feature_predictor(
        dict(type='mlp2', hidden_dim=5), input_dim=6, output_dim=4)
    linears = [m for m in mlp2.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (6, 5), (5, 4)]
    assert sum(p.numel() for p in mlp2.parameters()) == (
        6 * 5 + 5 + 5 * 4 + 4)


def test_feature_normalization_handles_masks_half_precision_and_zero():
    norm = nn.LayerNorm(2, elementwise_affine=False)
    tokens = torch.tensor([[[3.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, False]])
    prepared = prepare_teacher_fusion_input(tokens, valid, norm=norm)
    torch.testing.assert_close(prepared[0, 0], torch.tensor([1.0, -1.0]))
    assert torch.count_nonzero(prepared[0, 1]) == 0

    half = prepare_teacher_fusion_input(
        tokens.half(), valid, norm=nn.LayerNorm(
            2, elementwise_affine=False, eps=1e-6))
    assert half.dtype == torch.float16
    assert torch.isfinite(half).all()
    assert torch.count_nonzero(half[0, 1]) == 0


def test_valid_observed_nonfinite_or_zero_descriptors_are_rejected():
    norm = nn.LayerNorm(2, elementwise_affine=False)
    with pytest.raises(ValueError, match='finite'):
        prepare_teacher_fusion_input(
            torch.tensor([[[float('nan'), 1.0]]]),
            torch.tensor([[True]]), norm=norm)
    with pytest.raises(ValueError, match='zero'):
        prepare_teacher_fusion_input(
            torch.zeros(1, 1, 2), torch.tensor([[True]]), norm=norm)


def test_masked_cosine_values_reduction_and_gradients():
    prediction = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[[0.0, 2.0], [2.0, 0.0]], [[3.0, 0.0], [0.0, 1.0]]])
    valid = torch.tensor([[True, True], [False, False]])
    loss = masked_teacher_cosine_loss(prediction, target, valid)
    # One fully invalid sample contributes zero; the other has two
    # orthogonal tokens, so its per-sample mean is one.
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[0].abs().sum() > 0
    assert prediction.grad[1].abs().sum() == 0
    assert target.grad is None

    all_invalid = masked_teacher_cosine_loss(
        prediction.detach()[1:], target[1:], torch.zeros(1, 2, dtype=torch.bool))
    torch.testing.assert_close(all_invalid, torch.zeros(()))


def test_predictor_matches_l2_cosine_definition():
    prediction = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]],
                              requires_grad=True)
    target = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])
    valid = torch.tensor([[True, True]])
    actual = masked_teacher_cosine_loss(prediction, target, valid)
    expected = 1 - F.cosine_similarity(
        F.normalize(prediction, dim=-1), F.normalize(target, dim=-1), dim=-1)
    torch.testing.assert_close(actual, expected.mean())


def test_representation_fingerprint_ignores_runtime_details_and_loss_weight():
    base = spec()
    same = spec(loss_weight=0.3)
    assert teacher_representation_fingerprint(base) == (
        teacher_representation_fingerprint(same))
    assert len(teacher_representation_fingerprint(base)) == 32
    changed = spec(checkpoint_sha256='b' * 64)
    assert teacher_representation_fingerprint(base) != (
        teacher_representation_fingerprint(changed))


def test_policy_digest_changes_for_architecture_and_identity_only():
    base = config()
    changed_predictor = config(predictor=dict(type='linear'))
    changed_switch = config(current_conditioning=False)
    changed_weight = config(encoders=[spec(loss_weight=0.4)])
    changed_identity = config(
        encoders=[spec(checkpoint_sha256='b' * 64)])
    assert teacher_policy_digest(base) != teacher_policy_digest(
        changed_predictor)
    assert teacher_policy_digest(base) != teacher_policy_digest(
        changed_switch)
    assert teacher_policy_digest(base) == teacher_policy_digest(changed_weight)
    assert teacher_policy_digest(base) != teacher_policy_digest(
        changed_identity)

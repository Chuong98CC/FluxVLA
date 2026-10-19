"""CPU regression tests for the PatchWAM RoboTwin FK pose objective."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from fluxvla.engines import build_head_from_cfg, build_vla_from_cfg
from fluxvla.models.third_party_models.patchwam.codec import (
    RepeatedActionCodec)
from fluxvla.models.third_party_models.patchwam.fk_pose import (
    ALOHA_ORIENTATION_CORRECTION,
    ALOHA_TCP_OFFSET_M,
    AlohaFk,
    FkPoseObjective,
    FkPoseStatistics,
    PoseBoundsAccumulator,
    compute_fk_pose_loss,
    default_aloha_urdf_path,
    episode_pose_activity,
    file_sha256,
)
from fluxvla.models.third_party_models.patchwam.flow import masked_sample_mse
from fluxvla.models.heads import patchwam_head  # noqa: F401
from fluxvla.models.heads import teacher_forcing_patchwam_head  # noqa: F401
from fluxvla.models.vlas import patchwam_vla  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[3]
URDF = REPO_ROOT / (
    'src/RoboTwin/assets/embodiments/aloha-agilex/urdf/'
    'arx5_description_isaac.urdf')
_FORCE_MISSING_FK_ASSETS = (
    os.environ.get('FLUXVLA_TEST_FORCE_MISSING_FK_ASSETS') == '1')


def _runtime_available():
    return (not _FORCE_MISSING_FK_ASSETS
            and importlib.util.find_spec('pytorch_kinematics') is not None)


def _asset_available():
    return not _FORCE_MISSING_FK_ASSETS and URDF.is_file()


def _fk_skip_reason():
    missing = []
    if not _runtime_available():
        missing.append('pytorch_kinematics')
    if not _asset_available():
        missing.append(str(URDF))
    return None if not missing else 'Missing optional FK test dependency: ' + (
        ', '.join(missing))


requires_asset = pytest.mark.skipif(
    not _asset_available(),
    reason=f'Missing optional ALOHA URDF asset: {URDF}')
requires_fk = pytest.mark.skipif(
    _fk_skip_reason() is not None, reason=_fk_skip_reason() or '')


def test_optional_fk_skip_reason_names_missing_parts(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__], '_runtime_available', lambda: False)
    monkeypatch.setattr(
        sys.modules[__name__], '_asset_available', lambda: False)
    reason = _fk_skip_reason()
    assert 'pytorch_kinematics' in reason
    assert str(URDF) in reason


@pytest.fixture(scope='module')
def aloha() -> AlohaFk:
    return AlohaFk(URDF, dtype=torch.float64)


def _stats_payload(root: str = '/tmp/fk-root',
                   episodes=(0,), *, digest: str = 'a' * 64) -> dict:
    return {
        'schema_version': 1,
        'kind': 'patchwam_aloha_fk_pose_stats',
        'robot': {
            'urdf_path': str(URDF),
            'urdf_sha256': digest,
            'root_link': 'footprint',
            'end_links': {'left': 'fl_link6', 'right': 'fr_link6'},
            'joint_names': {
                'left': [f'fl_joint{i}' for i in range(1, 7)],
                'right': [f'fr_joint{i}' for i in range(1, 7)],
            },
        },
        'tcp': {
            'offset_m': ALOHA_TCP_OFFSET_M,
            'local_axis': [1.0, 0.0, 0.0],
            'orientation_correction': [
                list(row) for row in ALOHA_ORIENTATION_CORRECTION],
        },
        'action_layout': {
            'dim': 14,
            'arms': {'left': [0, 6], 'right': [7, 13]},
            'grippers': [6, 13],
        },
        'thresholds': {'translation_spread_m': 0.01,
                       'orientation_spread': 0.05},
        'training': {'roots': [root], 'num_episodes': len(episodes)},
        'pose_bounds': {
            arm: {'min': [-3.0, -3.0, -3.0],
                  'max': [3.0, 3.0, 3.0],
                  'center': [0.0, 0.0, 0.0],
                  'scale': 3.0}
            for arm in ('left', 'right')
        },
        'episodes': {
            root: {
                str(episode): {
                    'active_arm': [True, True],
                    'translation_spread_m': [0.1, 0.1],
                    'orientation_spread': [0.1, 0.1],
                } for episode in episodes
            },
        },
    }


def _write_stats(path: Path, **kwargs) -> Path:
    kwargs.setdefault('digest', file_sha256(URDF))
    path.write_text(json.dumps(_stats_payload(**kwargs)), encoding='utf-8')
    return path


def _loss_inputs(batch: int = 2, horizon: int = 4, copies: int = 9):
    target = torch.zeros(batch, horizon, 14)
    predicted = target[:, :, None, :].expand(
        batch, horizon, copies, 14).clone().requires_grad_(True)
    return (
        predicted,
        target,
        torch.ones(batch, 2, dtype=torch.bool),
        torch.ones(batch, horizon, 14, dtype=torch.bool),
        torch.zeros(batch, 14),
        torch.ones(batch, 14),
        torch.zeros(2, 3),
        torch.ones(2),
    )


@pytest.mark.parametrize('token_dim,scale',
                         [(56, 0.5), (112, 1.75), (128, 0.25)])
def test_codec_unpack_is_coordinate_major(token_dim, scale):
    codec = RepeatedActionCodec(14, token_dim, scale=scale)
    actions = torch.randn(2, 5, 14)
    tokens = codec.encode(actions)
    unpacked = codec.unpack(tokens)
    assert unpacked.shape == (2, 5, codec.copies, 14)
    torch.testing.assert_close(
        unpacked.mean(dim=-2), codec.decode(tokens), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        unpacked, actions[:, :, None, :].expand_as(unpacked))
    poisoned = tokens.clone()
    poisoned[..., codec.occupied:] = float('nan')
    torch.testing.assert_close(codec.unpack(poisoned), unpacked)


@requires_fk
def test_aloha_chain_and_robotwin_tcp_transform(aloha):
    zero = torch.zeros(1, 6, dtype=torch.float64)
    correction = torch.tensor(
        ALOHA_ORIENTATION_CORRECTION, dtype=torch.float64)
    for arm, expected in (
            ('left', (0.4562, 0.3003, 0.9420)),
            ('right', (0.4572, -0.3052, 0.9410))):
        position, rotation = aloha.tcp_pose(zero, arm)
        torch.testing.assert_close(
            position[0], torch.tensor(expected, dtype=torch.float64),
            rtol=0, atol=2e-3)
        chain = aloha._chains[arm]
        link = chain.forward_kinematics_tensor(zero)[
            aloha._end_indices[arm], 0]
        torch.testing.assert_close(
            rotation[0], link[:3, :3] @ correction, rtol=0, atol=1e-9)


@requires_fk
def test_tcp_offset_is_along_the_corrected_local_x_axis(aloha):
    zero = torch.zeros(1, 6, dtype=torch.float64)
    position, rotation = aloha.tcp_pose(zero, 'left')
    # The correction flips y/z, so local x is unchanged and the TCP offset is
    # exactly 0.12 m along the endpoint x axis.
    chain = aloha._chains['left']
    index = aloha._end_indices['left']
    link = chain.forward_kinematics_tensor(zero)[index, 0]
    torch.testing.assert_close(
        position[0] - link[:3, 3],
        rotation[0] @ torch.tensor([0.12, 0.0, 0.0],
                                   dtype=torch.float64))
    assert torch.isfinite(rotation).all()


@requires_asset
def test_default_urdf_resolution_respects_robotwin_root(tmp_path):
    assert default_aloha_urdf_path().is_file()
    assert default_aloha_urdf_path(
        robotwin_root=str(REPO_ROOT / 'src/RoboTwin')) == URDF


@requires_fk
def test_identity_actions_have_zero_pose_loss(aloha):
    inputs = _loss_inputs()
    output = compute_fk_pose_loss(
        aloha, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
        inputs[5], 1e-8, inputs[6], inputs[7])
    zero = torch.zeros((), dtype=output['loss_fk'].dtype)
    torch.testing.assert_close(output['loss_fk'], zero)
    torch.testing.assert_close(output['loss_fk_position'], zero)
    torch.testing.assert_close(output['loss_fk_rotation'], zero)
    assert float(output['loss_fk_active_arm_timestep_pairs']) == 16
    assert float(output['loss_fk_active_arm_timestep_copy_tuples']) == 144


@requires_fk
def test_opposite_copies_cancel_mean_but_not_fk(aloha):
    predicted, target, active, valid, mean, std, center, scale = (
        _loss_inputs(batch=1, horizon=1, copies=8))
    with torch.no_grad():
        predicted[..., 0, 0] = 0.25
        predicted[..., 1, 0] = -0.25
    output = compute_fk_pose_loss(
        aloha, predicted, target, active, valid, mean, std, 1e-8,
        center, scale)
    mean_mse = masked_sample_mse(
        predicted.mean(dim=-2), target, valid).mean()
    torch.testing.assert_close(mean_mse, torch.zeros(()), atol=1e-7,
                               rtol=0)
    assert float(output['loss_fk'].detach()) > 0
    output['loss_fk'].backward()
    grad = predicted.grad[0, 0, :, 0]
    assert torch.isfinite(grad).all()
    assert grad[0].abs() > 0 and grad[1].abs() > 0
    assert grad[0].sign() != grad[1].sign()
    assert grad[2:].abs().sum() == 0


@requires_fk
def test_wrist_rotation_and_upstream_position_sensitivity(aloha):
    predicted, target, active, valid, mean, std, center, scale = (
        _loss_inputs(batch=1, horizon=1, copies=2))
    wrist = predicted.clone()
    with torch.no_grad():
        wrist[..., 5] = 0.2
    wrist_out = compute_fk_pose_loss(
        aloha, wrist, target, active, valid, mean, std, 1e-8, center, scale)
    assert float(wrist_out['loss_fk_rotation'].detach()) > 0

    upstream = predicted.clone()
    with torch.no_grad():
        upstream[..., 1] = 0.2
    upstream_out = compute_fk_pose_loss(
        aloha, upstream, target, active, valid, mean, std, 1e-8,
        center, scale)
    assert float(upstream_out['loss_fk_position'].detach()) > 0


@requires_fk
def test_inactive_and_padded_values_cannot_contaminate_fk(aloha):
    predicted, target, active, valid, mean, std, center, scale = (
        _loss_inputs(batch=2, horizon=3, copies=9))
    active[1] = False
    valid[0, -1] = False
    valid[0, -1, :2] = True
    with torch.no_grad():
        predicted[1] = float('nan')
        predicted[0, -1] = float('nan')
        target[1] = float('nan')
        target[0, -1] = float('nan')
        target[0, 0, 0] = 0.1
    output = compute_fk_pose_loss(
        aloha, predicted, target, active, valid, mean, std, 1e-8,
        center, scale)
    assert torch.isfinite(output['loss_fk']).all()
    output['loss_fk'].backward()
    assert predicted.grad[1].abs().sum() == 0
    assert predicted.grad[0, -1].abs().sum() == 0
    assert predicted.grad[0, 0].abs().sum() > 0


@requires_fk
def test_all_invalid_batch_returns_differentiable_zero(aloha):
    predicted, target, active, valid, mean, std, center, scale = (
        _loss_inputs(batch=2, horizon=2, copies=9))
    active[:] = False
    with torch.no_grad():
        predicted.fill_(float('nan'))
        target.fill_(float('nan'))
    output = compute_fk_pose_loss(
        aloha, predicted, target, active, valid, mean, std, 1e-8,
        center, scale)
    assert torch.isfinite(output['loss_fk']).all()
    output['loss_fk'].backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad.abs().sum() == 0
    assert float(output['loss_fk_active_arm_timestep_pairs']) == 0


@requires_fk
def test_episode_activity_labels_single_dual_stationary_and_wrist(aloha):
    stationary = torch.zeros(12, 14, dtype=torch.float64)
    result = episode_pose_activity(stationary, aloha)
    assert result.active_arm == (False, False)

    left = stationary.clone()
    left[:, 1] = torch.linspace(-0.3, 0.3, 12, dtype=torch.float64)
    result = episode_pose_activity(left, aloha)
    assert result.active_arm == (True, False)

    both = left.clone()
    both[:, 8] = torch.linspace(0.3, -0.3, 12, dtype=torch.float64)
    result = episode_pose_activity(both, aloha)
    assert result.active_arm == (True, True)

    wrist = stationary.clone()
    wrist[:, 5] = torch.linspace(-0.4, 0.4, 12, dtype=torch.float64)
    result = episode_pose_activity(wrist, aloha)
    assert result.active_arm == (True, False)
    assert result.translation_spread_m[0] < 0.01
    assert result.orientation_spread[0] > 0.05


@requires_fk
def test_pose_bounds_ignore_inactive_frames(aloha):
    near = torch.zeros(4, 6, dtype=torch.float64)
    near[:, 0] = torch.linspace(0.0, 0.03, 4, dtype=torch.float64)
    far = near.clone()
    far[:, 1] = 1.4
    near_position, _ = aloha.tcp_pose(near, 'left')
    right_position, _ = aloha.tcp_pose(near, 'right')
    far_position, _ = aloha.tcp_pose(far, 'left')
    accumulator = PoseBoundsAccumulator()
    accumulator.add(0, near_position)
    accumulator.add(1, right_position)
    bounds = accumulator.finalize()
    assert bounds['left']['min'][0] <= near_position[:, 0].min().item()
    assert bounds['left']['max'][0] >= near_position[:, 0].max().item()
    low = torch.tensor(bounds['left']['min'], dtype=torch.float64)
    high = torch.tensor(bounds['left']['max'], dtype=torch.float64)
    inside = ((far_position >= low) & (far_position <= high)).all(dim=-1)
    assert not inside.any()


def test_statistics_loader_rejects_malformed_and_ambiguous_artifacts():
    payload = _stats_payload()
    stats = FkPoseStatistics.from_mapping(payload)
    assert stats.active_arm('/tmp/fk-root', 0) == (True, True)

    bad = copy.deepcopy(payload)
    bad['episodes']['/tmp/fk-root']['0']['active_arm'] = [True, 1]
    with pytest.raises(ValueError, match='active_arm'):
        FkPoseStatistics.from_mapping(bad)

    bad = copy.deepcopy(payload)
    bad['training']['roots'] = ['/tmp/fk-root', '/tmp/fk-root/']
    with pytest.raises(ValueError, match='canonical root twice'):
        FkPoseStatistics.from_mapping(bad)

    bad = copy.deepcopy(payload)
    bad['episodes']['/tmp/fk-root/.'] = bad['episodes']['/tmp/fk-root']
    with pytest.raises(ValueError, match='ambiguous root'):
        FkPoseStatistics.from_mapping(bad)

    bad = copy.deepcopy(payload)
    bad['pose_bounds']['left']['center'] = [0.1, 0.0, 0.0]
    with pytest.raises(ValueError, match='midpoint'):
        FkPoseStatistics.from_mapping(bad)

    with pytest.raises(KeyError, match='No FK episode label'):
        stats.active_arm('/tmp/fk-root', 99)


@requires_asset
def test_robot_identity_mismatch_is_rejected(tmp_path):
    payload = _stats_payload()
    payload['robot']['urdf_sha256'] = 'a' * 64
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='robot identity'):
        FkPoseObjective.from_path(path, urdf_path=str(URDF))


def _tiny_head(**overrides):
    cfg = dict(
        type='PatchWAMHead',
        denoiser=dict(type='PatchWAMTinyDenoiser', token_dim=128,
                      text_dim=32, width=32, heads=4, depth=2),
        action_dim=14,
        text_dim=32,
        proprio_dim=14,
        token_dim=128,
    )
    cfg.update(overrides)
    return cfg


def _batch(batch: int = 2, horizon: int = 16):
    return {
        'reference_tokens': torch.randn(batch, 4, 128),
        'future_tokens': torch.randn(batch, 4, 128),
        'text_tokens': torch.randn(batch, 3, 32),
        'action': torch.randn(batch, horizon, 14),
        'proprio': torch.randn(batch, horizon, 14),
    }


@requires_fk
def test_head_fk_only_backpropagates_to_model_parameters(tmp_path):
    stats = _write_stats(tmp_path / 'stats.json')
    head = build_head_from_cfg(_tiny_head(
        video_weight=0.0,
        action_weight=1.0,
        action_loss=dict(token_velocity_weight=0.0,
                         mean_clean_weight=0.0),
        fk_loss=dict(weight=1.0, stats_path=str(stats)),
    ))
    batch = _batch()
    batch.update({
        'active_arm': torch.tensor([[True, True], [True, False]]),
        'fk_action_mean': torch.zeros(2, 14),
        'fk_action_std': torch.ones(2, 14),
        'fk_action_epsilon': torch.full((2,), 1e-8),
    })
    output = head(batch, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(output['loss'], output['loss_fk'])
    assert float(output['loss_fk'].detach()) > 0
    output['loss'].backward()
    gradient = head.denoiser.token_input.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_head_mean_clean_matches_codec_decode(tmp_path):
    head = build_head_from_cfg(_tiny_head(
        action_loss=dict(token_velocity_weight=0.0,
                         mean_clean_weight=1.0)))
    batch = _batch(batch=1, horizon=2)
    desired = torch.randn(1, 2, 9, 14)

    def fake_velocity(batch, x, u, sigma, **kwargs):
        tokens = torch.zeros_like(u)
        tokens[..., :head.codec.occupied] = (
            desired.transpose(-1, -2).reshape(
                1, 2, head.codec.occupied) * head.codec.scale)
        return torch.zeros_like(x), u - tokens

    head._velocity = fake_velocity
    output = head(
        batch, sigma=torch.ones(1),
        generator=torch.Generator().manual_seed(4))
    expected = masked_sample_mse(
        desired.mean(dim=-2), batch['action'],
        torch.ones(1, 2, 14, dtype=torch.bool)).mean()
    torch.testing.assert_close(
        output['loss_action_mean_clean'], expected)
    torch.testing.assert_close(
        output['loss_action_mean_clean'],
        masked_sample_mse(
            head.codec.decode(
                head.codec.encode(desired.mean(dim=-2))),
            batch['action'],
            torch.ones(1, 2, 14, dtype=torch.bool)).mean())


@requires_fk
def test_head_opposite_copies_do_not_cancel_fk(tmp_path):
    stats = _write_stats(tmp_path / 'stats.json')
    head = build_head_from_cfg(_tiny_head(
        video_weight=0.0,
        action_weight=1.0,
        action_loss=dict(token_velocity_weight=0.0,
                         mean_clean_weight=1.0),
        fk_loss=dict(weight=1.0, stats_path=str(stats)),
    ))
    batch = _batch(batch=1, horizon=1)
    batch['action'].zero_()
    desired = torch.zeros(1, 1, 9, 14)
    desired[..., 0, 0] = 0.2
    desired[..., 4, 0] = -0.2

    def fake_velocity(batch, x, u, sigma, **kwargs):
        tokens = torch.zeros_like(u)
        tokens[..., :head.codec.occupied] = (
            desired.transpose(-1, -2).reshape(
                1, 1, head.codec.occupied) * head.codec.scale)
        return torch.zeros_like(x), u - tokens

    head._velocity = fake_velocity
    batch.update({
        'active_arm': torch.ones(1, 2, dtype=torch.bool),
        'fk_action_mean': torch.zeros(1, 14),
        'fk_action_std': torch.ones(1, 14),
        'fk_action_epsilon': torch.full((1,), 1e-8),
    })
    output = head(
        batch, sigma=torch.ones(1),
        generator=torch.Generator().manual_seed(4))
    assert float(output['loss_action_mean_clean']) < 1e-10
    assert float(output['loss_fk'].detach()) > 0
    torch.testing.assert_close(
        output['loss'], output['loss_action'] + output['loss_fk'])


@requires_fk
def test_weighted_action_and_fk_total_is_composed_once(tmp_path):
    stats = _write_stats(tmp_path / 'stats.json')
    head = build_head_from_cfg(_tiny_head(
        video_weight=0.0,
        action_weight=2.0,
        action_loss=dict(token_velocity_weight=3.0,
                         mean_clean_weight=5.0),
        fk_loss=dict(weight=7.0, stats_path=str(stats)),
    ))
    batch = _batch()
    batch.update({
        'active_arm': torch.ones(2, 2, dtype=torch.bool),
        'fk_action_mean': torch.zeros(2, 14),
        'fk_action_std': torch.ones(2, 14),
        'fk_action_epsilon': torch.full((2,), 1e-8),
    })
    output = head(batch, generator=torch.Generator().manual_seed(9))
    expected = (
        2.0 * (3.0 * output['loss_action_token_velocity']
               + 5.0 * output['loss_action_mean_clean'])
        + 7.0 * output['loss_fk'])
    torch.testing.assert_close(output['loss'], expected)


def test_runner_collects_all_fk_diagnostics():
    from fluxvla.engines.runners.base_train_runner import BaseTrainRunner

    metrics = BaseTrainRunner._collect_output_loss_metrics({
        'loss': torch.zeros(()),
        'loss_fk': torch.ones(()),
        'loss_fk_tcp_error_m': torch.full((), 0.01),
        'loss_fk_tcp_error_mean_action_m': torch.full((), 0.02),
        'loss_fk_active_arm_timestep_pairs': torch.full((), 16.0),
        'loss_fk_active_arm_timestep_copy_tuples': torch.full((), 144.0),
    })
    assert set(metrics) == {
        'loss_fk', 'loss_fk_tcp_error_m',
        'loss_fk_tcp_error_mean_action_m',
        'loss_fk_active_arm_timestep_pairs',
        'loss_fk_active_arm_timestep_copy_tuples',
    }


def test_runner_batch_fp32_keys_are_validated():
    from fluxvla.engines.runners.base_train_runner import BaseTrainRunner

    assert BaseTrainRunner._normalize_batch_fp32_keys(None) == frozenset()
    assert BaseTrainRunner._normalize_batch_fp32_keys(
        ['a', 'a', 'b']) == frozenset({'a', 'b'})
    with pytest.raises(TypeError, match='sequence'):
        BaseTrainRunner._normalize_batch_fp32_keys('fk_action_mean')
    with pytest.raises(TypeError, match='nonempty strings'):
        BaseTrainRunner._normalize_batch_fp32_keys([''])


@requires_fk
def test_runner_prepare_batch_preserves_fk_statistics_in_fp32(tmp_path):
    from fluxvla.engines.runners.base_train_runner import BaseTrainRunner
    from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner

    stats = _write_stats(tmp_path / 'stats.json')
    head = build_head_from_cfg(_tiny_head(
        video_weight=0.0,
        action_weight=0.0,
        action_loss=dict(token_velocity_weight=0.0,
                         mean_clean_weight=0.0),
        fk_loss=dict(weight=1.0, stats_path=str(stats)),
    ))
    batch = _batch()
    batch.update({
        'active_arm': torch.ones(2, 2, dtype=torch.bool),
        'fk_action_mean': torch.zeros(2, 14),
        'fk_action_std': torch.ones(2, 14),
        'fk_action_epsilon': torch.full((2,), 1e-8),
    })
    runner = object.__new__(DDPTrainRunner)
    runner.batch_fp32_keys = BaseTrainRunner._normalize_batch_fp32_keys([
        'fk_action_mean', 'fk_action_std', 'fk_action_epsilon'])
    prepared = BaseTrainRunner._prepare_batch(
        runner, batch, torch.device('cpu'), torch.bfloat16)
    assert prepared['action'].dtype == torch.bfloat16
    assert prepared['proprio'].dtype == torch.bfloat16
    for key in ('fk_action_mean', 'fk_action_std', 'fk_action_epsilon'):
        assert prepared[key].dtype == torch.float32
    assert torch.equal(
        prepared['fk_action_epsilon'],
        torch.full((2,), 1e-8))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = head(
            prepared, generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(output['loss_fk']).all()
    # CPU DNNL does not provide a bf16 backward on this host. The FK-only
    # gradient path is covered by the FP32 tiny-network test above.


def test_head_fk_disabled_is_inert(tmp_path):
    plain = build_head_from_cfg(_tiny_head())
    zero = build_head_from_cfg(_tiny_head(
        fk_loss=dict(weight=0.0,
                     stats_path=str(tmp_path / 'missing.json'))))
    assert set(plain.state_dict()) == set(zero.state_dict())
    output = zero(_batch(), generator=torch.Generator().manual_seed(4))
    assert 'loss_fk' not in output


@requires_fk
def test_head_fk_enabled_requires_assets_and_metadata(tmp_path):
    enabled = build_head_from_cfg(_tiny_head(
        video_weight=0.0,
        action_loss=dict(token_velocity_weight=0.0,
                         mean_clean_weight=0.0),
        fk_loss=dict(weight=1.0, stats_path=str(tmp_path / 'missing.json'))))
    with pytest.raises(FileNotFoundError, match='FK pose statistics'):
        batch = _batch()
        batch.update({
            'active_arm': torch.ones(2, 2, dtype=torch.bool),
            'fk_action_mean': torch.zeros(2, 14),
            'fk_action_std': torch.ones(2, 14),
        })
        enabled(batch, generator=torch.Generator().manual_seed(4))
    stats = _write_stats(tmp_path / 'stats.json')
    enabled = build_head_from_cfg(_tiny_head(
        video_weight=0.0,
        action_loss=dict(token_velocity_weight=0.0,
                         mean_clean_weight=0.0),
        fk_loss=dict(weight=1.0, stats_path=str(stats))))
    with pytest.raises(ValueError, match='missing'):
        enabled(_batch(), generator=torch.Generator().manual_seed(4))


def test_teacher_forcing_rejects_nondefault_objectives():
    with pytest.raises(ValueError, match='Separate PatchWAM'):
        build_head_from_cfg(dict(
            _tiny_head(
                type='TeacherForcingPatchWAMHead',
                teacher_forcing=dict(
                    encoders=[dict(
                        name='radio', feature_dim=8,
                        representation_id='fake',
                        checkpoint_sha256='e' * 64,
                        stream='spatial_features',
                        preprocessing='unit_range_once',
                        input_size=(8, 12), native_grid=(2, 3),
                        loss_weight=0.1)],
                    predictor=dict(type='linear')),
                action_loss=dict(token_velocity_weight=1.0,
                                 mean_clean_weight=0.5)),
        ))


@requires_fk
def test_vla_forward_passes_fk_metadata(tmp_path):
    stats = _write_stats(tmp_path / 'stats.json')
    vla = build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        action_horizon=16,
        num_inference_steps=2,
        vla_head=_tiny_head(
            video_weight=0.0,
            action_weight=0.0,
            action_loss=dict(token_velocity_weight=0.0,
                             mean_clean_weight=0.0),
            fk_loss=dict(weight=1.0, stats_path=str(stats))),
    ))
    batch = _batch()
    output = vla(
        reference_tokens=batch['reference_tokens'],
        future_tokens=batch['future_tokens'],
        text_tokens=batch['text_tokens'],
        actions=batch['action'],
        states=batch['proprio'],
        action_masks=torch.ones(2, 16, dtype=torch.bool),
        active_arm=torch.tensor([[True, True], [True, False]]),
        fk_action_mean=torch.zeros(2, 14),
        fk_action_std=torch.ones(2, 14),
        fk_action_epsilon=torch.full((2,), 1e-8),
    )
    assert 'loss_fk' in output and torch.isfinite(output['loss'])
    output['loss'].backward()


def test_enabled_fk_does_not_touch_training_assets_at_inference(tmp_path):
    vla = build_vla_from_cfg(dict(
        type='PatchWAMVLA',
        action_horizon=2,
        num_inference_steps=1,
        vla_head=_tiny_head(
            fk_loss=dict(weight=1.0,
                         stats_path=str(tmp_path / 'missing.json'))),
    ))
    batch = _batch(batch=1, horizon=2)
    prediction = vla.predict_action(
        reference_tokens=batch['reference_tokens'],
        text_tokens=batch['text_tokens'],
        states=batch['proprio'][:, 0],
        seed=3,
    )
    assert prediction.shape == (1, 2, 14)

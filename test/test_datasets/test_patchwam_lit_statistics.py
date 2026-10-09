# Copyright 2026 Limx Dynamics
"""Format/parity tests for native -> RoboTwin evaluation statistics."""

import json

import numpy as np
import pytest

from fluxvla.datasets.utils.statistics import (
    convert_robotwin_statistics_file, robotwin_statistics_from_native)
from fluxvla.engines.runners.robotwin_eval_runner import load_robotwin_stats
from fluxvla.transforms.normalize import NormalizeStatesAndActions

DIM = 14


def _native_payload(statistic_name='robotwin'):
    return {statistic_name: {
        'proprio': {
            'min': [0.0] * DIM,
            'max': [20.0] * DIM,
            'mean': [10.0] * DIM,
            'std': [5.0] * DIM,
            'q01': [1.0] * DIM,
            'q99': [19.0] * DIM,
        },
        'action': {
            'min': [-2.0] * DIM,
            'max': [2.0] * DIM,
            'mean': [0.5] * DIM,
            'std': [0.25] * DIM,
        },
        'timestamp': {'min': [0.0], 'max': [1.0], 'mean': [0.5],
                      'std': [0.25]},
    }}


def _released_payload():
    return {
        'state': {'default': {'global_mean': [10.0] * DIM,
                              'global_std': [5.0] * DIM}},
        'action': {'default': {'global_mean': [0.5] * DIM,
                               'global_std': [0.25] * DIM}},
    }


def test_native_payload_converts_to_released_schema():
    converted = robotwin_statistics_from_native(_native_payload())
    assert sorted(converted) == ['action', 'state']
    assert converted['state']['default']['global_mean'] == [10.0] * DIM
    assert converted['state']['default']['global_std'] == [5.0] * DIM
    assert converted['action']['default']['global_mean'] == [0.5] * DIM
    assert converted['action']['default']['global_std'] == [0.25] * DIM
    explicit = robotwin_statistics_from_native(
        _native_payload(), statistic_name='robotwin')
    assert explicit == converted


def test_loader_accepts_native_and_matches_released_codecs(tmp_path):
    native_path = tmp_path / 'dataset_statistics.json'
    native_path.write_text(json.dumps(_native_payload()), encoding='utf-8')
    released_path = tmp_path / 'dataset_stats.json'
    released_path.write_text(json.dumps(_released_payload()), encoding='utf-8')
    native_state, native_action = load_robotwin_stats(str(native_path))
    released_state, released_action = load_robotwin_stats(str(released_path))
    for native, released in ((native_state, released_state),
                             (native_action, released_action)):
        np.testing.assert_allclose(native.mean, released.mean)
        np.testing.assert_allclose(native.std, released.std)
        np.testing.assert_allclose(native.scale, released.scale)
        np.testing.assert_allclose(native.offset, released.offset)
    sample = np.full(DIM, 12.5)
    np.testing.assert_allclose(native_state.normalize(sample),
                               released_state.normalize(sample))


def test_eval_codec_matches_the_training_transform(tmp_path):
    native = _native_payload()
    native_path = tmp_path / 'dataset_statistics.json'
    native_path.write_text(json.dumps(native), encoding='utf-8')
    state_codec, _ = load_robotwin_stats(str(native_path))
    train_transform = NormalizeStatesAndActions(
        state_key='proprio', action_key='action', norm_type='mean_std',
        normalization_epsilon=1e-8, statistics_key='stats')
    raw = np.linspace(-5.0, 25.0, DIM, dtype=np.float32)
    out = train_transform({
        'states': raw.copy(), 'stats': {'proprio': native['robotwin']['proprio'],
                                        'action': native['robotwin']['action']},
    })
    expected = (raw - 10.0) / (5.0 + 1e-8)
    np.testing.assert_allclose(out['states'], expected, rtol=1e-5)
    np.testing.assert_allclose(state_codec.normalize(raw), expected, rtol=1e-5)


def test_loader_rejects_incomplete_native_payloads(tmp_path):
    broken = _native_payload()
    del broken['robotwin']['proprio']
    path = tmp_path / 'broken.json'
    path.write_text(json.dumps(broken), encoding='utf-8')
    with pytest.raises(KeyError, match='proprio'):
        load_robotwin_stats(str(path))
    ambiguous = {'run_a': _native_payload()['robotwin'],
                 'run_b': _native_payload()['robotwin']}
    path.write_text(json.dumps(ambiguous), encoding='utf-8')
    with pytest.raises(KeyError, match='statistic_name'):
        load_robotwin_stats(str(path))
    # The released schema keeps its original validation behavior.
    path.write_text(json.dumps({'state': {'default': {}}}), encoding='utf-8')
    with pytest.raises(KeyError):
        load_robotwin_stats(str(path))


def test_conversion_file_command_round_trip(tmp_path):
    native_path = tmp_path / 'dataset_statistics.json'
    native_path.write_text(json.dumps(_native_payload()), encoding='utf-8')
    out_path = tmp_path / 'robotwin_dataset_stats.json'
    converted = convert_robotwin_statistics_file(
        str(native_path), str(out_path), statistic_name='robotwin')
    on_disk = json.loads(out_path.read_text(encoding='utf-8'))
    assert on_disk == converted
    state, _ = load_robotwin_stats(str(out_path))
    np.testing.assert_allclose(state.mean, [10.0] * DIM)
    np.testing.assert_allclose(state.std, [5.0] * DIM)

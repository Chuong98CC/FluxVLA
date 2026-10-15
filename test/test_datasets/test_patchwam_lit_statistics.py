# Copyright 2026 Limx Dynamics
"""Format/parity tests for native -> released RoboTwin statistics."""

import json

import numpy as np
import pytest

from fluxvla.datasets.utils.statistics import (
    convert_robotwin_statistics_file, robotwin_statistics_from_native)
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


def test_conversion_carries_the_training_mean_and_std():
    native = _native_payload()
    converted = robotwin_statistics_from_native(native)
    np.testing.assert_allclose(converted['state']['default']['global_mean'],
                               native['robotwin']['proprio']['mean'])
    np.testing.assert_allclose(converted['state']['default']['global_std'],
                               native['robotwin']['proprio']['std'])
    np.testing.assert_allclose(converted['action']['default']['global_mean'],
                               native['robotwin']['action']['mean'])
    np.testing.assert_allclose(converted['action']['default']['global_std'],
                               native['robotwin']['action']['std'])


def test_training_transform_uses_the_converted_statistics():
    native = _native_payload()
    train_transform = NormalizeStatesAndActions(
        state_key='proprio', action_key='action', norm_type='mean_std',
        normalization_epsilon=1e-8, statistics_key='stats')
    raw = np.linspace(-5.0, 25.0, DIM, dtype=np.float32)
    out = train_transform(
        {
            'states': raw.copy(),
            'stats': {
                'proprio': native['robotwin']['proprio'],
                'action': native['robotwin']['action'],
            },
        })
    expected = (raw - 10.0) / (5.0 + 1e-8)
    np.testing.assert_allclose(out['states'], expected, rtol=1e-5)


def test_conversion_rejects_incomplete_native_payloads():
    broken = _native_payload()
    del broken['robotwin']['proprio']
    with pytest.raises(KeyError, match='proprio'):
        robotwin_statistics_from_native(broken)
    ambiguous = {'run_a': _native_payload()['robotwin'],
                 'run_b': _native_payload()['robotwin']}
    with pytest.raises(KeyError, match='statistic_name'):
        robotwin_statistics_from_native(ambiguous)
    with pytest.raises(KeyError, match='missing-group'):
        robotwin_statistics_from_native(ambiguous,
                                        statistic_name='missing-group')


def test_conversion_file_command_round_trip(tmp_path):
    native_path = tmp_path / 'dataset_statistics.json'
    native_path.write_text(json.dumps(_native_payload()), encoding='utf-8')
    out_path = tmp_path / 'robotwin_dataset_stats.json'
    converted = convert_robotwin_statistics_file(
        str(native_path), str(out_path), statistic_name='robotwin')
    on_disk = json.loads(out_path.read_text(encoding='utf-8'))
    assert on_disk == converted
    assert on_disk['state']['default']['global_mean'] == [10.0] * DIM
    assert on_disk['state']['default']['global_std'] == [5.0] * DIM

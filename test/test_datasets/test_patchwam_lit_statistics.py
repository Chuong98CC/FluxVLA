# Copyright 2026 Limx Dynamics
"""Format/parity tests for native -> released RoboTwin statistics."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fluxvla.datasets.utils.statistics import (
    convert_released_robotwin_statistics_file, convert_robotwin_statistics_file,
    released_robotwin_statistics_to_native, robotwin_statistics_from_native)
from fluxvla.transforms.normalize import NormalizeStatesAndActions

DIM = 14


def _released_payload():
    return {
        'state': {'default': {
            'global_mean': [10.0] * DIM,
            'global_std': [5.0] * DIM,
            'stepwise_mean': [[10.0] * DIM],
        }},
        'action': {'default': {
            'global_mean': [0.5] * DIM,
            'global_std': [0.25] * DIM,
        }},
        'num_episodes': 2,
    }


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


def test_released_payload_converts_to_native_schema():
    converted = released_robotwin_statistics_to_native(_released_payload())
    assert list(converted) == ['robotwin']
    group = converted['robotwin']
    assert sorted(group) == ['action', 'proprio']
    assert group['proprio']['mean'] == [10.0] * DIM
    assert group['proprio']['std'] == [5.0] * DIM
    assert group['action']['mean'] == [0.5] * DIM
    assert group['action']['std'] == [0.25] * DIM


def test_released_to_native_round_trips_through_the_released_schema():
    released = _released_payload()
    native = released_robotwin_statistics_to_native(released)
    # The existing util converts the other direction; the global z-score fields
    # must survive a full round trip unchanged.
    back = robotwin_statistics_from_native(
        {name: group for name, group in native.items()})
    assert back == {
        'state': {'default': {'global_mean': [10.0] * DIM,
                              'global_std': [5.0] * DIM}},
        'action': {'default': {'global_mean': [0.5] * DIM,
                               'global_std': [0.25] * DIM}},
    }


def test_released_conversion_drives_the_training_normalizer():
    native = released_robotwin_statistics_to_native(_released_payload())
    transform = NormalizeStatesAndActions(
        state_key='proprio', action_key=None, norm_type='mean_std',
        statistics_key='norm_stats', normalization_epsilon=1e-8,
        output_dtype='float32')
    raw = np.linspace(-5.0, 25.0, DIM, dtype=np.float32)
    out = transform({'states': raw.copy(), 'norm_stats': native['robotwin']})
    np.testing.assert_allclose(out['states'], (raw - 10.0) / (5.0 + 1e-8),
                               rtol=1e-6)


def test_released_conversion_rejects_incomplete_payloads():
    broken = _released_payload()
    del broken['state']['default']['global_std']
    with pytest.raises(KeyError, match='global_std'):
        released_robotwin_statistics_to_native(broken)
    mismatched = _released_payload()
    mismatched['action']['default']['global_std'] = [0.25] * (DIM - 1)
    with pytest.raises(ValueError, match='global_mean/global_std'):
        released_robotwin_statistics_to_native(mismatched)
    nonfinite = _released_payload()
    nonfinite['state']['default']['global_mean'][0] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        released_robotwin_statistics_to_native(nonfinite)
    with pytest.raises(KeyError):
        released_robotwin_statistics_to_native({'state': {}})


def test_released_conversion_file_round_trip(tmp_path):
    released_path = tmp_path / 'dataset_stats.json'
    released_path.write_text(json.dumps(_released_payload()), encoding='utf-8')
    out_path = tmp_path / 'dataset_statistics.json'
    converted = convert_released_robotwin_statistics_file(
        str(released_path), str(out_path))
    on_disk = json.loads(out_path.read_text(encoding='utf-8'))
    assert on_disk == converted
    assert on_disk['robotwin']['proprio']['mean'] == [10.0] * DIM
    # The released source file is never modified.
    assert json.loads(released_path.read_text(encoding='utf-8')) == \
        _released_payload()


def test_released_conversion_refuses_to_overwrite_its_source(tmp_path):
    released = tmp_path / 'dataset_stats.json'
    released.write_text(json.dumps(_released_payload()), encoding='utf-8')
    original = released.read_bytes()
    with pytest.raises(ValueError, match='onto itself'):
        convert_released_robotwin_statistics_file(str(released), str(released))
    assert released.read_bytes() == original
    for alias in (tmp_path / 'alias_link.json', tmp_path / 'alias_hard.json'):
        if alias.name.endswith('link.json'):
            alias.symlink_to(released)
        else:
            os.link(released, alias)
        # A symlink resolves to the source path ("onto itself"); a hardlink is
        # caught by the samefile check ("alias").
        with pytest.raises(ValueError, match='onto itself|alias'):
            convert_released_robotwin_statistics_file(str(released),
                                                      str(alias))
        assert released.read_bytes() == original


def test_native_conversion_refuses_to_overwrite_its_source(tmp_path):
    native = tmp_path / 'dataset_statistics.json'
    native.write_text(json.dumps(_native_payload()), encoding='utf-8')
    original = native.read_bytes()
    with pytest.raises(ValueError, match='onto itself'):
        convert_robotwin_statistics_file(str(native), str(native))
    assert native.read_bytes() == original


def _load_converter_tool():
    path = (Path(__file__).resolve().parents[2] / 'tools' /
            'convert_patchwam_robotwin_released.py')
    spec = importlib.util.spec_from_file_location(
        'convert_patchwam_robotwin_released', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_conversion_refuses_to_overwrite_its_source(tmp_path):
    source = tmp_path / 'step_140000.pt'
    source.write_bytes(b'released-patchwam-checkpoint')
    original = source.read_bytes()
    tool = _load_converter_tool()
    # The guard runs before the config is read or the model is built, so the
    # dummy config path is never touched and no GPU is required.
    with pytest.raises(ValueError, match='onto itself'):
        tool.convert_checkpoint('missing/config.py', str(source), str(source),
                                'cpu')
    alias = tmp_path / 'step_alias.pt'
    alias.symlink_to(source)
    with pytest.raises(ValueError, match='onto itself|alias'):
        tool.convert_checkpoint('missing/config.py', str(source), str(alias),
                                'cpu')
    assert source.read_bytes() == original

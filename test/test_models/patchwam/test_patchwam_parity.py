# Copyright 2026 Limx Dynamics
"""Source parity for native normalization and inverse action decoding.

Compares the native ``NormalizeStatesAndActions`` / ``DenormalizeLiberoAction``
against the sibling PatchWAM ``RobotFeatureCodec`` / ``FeatureScaler`` on the
same fixture statistics.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from fluxvla.transforms.normalize import (DenormalizeLiberoAction,
                                          NormalizeStatesAndActions)

SOURCE_DATA = Path('/home/chuong/workspace/PatchWAM/src/patchwam/data')
pytestmark = pytest.mark.skipif(
    not (SOURCE_DATA / 'scaling.py').is_file(),
    reason='sibling PatchWAM source is unavailable')


def _source_scaling():
    package = types.ModuleType('pwsrc_data')
    package.__path__ = [str(SOURCE_DATA)]
    sys.modules['pwsrc_data'] = package
    return importlib.import_module('pwsrc_data.scaling')


def _statistics():
    return {
        'states': {
            'min': [0.0] * 8,
            'max': [4.0] * 8,
            'mean': [2.0] * 8,
            'std': [1.0] * 8,
        },
        'action': {
            'min': [-1.0] * 7,
            'max': [1.0] * 7,
            'mean': [0.0] * 7,
            'std': [0.5] * 7,
        },
    }


def _shape_meta():
    return {
        'action': [{'key': 'default', 'shape': 7}],
        'state': [{'key': 'default', 'shape': 8}],
    }


@pytest.mark.parametrize('norm_type,source_mode', [('min_max', 'min/max'),
                                                   ('mean_std', 'z-score')])
def test_normalization_matches_source_codec(norm_type, source_mode):
    scaling = _source_scaling()
    stats = _statistics()
    codec = scaling.RobotFeatureCodec(
        _shape_meta(),
        {'action': {'default': stats['action']},
         'state': {'default': stats['states']}},
        mode=source_mode)
    states = np.array([0.5, 1.5, 2.5, 3.5, 1.0, 2.0, 3.0, 0.0],
                      dtype=np.float32)
    actions = np.array([-0.5, 0.5, 0.0, 0.25, -0.25, 1.0, -1.0],
                       dtype=np.float32)
    source = codec.encode({
        'action': {'default': torch.from_numpy(actions)},
        'state': {'default': torch.from_numpy(states)},
    })
    data = NormalizeStatesAndActions(
        state_key='states', action_key='action', action_dim=7, state_dim=8,
        norm_type=norm_type, statistics_key='stats',
        normalization_epsilon=1e-8)(
            {'states': states, 'actions': actions, 'stats': stats})
    torch.testing.assert_close(torch.from_numpy(data['states']).float(),
                               source['state']['default'].float(),
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.from_numpy(data['actions']).float(),
                               source['action']['default'].float(),
                               rtol=1e-6, atol=1e-6)


def test_full_horizon_inverse_decoding_matches_source_scaler():
    scaling = _source_scaling()
    stats = _statistics()
    scaler = scaling.FeatureScaler(stats['action'], mode='min/max')
    horizon = 16
    normalized = torch.linspace(-1.0, 1.0, horizon * 7).reshape(horizon, 7)
    decoder = DenormalizeLiberoAction(
        {'test': {'action': stats['action']}},
        norm_type='min_max', normalize_gripper_action=False,
        invert_gripper_action=False)
    decoded = decoder({'action': normalized.numpy(), 'norm_stats_key': 'test'})
    assert decoded.shape == (horizon, 7)
    expected = scaler.decode(normalized)
    np.testing.assert_allclose(decoded, expected, rtol=1e-6, atol=1e-6)

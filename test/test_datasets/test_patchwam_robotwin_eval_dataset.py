"""PatchWAM RoboTwin evaluation preprocessing through the canonical policy.

The RoboTwin evaluator is now the pull request's ``RobotwinEvalRunner`` with
its in-process ``LocalFluxVLAPolicy``. PatchWAM keeps its own observation
shaping at the dataset boundary; these tests drive the real canonical
``predict`` path with lightweight synthetic cameras, state and a fake model so
the observable batch and denormalized actions are checked rather than the
config text.
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from fluxvla.datasets.patchwam_robotwin_eval_dataset import (
    PatchWAMRoboTwinEvalDataset)
from fluxvla.engines import DATASETS
from fluxvla.engines.runners import robotwin_eval_runner as runner_module
from fluxvla.models.third_party_models.patchwam.encoding import (
    encode_observations)
from fluxvla.transforms.normalize import NormalizeStatesAndActions

DIM = 14
STATISTIC_NAME = 'robotwin'

_PROPRIO_MEAN = 1.0
_PROPRIO_STD = 2.0
_ACTION_MEAN = 0.5
_ACTION_STD = 0.25


def _stats(proprio_std=_PROPRIO_STD):
    return {STATISTIC_NAME: {
        'proprio': {
            'mean': [_PROPRIO_MEAN] * DIM,
            'std': [proprio_std] * DIM,
            'min': [0.0] * DIM,
            'max': [20.0] * DIM,
            'q01': [1.0] * DIM,
            'q99': [19.0] * DIM,
        },
        'action': {
            'mean': [_ACTION_MEAN] * DIM,
            'std': [_ACTION_STD] * DIM,
        },
    }}


def _write_stats(tmp_path, name='dataset_statistics.json', **stats_kwargs):
    path = tmp_path / name
    path.write_text(json.dumps(_stats(**stats_kwargs)), encoding='utf-8')
    return path


def _observation(value_head=0, value_left=64, value_right=192, qpos=None):
    return {
        'cam_high': np.full((240, 320, 3), value_head, dtype=np.uint8),
        'cam_left_wrist': np.full((240, 320, 3), value_left, dtype=np.uint8),
        'cam_right_wrist': np.full((240, 320, 3), value_right, dtype=np.uint8),
        'qpos': np.full(DIM, 5.0, dtype=np.float32) if qpos is None else qpos,
        'task_description': 'adjust the bottle',
    }


def _robotwin_observation(value_head=0, value_left=64, value_right=192,
                          qpos=None):
    """The raw RoboTwin observation ``_build_observation`` translates."""
    return {
        'observation': {
            'head_camera': {
                'rgb': np.full((240, 320, 3), value_head, dtype=np.uint8)},
            'left_camera': {
                'rgb': np.full((240, 320, 3), value_left, dtype=np.uint8)},
            'right_camera': {
                'rgb': np.full((240, 320, 3), value_right, dtype=np.uint8)},
        },
        'joint_action': {
            'vector': (np.full(DIM, 5.0, dtype=np.float32)
                       if qpos is None else np.asarray(qpos, dtype=np.float32)),
        },
    }


def _dataset(stats_path, device='cpu'):
    return PatchWAMRoboTwinEvalDataset(
        norm_stats=str(stats_path),
        statistic_name=STATISTIC_NAME,
        device=device,
        transforms=[
            dict(type='ProcessEvalInputs',
                 img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist']),
            dict(type='PatchWAMEvalFrames'),
            dict(type='PatchWAMCleanVL', image_size=448, head_index=0),
            dict(type='PatchWAMEvalPack', layout='robotwin',
                 video_size=[288, 256], camera_size=[240, 320],
                 robotwin_layout='compact_288x256', head_index=0),
            dict(type='NormalizeStatesAndActions', state_key='proprio',
                 action_key=None, norm_type='mean_std', state_dim=DIM,
                 statistics_key='norm_stats', normalization_epsilon=1e-8,
                 output_dtype='float32'),
        ])


def test_dataset_is_registered_under_its_config_name():
    assert DATASETS.get('PatchWAMRoboTwinEvalDataset') is \
        PatchWAMRoboTwinEvalDataset


def test_dataset_builds_the_compact_robotwin_mosaic(tmp_path):
    dataset = _dataset(_write_stats(tmp_path))
    batch = dataset(_observation())
    images = batch['images']
    assert tuple(images.shape) == (1, 3, 1, 288, 256)
    assert images.dtype == torch.float32
    assert float(images.min()) >= -1.0 and float(images.max()) <= 1.0
    # Head camera occupies the top band, wrist cameras the bottom left/right.
    head = 0.0 / 255.0 * 2.0 - 1.0
    left = 64.0 / 255.0 * 2.0 - 1.0
    right = 192.0 / 255.0 * 2.0 - 1.0
    assert images[0, 0, 0, 0, 0].item() == pytest.approx(head, abs=1e-5)
    assert images[0, 0, 0, -1, 0].item() == pytest.approx(left, abs=1e-5)
    assert images[0, 0, 0, -1, -1].item() == pytest.approx(right, abs=1e-5)
    assert batch['img_masks'].tolist() == [[True]]
    assert batch['task_description'] == ['adjust the bottle']


def test_dataset_keeps_the_raw_state_for_delta_actions(tmp_path):
    dataset = _dataset(_write_stats(tmp_path))
    qpos = np.linspace(-5.0, 25.0, DIM, dtype=np.float32)
    dataset(_observation(qpos=qpos))
    np.testing.assert_allclose(dataset.last_raw_state, qpos)


@pytest.mark.parametrize('proprio_std', [2.0, 0.0, 1e-9])
def test_state_normalization_matches_the_training_normalizer(tmp_path,
                                                            proprio_std):
    """Evaluation must reuse the trainer's epsilon, not a lookalike."""
    stats_path = _write_stats(
        tmp_path, name=f'stats_{proprio_std}.json', proprio_std=proprio_std)
    dataset = _dataset(stats_path)
    qpos = np.linspace(-5.0, 25.0, DIM, dtype=np.float32)
    batch = dataset(_observation(qpos=qpos))
    assert tuple(batch['states'].shape) == (1, DIM)

    trainer = NormalizeStatesAndActions(
        state_key='proprio', action_key='action', norm_type='mean_std',
        statistics_key='stats', normalization_epsilon=1e-8)
    expected = trainer({
        'states': qpos.copy(),
        'actions': np.zeros((1, DIM), dtype=np.float32),
        'stats': _stats(proprio_std=proprio_std)[STATISTIC_NAME],
    })['states']
    np.testing.assert_allclose(batch['states'].numpy()[0], expected,
                               rtol=1e-6)

    if proprio_std == 0.0:
        # The 1e-6 epsilon of the replaced StateFromInputs transform would be
        # off by 100x here; the shared trainer epsilon must be used.
        np.testing.assert_allclose(batch['states'].numpy()[0],
                                   (qpos - _PROPRIO_MEAN) / 1e-8, rtol=1e-6)


def test_dark_uint8_frames_are_not_mistaken_for_white(tmp_path):
    """A max-sample of 1 is a dark frame, not a normalized image."""
    dataset = _dataset(_write_stats(tmp_path))
    batch = dataset(_observation(value_head=1, value_left=1, value_right=1))
    dark = 1.0 / 255.0 * 2.0 - 1.0
    images = batch['images']
    assert float(images.max()) == pytest.approx(dark, abs=1e-4)
    assert float(images.max()) < 0.0


def test_dataset_emits_the_unaugmented_vl_current(tmp_path):
    dataset = _dataset(_write_stats(tmp_path))
    batch = dataset(_observation())
    current = batch['vl_current']
    assert tuple(current.shape) == (1, 3, 448, 448)
    # Head view is black in the fixture, so the clean view is -1 everywhere.
    assert float(current.min()) == pytest.approx(-1.0, abs=1e-6)
    assert float(current.max()) == pytest.approx(-1.0, abs=1e-6)
    assert tuple(batch['vl_history'].shape) == (1, 0, 3, 448, 448)
    assert batch['vl_history_valid'].shape == (1, 0)


class _FakeAutoencoder(nn.Module):
    """Tiny stand-in for the frozen FLUX VAE frontend."""

    def __init__(self, channels=8, pool=8):
        super().__init__()
        self.proj = nn.Conv2d(3, int(channels), 1)
        self.channels = int(channels)
        self.pool = int(pool)

    def encode(self, frames):
        return self.proj(torch.nn.functional.avg_pool2d(frames, self.pool))


class _RequiresImagesTextEncoder(nn.Module):
    """Encoder double that records the VL condition it is handed."""

    requires_images = True

    def __init__(self, tokens=4, dim=16):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.tokens = int(tokens)
        self.dim = int(dim)
        self.calls = []

    def forward(self, prompts, current=None, history=None,
                history_valid=None, subtasks=None):
        self.calls.append(
            dict(prompts=list(prompts), current=current, history=history,
                 history_valid=history_valid))
        return torch.zeros(len(prompts), self.tokens, self.dim)


def test_encode_observations_accepts_the_eval_batch(tmp_path):
    """The real encoder path needs the clean head view this adapter emits."""
    dataset = _dataset(_write_stats(tmp_path))
    batch = dataset(_observation())
    # ``PatchWAMVLA._prepare_batch`` promotes the instruction to ``prompt``
    # before the backbone runs; mirror that here.
    batch['prompt'] = batch['task_description']
    encoder = _RequiresImagesTextEncoder()
    encoded = encode_observations(
        batch,
        autoencoder=_FakeAutoencoder(),
        text_encoder=encoder,
        token_dim=8,
        text_dim=16,
        history_pool_size=4,
        target=False)
    # Without `vl_current` this call raises; reaching the encoder proves it.
    assert encoder.calls
    current = encoder.calls[0]['current']
    assert tuple(current.shape) == (1, 3, 448, 448)
    assert float(current.min()) == pytest.approx(-1.0, abs=1e-6)
    assert encoded['reference_tokens'].shape[0] == 1


class _FakeVLA(nn.Module):
    """Stand-in returning a fixed normalized action chunk."""

    def __init__(self, normalized_value=0.0, horizon=16):
        super().__init__()
        self.normalized_value = normalized_value
        self.horizon = horizon
        self.calls = 0
        self.batches = []

    def predict_action(self, **batch):
        self.calls += 1
        self.batches.append(batch)
        return torch.full((1, self.horizon, DIM), self.normalized_value)


def test_canonical_policy_denormalizes_patchwam_actions(tmp_path):
    from fluxvla.engines import build_transform_from_cfg

    stats_path = _write_stats(tmp_path)
    policy = object.__new__(runner_module.LocalFluxVLAPolicy)
    policy.dataset = _dataset(stats_path)
    policy.denormalize_action = build_transform_from_cfg(
        dict(type='DenormalizePrivateAction', norm_stats=str(stats_path),
             statistic_name=STATISTIC_NAME, norm_type='mean_std',
             action_dim=DIM))
    policy.unnorm_key = STATISTIC_NAME
    policy.device = torch.device('cpu')
    policy.eval_chunk_size = 10
    policy.mixed_precision_dtype = torch.bfloat16
    policy.enable_mixed_precision_training = False
    policy.max_episode_steps = None
    policy.vla = _FakeVLA(normalized_value=2.0)

    actions = policy.predict(_robotwin_observation(), 'adjust the bottle')

    # One canonical predict call, mosaic batch delivered to the model.
    assert policy.vla.calls == 1
    batch = policy.vla.batches[0]
    assert tuple(batch['images'].shape) == (1, 3, 1, 288, 256)
    assert tuple(batch['states'].shape) == (1, DIM)
    assert batch['task_description'] == ['adjust the bottle']
    # mean/std denormalization then truncation to eval_chunk_size.
    assert actions.shape == (10, DIM)
    np.testing.assert_allclose(actions,
                               2.0 * _ACTION_STD + _ACTION_MEAN, rtol=1e-5)

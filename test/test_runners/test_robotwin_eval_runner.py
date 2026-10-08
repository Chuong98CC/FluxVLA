# Copyright 2026 Limx Dynamics
"""Native RoboTwin evaluation runner tests (no simulator required)."""

import base64
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

from fluxvla.engines import HEADS, RUNNERS, VLAS, VLM_BACKBONES
from fluxvla.engines.runners.robotwin_eval_runner import (
    ROBOTWIN_MOSAIC_SIZE_HW, ROBOTWIN_PROMPT_TEMPLATE, RoboTwinEvalRunner,
    _FieldNormalizer, _SimBridge, compose_compact_mosaic,
    load_robotwin_stats)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stats():
    return {
        'state': {'default': {
            'global_mean': [0.5] * 14,
            'global_std': [0.25] * 14,
        }},
        'action': {'default': {
            'global_mean': [-0.25] * 14,
            'global_std': [0.5] * 14,
        }},
    }


def test_field_normalizer_matches_released_formula_and_clamps():
    codec = _FieldNormalizer(_stats()['state']['default'], 'state')
    value = np.array([0.5] * 14, dtype=np.float32)
    normalized = codec.normalize(value)
    assert np.allclose(normalized, 0.0, atol=1e-4)
    assert np.allclose(codec.denormalize(normalized), value, atol=1e-3)
    extreme = codec.normalize(np.array([1e6] * 14, dtype=np.float32))
    assert extreme.max() <= 5.0
    assert extreme.min() >= -5.0


def test_load_robotwin_stats_rejects_missing_fields(tmp_path):
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps({'state': {'default': {}}}))
    with pytest.raises(KeyError):
        load_robotwin_stats(str(path))


def test_compose_compact_mosaic_geometry():
    head = np.zeros((240, 320, 3), dtype=np.uint8)
    left = np.full((240, 320, 3), 64, dtype=np.uint8)
    right = np.full((240, 320, 3), 192, dtype=np.uint8)
    mosaic = compose_compact_mosaic(head, left, right)
    assert mosaic.shape == (*ROBOTWIN_MOSAIC_SIZE_HW, 3)
    assert mosaic.dtype == np.uint8
    # Head occupies the top half, wrist cameras the bottom-left/right.
    assert mosaic[0, 0].tolist() == [0, 0, 0]
    assert mosaic[-1, 0].tolist() == [64, 64, 64]
    assert mosaic[-1, -1].tolist() == [192, 192, 192]


def _raster(value=0):
    array = np.full((8, 8, 3), value, dtype=np.uint8)
    return {
        'shape': list(array.shape),
        'data': base64.b64encode(
            zlib.compress(array.tobytes())).decode('ascii'),
    }


def _observation(step, step_limit, success=False):
    return {
        'ok': True,
        'instruction': 'Adjust the bottle.',
        'seed': 1,
        'state': [0.0] * 14,
        'images': {
            'head_camera': _raster(0),
            'left_camera': _raster(32),
            'right_camera': _raster(64),
        },
        'success': success,
        'step': step,
        'step_limit': step_limit,
    }


class FakeBridge:
    """Deterministic stand-in for the simulator worker."""

    def __init__(self, retryable_resets=1, steps_to_success=5, step_limit=10):
        self.retryable_resets = retryable_resets
        self.steps_to_success = steps_to_success
        self.step_limit = step_limit
        self.resets = []
        self.steps = []

    def request_raw(self, payload):
        if payload['op'] == 'reset':
            self.resets.append(payload)
            if len(self.resets) <= self.retryable_resets:
                return {'ok': False, 'retryable': True, 'error': 'unstable'}
            return _observation(0, self.step_limit)
        raise AssertionError(payload)

    def request(self, payload):
        assert payload['op'] == 'step'
        self.steps.append(np.asarray(payload['action'], dtype=np.float32))
        success = len(self.steps) >= self.steps_to_success
        return _observation(len(self.steps), self.step_limit,
                            success=success) | {'done': success}


class FakeVLA(nn.Module):
    def __init__(self, action_dim=14):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.prompts = []

    def predict_action(self, *, images, states, task_description,
                       action_horizon, num_inference_steps, seed):
        self.calls += 1
        self.prompts.append(task_description[0])
        assert images.shape == (1, 3, *ROBOTWIN_MOSAIC_SIZE_HW)
        assert states.shape == (1, 14)
        assert action_horizon == 16
        assert num_inference_steps == 4
        return torch.full((1, action_horizon, 14), 0.5)


def _runner(tmp_path, stats_path, **overrides):
    runner = RoboTwinEvalRunner(cfg={'model': {'type': 'PatchWAMVLA'}},
                                **overrides)
    runner.bridge = FakeBridge()
    runner.vla = FakeVLA()
    state, action = load_robotwin_stats(str(stats_path))
    runner.state_normalizer, runner.action_normalizer = state, action
    runner.run_dir = tmp_path
    return runner


def _write_stats(tmp_path):
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps(_stats()))
    return path


def test_runner_retries_unstable_seeds_and_replans(tmp_path):
    stats = _write_stats(tmp_path)
    runner = _runner(tmp_path, stats, seed=7, dataset_stats_path=str(stats),
                     action_horizon=16, replan_steps=4,
                     num_inference_steps=4, episodes_per_task=1)
    runner.bridge = FakeBridge(retryable_resets=2, steps_to_success=5,
                               step_limit=10)
    runner.vla = FakeVLA()
    record = runner._run_episode(0)
    # Two rejected seeds, then the third candidate is used.
    assert record['seed_attempts'] == 3
    assert record['seed'] == 100000 * (1 + 7) + 2
    assert record['success'] is True
    # Five simulator steps; with replan=4 that needs two predictions.
    assert record['predictions'] == 2
    assert runner.vla.calls == 2
    assert all(prompt.startswith('A video recorded')
               for prompt in runner.vla.prompts)
    assert runner.vla.prompts[0] == ROBOTWIN_PROMPT_TEMPLATE.format(
        task='Adjust the bottle.')
    # One take_action per step, never batched.
    assert len(runner.bridge.steps) == 5


def test_runner_writes_results_json(tmp_path):
    stats = _write_stats(tmp_path)
    runner = _runner(tmp_path, stats, seed=0, dataset_stats_path=str(stats),
                     action_horizon=16, replan_steps=16,
                     num_inference_steps=4)
    runner.bridge = FakeBridge(retryable_resets=0, steps_to_success=2,
                               step_limit=4)
    runner.vla = FakeVLA()
    runner.run()
    payload = json.loads((tmp_path / 'results.json').read_text())
    assert payload['successes'] == 1
    assert payload['success_rate'] == 1.0
    assert payload['protocol']['replan_steps'] == 16
    assert payload['normalization']['action']['std'][0] == pytest.approx(0.5)


def test_runner_rejects_missing_prerequisites():
    runner = RoboTwinEvalRunner(cfg={'model': {}}, task_name='adjust_bottle')
    with pytest.raises(ValueError, match='robotwin_root'):
        runner.run_setup()


def test_runner_keeps_distinct_monotonic_seeds_across_episodes(tmp_path):
    stats = _write_stats(tmp_path)
    runner = _runner(tmp_path, stats, seed=42, dataset_stats_path=str(stats),
                     action_horizon=16, replan_steps=16,
                     num_inference_steps=4, episodes_per_task=3)
    runner.bridge = FakeBridge(retryable_resets=1, steps_to_success=1,
                               step_limit=2)
    runner.vla = FakeVLA()
    records = [runner._run_episode(index) for index in range(3)]
    seeds = [record['seed'] for record in records]
    assert len(set(seeds)) == len(seeds) == 3
    assert seeds == sorted(seeds)
    # The first rejected candidate is never retried by a later episode.
    assert seeds[0] == 100000 * (1 + 42) + 1
    assert seeds[1] == 100000 * (1 + 42) + 2
    assert records[0]['seed_candidates_start'] == 100000 * (1 + 42)
    assert records[0]['seed_attempts'] == 2
    assert records[1]['seed_attempts'] == 1


def test_runner_records_checkpoint_and_config_metadata(tmp_path):
    stats = _write_stats(tmp_path)
    runner = _runner(tmp_path, stats, dataset_stats_path=str(stats))
    assert runner._resolve_checkpoint_path(
        {'pretrained_name_or_path': '/models/step_140000.pt'}) == \
        '/models/step_140000.pt'

    class _Cfg:
        pretrained_name_or_path = '/models/other.pt'

    assert runner._resolve_checkpoint_path(_Cfg()) == '/models/other.pt'
    runner.ckpt_path = '/override/step_140000.pt'
    assert runner._resolve_checkpoint_path(_Cfg()) == \
        '/override/step_140000.pt'

    runner.resolved_checkpoint = '/models/step_140000.pt'
    runner.config_path = 'configs/patchwam/patchwam_robotwin_c2r_eval.py'
    runner._write_results([], 0)
    payload = json.loads((tmp_path / 'results.json').read_text())
    assert payload['checkpoint'] == '/models/step_140000.pt'
    assert payload['config_path'].endswith(
        'patchwam_robotwin_c2r_eval.py')
    assert payload['seeds']['initial'] == 100000 * (1 + runner.seed)
    assert 'monotonic cursor' in payload['protocol']['seed_scheme']


def test_bridge_times_out_and_terminates_hung_worker(tmp_path):
    script = tmp_path / 'hang.py'
    script.write_text('import time\ntime.sleep(120)\n')
    bridge = _SimBridge(sys.executable, str(script), str(tmp_path),
                        response_timeout=0.5)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match='did not respond'):
        bridge.request_raw({'op': 'reset'})
    assert time.monotonic() - start < 15
    bridge.close()
    assert bridge.process.poll() is not None


def test_robotwin_eval_config_wiring():
    cfg = Config.fromfile(str(
        REPO_ROOT / 'configs/patchwam/patchwam_robotwin_c2r_eval.py'))
    assert cfg.eval.type == 'RoboTwinEvalRunner'
    assert RUNNERS.get(cfg.eval.type) is RoboTwinEvalRunner
    assert VLAS.get(cfg.model.type) is not None
    assert VLM_BACKBONES.get(cfg.model.vlm_backbone.type) is not None
    assert HEADS.get(cfg.model.vla_head.denoiser.type) is not None
    assert cfg.model.vla_head.pack_proprio_after_text is True
    assert cfg.model.vlm_backbone.tokenizer_path.endswith('tokenizer')
    assert cfg.model.pretrained_name_or_path.endswith('step_140000.pt')

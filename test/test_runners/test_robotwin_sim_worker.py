# Copyright 2026 Limx Dynamics
"""RoboTwin simulator worker lifecycle tests (stubbed simulator, no sapien)."""

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPO_ROOT / 'tools' / 'robotwin_sim_worker.py'


@pytest.fixture()
def worker_module():
    spec = importlib.util.spec_from_file_location('robotwin_sim_worker',
                                                  WORKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubTask:
    """Records lifecycle calls for one fake RoboTwin rollout."""

    instances = []
    plan_success = True

    def __init__(self):
        self.closed = 0
        self.actions = []
        StubTask.instances.append(self)

    def setup_demo(self, **kwargs):
        pass

    def play_once(self):
        return {'info': {}}

    def check_success(self):
        return True

    def set_instruction(self, instruction=None):
        pass

    def take_action(self, action, action_type='qpos'):
        self.actions.append(np.asarray(action))

    def get_obs(self):
        return {
            'joint_action': {'vector': [0.0] * 14},
            'observation': {
                name: {'rgb': np.zeros((4, 4, 3), dtype=np.uint8)}
                for name in ('head_camera', 'left_camera', 'right_camera')
            },
        }

    def close_env(self, **_kwargs):
        self.closed += 1


@pytest.fixture()
def stubbed_worker(worker_module, monkeypatch):
    StubTask.instances = []
    namespace = type('StubEnvs', (), {'stub_task': StubTask})
    monkeypatch.setattr(worker_module.importlib, 'import_module',
                        lambda name: namespace)
    worker = worker_module.RoboTwinTask('/tmp/robotwin-root')
    monkeypatch.setattr(worker, '_runtime_arguments',
                        lambda config, overrides: ({}, 'stub'))
    return worker


def _reset(stubbed_worker, seed=1):
    return stubbed_worker.reset({
        'task': 'stub_task',
        'config': 'demo_clean',
        'seed': seed,
        'instruction': 'Adjust the bottle.',
        'expert_check': False,
    })


def test_reset_closes_previous_rollout(stubbed_worker):
    _reset(stubbed_worker, seed=1)
    first = stubbed_worker.task
    _reset(stubbed_worker, seed=2)
    second = stubbed_worker.task
    assert first is not second
    assert first.closed == 1
    assert second.closed == 0
    stubbed_worker.close()
    assert second.closed == 1
    assert stubbed_worker.task is None


def test_serve_closes_task_on_input_eof(worker_module, stubbed_worker,
                                        monkeypatch):
    stream_in = io.StringIO(json.dumps({
        'op': 'reset', 'task': 'stub_task', 'config': 'demo_clean',
        'seed': 5, 'instruction': 'Adjust the bottle.', 'expert_check': False,
    }) + '\n')
    stream_out = io.StringIO()
    monkeypatch.setattr(worker_module.importlib, 'import_module',
                        lambda name: type('StubEnvs', (), {
                            'stub_task': StubTask}))

    original_task = worker_module.RoboTwinTask

    def _make_worker(root):
        worker = original_task(root)
        worker._runtime_arguments = lambda config, overrides: ({}, 'stub')
        return worker

    monkeypatch.setattr(worker_module, 'RoboTwinTask', _make_worker)
    worker_module.serve(stream_in, stream_out, '/tmp/robotwin-root')
    responses = [json.loads(line) for line in stream_out.getvalue().splitlines()]
    assert responses and responses[0]['ok'] is True
    assert len(StubTask.instances) == 1
    # The stub rollout must be released when stdin closes.
    assert StubTask.instances[0].closed == 1


class _GeneratorStub:
    """Records the requested variant-pool size and returns fixed variants."""

    def __init__(self, variants):
        self.variants = list(variants)
        self.calls = []

    def __call__(self, task_name, episodes, max_descriptions):
        self.calls.append(max_descriptions)
        return [{'unseen': list(self.variants), 'seen': list(self.variants)}]


def _install_generator(worker_module, monkeypatch, variants):
    stub = _GeneratorStub(variants)
    module = types.ModuleType('generate_episode_instructions')
    module.generate_episode_descriptions = stub
    monkeypatch.setitem(sys.modules, 'generate_episode_instructions', module)
    worker = worker_module.RoboTwinTask('/tmp/robotwin-root')
    return worker, stub


def test_seeded_instruction_selection_is_deterministic(worker_module,
                                                       monkeypatch):
    worker, stub = _install_generator(worker_module, monkeypatch,
                                      ['a', 'b', 'c'])
    request = {
        'seed': 5,
        'instruction_type': 'unseen',
        'instruction_selection': 'seeded',
        'instruction_pool_size': 3,
    }
    first = worker._instruction(request, {})
    second = worker._instruction(request, {})
    assert first == second
    assert first in ('a', 'b', 'c')
    # The seeded path only ever asks the generator for a single variant.
    assert stub.calls == [1, 1]


def test_official_instruction_selection_uses_global_choice(worker_module,
                                                           monkeypatch):
    worker, stub = _install_generator(worker_module, monkeypatch,
                                      ['a', 'b', 'c'])
    request = {
        'seed': 5,
        'instruction_type': 'unseen',
        'instruction_selection': 'official',
        'instruction_pool_size': 7,
    }
    np.random.seed(0)
    expected = str(np.random.choice(['a', 'b', 'c']))
    np.random.seed(0)
    assert worker._instruction(request, {}) == expected
    # Official sampling sizes the pool like the released ``test_num``.
    assert stub.calls == [7]


def test_unknown_instruction_selection_is_rejected(worker_module):
    worker = worker_module.RoboTwinTask('/tmp/robotwin-root')
    with pytest.raises(ValueError, match='instruction_selection'):
        worker._instruction({'seed': 1, 'instruction_selection': 'bogus'}, {})

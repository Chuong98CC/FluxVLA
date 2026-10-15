"""RoboTwin evaluation registry and config routing.

RoboTwin evaluation is the shared FluxVLA evaluator only: the pull request's
``RobotwinEvalRunner`` embeds RoboTwin's own harness in this process and
implements its ``eval_policy.py`` policy contract. The independently developed
FluxVLA runner and its subprocess simulator bridge are gone, so these tests pin
the single registration and the routing of every PatchWAM RoboTwin config.
"""

from pathlib import Path

import json
import pytest
from mmengine.config import Config

from fluxvla.engines import DATASETS, RUNNERS, TRANSFORMS
from fluxvla.engines.runners import robotwin_eval_runner as runner_module

REPO_ROOT = Path(__file__).resolve().parents[2]

PATCHWAM_ROBOTWIN_CONFIGS = (
    'configs/patchwam/patchwam_robotwin_eval.py',
    'configs/patchwam/patchwam_robotwin_official_benchmark.py',
    'configs/patchwam/patchwam_lit_eval_robotwin.py',
    'configs/patchwam/patchwam_lit_official_benchmark.py',
)

RELEASED_CONFIG = 'configs/patchwam/patchwam_robotwin_released_eval.py'


def test_only_the_shared_robotwin_runner_is_registered():
    assert RUNNERS.get('RobotwinEvalRunner') is \
        runner_module.RobotwinEvalRunner
    # The retired FluxVLA runner must not survive under any name.
    assert RUNNERS.get('RoboTwinEvalRunner') is None
    assert not hasattr(runner_module, 'RoboTwinEvalRunner')
    assert not hasattr(runner_module, '_SimBridge')


def test_robotwin_policy_entrypoints_are_module_level():
    # RoboTwin resolves the policy module and reads these attributes.
    for name in ('get_model', 'eval', 'reset_model', 'close_model'):
        assert callable(getattr(runner_module, name))


@pytest.mark.parametrize('config_path', PATCHWAM_ROBOTWIN_CONFIGS)
def test_patchwam_robotwin_configs_route_to_the_shared_runner(config_path):
    cfg = Config.fromfile(str(REPO_ROOT / config_path))
    assert cfg.eval.type == 'RobotwinEvalRunner'
    assert RUNNERS.get(cfg.eval.type) is runner_module.RobotwinEvalRunner
    # The canonical runner consumes these fields directly.
    for field in ('model_family', 'task_list', 'task_suite_name', 'dataset',
                  'denormalize_action'):
        assert field in cfg.eval, f'{config_path} is missing eval.{field}'
    assert DATASETS.get(cfg.eval.dataset.type) is not None
    assert TRANSFORMS.get(cfg.eval.denormalize_action.type) is not None


def test_patchwam_robotwin_eval_preprocessing_is_patchwam_owned():
    cfg = Config.fromfile(
        str(REPO_ROOT / 'configs/patchwam/patchwam_robotwin_eval.py'))
    assert cfg.eval.dataset.type == 'PatchWAMRoboTwinEvalDataset'
    steps = [step['type'] for step in cfg.eval.dataset.transforms]
    assert steps == ['ProcessEvalInputs', 'PatchWAMEvalFrames',
                     'PatchWAMCleanVL', 'PatchWAMEvalPack',
                     'NormalizeStatesAndActions']
    # Training-only augmentation must not appear in the evaluation chain.
    assert not any('Augment' in step for step in steps)
    state_step = cfg.eval.dataset.transforms[-1]
    # The trainer's own normalizer and epsilon are reused, not an equivalent.
    assert state_step['normalization_epsilon'] == 1e-8
    assert state_step['statistics_key'] == 'norm_stats'


def test_official_benchmark_configs_use_the_released_protocol():
    manifest = json.loads(
        (REPO_ROOT / 'configs/patchwam/'
         'patchwam_robotwin_official_benchmark_tasks.json').read_text(
             encoding='utf-8'))
    for config_path in (
            'configs/patchwam/patchwam_robotwin_official_benchmark.py',
            'configs/patchwam/patchwam_lit_official_benchmark.py'):
        cfg = Config.fromfile(str(REPO_ROOT / config_path))
        assert cfg.eval.seed == 0
        assert list(cfg.eval.task_suite_name) == ['clean', 'random']
        assert cfg.eval.num_trials_per_task == 100
        # The inlined list must match the recorded released task manifest.
        assert list(cfg.eval.task_list) == list(manifest['tasks'])


def test_released_config_defaults_to_a_real_task_list(monkeypatch):
    """A bare string would be one task with a space in its name."""
    monkeypatch.delenv('PATCHWAM_ROBOTWIN_TASKS', raising=False)
    monkeypatch.delenv('PATCHWAM_EPISODES', raising=False)
    monkeypatch.delenv('PATCHWAM_TASK_SUITE', raising=False)
    cfg = Config.fromfile(str(REPO_ROOT / RELEASED_CONFIG))
    assert list(cfg.eval.task_list) == ['adjust_bottle', 'rotate_qrcode']
    assert cfg.eval.task_suite_name == 'clean'
    assert cfg.eval.num_trials_per_task == 1


def test_released_config_task_list_follows_the_env_override(monkeypatch):
    monkeypatch.setenv('PATCHWAM_ROBOTWIN_TASKS',
                       'scan_object rotate_qrcode shake_bottle')
    monkeypatch.setenv('PATCHWAM_TASK_SUITE', 'random')
    monkeypatch.setenv('PATCHWAM_EPISODES', '3')
    cfg = Config.fromfile(str(REPO_ROOT / RELEASED_CONFIG))
    assert list(cfg.eval.task_list) == [
        'scan_object', 'rotate_qrcode', 'shake_bottle']
    assert cfg.eval.task_suite_name == 'random'
    assert cfg.eval.num_trials_per_task == 3


def test_released_config_is_routed_to_the_shared_runner(monkeypatch):
    monkeypatch.delenv('PATCHWAM_ROBOTWIN_TASKS', raising=False)
    cfg = Config.fromfile(str(REPO_ROOT / RELEASED_CONFIG))
    assert cfg.eval.type == 'RobotwinEvalRunner'
    assert RUNNERS.get(cfg.eval.type) is runner_module.RobotwinEvalRunner
    assert cfg.eval.dataset.type == 'PatchWAMRoboTwinEvalDataset'
    # The released checkpoint is text-only conditioned, not the Qwen3-VL head.
    assert cfg.model.vlm_backbone.text_encoder_kind == 'qwen3'
    assert cfg.model.vla_head.pack_proprio_after_text is True

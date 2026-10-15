"""Merge regression: both RoboTwin runner implementations must coexist.

PR 134 was an add/add conflict on
``fluxvla/engines/runners/robotwin_eval_runner.py``: dev1 added
``RoboTwinEvalRunner`` (policy in FluxVLA, simulator behind
``tools/robotwin_sim_worker.py``) and the pull request added
``RobotwinEvalRunner`` (RoboTwin's evaluation harness embedded in-process).

The merged ``runners/__init__.py`` imports both names and each config family
selects one of them, so resolving that conflict by keeping a single side
leaves the other family unbuildable. These tests fail on the old
wholesale-ours/theirs resolutions and pass only on the union.
"""

from pathlib import Path

from fluxvla.engines import RUNNERS
from fluxvla.engines.runners import robotwin_eval_runner as runner_module

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_both_robotwin_runners_are_registered():
    assert RUNNERS.get('RoboTwinEvalRunner') is \
        runner_module.RoboTwinEvalRunner
    assert RUNNERS.get('RobotwinEvalRunner') is \
        runner_module.RobotwinEvalRunner


def test_robotwin_policy_entrypoints_survive_the_merge():
    # RoboTwin resolves the policy module and reads these attributes.
    for name in ('get_model', 'eval', 'reset_model', 'close_model'):
        assert callable(getattr(runner_module, name))


def test_both_robotwin_config_families_resolve():
    from mmengine.config import Config

    native = Config.fromfile(
        str(REPO_ROOT / 'configs/patchwam/patchwam_robotwin_c2r_eval.py'))
    in_process = Config.fromfile(
        str(REPO_ROOT /
            'configs/fastwam/fastwam_robotwin_all_data_full_finetune.py'))
    native_type = native.eval.type
    in_process_type = in_process.eval.runner.type
    assert native_type == 'RoboTwinEvalRunner'
    assert in_process_type == 'RobotwinEvalRunner'
    assert RUNNERS.get(native_type) is not None
    assert RUNNERS.get(in_process_type) is not None

"""Check launcher arguments at the shared manager process boundary."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def launcher(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / 'my_scripts').mkdir()
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'bin').mkdir()
    bash = shutil.which('bash')
    assert bash is not None
    # Bound recursive self-launches without ever starting another shell chain.
    guarded_bash = tmp_path / 'bin' / 'bash'
    guarded_bash.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = "my_scripts/eval_patchwam.sh" ]; then\n'
        '  echo "recursive evaluation launcher invocation" >&2\n'
        '  exit 88\n'
        'fi\n'
        f'exec {shlex.quote(bash)} "$@"\n')
    guarded_bash.chmod(0o755)
    script = tmp_path / 'my_scripts' / 'eval_patchwam.sh'
    shutil.copyfile(repo / 'my_scripts' / 'eval_patchwam.sh', script)
    manager = tmp_path / 'scripts' / 'eval_robotwin_manager.sh'
    manager.write_text(
        f"exec {shlex.quote(sys.executable)} - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "print(json.dumps(dict(args=sys.argv[1:], "
        "tasks=os.environ.get('TASK_NAMES'), "
        "condition=os.environ.get('TASK_SUITE_NAME'), "
        "trials=os.environ.get('NUM_TRIALS_PER_TASK'), "
        "output=os.environ.get('OUTPUT_DIR'))))\n"
        "PY\n")
    config = tmp_path / 'eval config.py'
    config.write_text('eval = {}\n')
    env = os.environ.copy()
    env['PATH'] = str(tmp_path / 'bin') + os.pathsep + env.get('PATH', '')
    for name in ('PATCHWAM_CHECKPOINT', 'PATCHWAM_EVAL_CONFIG', 'TASK_NAMES',
                 'OUTPUT_DIR'):
        env.pop(name, None)

    def run(*args, extra_env=None):
        return subprocess.run(
            [bash, str(script), '--config', str(config), *args],
            env={**env, **(extra_env or {})},
            capture_output=True, text=True, check=False, timeout=5)

    return run


@pytest.mark.parametrize('task_args', [
    ['--task', 'rotate_qrcode', '--task', 'scan_object'],
    ['--task', 'rotate_qrcode scan_object'],
])
def test_requested_tasks_reach_the_shared_manager(launcher, task_args):
    result = launcher('--ckpt', 'model checkpoint.pt', *task_args)
    assert result.returncode == 0, result.stderr
    dispatched = json.loads(result.stdout)
    assert dispatched['tasks'] == 'rotate_qrcode scan_object'
    assert dispatched['args'][1] == 'model checkpoint.pt'


def test_suite_root_sets_manager_output_without_splitting_arguments(launcher):
    output = 'runs/evaluation/new suite'
    result = launcher(
        '--ckpt', 'model.pt', '--suite-root', output,
        '--', '--cfg-options', 'eval.eval_chunk_size=10',
        extra_env={'OUTPUT_DIR': 'previous output'})
    assert result.returncode == 0, result.stderr
    dispatched = json.loads(result.stdout)
    assert dispatched['output'] == output
    assert dispatched['args'][2:] == [
        '--cfg-options', 'eval.eval_chunk_size=10']


def test_checkpoint_environment_default_is_forwarded(launcher):
    result = launcher(
        '--task', 'rotate_qrcode',
        extra_env={'PATCHWAM_CHECKPOINT': 'configured checkpoint.pt'})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['args'][1] == 'configured checkpoint.pt'


def test_help_does_not_launch_the_script_or_manager(launcher):
    result = launcher('--help')
    assert result.returncode == 0, result.stderr
    assert 'PatchWAM RoboTwin' in result.stdout
    assert not result.stderr


def test_clean_quick_check_dispatches_once_with_requested_trials(launcher):
    result = launcher(
        '--suite-root', 'runs/evaluation/patchwam_quick_check',
        '--task', 'rotate_qrcode', '--condition', 'clean', '--episodes', '10')
    assert result.returncode == 0, result.stderr
    dispatched = json.loads(result.stdout)
    assert dispatched['tasks'] == 'rotate_qrcode'
    assert dispatched['condition'] == 'clean'
    assert dispatched['trials'] == '10'
    assert dispatched['output'] == 'runs/evaluation/patchwam_quick_check'

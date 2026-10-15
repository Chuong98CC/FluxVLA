"""Worker-summary selection when a retry reuses the output directory.

A retry shares OUTPUT_DIR with earlier attempts, so the manager must pick the
attempt that belongs to the checkpoint it is about to evaluate. These tests run
the real ``worker_summary_path`` helper (extracted verbatim from the manager)
against fake summary trees; no simulator or GPU is involved.
"""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGER = REPO_ROOT / 'scripts' / 'eval_robotwin_manager.sh'
SUMMARIZER = REPO_ROOT / 'tools' / 'summarize_robotwin_eval_results.py'


def _extract_function(name: str) -> str:
    lines = MANAGER.read_text(encoding='utf-8').splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith(f'{name}() {{'))
    end = next(i for i in range(start, len(lines)) if lines[i] == '}')
    return '\n'.join(lines[start:end + 1])


def _write_summary(path: Path, successes: int, episodes: int,
                   checkpoint: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            'ckpt': checkpoint,
            'task_results': {
                'rotate_qrcode': {
                    'status': 'COMPLETED' if episodes else 'MISSING',
                    'successes': successes,
                    'total_episodes': episodes,
                    'success_rate': (successes / episodes * 100
                                     if episodes else None),
                }}}),
        encoding='utf-8')


def _run_selector(root: Path, tag: str, task: str = 'rotate_qrcode') -> str:
    script = '\n'.join([
        'set -euo pipefail',
        'declare -A JOB_ROOT',
        f'JOB_ROOT[{task}]={shlex.quote(str(root))}',
        f'ckpt_tag={shlex.quote(tag)}',
        _extract_function('worker_summary_path'),
        f'worker_summary_path {shlex.quote(task)}',
    ])
    # The manager helper runs under the same PATH contract as the manager
    # itself: `python` must resolve to the configured interpreter.
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(
        [os.path.dirname(sys.executable), env.get('PATH', '')])
    result = subprocess.run(['bash', '-c', script], capture_output=True,
                            text=True, check=False, timeout=60, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def attempt_tree(tmp_path):
    """Old-attempt tree (0 episodes) that is *newer* by mtime than the new one."""
    root = tmp_path / 'run'
    old = root / 'workers' / 'rotate_qrcode' / 'eval_runs' / 'step_140000' \
        / 'EVAL-old-attempt' / 'summary.json'
    new = root / 'workers' / 'rotate_qrcode' / 'eval_runs' \
        / 'step_140000_native' / 'EVAL-new-attempt' / 'summary.json'
    _write_summary(old, 0, 0)
    _write_summary(new, 3, 10)
    os.utime(new, (1_700_000_000, 1_700_000_000))
    os.utime(old, (1_800_000_000, 1_800_000_000))
    return root, old, new


def test_selector_ignores_a_newer_summary_from_another_checkpoint(attempt_tree):
    root, old, new = attempt_tree
    selected = _run_selector(root, 'step_140000_native')
    assert Path(selected) == new
    assert 'step_140000/' not in selected


def test_selector_returns_nothing_when_no_attempt_matches_the_checkpoint(
        attempt_tree):
    root, old, _ = attempt_tree
    assert _run_selector(root, 'a_different_checkpoint') == ''


def test_selector_prefers_the_newest_attempt_of_the_same_checkpoint(
        attempt_tree):
    root, _, new = attempt_tree
    older = new.parent.parent / 'EVAL-earlier-attempt' / 'summary.json'
    _write_summary(older, 1, 10)
    os.utime(older, (1_600_000_000, 1_600_000_000))
    assert Path(_run_selector(root, 'step_140000_native')) == new


def _load_summarizer():
    spec = importlib.util.spec_from_file_location('summarize_robotwin_eval',
                                                  SUMMARIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregator_picks_the_newest_attempt(tmp_path):
    root = tmp_path / 'run'
    older = root / 'workers' / 'rotate_qrcode' / 'eval_runs' / 'ckpt' \
        / 'EVAL-earlier' / 'summary.json'
    newer = root / 'workers' / 'rotate_qrcode' / 'eval_runs' / 'ckpt' \
        / 'EVAL-later' / 'summary.json'
    _write_summary(older, 0, 0)
    _write_summary(newer, 3, 10)
    os.utime(older, (1_600_000_000, 1_600_000_000))
    os.utime(newer, (1_700_000_000, 1_700_000_000))
    summaries = _load_summarizer()._load_worker_summaries(root)
    stats = summaries['rotate_qrcode']['task_results']['rotate_qrcode']
    assert stats['successes'] == 3
    assert stats['total_episodes'] == 10


def test_aggregator_ignores_a_newer_different_checkpoint(tmp_path):
    root = tmp_path / 'run'
    workers = root / 'workers' / 'rotate_qrcode' / 'eval_runs'
    current = workers / 'current' / 'attempt' / 'summary.json'
    other = workers / 'other' / 'attempt' / 'summary.json'
    checkpoint = str(tmp_path / 'current.safetensors')
    _write_summary(current, 3, 10, checkpoint)
    _write_summary(other, 9, 10, str(tmp_path / 'other.safetensors'))
    os.utime(current, (1_600_000_000, 1_600_000_000))
    os.utime(other, (1_800_000_000, 1_800_000_000))
    summaries = _load_summarizer()._load_worker_summaries(root, checkpoint)
    stats = summaries['rotate_qrcode']['task_results']['rotate_qrcode']
    assert stats['successes'] == 3
